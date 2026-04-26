#!/usr/bin/env -S uv run python
"""MNIST inference on Verilator simulation via the TVM runtime transport.

Loads the int8 ONNX model weights, runs both GEMM layers on the sim,
and compares against a CPU reference that models the hardware epilogue.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tvm"))

from runtime import TcpTransport, pack_weight_rows

log = logging.getLogger("tvm_sim_test")

# ---------------------------------------------------------------------------
# Epilogue reference (matches hardware: SRDHM + RDBPOT + clamp)
# ---------------------------------------------------------------------------

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
MEM_ALIGN = 32


def align_up(v, a=MEM_ALIGN):
    return (v + a - 1) & -a


def ref_srdhm(a, b):
    if a == INT32_MIN and b == INT32_MIN:
        return INT32_MAX
    ab = a * b
    nudge = (1 << 30) if ab >= 0 else (1 - (1 << 30))
    return max(INT32_MIN, min(INT32_MAX, (ab + nudge) >> 31))


def ref_rdbpot(x, exponent):
    if exponent == 0:
        return x
    mask = (1 << exponent) - 1
    remainder = x & mask
    threshold = (mask >> 1) + ((x >> 31) & 1)
    return (x >> exponent) + (1 if remainder > threshold else 0)


def cpu_requantize(acc, bias, multiplier, shift, output_offset, act_min, act_max):
    x = acc.astype(np.int64) + bias.astype(np.int64)
    out = np.zeros(acc.shape, dtype=np.int8)
    for r in range(acc.shape[0]):
        for c in range(acc.shape[1]):
            val = int(x[r, c])
            val = ref_srdhm(val, int(multiplier[c]))
            val = ref_rdbpot(val, int(shift[c]))
            val += output_offset
            out[r, c] = max(act_min, min(act_max, val))
    return out


# ---------------------------------------------------------------------------
# Full GEMM on sim (handles N-tiling and M-padding automatically)
# ---------------------------------------------------------------------------

def run_gemm_on_sim(transport, lhs_orig, rhs, *,
                    bias=None, multiplier=None, shift=None,
                    output_offset=0, activation_min=-128, activation_max=127,
                    tile=8, cfu_word_bits=64, cfu_store_depth_words=512,
                    tensor_pool_base=0x40010100):
    """Execute a full GEMM (any M,N) on the sim via the transport."""

    from shared.ir import build_pipelined_gemm_program, plan_memory

    m_orig, k = lhs_orig.shape
    _k2, n = rhs.shape
    assert k == _k2

    if bias is None:
        bias = np.zeros(n, dtype=np.int32)
    if multiplier is None:
        multiplier = np.ones(n, dtype=np.int32) * (1 << 30)
    if shift is None:
        shift = np.zeros(n, dtype=np.int32)

    # Zero-pad M to tile multiple (hardware requires full-width DMA rows)
    m = ((m_orig + tile - 1) // tile) * tile
    lhs = np.zeros((m, k), dtype=np.int8)
    lhs[:m_orig, :] = lhs_orig

    input_data = pack_input_tiles(lhs, tile)
    weight_data = pack_weight_rows(rhs)
    bias_data = bias.astype(np.int32).tobytes()
    mult_data = multiplier.astype(np.int32).tobytes()
    shift_data = shift.astype(np.int32).tobytes()
    output_size = m * n

    base = align_up(tensor_pool_base, MEM_ALIGN)
    input_addr = base
    weight_addr = input_addr + align_up(len(input_data))
    output_addr = weight_addr + align_up(len(weight_data))
    bias_addr = output_addr + align_up(output_size)
    mult_addr = bias_addr + align_up(n * 4)
    shift_addr = mult_addr + align_up(n * 4)

    layout = plan_memory(input_addr, weight_addr, output_addr,
                         bias_addr, mult_addr, shift_addr)
    program = build_pipelined_gemm_program(
        layout, m, k, n, tile,
        act_tensor_id=0, wgt_tensor_id=1, out_tensor_id=2,
        bias_id=3, mult_id=4, shift_id=5,
        cfu_word_bits=cfu_word_bits,
        cfu_store_depth_words=cfu_store_depth_words,
    )

    # Patch epilogue bytes in-place
    epi_bytes = bytearray(program)
    for i in range(104, len(epi_bytes) - 4):
        if epi_bytes[i] == 0x05:  # SET_EPILOGUE
            epi_bytes[i + 8] = output_offset & 0xFF
            epi_bytes[i + 9] = activation_min & 0xFF
            epi_bytes[i + 10] = activation_max & 0xFF
            break
    program = bytes(epi_bytes)

    transport.write_mem(input_addr, input_data)
    transport.write_mem(weight_addr, weight_data)
    transport.write_mem(bias_addr, bias_data)
    transport.write_mem(mult_addr, mult_data)
    transport.write_mem(shift_addr, shift_data)

    cycles = transport.exec_program(program)
    log.info("  GEMM %dx%dx%d: %d cycles", m_orig, k, n, cycles)

    out_bytes = transport.read_mem(output_addr, output_size)
    full = np.frombuffer(out_bytes, dtype=np.int8).reshape(m, n)
    return full[:m_orig, :]


def pack_input_tiles(matrix, tile):
    """Pack [M, K] into [K, tile] HW layout, zero-padding partial tiles."""
    _m, _k = matrix.shape
    chunks = []
    for m_base in range(0, _m, tile):
        ts = matrix[m_base: m_base + tile, :].T  # [K, tile_m]
        if ts.shape[1] < tile:
            pad = tile - ts.shape[1]
            ts = np.pad(ts, ((0, 0), (0, pad)), mode="constant")
        chunks.append(np.ascontiguousarray(ts))
    return np.concatenate(chunks).astype(np.int8).tobytes()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MNIST inference on Verilator sim")
    parser.add_argument("--tcp", default="tcp://127.0.0.1:21450")
    parser.add_argument("--onnx", default=str(REPO_ROOT / "models/out/mnist_int8.onnx"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-tolerance", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        log.error("ONNX model not found: %s", onnx_path)
        log.error("Run: uv run python -m models.mnist")
        return 1

    # Load weights
    import onnx
    model = onnx.load(str(onnx_path))
    weights = {}
    for init in model.graph.initializer:
        weights[init.name] = onnx.numpy_helper.to_array(init)

    w0 = weights["net.1.weight_quantized"]  # [256, 784] int8
    b0 = weights["net.1.bias_quantized"]    # [256] int32
    w1 = weights["net.3.weight_quantized"]  # [10, 256] int8
    b1 = weights["net.3.bias_quantized"]    # [10] int32

    log.info("Weights: L0=%s b0=%s  L1=%s b1=%s", w0.shape, b0.shape, w1.shape, b1.shape)

    # Random test input
    rng = np.random.default_rng(args.seed)
    input_img = rng.integers(-128, 128, size=(1, 784), dtype=np.int8)

    # --- CPU reference ---
    n0, n1 = 256, 10
    mult0 = np.ones(n0, dtype=np.int32) * (1 << 30)
    shift0 = np.zeros(n0, dtype=np.int32)
    cpu_acc0 = input_img.astype(np.int32) @ w0.T.astype(np.int32)
    cpu_h = cpu_requantize(cpu_acc0, b0, mult0, shift0, 0, 0, 127)

    mult1 = np.ones(n1, dtype=np.int32) * (1 << 30)
    shift1 = np.zeros(n1, dtype=np.int32)
    cpu_acc1 = cpu_h.astype(np.int32) @ w1.T.astype(np.int32)
    cpu_out = cpu_requantize(cpu_acc1, b1, mult1, shift1, 0, -128, 127)
    log.info("CPU: pred=%d logits=%s", int(cpu_out.argmax()),
             str(cpu_out.flatten().tolist()))

    # --- Sim inference ---
    transport = TcpTransport(args.tcp, timeout_s=600)
    try:
        sim_h = run_gemm_on_sim(
            transport, input_img, w0.T.astype(np.int8),
            bias=b0.astype(np.int32), multiplier=mult0, shift=shift0,
            output_offset=0, activation_min=0, activation_max=127,
        )

        sim_out = run_gemm_on_sim(
            transport, sim_h, w1.T.astype(np.int8),
            bias=b1.astype(np.int32), multiplier=mult1, shift=shift1,
            output_offset=0, activation_min=-128, activation_max=127,
        )
    finally:
        transport.close()

    log.info("Sim: pred=%d logits=%s", int(sim_out.argmax()),
             str(sim_out.flatten().tolist()))

    # Compare
    tol = args.verify_tolerance
    delta_h = np.abs(cpu_h.astype(np.int32) - sim_h.astype(np.int32))
    delta_out = np.abs(cpu_out.astype(np.int32) - sim_out.astype(np.int32))

    log.info("Layer 0: max|Δ|=%d, within±%d: %.1f%%",
             int(delta_h.max()), tol, 100 * (delta_h <= tol).mean())
    log.info("Layer 1: max|Δ|=%d, within±%d: %.1f%%",
             int(delta_out.max()), tol, 100 * (delta_out <= tol).mean())

    same_pred = int(cpu_out.argmax()) == int(sim_out.argmax())
    pass_tol = int(delta_h.max()) <= tol and int(delta_out.max()) <= tol

    if same_pred or pass_tol:
        log.info("PASS")
        return 0
    else:
        log.error("FAIL: mismatched prediction or tolerance exceeded")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
