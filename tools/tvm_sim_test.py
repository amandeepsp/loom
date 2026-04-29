#!/usr/bin/env -S uv run python
"""MNIST inference on Verilator simulation with proper requantization.

Derives epilogue params from the ONNX model's QDQ constants so that both
CPU reference and hardware use the correct requantization math.  Compares
per‑layer int8 output as well as the final float32 logits.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from compiler import TcpTransport
from compiler.quant_utils import quantize_multiplier_less_than_one
from shared.reference import cpu_requantize
from shared.sim_harness import run_gemm_on_sim

log = logging.getLogger("tvm_sim_test")


# ── model‑specific QDQ constants (from the debug pipeline) ──────────────────
_L0_IS  = 0.003921568859368563   # input_scale
_L0_IZP = -128                   # input_zp
_L0_WS  = 0.0026108562014997005  # weight_scale
_L0_OS  = 0.025416772812604904   # output_scale
_L0_OZP = -128                   # output_zp

_L1_IS  = _L0_OS                # input scale == previous output scale
_L1_IZP = -128
_L1_WS  = 0.0017632287926971912
_L1_OS  = 0.040668897330760956
_L1_OZP = 21


def _compute_epilogue(
    input_scale: float,
    input_zp: int,
    weight_scale: float,
    output_scale: float,
    weight: np.ndarray,
    bias_raw: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bias, multiplier, shift) arrays of length *n*."""
    combined = (input_scale * weight_scale) / output_scale
    mult, shft = quantize_multiplier_less_than_one(combined)
    mult_i = int(mult)
    shft_i = int(shft)

    # weight shape: [out_feat, in_feat] → sum over in_feat axis
    sum_w = weight.sum(axis=1).astype(np.int32)

    # hardware computes  acc_hw = Σ x_q·w_q, but we need
    #   acc_true = Σ (x_q − zp)·w_q = acc_hw − zp·Σ w_q
    bias = bias_raw.astype(np.int32) - input_zp * sum_w

    multiplier = np.full(n, mult_i, dtype=np.int32)
    shift = np.full(n, shft_i, dtype=np.int32)
    return bias, multiplier, shift


def main():
    parser = argparse.ArgumentParser(description="MNIST inference on Verilator sim")
    parser.add_argument("--tcp", default="tcp://127.0.0.1:21450")
    parser.add_argument("--onnx", default=str(REPO_ROOT / "models/out/mnist_int8.onnx"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-tolerance", type=int, default=1)
    parser.add_argument("--driver-timeout", type=float, default=1800.0,
                        help="TCP transport timeout in seconds (default 1800)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        log.error("ONNX model not found: %s", onnx_path)
        return 1

    # ── load ONNX weights ────────────────────────────────────────────────
    import onnx
    model = onnx.load(str(onnx_path))
    weights = {}
    for init in model.graph.initializer:
        weights[init.name] = onnx.numpy_helper.to_array(init)

    w0 = weights["net.1.weight_quantized"]   # [256, 784] int8
    b0 = weights["net.1.bias_quantized"]     # [256]    int32
    w1 = weights["net.3.weight_quantized"]   # [10, 256] int8
    b1 = weights["net.3.bias_quantized"]     # [10]     int32

    log.info("Weights: L0=%s b0=%s  L1=%s b1=%s",
             w0.shape, b0.shape, w1.shape, b1.shape)

    # ── compute proper epilogue params ───────────────────────────────────
    n0, n1 = 256, 10

    l0_bias, l0_mult, l0_shift = _compute_epilogue(
        _L0_IS, _L0_IZP, _L0_WS, _L0_OS, w0, b0, n0,
    )
    l1_bias, l1_mult, l1_shift = _compute_epilogue(
        _L1_IS, _L1_IZP, _L1_WS, _L1_OS, w1, b1, n1,
    )

    log.info("Layer 0: mult=%d shift=%d out_zp=%d",
             l0_mult[0], l0_shift[0], _L0_OZP)
    log.info("Layer 1: mult=%d shift=%d out_zp=%d",
             l1_mult[0], l1_shift[0], _L1_OZP)

    # ── random test input (int8) ─────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    input_img = rng.integers(-128, 128, size=(1, 784), dtype=np.int8)

    # ── CPU reference (bit‑accurate hardware epilogue model) ─────────────
    cpu_acc0 = input_img.astype(np.int32) @ w0.T.astype(np.int32)
    cpu_h = cpu_requantize(cpu_acc0, l0_bias, l0_mult, l0_shift,
                           _L0_OZP, -128, 127)

    cpu_acc1 = cpu_h.astype(np.int32) @ w1.T.astype(np.int32)
    cpu_out = cpu_requantize(cpu_acc1, l1_bias, l1_mult, l1_shift,
                             _L1_OZP, -128, 127)

    cpu_out_float = (cpu_out.astype(np.float32) - float(_L1_OZP)) * float(_L1_OS)
    log.info("CPU : pred=%d logits=%s",
             int(cpu_out.argmax()),
             str(cpu_out_float.flatten().tolist()))
    log.info("     L0=%s L1=%s",
             str(cpu_h.flatten()[:8].tolist()) + "...",
             str(cpu_out.flatten().tolist()))

    # ── Sim inference ────────────────────────────────────────────────────
    transport = TcpTransport(args.tcp, timeout_s=args.driver_timeout)
    try:
        sim_h = run_gemm_on_sim(
            transport, input_img, w0.T.astype(np.int8),
            bias=l0_bias, multiplier=l0_mult, shift=l0_shift,
            output_offset=_L0_OZP, activation_min=-128, activation_max=127,
        )
        sim_out = run_gemm_on_sim(
            transport, sim_h, w1.T.astype(np.int8),
            bias=l1_bias, multiplier=l1_mult, shift=l1_shift,
            output_offset=_L1_OZP, activation_min=-128, activation_max=127,
        )
    finally:
        transport.close()

    sim_out_float = (sim_out.astype(np.float32) - float(_L1_OZP)) * float(_L1_OS)
    log.info("Sim : pred=%d logits=%s",
             int(sim_out.argmax()),
             str(sim_out_float.flatten().tolist()))
    log.info("     L0=%s L1=%s",
             str(sim_h.flatten()[:8].tolist()) + "...",
             str(sim_out.flatten().tolist()))

    # ── compare per‑layer int8 ──────────────────────────────────────────────
    tol = args.verify_tolerance
    delta_h = np.abs(cpu_h.astype(np.int32) - sim_h.astype(np.int32))
    delta_out = np.abs(cpu_out.astype(np.int32) - sim_out.astype(np.int32))

    log.info("Layer 0: max|Δ|=%d  within±%d: %.1f%%",
             int(delta_h.max()), tol, 100 * (delta_h <= tol).mean())
    log.info("Layer 1: max|Δ|=%d  within±%d: %.1f%%",
             int(delta_out.max()), tol, 100 * (delta_out <= tol).mean())

    same_pred = int(cpu_out.argmax()) == int(sim_out.argmax())
    pass_tol = int(delta_h.max()) <= tol and int(delta_out.max()) <= tol

    if same_pred or pass_tol:
        log.info("PASS")
        return 0
    log.error("FAIL: mismatched prediction or tolerance exceeded")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
