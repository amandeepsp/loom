#!/usr/bin/env -S uv run python
"""GEMM end-to-end tests: non-pipelined and pipelined variants.

Usage:
    accel-e2e-gemm /dev/ttyUSB1 non-pipelined
    accel-e2e-gemm /dev/ttyUSB1 pipelined
    accel-e2e-gemm /dev/ttyUSB1 all
"""

import argparse
import logging
import os
import pathlib
import struct
import tempfile

import numpy as np

from shared.ir import (
    build_gemm_program,
    plan_memory,
)
from shared.layout import align_up, pack_input_tiles, pack_weight_rows
from shared.protocol import SerialTransport, TcpTransport
from shared.reference import INT32_MAX, INT32_MIN, ref_rdbpot, ref_srdhm

log = logging.getLogger("accel.e2e")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DRIVER_BIN = REPO_ROOT / "zig-out" / "bin" / "driver"
DEFAULT_CFU_WORD_BITS = int(os.environ.get("ACCEL_CFU_WORD_BITS", "64"))
DEFAULT_CFU_STORE_DEPTH_WORDS = int(
    os.environ.get("ACCEL_CFU_STORE_DEPTH_WORDS", "512")
)
DEFAULT_DRIVER_TIMEOUT_S = float(os.environ.get("ACCEL_DRIVER_TIMEOUT_S", "120"))
TENSOR_POOL_BASE = 0x40010000
MEM_ALIGN = 32


def run(cmd: list[str], *, timeout_s: float) -> str:
    log.debug("driver cmd: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        log.error("driver subprocess timed out after %ds: %s", timeout_s, " ".join(cmd))
        raise SystemExit(124) from exc
    if proc.returncode != 0:
        log.error("driver failed with code %d", proc.returncode)
        log.error("stderr: %s", proc.stderr)
        raise SystemExit(proc.returncode)
    return proc.stdout + proc.stderr


def pack_i8(arr: np.ndarray) -> bytes:
    return arr.astype(np.int8).tobytes()


def pack_i32(arr: np.ndarray) -> bytes:
    return arr.astype(np.int32).tobytes()


def compute_expected(
    input_matrix: np.ndarray,
    weight_matrix: np.ndarray,
    bias: np.ndarray,
    multiplier: np.ndarray,
    shift: np.ndarray,
) -> np.ndarray:
    """GEMM with SRDHM + RDBPOT epilogue."""
    acc = input_matrix.astype(np.int32) @ weight_matrix.astype(np.int32)
    m, n = acc.shape
    out = np.empty((m, n), dtype=np.int8)
    for col in range(n):
        b = int(bias[col])
        mult = int(multiplier[col])
        sh = int(shift[col])
        for row in range(m):
            val = int(acc[row, col]) + b
            val = ref_srdhm(val, mult)
            val = ref_rdbpot(val, sh)
            out[row, col] = max(-128, min(127, val))
    return out


def generate_input_matrix(m: int, k: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(-128, 128, size=(m, k), dtype=np.int8)


def generate_weight_matrix(k: int, n: int, seed: int = 137) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(-128, 128, size=(k, n), dtype=np.int8)


def generate_epilogue_params(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple params: bias=0, mult=256, shift=0."""
    bias = np.zeros(n, dtype=np.int32)
    multiplier = np.ones(n, dtype=np.int32) * 256
    shift = np.zeros(n, dtype=np.int32)
    return bias, multiplier, shift


def verify_gemm_output(
    expected: bytes,
    actual: bytes,
    m: int,
    n: int,
    tolerance: int,
) -> bool:
    """Verify GEMM output against expected values."""
    if len(expected) != len(actual):
        log.error(
            "output size mismatch: expected %d, got %d", len(expected), len(actual)
        )
        return False
    if tolerance < 0 or tolerance > 127:
        raise ValueError("verify tolerance must be in 0..127")

    exp_arr = np.frombuffer(expected, dtype=np.int8).astype(np.int32)
    act_arr = np.frombuffer(actual, dtype=np.int8).astype(np.int32)
    delta = np.abs(exp_arr - act_arr)

    exact = int(np.sum(delta == 0))
    off_by_one = int(np.sum(delta == 1))
    max_abs = int(delta.max())
    total = len(expected)

    log.info(
        "verify: %d/%d exact, %d cells with |Δ|=1, max |Δ|=%d, tolerance=%d",
        exact,
        total,
        off_by_one,
        max_abs,
        tolerance,
    )

    if max_abs <= tolerance:
        if max_abs == 0:
            log.info("output verification PASSED (exact match)")
        else:
            log.info("output verification PASSED (tolerance=%d)", tolerance)
        return True

    fail_mask = delta > tolerance
    fail_indices = np.where(fail_mask)[0].tolist()
    log.error("FAILED: %d/%d cells exceed tolerance", len(fail_indices), total)
    for i in fail_indices[:16]:
        row, col = divmod(i, n)
        es = int(exp_arr[i])
        ac = int(act_arr[i])
        log.info(
            "  mismatch (%d,%d): expected %d, got %d, |Δ|=%d",
            row,
            col,
            es,
            ac,
            abs(es - ac),
        )
    if len(fail_indices) > 16:
        log.info("  ... and %d more", len(fail_indices) - 16)
    if off_by_one > 0 and max_abs == 1:
        log.info("hint: failures are all |Δ|=1 — typical of rounding differences")

    return False


def open_transport(port: str, timeout_s: float):
    """Return a TcpTransport or SerialTransport depending on port format."""
    if port.startswith("tcp://"):
        return TcpTransport(port, timeout_s=timeout_s)
    return SerialTransport(port, baud_rate=115200, timeout_s=timeout_s)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Run GEMM end-to-end test against accel board.",
    )
    parser.add_argument("port", default="/dev/ttyUSB1")
    parser.add_argument(
        "variant",
        choices=["non-pipelined", "pipelined", "all"],
        default="all",
    )
    parser.add_argument("--driver", default=str(DRIVER_BIN))
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--cfu-word-bits", type=int, default=DEFAULT_CFU_WORD_BITS)
    parser.add_argument(
        "--cfu-store-depth-words", type=int, default=DEFAULT_CFU_STORE_DEPTH_WORDS
    )
    parser.add_argument(
        "--driver-timeout", type=float, default=DEFAULT_DRIVER_TIMEOUT_S
    )
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--verify-tolerance", type=int, default=0)
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.DEBUG if args.verbose else logging.INFO)

    driver = pathlib.Path(args.driver)
    if not args.port.startswith("tcp://") and not driver.is_file():
        log.error("missing driver: %s", driver)
        raise SystemExit(1)

    m, k, n = args.m, args.k, args.n
    tile = args.cfu_word_bits // 8
    cfu_word_bits = args.cfu_word_bits
    cfu_store_depth_words = args.cfu_store_depth_words

    log.info("variant=%s M=%d K=%d N=%d tile=%d", args.variant, m, k, n, tile)

    input_matrix = generate_input_matrix(m, k)
    weight_matrix = generate_weight_matrix(k, n)
    bias, multiplier, shift = generate_epilogue_params(n)

    input_data = pack_input_tiles(input_matrix, tile)
    weight_data = pack_weight_rows(weight_matrix)
    bias_data = pack_i32(bias)
    mult_data = pack_i32(multiplier)
    shift_data = pack_i32(shift)

    output_size = m * n
    input_addr = TENSOR_POOL_BASE + 256
    weight_addr = input_addr + len(input_data)
    output_addr = weight_addr + len(weight_data)
    bias_addr = output_addr + output_size
    mult_addr = bias_addr + n * 4
    shift_addr = mult_addr + n * 4

    layout = plan_memory(
        input_addr=input_addr,
        weight_addr=weight_addr,
        output_addr=output_addr,
        bias_addr=bias_addr,
        mult_addr=mult_addr,
        shift_addr=shift_addr,
    )

    expected = compute_expected(input_matrix, weight_matrix, bias, multiplier, shift)
    expected_bytes = np.ascontiguousarray(expected).astype(np.int8).tobytes()

    to = args.driver_timeout
    transport = open_transport(args.port, to)

    log.info("uploading test data...")
    transport.write_mem(input_addr, input_data)
    transport.write_mem(weight_addr, weight_data)
    transport.write_mem(bias_addr, bias_data)
    transport.write_mem(mult_addr, mult_data)
    transport.write_mem(shift_addr, shift_data)

    if args.variant == "all":
        variants = ["non-pipelined", "pipelined"]
    else:
        variants = [args.variant]
    all_passed = True

    for variant in variants:
        log.info("=== %s ===", variant)

        if variant == "non-pipelined":
            k_tile = cfu_store_depth_words // (cfu_word_bits // 32)
        else:
            k_tile = None

        program = build_gemm_program(
            layout=layout,
            m=m,
            k=k,
            n=n,
            tile=tile,
            act_tensor_id=0,
            wgt_tensor_id=1,
            out_tensor_id=2,
            bias_id=3,
            mult_id=4,
            shift_id=5,
            cfu_word_bits=cfu_word_bits,
            cfu_store_depth_words=cfu_store_depth_words,
            k_tile=k_tile,
        )

        log.debug("program (%d bytes): %s", len(program), program.hex())

        cycles = transport.exec_program(program)
        log.info("exec ok, cycles=%d", cycles)

        if args.no_verify:
            log.info("%s: PASSED (no verify)", variant)
            continue

        actual = transport.read_mem(output_addr, output_size)

        if verify_gemm_output(expected_bytes, actual, m, n, args.verify_tolerance):
            log.info("%s: PASSED", variant)
        else:
            all_passed = False

    transport.close()
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
