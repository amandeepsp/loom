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
import subprocess
import tempfile

import numpy as np

from shared.ir import (
    build_non_pipelined_gemm_program,
    build_pipelined_gemm_program,
    plan_memory,
)

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


def align_up(value: int, alignment: int = MEM_ALIGN) -> int:
    return (value + alignment - 1) & -alignment


def pack_i8(arr: np.ndarray) -> bytes:
    return arr.astype(np.int8).tobytes()


def pack_i32(arr: np.ndarray) -> bytes:
    return arr.astype(np.int32).tobytes()


def pack_input_tiles(matrix: np.ndarray, tile: int) -> bytes:
    """Pack [M, K] activation matrix into [K, tile] scratchpad layout.

    The hardware expects [K, tile] column-major layout per M-tile.
    For each M-tile, we create K rows of tile values each.
    Layout: [K/tile chunks][K][tile] where K/tile chunks are for each M-tile.
    """
    m, k = matrix.shape
    chunks = []
    for m_base in range(0, m, tile):
        tile_slice = matrix[m_base : m_base + tile, :].T  # [K, tile]
        chunks.append(np.ascontiguousarray(tile_slice))
    return np.concatenate(chunks).astype(np.int8).tobytes()


def pack_weight_rows(matrix: np.ndarray) -> bytes:
    return np.ascontiguousarray(matrix).astype(np.int8).tobytes()


INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


def ref_srdhm(a: int, b: int) -> int:
    """Saturating Rounding Doubling High Multiply."""
    if a == INT32_MIN and b == INT32_MIN:
        return INT32_MAX
    ab = a * b
    nudge = (1 << 30) if ab >= 0 else (1 - (1 << 30))
    result = (ab + nudge) >> 31
    return max(INT32_MIN, min(INT32_MAX, result))


def ref_rdbpot(x: int, exponent: int) -> int:
    """Rounding Divide by Power of Two."""
    if exponent == 0:
        return x
    mask = (1 << exponent) - 1
    remainder = x & mask
    sign_bit = (x >> 31) & 1
    threshold = (mask >> 1) + sign_bit
    rounding = 1 if remainder > threshold else 0
    return (x >> exponent) + rounding


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


def write_blob(
    driver: pathlib.Path,
    port: str,
    addr: int,
    blob: bytes,
    suffix: str,
    *,
    timeout_s: float,
) -> None:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(blob)
        handle.flush()
        temp_path = pathlib.Path(handle.name)
    try:
        run(
            [str(driver), port, "write-file", hex(addr), str(temp_path)],
            timeout_s=timeout_s,
        )
    finally:
        temp_path.unlink(missing_ok=True)


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
    if not driver.is_file():
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

    log.info("uploading test data...")
    write_blob(driver, args.port, input_addr, input_data, ".input.bin", timeout_s=to)
    write_blob(driver, args.port, weight_addr, weight_data, ".wgt.bin", timeout_s=to)
    write_blob(driver, args.port, bias_addr, bias_data, ".bias.bin", timeout_s=to)
    write_blob(driver, args.port, mult_addr, mult_data, ".mult.bin", timeout_s=to)
    write_blob(driver, args.port, shift_addr, shift_data, ".shift.bin", timeout_s=to)

    if args.variant == "all":
        variants = ["non-pipelined", "pipelined"]
    else:
        variants = [args.variant]
    all_passed = True

    for variant in variants:
        log.info("=== %s ===", variant)

        if variant == "non-pipelined":
            program = build_non_pipelined_gemm_program(
                layout=layout,
                m=m,
                k=k,
                n=n,
                act_tensor_id=0,
                wgt_tensor_id=1,
                out_tensor_id=2,
                bias_id=3,
                mult_id=4,
                shift_id=5,
                tile=tile,
                k_tile=cfu_store_depth_words // (cfu_word_bits // 32),
            )
        else:
            program = build_pipelined_gemm_program(
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
            )

        log.debug("program (%d bytes): %s", len(program), program.hex())

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(program)
            f.flush()
            prog_path = f.name

        try:
            out = run([str(driver), args.port, "exec-bin", prog_path], timeout_s=to)
            if "exec-bin ok" not in out:
                log.error("%s: FAILED (exec error)", variant)
                all_passed = False
                continue

            if args.no_verify:
                log.info("%s: PASSED (no verify)", variant)
                continue

            with tempfile.NamedTemporaryFile(suffix=".out.bin", delete=False) as f:
                out_path = f.name

            run(
                [
                    str(driver),
                    args.port,
                    "read-file",
                    hex(output_addr),
                    str(output_size),
                    out_path,
                ],
                timeout_s=to,
            )
            actual = pathlib.Path(out_path).read_bytes()
            pathlib.Path(out_path).unlink(missing_ok=True)

            if verify_gemm_output(expected_bytes, actual, m, n, args.verify_tolerance):
                log.info("%s: PASSED", variant)
            else:
                all_passed = False
        finally:
            pathlib.Path(prog_path).unlink(missing_ok=True)

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
