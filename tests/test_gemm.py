#!/usr/bin/env -S uv run python
"""End-to-end GEMM on the board or Verilator sim, via a compiled `.lmb`.

    python -m tests.test_gemm /dev/ttyUSB1 --m 8 --k 384 --n 8
    python -m tests.test_gemm tcp://127.0.0.1:21450 --m 20 --k 100 --n 24

The GEMM is compiled with `compiler.compile_gemm` into the same model-binary
format real models use, so this exercises the production path (affine
K-tiling with DMA/compute overlap, M/N tails, epilogue) and reports on-chip
cycles without UART streaming in the measurement.

Epilogue parameters are chosen per column so outputs span the int8 range
(random bias, non-zero output offset, ReLU-style lower clamp); trivially
small scales would make nearly every output 0 and hide datapath bugs.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from compiler import compile_gemm
from compiler.backend.runtime import ModelBinaryArtifact, ModelBinaryRuntime
from compiler.quant_utils import quantize_multiplier_less_than_one
from shared.protocol import PerfCounters, SerialTransport, TcpTransport
from shared.reference import cpu_requantize

log = logging.getLogger("loom.e2e")

REPO_ROOT = Path(__file__).resolve().parents[1]
HW_CSR = REPO_ROOT / "build/sipeed_tang_nano_20k/csr.json"
SIM_CSR = REPO_ROOT / "build/sim/csr.json"
DEFAULT_CFU_WORD_BITS = int(os.environ.get("LOOM_CFU_WORD_BITS", "64"))
DEFAULT_CFU_STORE_DEPTH_WORDS = int(os.environ.get("LOOM_CFU_STORE_DEPTH_WORDS", "512"))


@dataclass(frozen=True)
class GemmCase:
    input: np.ndarray        # [m, k] int8
    weight: np.ndarray       # [k, n] int8
    bias: np.ndarray         # [n] int32
    multiplier: np.ndarray   # [n] int32
    shift: np.ndarray        # [n] int32, residual shift after SRDHM
    output_zero_point: int
    activation_min: int
    activation_max: int

    def expected(self) -> np.ndarray:
        acc = self.input.astype(np.int32) @ self.weight.astype(np.int32)
        return cpu_requantize(
            acc, self.bias, self.multiplier, self.shift,
            self.output_zero_point, self.activation_min, self.activation_max,
        )


@dataclass
class GemmResult:
    m: int
    k: int
    n: int
    cycles: int
    max_delta: int
    exact: int
    total: int
    counters: PerfCounters | None = None
    nonzero_outputs: int = 0
    lmb_bytes: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def gops(self) -> float:
        return 2 * self.m * self.k * self.n * 48e6 / self.cycles / 1e9


def make_case(m: int, k: int, n: int, seed: int = 0) -> GemmCase:
    rng = np.random.default_rng(seed)
    a = rng.integers(-128, 128, size=(m, k), dtype=np.int8)
    w = rng.integers(-128, 128, size=(k, n), dtype=np.int8)
    acc = a.astype(np.int64) @ w.astype(np.int64)
    bias = rng.integers(-2000, 2000, size=n).astype(np.int32)
    # Map roughly +/-3 sigma of each column onto +/-100 output steps.
    spread = np.maximum(acc.std(axis=0), 1.0) * 3
    multiplier = np.empty(n, dtype=np.int32)
    shift = np.empty(n, dtype=np.int32)
    for col in range(n):
        mult, tosa_shift = quantize_multiplier_less_than_one(min(100.0 / spread[col], 0.999))
        multiplier[col], shift[col] = int(mult), int(tosa_shift) - 31
    return GemmCase(a, w, bias, multiplier, shift,
                    output_zero_point=-10, activation_min=-60, activation_max=127)


def run_gemm(
    transport,
    m: int,
    k: int,
    n: int,
    *,
    seed: int = 0,
    csr_json: Path = HW_CSR,
    cfu_word_bits: int = DEFAULT_CFU_WORD_BITS,
    cfu_store_depth_words: int = DEFAULT_CFU_STORE_DEPTH_WORDS,
    read_counters: bool = True,
) -> GemmResult:
    """Compile, upload, and run one GEMM; return cycles + verification stats."""

    case = make_case(m, k, n, seed)
    compiled = compile_gemm(
        case.weight, m=m, bias=case.bias, multiplier=case.multiplier, shift=case.shift,
        output_zero_point=case.output_zero_point,
        activation_min=case.activation_min, activation_max=case.activation_max,
        tile=cfu_word_bits // 8, cfu_word_bits=cfu_word_bits,
        cfu_store_depth_words=cfu_store_depth_words, csr_json=csr_json,
    )
    artifact = ModelBinaryArtifact.from_bytes(compiled.to_bytes())
    runtime = ModelBinaryRuntime(artifact, transport)

    if read_counters:
        transport.perf_counters(reset=True)
    run = runtime.run_quantized(case.input)
    counters = transport.perf_counters() if read_counters else None

    expected = case.expected()
    delta = np.abs(run.output.astype(np.int32) - expected.astype(np.int32))
    if delta.any():
        rows, cols = np.nonzero(delta)
        tiles = sorted({(int(r) // 8, int(c) // 8) for r, c in zip(rows, cols)})
        log.info("mismatches at %d positions; (m_tile, n_tile) affected: %s; first: %s",
                 len(rows), tiles, list(zip(rows[:8].tolist(), cols[:8].tolist())))
    return GemmResult(
        m=m, k=k, n=n,
        cycles=run.cycles,
        max_delta=int(delta.max()),
        exact=int((delta == 0).sum()),
        total=int(delta.size),
        counters=counters,
        nonzero_outputs=int((expected != case.output_zero_point).sum()),
        lmb_bytes=len(artifact.blob),
    )


def run_explicit_gemm(
    transport,
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    bias: np.ndarray,
    multiplier: np.ndarray,
    shift: np.ndarray,
    output_offset: int = 0,
    activation_min: int = -128,
    activation_max: int = 127,
    csr_json: Path = HW_CSR,
) -> np.ndarray:
    """Run `lhs[m, k] @ rhs[k, n]` with caller-supplied epilogue params; return int8 [m, n]."""

    compiled = compile_gemm(
        rhs, m=lhs.shape[0], bias=bias, multiplier=multiplier, shift=shift,
        output_zero_point=output_offset,
        activation_min=activation_min, activation_max=activation_max, csr_json=csr_json,
    )
    runtime = ModelBinaryRuntime(ModelBinaryArtifact.from_bytes(compiled.to_bytes()), transport)
    return runtime.run_quantized(np.ascontiguousarray(lhs, dtype=np.int8)).output


def open_transport(port: str, timeout_s: float):
    if port.startswith("tcp://"):
        return TcpTransport(port, timeout_s=timeout_s)
    return SerialTransport(port, timeout_s=timeout_s)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Run a GEMM end-to-end on the board or sim.")
    parser.add_argument("port", help="serial device or tcp://host:port")
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csr-json", type=Path, default=None,
                        help="default: sim csr for tcp:// ports, hw csr otherwise")
    parser.add_argument("--cfu-word-bits", type=int, default=DEFAULT_CFU_WORD_BITS)
    parser.add_argument("--cfu-store-depth-words", type=int, default=DEFAULT_CFU_STORE_DEPTH_WORDS)
    parser.add_argument("--timeout", "--driver-timeout", dest="timeout", type=float, default=600)
    parser.add_argument("--verify-tolerance", type=int, default=0)
    args = parser.parse_args()

    csr = args.csr_json or (SIM_CSR if args.port.startswith("tcp://") else HW_CSR)
    transport = open_transport(args.port, args.timeout)
    try:
        r = run_gemm(
            transport, args.m, args.k, args.n, seed=args.seed, csr_json=csr,
            cfu_word_bits=args.cfu_word_bits, cfu_store_depth_words=args.cfu_store_depth_words,
        )
    finally:
        transport.close()

    log.info("M=%d K=%d N=%d  lmb=%d bytes", r.m, r.k, r.n, r.lmb_bytes)
    log.info("cycles=%d (%.3f GOPS on-chip)  counters=%s", r.cycles, r.gops, r.counters)
    log.info("verify: %d/%d exact, max|Δ|=%d, %d/%d outputs away from zero point",
             r.exact, r.total, r.max_delta, r.nonzero_outputs, r.total)
    if r.max_delta > args.verify_tolerance:
        log.error("FAILED: max|Δ|=%d > tolerance %d", r.max_delta, args.verify_tolerance)
        return 1
    log.info("PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
