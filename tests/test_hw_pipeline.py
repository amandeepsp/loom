#!/usr/bin/env python
"""Run the compiled MNIST model binary against hardware or Verilator sim."""

from __future__ import annotations

import argparse
from pathlib import Path

from compiler import SerialTransport, TcpTransport

from tests.mnist_model_binary import DEFAULT_ONNX, run_mnist_model_binary

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile ONNX to a loom model binary and run it on hardware or sim.",
    )
    parser.add_argument("--port", default="/dev/ttyUSB1")
    parser.add_argument("--baud-rate", type=int, default=115200)
    parser.add_argument("--onnx", default=str(DEFAULT_ONNX))
    parser.add_argument("--csr-json", default=str(REPO_ROOT / "build/sipeed_tang_nano_20k/csr.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument("--model-addr", type=lambda text: int(text, 0), default=None)
    args = parser.parse_args()

    if args.port.startswith("tcp://"):
        transport = TcpTransport(args.port, timeout_s=1800)
        target_label = "sim"
    else:
        transport = SerialTransport(
            port=args.port,
            baud_rate=args.baud_rate,
            timeout_s=1800,
        )
        target_label = "hw"
    try:
        result = run_mnist_model_binary(
            transport,
            onnx_path=Path(args.onnx),
            csr_json=Path(args.csr_json),
            seed=args.seed,
            model_addr=args.model_addr,
        )
    finally:
        transport.close()

    print(f"model: addr=0x{result.model_addr:x} bytes={result.model_len} cycles={result.cycles}")
    print(f"cpu output: {result.cpu_output.flatten().tolist()}")
    print(f"{target_label} output:  {result.loom_output.flatten().tolist()}")
    print(f"hidden max delta: {result.hidden_max_delta}")
    print(f"output max delta: {result.output_max_delta}")
    print(f"prediction: cpu={result.cpu_pred} {target_label}={result.loom_pred}")

    if result.hidden_max_delta > args.tolerance:
        raise AssertionError(
            f"hidden max delta {result.hidden_max_delta} > tolerance {args.tolerance}"
        )
    if result.output_max_delta > args.tolerance:
        raise AssertionError(
            f"output max delta {result.output_max_delta} > tolerance {args.tolerance}"
        )
    if result.loom_pred != result.cpu_pred:
        raise AssertionError(
            f"prediction mismatch: cpu={result.cpu_pred} hw={result.loom_pred}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
