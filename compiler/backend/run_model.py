"""CLI for running a precompiled loom model binary artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from shared.protocol import DEFAULT_BAUD, SerialTransport, TcpTransport

from .runtime import ModelBinaryArtifact, ModelBinaryRuntime


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a precompiled loom model binary.")
    parser.add_argument("model", help="Path to a .lmb model binary.")
    parser.add_argument("input", help="Raw int8 input tensor bytes.")
    parser.add_argument("--port", default="/dev/ttyUSB1")
    parser.add_argument("--baud-rate", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--model-addr", type=lambda text: int(text, 0), default=None)
    parser.add_argument("-o", "--output", help="Optional path for raw int8 output tensor bytes.")
    parser.add_argument("--expect-output", help="Optional raw int8 expected output bytes.")
    parser.add_argument("--tolerance", type=int, default=0)
    parser.add_argument(
        "--check-prediction",
        action="store_true",
        help="Require argmax(output) to match argmax(expect-output).",
    )
    args = parser.parse_args()

    artifact = ModelBinaryArtifact.from_file(args.model)
    input_shape = tuple(artifact.manifest["runtime_contract"]["input"]["shape"])
    input_bytes = Path(args.input).read_bytes()
    input_arr = np.frombuffer(input_bytes, dtype=np.int8)
    if input_arr.size != _num_elements(input_shape):
        raise ValueError(
            f"input file has {input_arr.size} int8 elements, expected {_num_elements(input_shape)}"
        )
    input_arr = input_arr.reshape(input_shape)

    if args.port.startswith("tcp://"):
        transport = TcpTransport(args.port, timeout_s=args.timeout)
        target_label = "sim"
    else:
        transport = SerialTransport(
            port=args.port,
            baud_rate=args.baud_rate,
            timeout_s=args.timeout,
        )
        target_label = "hw"

    try:
        runtime = ModelBinaryRuntime(artifact, transport, model_addr=args.model_addr)
        result = runtime.run_quantized(input_arr)
    finally:
        transport.close()

    if args.output:
        Path(args.output).write_bytes(np.ascontiguousarray(result.output).tobytes())

    print(f"model: addr=0x{result.model_addr:x} bytes={len(artifact.blob)} cycles={result.cycles}")
    print(f"{target_label} output: {result.output.flatten().tolist()}")
    if args.expect_output:
        expected = _load_expected_output(args.expect_output, result.output.shape)
        delta = np.abs(result.output.astype(np.int32) - expected.astype(np.int32))
        max_delta = int(delta.max()) if delta.size else 0
        print(f"output max delta: {max_delta}")
        if max_delta > args.tolerance:
            raise AssertionError(f"output max delta {max_delta} > tolerance {args.tolerance}")
        if args.check_prediction:
            expected_pred = int(expected.argmax())
            actual_pred = int(result.output.argmax())
            print(f"prediction: expected={expected_pred} actual={actual_pred}")
            if actual_pred != expected_pred:
                raise AssertionError(
                    f"prediction mismatch: expected={expected_pred} actual={actual_pred}"
                )
    return 0


def _num_elements(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def _load_expected_output(path: str, shape: tuple[int, ...]) -> np.ndarray:
    data = Path(path).read_bytes()
    arr = np.frombuffer(data, dtype=np.int8)
    expected = _num_elements(shape)
    if arr.size != expected:
        raise ValueError(
            f"expected output file has {arr.size} int8 elements, expected {expected}"
        )
    return arr.reshape(shape)


if __name__ == "__main__":
    raise SystemExit(main())
