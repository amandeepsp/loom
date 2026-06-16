"""Prepare MNIST model-binary artifact inputs for loom-run-model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from compiler.backend import compile_onnx_model
from compiler.quant_utils import quantize_multiplier_less_than_one
from shared.reference import cpu_requantize

_L0_IS = 0.003921568859368563
_L0_IZP = -128
_L0_WS = 0.0026108562014997005
_L0_OS = 0.025416772812604904
_L0_OZP = -128

_L1_IS = _L0_OS
_L1_IZP = -128
_L1_WS = 0.0017632287926971912
_L1_OS = 0.040668897330760956
_L1_OZP = 21


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile MNIST .lmb and emit deterministic raw int8 IO files.",
    )
    parser.add_argument("--onnx", default="models/out/mnist_int8.onnx")
    parser.add_argument("--csr-json", default="build/sipeed_tang_nano_20k/csr.json")
    parser.add_argument("--model", default="build/mnist.lmb")
    parser.add_argument("--input", default="build/mnist_input_i8.bin")
    parser.add_argument("--expected", default="build/mnist_expected_i8.bin")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--debug-sections",
        action="store_true",
        help="Include diagnostic ASM/xDSL IR sections in the generated .lmb.",
    )
    args = parser.parse_args()

    onnx_path = Path(args.onnx)
    input_img, expected = _mnist_reference(onnx_path, args.seed)
    compiled = compile_onnx_model(onnx_path, csr_json=args.csr_json)
    model_blob = compiled.to_bytes(include_debug_sections=args.debug_sections)

    for output in (args.model, args.input, args.expected):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.model).write_bytes(model_blob)
    Path(args.input).write_bytes(np.ascontiguousarray(input_img).tobytes())
    Path(args.expected).write_bytes(np.ascontiguousarray(expected).tobytes())

    deployment = compiled.manifest()["deployment"]
    print(f"model: {args.model} bytes={len(model_blob)} addr=0x{deployment['model_addr']:x}")
    print(f"input: {args.input} shape={list(input_img.shape)}")
    print(f"expected: {args.expected} shape={list(expected.shape)} pred={int(expected.argmax())}")
    return 0


def _mnist_reference(onnx_path: Path, seed: int) -> tuple[np.ndarray, np.ndarray]:
    import onnx

    model = onnx.load(str(onnx_path))
    weights = {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}
    w0 = weights["net.1.weight_quantized"]
    b0 = weights["net.1.bias_quantized"]
    w1 = weights["net.3.weight_quantized"]
    b1 = weights["net.3.bias_quantized"]

    l0_bias, l0_mult, l0_shift = _epilogue_params(_L0_IS, _L0_IZP, _L0_WS, _L0_OS, w0, b0, 256)
    l1_bias, l1_mult, l1_shift = _epilogue_params(_L1_IS, _L1_IZP, _L1_WS, _L1_OS, w1, b1, 10)

    rng = np.random.default_rng(seed)
    input_img = rng.integers(-128, 128, size=(1, 784), dtype=np.int8)
    cpu_acc0 = input_img.astype(np.int32) @ w0.T.astype(np.int32)
    cpu_hidden = cpu_requantize(cpu_acc0, l0_bias, l0_mult, l0_shift, _L0_OZP, -128, 127)
    cpu_acc1 = cpu_hidden.astype(np.int32) @ w1.T.astype(np.int32)
    cpu_output = cpu_requantize(cpu_acc1, l1_bias, l1_mult, l1_shift, _L1_OZP, -128, 127)
    return input_img, cpu_output


def _epilogue_params(
    input_scale: float,
    input_zp: int,
    weight_scale: float,
    output_scale: float,
    weight: np.ndarray,
    bias_raw: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    combined = (input_scale * weight_scale) / output_scale
    mult, shft = quantize_multiplier_less_than_one(combined)
    sum_w = weight.sum(axis=1).astype(np.int32)
    bias = bias_raw.astype(np.int32) - input_zp * sum_w
    return (
        bias,
        np.full(n, int(mult), dtype=np.int32),
        np.full(n, int(shft), dtype=np.int32),
    )


if __name__ == "__main__":
    raise SystemExit(main())
