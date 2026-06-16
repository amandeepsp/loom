"""Shared MNIST model-binary runner for sim and hardware."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from compiler import ModelBinaryArtifact, ModelBinaryRuntime, compile_onnx_model
from compiler.quant_utils import quantize_multiplier_less_than_one
from shared.reference import cpu_requantize

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX = REPO_ROOT / "models/out/mnist_int8.onnx"

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


@dataclass(frozen=True)
class MnistModelBinaryResult:
    cpu_hidden: np.ndarray
    cpu_output: np.ndarray
    loom_hidden: np.ndarray
    loom_output: np.ndarray
    cycles: int
    model_addr: int
    model_len: int

    @property
    def hidden_max_delta(self) -> int:
        return int(np.abs(self.cpu_hidden.astype(np.int32) - self.loom_hidden.astype(np.int32)).max())

    @property
    def output_max_delta(self) -> int:
        return int(np.abs(self.cpu_output.astype(np.int32) - self.loom_output.astype(np.int32)).max())

    @property
    def cpu_pred(self) -> int:
        return int(self.cpu_output.argmax())

    @property
    def loom_pred(self) -> int:
        return int(self.loom_output.argmax())


def run_mnist_model_binary(
    transport,
    *,
    onnx_path: Path = DEFAULT_ONNX,
    csr_json: Path | None = None,
    seed: int = 42,
    model_addr: int | None = None,
) -> MnistModelBinaryResult:
    """Compile MNIST ONNX, run the model binary, and return CPU/sim outputs."""

    import onnx

    onnx_path = Path(onnx_path)
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

    compiled = compile_onnx_model(onnx_path, csr_json=csr_json)
    artifact = ModelBinaryArtifact.from_bytes(compiled.to_bytes(include_debug_sections=False))
    runtime = ModelBinaryRuntime(artifact, transport, model_addr=model_addr)
    run = runtime.run_quantized(input_img)

    hidden_buf = artifact.buffer("gemm0.output")
    first_kernel = artifact.manifest["kernels"][0]
    hidden = np.frombuffer(
        transport.read_mem(hidden_buf.addr, hidden_buf.size),
        dtype=np.int8,
    ).reshape(first_kernel["padded_m"], first_kernel["n"])[:1].copy()

    return MnistModelBinaryResult(
        cpu_hidden=cpu_hidden,
        cpu_output=cpu_output,
        loom_hidden=hidden,
        loom_output=run.output,
        cycles=run.cycles,
        model_addr=run.model_addr,
        model_len=len(artifact.blob),
    )


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
