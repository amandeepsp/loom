"""Shared MNIST model-binary runner for sim and hardware.

The reference is the source ONNX model run through ONNX Runtime on a real
MNIST test digit, so it cannot drift from the model (earlier versions used
hand-copied quantization constants that went stale when the model was
retrained, and random-noise inputs that saturated every logit).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx

from compiler import ModelBinaryArtifact, ModelBinaryRuntime, compile_onnx_model
from compiler.backend.reference import OnnxReference, load_sample

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX = REPO_ROOT / "models/out/mnist_int8.onnx"
HIDDEN_TENSOR = "/net/net.1/Gemm_output_0_quantized"


@dataclass(frozen=True)
class MnistModelBinaryResult:
    label: int
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


def mnist_reference(onnx_path: Path = DEFAULT_ONNX) -> OnnxReference:
    return OnnxReference(onnx_path, extra_outputs=(HIDDEN_TENSOR,))


def mnist_digit(reference: OnnxReference, index: int) -> tuple[np.ndarray, int]:
    """int8 [1, 784] input for MNIST test image `index`, and its label."""

    sample = load_sample("mnist", index, reference)
    return sample.int8_input.reshape(1, 784), sample.label


def onnx_hidden_and_output(reference: OnnxReference, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    output, hidden = reference.run_int8_tensors(x, (HIDDEN_TENSOR,))
    return hidden.reshape(1, -1), output.reshape(1, -1)


def qgemm_output_zero_points(onnx_path: Path = DEFAULT_ONNX) -> list[int]:
    """Output zero point of each QGemm, in graph order."""

    model = onnx.load(str(onnx_path))
    consts = {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "Constant":
            consts[node.output[0]] = onnx.numpy_helper.to_array(node.attribute[0].t)
    return [int(consts[n.input[8]]) for n in model.graph.node if n.op_type == "QGemm"]


def compiled_layer_params(compiled) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(bias, multiplier, shift) int32 vectors the compiler emitted per GEMM."""

    buffers = {b.name: b for b in compiled.buffers}

    def vec(name: str) -> np.ndarray:
        return np.frombuffer(buffers[name].data, dtype="<i4").astype(np.int32)

    return [
        (vec(f"gemm{i}.bias"), vec(f"gemm{i}.multiplier"), vec(f"gemm{i}.shift"))
        for i in range(len(compiled.kernels))
    ]


def run_mnist_model_binary(
    transport,
    *,
    onnx_path: Path = DEFAULT_ONNX,
    csr_json: Path | None = None,
    image: int = 0,
    model_addr: int | None = None,
) -> MnistModelBinaryResult:
    """Compile MNIST ONNX, run it on a real digit, and return ONNX/loom outputs."""

    onnx_path = Path(onnx_path)
    reference = mnist_reference(onnx_path)
    input_img, label = mnist_digit(reference, image)
    cpu_hidden, cpu_output = onnx_hidden_and_output(reference, input_img)

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
        label=label,
        cpu_hidden=cpu_hidden,
        cpu_output=cpu_output,
        loom_hidden=hidden,
        loom_output=run.output,
        cycles=run.cycles,
        model_addr=run.model_addr,
        model_len=len(artifact.blob),
    )
