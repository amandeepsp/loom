"""Public compiler entry points for the xDSL model-binary pipeline."""

from shared.protocol import SerialTransport, TcpTransport

from .backend import (
    CompiledModel,
    ModelBinaryArtifact,
    ModelBinaryRuntime,
    ModelRunResult,
    ModelRuntime,
    compile_onnx_model,
    compile_tosa_module,
    lower_tosa_to_affine,
    lower_tosa_to_loom,
    lower_tosa_to_riscv,
    model_binary_manifest,
    model_binary_sections,
)
from .frontend import emit_tosa_ir, load_model


__all__ = [
    "CompiledModel",
    "ModelBinaryArtifact",
    "ModelBinaryRuntime",
    "ModelRunResult",
    "ModelRuntime",
    "SerialTransport",
    "TcpTransport",
    "compile_onnx_model",
    "compile_tosa_module",
    "emit_tosa_ir",
    "load_model",
    "lower_tosa_to_affine",
    "lower_tosa_to_loom",
    "lower_tosa_to_riscv",
    "model_binary_manifest",
    "model_binary_sections",
]
