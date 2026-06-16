"""Backend lowering from xDSL TOSA IR to loom model binaries."""

from .model import (
    Buffer,
    CompiledModel,
    GemmKernel,
    model_binary_manifest,
    model_binary_sections,
    compile_onnx_model,
    compile_tosa_module,
    lower_tosa_to_affine,
    lower_tosa_to_loom,
    lower_tosa_to_riscv,
)
from .riscv import custom0, encode_r_type
from .runtime import ModelBinaryArtifact, ModelBinaryRuntime, ModelRunResult, ModelRuntime

__all__ = [
    "Buffer",
    "CompiledModel",
    "GemmKernel",
    "ModelBinaryArtifact",
    "ModelBinaryRuntime",
    "ModelRunResult",
    "ModelRuntime",
    "compile_onnx_model",
    "compile_tosa_module",
    "custom0",
    "encode_r_type",
    "lower_tosa_to_affine",
    "lower_tosa_to_loom",
    "lower_tosa_to_riscv",
    "model_binary_manifest",
    "model_binary_sections",
]
