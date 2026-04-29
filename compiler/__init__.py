"""Accel compiler: TVM Relax integration for the CFU accelerator.

This package contains the out-of-tree TVM lowering pipeline:
- patterns: DPL pattern matching for quantized matmul composites
- codegen: Lowering accel regions to runtime calls
- relax: Pass entrypoints (tiling, partitioning, lowering)
- runtime: Host runtime bridge (packed funcs, memory upload, execution)
- quant_utils: Epilogue multiplier/shift derivation
"""

from .codegen import lower_accel_regions, get_composite_constants
from .patterns import partition_for_accel_cfu, make_matmul_requant_pattern
from .relax import lower_pipeline
from .runtime import (
    AccelRuntime,
    RuntimeConfig,
    SerialTransport,
    TcpTransport,
    register_runtime_functions,
)

__all__ = [
    "lower_accel_regions",
    "get_composite_constants",
    "partition_for_accel_cfu",
    "make_matmul_requant_pattern",
    "lower_pipeline",
    "AccelRuntime",
    "RuntimeConfig",
    "SerialTransport",
    "TcpTransport",
    "register_runtime_functions",
]
