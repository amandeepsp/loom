"""Relax pass entrypoints for the out-of-tree accel integration."""

from __future__ import annotations

import tvm
from tvm import relax

from . import codegen, patterns


def lower_pipeline(mod: tvm.IRModule) -> tvm.IRModule:
    """Run the current out-of-tree lowering pipeline.

    Pipeline: partition → lower accel regions → LambdaLift.
    """
    mod = patterns.partition_for_accel_cfu(mod)
    mod = codegen.lower_accel_regions(mod)
    mod = relax.transform.LambdaLift()(mod)
    return mod


def register_relax_pipeline() -> None:
    """Register the accel lowering pipeline as a TVM global function.

    Called once at module import time.
    """

    @tvm.register_global_func("accel.relax.lower_pipeline", override=True)
    def _entry(mod: tvm.IRModule) -> tvm.IRModule:
        return lower_pipeline(mod)


register_relax_pipeline()
