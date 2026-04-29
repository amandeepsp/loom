"""Relax pass entrypoints for the out-of-tree accel integration.

This file intentionally lives under `tvm/` as source scaffolding and not as an
importable top-level `tvm` Python package. Creating `tvm/__init__.py` in this
repository would shadow the upstream TVM install from `../tvm`.
"""

from __future__ import annotations

from typing import Any

import tvm
from tvm import relax

from runtime import _load_local_module


def _as_int(value: Any) -> int | None:
    """Convert a static Relax shape dimension to `int` when possible."""

    try:
        return int(value)
    except TypeError:
        return None


@relax.expr_functor.mutator
class _MatmulTiler(relax.PyExprMutator):
    """Rewrite eligible matmuls into explicit `M/N` tiled Relax subgraphs."""

    def __init__(self, mod: tvm.IRModule, tile_m: int, tile_n: int) -> None:
        super().__init__(mod)
        self.mod_ = mod
        self.tile_m = tile_m
        self.tile_n = tile_n

    def transform(self) -> tvm.IRModule:
        """Return the rewritten module."""

        for global_var, func in self.mod_.functions.items():
            if not isinstance(func, relax.Function):
                continue

            attrs = func.attrs
            if attrs and ("Codegen" in attrs or "Composite" in attrs):
                self.builder_.update_func(global_var, func)
                continue

            updated_func = self.visit_expr(func)
            self.builder_.normalize(updated_func)
            self.builder_.update_func(global_var, updated_func)
        return self.builder_.get()

    def visit_call_(self, call: relax.Call) -> relax.Expr:
        call = super().visit_call_(call)
        if not _is_tileable_int32_matmul(call):
            return call

        lhs, rhs = call.args
        lhs_sinfo = lhs.struct_info
        rhs_sinfo = rhs.struct_info
        assert isinstance(lhs_sinfo, relax.TensorStructInfo)
        assert isinstance(rhs_sinfo, relax.TensorStructInfo)

        m = _as_int(lhs_sinfo.shape[0])
        k = _as_int(lhs_sinfo.shape[1])
        n = _as_int(rhs_sinfo.shape[1])
        if m is None or k is None or n is None:
            return call
        if m <= self.tile_m and n <= self.tile_n:
            return call

        row_tiles: list[relax.Expr] = []
        for m_begin in range(0, m, self.tile_m):
            m_end = min(m_begin + self.tile_m, m)
            lhs_tile = relax.op.strided_slice(
                lhs,
                axes=[0],
                begin=[m_begin],
                end=[m_end],
                strides=[1],
                assume_inbound=True,
            )

            col_tiles: list[relax.Expr] = []
            for n_begin in range(0, n, self.tile_n):
                n_end = min(n_begin + self.tile_n, n)
                rhs_tile = relax.op.strided_slice(
                    rhs,
                    axes=[1],
                    begin=[n_begin],
                    end=[n_end],
                    strides=[1],
                    assume_inbound=True,
                )
                col_tiles.append(relax.op.matmul(lhs_tile, rhs_tile, out_dtype="int32"))

            row_tiles.append(col_tiles[0] if len(col_tiles) == 1 else relax.op.concat(col_tiles, axis=1))

        return row_tiles[0] if len(row_tiles) == 1 else relax.op.concat(row_tiles, axis=0)


def _is_tileable_int32_matmul(call: relax.Call) -> bool:
    """Return whether `call` is a static-shape `int32` matmul we can tile."""

    matmul_op = tvm.ir.Op.get("relax.matmul")
    if call.op != matmul_op:
        return False
    if len(call.args) != 2:
        return False

    out_sinfo = call.struct_info
    if not isinstance(out_sinfo, relax.TensorStructInfo):
        return False
    if out_sinfo.dtype != "int32" or out_sinfo.shape is None:
        return False
    if len(out_sinfo.shape) != 2:
        return False

    for arg in call.args:
        sinfo = arg.struct_info
        if not isinstance(sinfo, relax.TensorStructInfo):
            return False
        if sinfo.dtype != "int32" or sinfo.shape is None:
            return False
        if len(sinfo.shape) != 2:
            return False

    return True


def tile_matmul_tiles(mod: tvm.IRModule, tile_m: int = 8, tile_n: int = 8) -> tvm.IRModule:
    """Rewrite eligible Relax programs to expose explicit `M/N` tiling.

    The current implementation only tiles static 2D `int32` matmuls. This
    matches the shape produced by the ONNX importer for the quantized MNIST
    graph and keeps the tile structure visible in Relax.
    """

    if tile_m <= 0 or tile_n <= 0:
        raise ValueError("tile sizes must be positive")
    return _MatmulTiler(mod, tile_m=tile_m, tile_n=tile_n).transform()


def lower_pipeline(
    mod: tvm.IRModule,
    *,
    tile_m: int = 8,
    tile_n: int = 8,
    enable_tiling: bool = True,
    enable_partitioning: bool = True,
) -> tvm.IRModule:
    """Run the current out-of-tree lowering pipeline.

    Pipeline order:

    1. optional explicit `M/N` tiling in Relax
    2. partition eligible accelerator regions
    3. lower partitioned calls to packed runtime calls
    """

    patterns_mod = _load_local_module("patterns", "patterns.py")
    codegen_mod = _load_local_module("codegen", "codegen.py")

    if enable_tiling:
        mod = tile_matmul_tiles(mod, tile_m=tile_m, tile_n=tile_n)
    if enable_partitioning:
        mod = patterns_mod.partition_for_accel_cfu(mod)
        mod = codegen_mod.lower_accel_regions(mod)
    mod = relax.transform.LambdaLift()(mod)
    return mod


def register_relax_pipeline() -> None:
    """Register out-of-tree Relax pass entrypoints for accel lowering."""

    @tvm.register_global_func("accel.relax.tile_matmul_tiles", override=True)
    def _tile_entry(mod: tvm.IRModule, tile_m: int = 8, tile_n: int = 8) -> tvm.IRModule:
        return tile_matmul_tiles(mod, tile_m=tile_m, tile_n=tile_n)

    @tvm.register_global_func("accel.relax.lower_pipeline", override=True)
    def _pipeline_entry(
        mod: tvm.IRModule,
        tile_m: int = 8,
        tile_n: int = 8,
        enable_tiling: bool = True,
        enable_partitioning: bool = True,
    ) -> tvm.IRModule:
        return lower_pipeline(
            mod,
            tile_m=tile_m,
            tile_n=tile_n,
            enable_tiling=enable_tiling,
            enable_partitioning=enable_partitioning,
        )
