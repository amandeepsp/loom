"""Extract and validate tiled GEMM schedules from affine/loom IR."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from xdsl.dialects import affine
from xdsl.dialects.builtin import AffineMapAttr, IntegerAttr
from xdsl.ir import Operation
from xdsl.ir.affine import AffineConstantExpr

from compiler.dialects.loom import (
    SetEpilogueOp,
    TileLoadActOp,
    TileLoadWgtOp,
    TileMmaOp,
    TileStoreOp,
)


class GemmScheduleSpec(Protocol):
    m: int
    k: int
    n: int
    tile: int


@dataclass(frozen=True)
class KRegion:
    lower: int
    upper: int
    step: int
    k_count: int


@dataclass(frozen=True)
class NRegion:
    lower: int
    upper: int
    step: int
    n_count: int
    m_region: "MRegion"


@dataclass(frozen=True)
class MRegion:
    lower: int
    upper: int
    step: int
    m_count: int
    k_regions: tuple[KRegion, ...]


@dataclass(frozen=True)
class AffineGemmSchedule:
    n_regions: tuple[NRegion, ...]


def extract_affine_schedule(ops: Iterable[Operation]) -> AffineGemmSchedule:
    n_regions: list[NRegion] = []
    for op in ops:
        if not isinstance(op, affine.ForOp):
            raise TypeError(f"unsupported top-level op in GEMM affine plan: {op.name}")
        n_regions.append(_extract_n_region(op))
    if not n_regions:
        raise ValueError("GEMM requires an affine tiled loop plan")
    return AffineGemmSchedule(tuple(n_regions))


def validate_schedule_against_spec(schedule: AffineGemmSchedule, spec: GemmScheduleSpec) -> None:
    n_regions = schedule.n_regions
    if n_regions[0].lower != 0 or n_regions[-1].upper != spec.n:
        raise ValueError("GEMM affine N regions do not cover the GEMM N dimension")
    expected_n_lower = 0
    for region in n_regions:
        if region.lower != expected_n_lower:
            raise ValueError("GEMM affine N regions are not contiguous")
        if region.step != region.n_count:
            raise ValueError("GEMM affine N loop step must equal n_count")
        if region.n_count <= 0 or region.n_count > spec.tile:
            raise ValueError("GEMM affine N count is invalid")
        expected_n_lower = region.upper

        m_region = region.m_region
        if m_region.lower != 0 or m_region.upper != spec.m:
            raise ValueError("GEMM affine M region does not cover the padded M dimension")
        if m_region.step != m_region.m_count:
            raise ValueError("GEMM affine M loop step must equal m_count")
        if m_region.m_count != spec.tile:
            raise ValueError("GEMM affine M count must equal the hardware tile")

        k_regions = m_region.k_regions
        if k_regions[0].lower != 0 or k_regions[-1].upper != spec.k:
            raise ValueError("GEMM affine K regions do not cover the GEMM K dimension")
        expected_k_lower = 0
        for k_region in k_regions:
            if k_region.lower != expected_k_lower:
                raise ValueError("GEMM affine K regions are not contiguous")
            if k_region.step != k_region.k_count:
                raise ValueError("GEMM affine K loop step must equal k_count")
            if k_region.k_count <= 0:
                raise ValueError("GEMM affine K count is invalid")
            expected_k_lower = k_region.upper


def _extract_n_region(n_loop: affine.ForOp) -> NRegion:
    n_lower, n_upper, n_step = _for_bounds(n_loop)
    body = _body_ops(n_loop)
    m_loops = [op for op in body if isinstance(op, affine.ForOp)]
    if len(m_loops) != 1:
        raise ValueError("GEMM affine plan must contain one M loop per N region")
    m_loop = m_loops[0]
    m_lower, m_upper, m_step = _for_bounds(m_loop)
    m_body = _body_ops(m_loop)

    epilogue = [op for op in m_body if isinstance(op, SetEpilogueOp)]
    stores = [op for op in m_body if isinstance(op, TileStoreOp)]
    if len(epilogue) != 1 or len(stores) != 1:
        raise ValueError("GEMM affine plan must contain one epilogue setup and one store")
    n_count = _int_attr(epilogue[0].n_count)
    if n_count != _int_attr(stores[0].n_count):
        raise ValueError("GEMM affine plan has mismatched N counts")
    m_count = _int_attr(stores[0].m_count)

    k_regions = tuple(_extract_k_region(op) for op in m_body if isinstance(op, affine.ForOp))
    if not k_regions:
        raise ValueError("GEMM affine plan must contain at least one K loop")
    return NRegion(
        n_lower,
        n_upper,
        n_step,
        n_count,
        MRegion(m_lower, m_upper, m_step, m_count, k_regions),
    )


def _extract_k_region(k_loop: affine.ForOp) -> KRegion:
    lower, upper, step = _for_bounds(k_loop)
    body = _body_ops(k_loop)
    act = [op for op in body if isinstance(op, TileLoadActOp)]
    wgt = [op for op in body if isinstance(op, TileLoadWgtOp)]
    mma = [op for op in body if isinstance(op, TileMmaOp)]
    if len(act) != 1 or len(wgt) != 1 or len(mma) != 1:
        raise ValueError("GEMM K loop must contain act load, weight load, and MMA")
    k_count = _int_attr(mma[0].k_count)
    if k_count != _int_attr(act[0].k_count) or k_count != _int_attr(wgt[0].k_count):
        raise ValueError("GEMM K loop has mismatched K counts")
    return KRegion(lower, upper, step, k_count)


def _body_ops(loop: affine.ForOp) -> tuple[Operation, ...]:
    return tuple(op for op in loop.body.block.ops if not isinstance(op, affine.YieldOp))


def _for_bounds(loop: affine.ForOp) -> tuple[int, int, int]:
    return (
        _constant_affine_bound(loop.lowerBoundMap),
        _constant_affine_bound(loop.upperBoundMap),
        _int_attr(loop.step),
    )


def _constant_affine_bound(bound: AffineMapAttr) -> int:
    results = bound.data.results
    if len(results) != 1 or not isinstance(results[0], AffineConstantExpr):
        raise ValueError("GEMM only supports constant affine loop bounds")
    return int(results[0].value)


def _int_attr(attr: IntegerAttr) -> int:
    return int(attr.value.data)
