"""Pattern-rewrite pipelines for loom backend lowering stages."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import ClassVar, Protocol

from xdsl.context import Context
from xdsl.dialects import affine
from xdsl.dialects.builtin import DenseArrayBase, IndexType, IntegerAttr, ModuleOp, i32
from xdsl.dialects.tosa import AddOp, RescaleOp
from xdsl.ir import Block, Operation, Region, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint

from compiler.dialects.loom import (
    GemmCandidateOp,
    GemmOp,
    RepackOp,
    SetEpilogueOp,
    TileLoadActOp,
    TileLoadWgtOp,
    TileMmaOp,
    TileStoreOp,
)
from compiler.dialects.loom_cpu import Pack2DToTilesOp
from compiler.frontend.tosa_ext import MatMulOp


class BufferLike(Protocol):
    name: str
    addr: int


@dataclass(frozen=True)
class PackFunctionBinding:
    source: str
    target: str
    function_name: str


@dataclass(frozen=True)
class PlannedGemmReplacement:
    repack: RepackOp
    gemm: GemmOp


class LowerRepackToCpuPackPattern(RewritePattern):
    """Lower high-level loom.repack plan ops to executable CPU layout ops."""

    def __init__(self, buffers: Iterable[BufferLike], bindings: Iterable[PackFunctionBinding]):
        self.buffers = {buf.name: buf for buf in buffers}
        self.bindings = {
            (binding.source, binding.target): binding.function_name
            for binding in bindings
        }

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: RepackOp, rewriter: PatternRewriter) -> None:
        source = op.source.data
        target = op.target.data
        src = self.buffers[source]
        dst = self.buffers[target]
        function_name = self.bindings[(source, target)]
        rewriter.replace_op(
            op,
            Pack2DToTilesOp(
                function_name,
                source,
                target,
                src_addr=src.addr,
                dst_addr=dst.addr,
                m=int(op.m.value.data),
                k=int(op.k.value.data),
                tile=int(op.tile.value.data),
            )
        )


class LowerTosaGemmToCandidatePattern(RewritePattern):
    """Recognize tosa.matmul -> tosa.add -> tosa.rescale as a loom candidate op."""

    def __init__(self):
        self.index = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: RescaleOp, rewriter: PatternRewriter) -> None:
        add = op.input.owner
        if not isinstance(add, AddOp):
            return
        matmul, bias = _split_add(add)
        if matmul is None or bias is None:
            return

        name = op.output.name_hint or f"gemm_{self.index}"
        self.index += 1
        rewriter.replace_op(
            op,
            GemmCandidateOp(
                name,
                matmul.output,
                bias,
                op.multiplier,
                op.shift,
                op.input_zp,
                op.output_zp,
                op.output.type,
            ),
        )


class LowerGemmToAffinePattern(RewritePattern):
    """Lower planned loom.gemm ops to tiled affine schedule loops."""

    def __init__(self, *, cfu_word_bits: int, cfu_store_depth_words: int):
        self.cfu_word_bits = cfu_word_bits
        self.cfu_store_depth_words = cfu_store_depth_words
        self.k_tile = compute_k_tile(
            cfu_word_bits=cfu_word_bits,
            cfu_store_depth_words=cfu_store_depth_words,
        )

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: GemmOp, rewriter: PatternRewriter) -> None:
        loops = _affine_kernel_plan(
            k=int(op.k.value.data),
            n=int(op.n.value.data),
            padded_m=int(op.padded_m.value.data),
            tile=int(op.tile.value.data),
            k_tile=self.k_tile,
        )
        rewriter.replace_op(op, loops)


class LowerGemmCandidateToLoomPlanPattern(RewritePattern):
    """Materialize planned loom ops from recognized GEMM candidate ops."""

    def __init__(self, plan_candidate: Callable[[GemmCandidateOp], PlannedGemmReplacement]):
        self.plan_candidate = plan_candidate
        self.planned: set[int] = set()

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: GemmCandidateOp, rewriter: PatternRewriter) -> None:
        key = id(op)
        if key in self.planned:
            return
        self.planned.add(key)
        planned = self.plan_candidate(op)
        rewriter.insert_op(
            [planned.repack, planned.gemm],
            InsertPoint.before(op),
        )


@dataclass(frozen=True)
class LowerRepackToCpuPackPass(ModulePass):
    """Pass wrapper for lowering loom.repack ops to executable CPU pack ops."""

    name: ClassVar[str] = "loom-lower-repack-to-cpu-pack"

    buffers: tuple[BufferLike, ...]
    bindings: tuple[PackFunctionBinding, ...]

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        del ctx
        apply_all_rewrites(
            op,
            [LowerRepackToCpuPackPattern(self.buffers, self.bindings)],
        )


@dataclass(frozen=True)
class LowerTosaGemmToCandidatesPass(ModulePass):
    """Pass wrapper for recognizing supported TOSA GEMM regions."""

    name: ClassVar[str] = "loom-lower-tosa-gemm-to-candidates"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        del ctx
        apply_all_rewrites(op, [LowerTosaGemmToCandidatePattern()])


@dataclass(frozen=True)
class LowerGemmCandidatesToLoomPlanPass(ModulePass):
    """Pass wrapper for replacing recognized GEMM candidates with loom plan ops."""

    name: ClassVar[str] = "loom-lower-gemm-candidates-to-loom-plan"

    plan_candidate: Callable[[GemmCandidateOp], PlannedGemmReplacement]

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        del ctx
        apply_all_rewrites(op, [LowerGemmCandidateToLoomPlanPattern(self.plan_candidate)])


@dataclass(frozen=True)
class LowerGemmToAffinePass(ModulePass):
    """Pass wrapper for lowering planned loom.gemm ops to affine loop schedules."""

    name: ClassVar[str] = "loom-lower-gemm-to-affine"

    cfu_word_bits: int
    cfu_store_depth_words: int

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        del ctx
        apply_all_rewrites(
            op,
            [
                LowerGemmToAffinePattern(
                    cfu_word_bits=self.cfu_word_bits,
                    cfu_store_depth_words=self.cfu_store_depth_words,
                )
            ],
        )


def apply_all_rewrites(module: ModuleOp, rewrites: list[RewritePattern]) -> ModuleOp:
    """Apply backend rewrite patterns greedily to an xDSL module."""

    merged_pattern = GreedyRewritePatternApplier(rewrites, dce_enabled=False)
    walker = PatternRewriteWalker(merged_pattern)
    walker.rewrite_module(module)
    return module


def compute_k_tile(*, cfu_word_bits: int, cfu_store_depth_words: int) -> int:
    """Return the K rows that fit in one activation/weight scratchpad fill."""

    if cfu_word_bits <= 0:
        raise ValueError("cfu_word_bits must be positive")
    if cfu_word_bits % 32 != 0:
        raise ValueError("cfu_word_bits must be a multiple of 32")
    if cfu_store_depth_words <= 0:
        raise ValueError("cfu_store_depth_words must be positive")

    words_per_line = cfu_word_bits // 32
    k_tile = cfu_store_depth_words // words_per_line
    if k_tile <= 0:
        raise ValueError("cfu_store_depth_words is too small for one CFU line")
    return k_tile


def lower_repack_to_cpu(module: ModuleOp, buffers: Iterable[BufferLike], bindings: Iterable[PackFunctionBinding]) -> ModuleOp:
    """Rewrite all loom.repack operations in a module to loom_cpu pack ops."""

    LowerRepackToCpuPackPass(tuple(buffers), tuple(bindings)).apply(Context(), module)
    return module


def lower_tosa_gemm_to_candidates(module: ModuleOp) -> ModuleOp:
    """Rewrite supported TOSA GEMM regions to loom.gemm_candidate operations."""

    LowerTosaGemmToCandidatesPass().apply(Context(), module)
    return module


def lower_gemm_candidates_to_loom(
    module: ModuleOp,
    plan_candidate: Callable[[GemmCandidateOp], PlannedGemmReplacement],
) -> ModuleOp:
    """Rewrite recognized GEMM candidates by materializing planned loom ops."""

    LowerGemmCandidatesToLoomPlanPass(plan_candidate).apply(Context(), module)
    return module


def lower_gemm_to_affine(
    module: ModuleOp,
    *,
    cfu_word_bits: int,
    cfu_store_depth_words: int,
) -> ModuleOp:
    """Rewrite planned loom.gemm ops to affine loop schedules."""

    LowerGemmToAffinePass(cfu_word_bits, cfu_store_depth_words).apply(Context(), module)
    return module


def walk_ops(module: ModuleOp, op_type: type[Operation]) -> Iterable[Operation]:
    for op in module.walk():
        if isinstance(op, op_type):
            yield op


def _split_add(add: AddOp) -> tuple[MatMulOp | None, SSAValue | None]:
    lhs, rhs = add.operands
    if isinstance(lhs.owner, MatMulOp):
        return lhs.owner, rhs
    if isinstance(rhs.owner, MatMulOp):
        return rhs.owner, lhs
    return None, None


def _affine_kernel_plan(*, k: int, n: int, padded_m: int, tile: int, k_tile: int) -> list[affine.ForOp]:
    top_loops: list[affine.ForOp] = []
    full_n = (n // tile) * tile
    n_tail = n - full_n
    if full_n:
        top_loops.append(_n_tile_loop(0, full_n, tile, tile, padded_m, k, tile, k_tile))
    if n_tail:
        top_loops.append(_n_tile_loop(full_n, n, n_tail, n_tail, padded_m, k, tile, k_tile))
    return top_loops


def _n_tile_loop(
    lower: int,
    upper: int,
    step: int,
    n_count: int,
    padded_m: int,
    k: int,
    tile: int,
    k_tile: int,
) -> affine.ForOp:
    def body(n_iv: SSAValue) -> list:
        return [
            _affine_for(
                0,
                padded_m,
                tile,
                lambda m_iv: [
                    SetEpilogueOp(n_iv, n_count=n_count),
                    *_k_tile_loops(n_iv, m_iv, k, k_tile),
                    TileStoreOp(m_iv, n_iv, m_count=tile, n_count=n_count),
                ],
            )
        ]

    return _affine_for(lower, upper, step, body)


def _k_tile_loops(n_iv: SSAValue, m_iv: SSAValue, k: int, k_tile: int) -> list:
    full_k = (k // k_tile) * k_tile
    tail_k = k - full_k
    ops: list = []
    if full_k:
        ops.append(
            _affine_for(
                0,
                full_k,
                k_tile,
                lambda k_iv: [
                    TileLoadActOp(m_iv, k_iv, k_count=k_tile),
                    TileLoadWgtOp(n_iv, k_iv, k_count=k_tile),
                    TileMmaOp(k_iv, k_count=k_tile),
                ],
            )
        )
    if tail_k:
        ops.append(
            _affine_for(
                full_k,
                k,
                tail_k,
                lambda k_iv: [
                    TileLoadActOp(m_iv, k_iv, k_count=tail_k),
                    TileLoadWgtOp(n_iv, k_iv, k_count=tail_k),
                    TileMmaOp(k_iv, k_count=tail_k),
                ],
            )
        )
    return ops


def _affine_for(lower: int, upper: int, step: int, body_builder) -> affine.ForOp:
    block = Block(arg_types=[IndexType()])
    iv = block.args[0]
    block.add_ops([*body_builder(iv), affine.YieldOp.get()])
    return affine.ForOp.create(
        properties={
            "lowerBoundMap": affine.AffineMapAttr.constant_map(lower),
            "upperBoundMap": affine.AffineMapAttr.constant_map(upper),
            "step": IntegerAttr(step, IndexType()),
            "operandSegmentSizes": DenseArrayBase.from_list(i32, [0, 0, 0]),
        },
        operands=[],
        regions=[Region(block)],
    )
