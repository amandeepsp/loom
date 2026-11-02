"""Lower scheduled affine GEMM loops toward xDSL RISC-V target IR."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import ClassVar

from xdsl.context import Context
from xdsl.dialects import riscv, riscv_scf, rv32
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, i32
from xdsl.ir import Block, Operation, SSAValue, Region
from xdsl.passes import ModulePass

from compiler.dialects import loom_riscv

from .affine_schedule import (
    AffineGemmSchedule,
    KRegion,
    MRegion,
    NRegion,
    extract_affine_schedule,
    validate_schedule_against_spec,
)
from .target import CsrMap, GemmTargetSpec, DmaRegs


@dataclass(frozen=True)
class RiscvLoweringContext:
    """Concrete target context needed after memory planning and CSR binding."""

    spec: GemmTargetSpec
    csr: CsrMap


@dataclass(frozen=True)
class LowerAffineKernelToRiscvScfPass(ModulePass):
    """Lower scheduled affine/tile GEMM loops into structured RISC-V target IR."""

    name: ClassVar[str] = "loom-lower-affine-kernel-to-riscv-scf"

    lowering_context: RiscvLoweringContext

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        del ctx
        schedule = extract_affine_schedule(tuple(op.body.block.ops))
        validate_schedule_against_spec(schedule, self.lowering_context.spec)
        lowered = _schedule_to_module(self.lowering_context, schedule)
        _replace_module_body(op, tuple(lowered.body.block.ops))


def lower_affine_kernel_to_riscv_scf(
    ctx: RiscvLoweringContext,
    ops: Iterable[Operation],
) -> ModuleOp:
    """Build structured RISC-V target IR from affine GEMM loop roots."""

    module = ModuleOp([op.clone() for op in ops])
    LowerAffineKernelToRiscvScfPass(ctx).apply(Context(), module)
    return module


def _replace_module_body(module: ModuleOp, new_ops: tuple[Operation, ...]) -> None:
    block = module.body.block
    for old_op in tuple(block.ops):
        block.detach_op(old_op)
        old_op.erase(safe_erase=False)
    source_block = new_ops[0].parent if new_ops else None
    if source_block is not None:
        for new_op in new_ops:
            source_block.detach_op(new_op)
    block.add_ops(new_ops)


def _schedule_to_module(ctx: RiscvLoweringContext, schedule: AffineGemmSchedule) -> ModuleOp:
    out: list[Operation] = []
    for region in schedule.n_regions:
        out.extend(_n_region_ops(ctx, region))
    return ModuleOp(out)


def _n_region_ops(ctx: RiscvLoweringContext, region: NRegion) -> list[Operation]:
    def n_body(n_iv: SSAValue) -> list[Operation]:
        # Epilogue params depend only on the N tile and live in a CFU memory
        # that nothing else writes, so upload them once per N tile rather
        # than once per (M, N) tile.
        return [
            *_epilogue_ops(ctx, region.n_count, n_iv),
            *_for_ops(
                0,
                region.m_region.upper,
                region.m_region.step,
                lambda m_iv: _m_region_body(ctx, region.n_count, region.m_region, n_iv, m_iv),
            ),
        ]

    return _for_ops(region.lower, region.upper, region.step, n_body)


def _m_region_body(
    ctx: RiscvLoweringContext,
    n_count: int,
    region: MRegion,
    n_iv: SSAValue,
    m_iv: SSAValue,
) -> list[Operation]:
    out: list[Operation] = []
    for k_region in region.k_regions:
        out.extend(
            _for_ops(
                k_region.lower,
                k_region.upper,
                k_region.step,
                lambda k_iv: _k_body(ctx, k_region, n_iv, m_iv, k_iv),
            )
        )
    out.append(loom_riscv.ComputeWaitOp())
    out.extend(_store_result_ops(ctx, n_count, m_iv, n_iv))
    return out


def _k_body(
    ctx: RiscvLoweringContext,
    region: KRegion,
    n_iv: SSAValue,
    m_iv: SSAValue,
    k_iv: SSAValue,
) -> list[Operation]:
    act_addr_ops, act_addr = _activation_dma_addr(ctx.spec, m_iv, k_iv)
    wgt_addr_ops, wgt_addr = _weight_dma_addr(ctx.spec, n_iv, k_iv)
    flags_ops, flags = _compute_flags(ctx.spec, region, k_iv)
    k_count = rv32.LiOp(region.k_count)
    return [
        *act_addr_ops,
        *wgt_addr_ops,
        *_dma_kick_ops(ctx.csr.act_dma, act_addr, region.k_count * ctx.spec.tile),
        *_dma_kick_ops(ctx.csr.wgt_dma, wgt_addr, region.k_count * ctx.spec.tile),
        loom_riscv.WaitPreviousComputeOp(k_iv),
        _dma_wait_loop(ctx.csr.act_dma.done),
        _dma_wait_loop(ctx.csr.wgt_dma.done),
        *_dma_drain_fence_ops(ctx.csr.act_dma.done),
        *_dma_stop_ops(ctx.csr.act_dma),
        *_dma_stop_ops(ctx.csr.wgt_dma),
        *flags_ops,
        k_count,
        loom_riscv.ComputeStartOp(flags, k_count.rd),
    ]


def _epilogue_ops(
    ctx: RiscvLoweringContext,
    n_count: int,
    n_iv: SSAValue,
) -> list[Operation]:
    return [
        *_for_ops(
            0,
            ctx.spec.tile,
            1,
            lambda row_iv: _for_ops(
                0,
                n_count,
                1,
                lambda col_iv: _epilogue_param_body(ctx, n_iv, row_iv, col_iv),
            ),
        ),
        out_zp := rv32.LiOp(ctx.spec.output_offset),
        act_min := rv32.LiOp(ctx.spec.activation_min),
        act_max := rv32.LiOp(ctx.spec.activation_max),
        loom_riscv.WriteConfigOp(out_zp.rd, register=0),
        loom_riscv.WriteConfigOp(act_min.rd, register=1),
        loom_riscv.WriteConfigOp(act_max.rd, register=2),
    ]


def _store_result_ops(
    ctx: RiscvLoweringContext,
    n_count: int,
    m_iv: SSAValue,
    n_iv: SSAValue,
) -> list[Operation]:
    """Copy one finished tile from the CFU result buffer to the output matrix.

    Per row: one row pointer (output + (m+row)*N + n) and one result-index
    base; the column loop is fully unrolled (n_count <= tile is a compile-time
    constant), so each element is addi + READ_RESULT + sb with an immediate
    offset instead of re-deriving its address from scratch.
    """
    return _for_ops(
        0,
        ctx.spec.tile,
        1,
        lambda row_iv: _store_result_row(ctx, n_count, m_iv, n_iv, row_iv),
    )


def _store_result_row(
    ctx: RiscvLoweringContext,
    n_count: int,
    m_iv: SSAValue,
    n_iv: SSAValue,
    row_iv: SSAValue,
) -> list[Operation]:
    m_row = riscv.AddOp(m_iv, row_iv)
    row_offset_ops, row_offset = _mul_const(m_row.rd, ctx.spec.n)
    col0 = riscv.AddOp(row_offset, n_iv)
    base = rv32.LiOp(ctx.spec.output_addr)
    row_ptr = riscv.AddOp(base.rd, col0.rd)
    zero = rv32.LiOp(0)
    index_ops, index_base = _tile_channel(ctx.spec, row_iv, zero.rd)
    out: list[Operation] = [m_row, *row_offset_ops, col0, base, row_ptr, zero, *index_ops]
    for col in range(n_count):
        if col == 0:
            index = index_base
        else:
            step = riscv.AddiOp(index_base, col)
            out.append(step)
            index = step.rd
        read = loom_riscv.ReadResultOp(index)
        out.extend([read, riscv.SbOp(row_ptr.rd, read.rd, col)])
    return out


def _epilogue_param_body(
    ctx: RiscvLoweringContext,
    n_iv: SSAValue,
    row_iv: SSAValue,
    col_iv: SSAValue,
) -> list[Operation]:
    channel_ops, channel = _tile_channel(ctx.spec, row_iv, col_iv)
    n_plus_col = riscv.AddOp(n_iv, col_iv)
    param_off = riscv.SlliOp(n_plus_col.rd, 2)
    out: list[Operation] = [*channel_ops, n_plus_col, param_off]
    for field, base_addr in (
        (0, ctx.spec.bias_addr),
        (1, ctx.spec.mult_addr),
        (2, ctx.spec.shift_addr),
    ):
        load_ops, value = _load_word_abs_indexed(base_addr, param_off.rd)
        out.extend([*load_ops, loom_riscv.WriteEpilogueParamOp(channel, value, field=field)])
    return out


def _activation_dma_addr(
    spec: GemmTargetSpec,
    m_iv: SSAValue,
    k_iv: SSAValue,
) -> tuple[list[Operation], SSAValue]:
    m_ops, m_offset = _mul_const(m_iv, spec.k)
    k_ops, k_offset = _mul_const(k_iv, spec.tile)
    offset = riscv.AddOp(m_offset, k_offset)
    base = rv32.LiOp(spec.input_addr)
    addr = riscv.AddOp(base.rd, offset.rd)
    return [*m_ops, *k_ops, offset, base, addr], addr.rd


def _weight_dma_addr(
    spec: GemmTargetSpec,
    n_iv: SSAValue,
    k_iv: SSAValue,
) -> tuple[list[Operation], SSAValue]:
    n_tile = riscv.SrliOp(n_iv, _log2_power_of_two(spec.tile))
    n_ops, n_offset = _mul_const(n_tile.rd, spec.k * spec.tile)
    k_ops, k_offset = _mul_const(k_iv, spec.tile)
    offset = riscv.AddOp(n_offset, k_offset)
    base = rv32.LiOp(spec.weight_addr)
    addr = riscv.AddOp(base.rd, offset.rd)
    return [n_tile, *n_ops, *k_ops, offset, base, addr], addr.rd


def _tile_channel(
    spec: GemmTargetSpec,
    row_iv: SSAValue,
    col_iv: SSAValue,
) -> tuple[list[Operation], SSAValue]:
    shift = _log2_power_of_two(spec.tile)
    if shift == 0:
        channel = riscv.AddOp(row_iv, col_iv)
        return [channel], channel.rd
    row_base = riscv.SlliOp(row_iv, shift)
    channel = riscv.AddOp(row_base.rd, col_iv)
    return [row_base, channel], channel.rd


def _load_word_abs_indexed(addr: int, offset: SSAValue) -> tuple[list[Operation], SSAValue]:
    base = rv32.LiOp(addr)
    absolute = riscv.AddOp(base.rd, offset)
    value = riscv.LwOp(absolute.rd, 0)
    return [base, absolute, value], value.rd


def _dma_kick_ops(regs: DmaRegs, base: SSAValue, length: int) -> list[Operation]:
    return [
        *_store_imm_abs(regs.enable, 0),
        *_store_imm_abs(regs.loop, 0),
        *_store_reg_abs(regs.base, base),
        *_store_imm_abs(regs.length, length),
        *_store_imm_abs(regs.enable, 1),
    ]


def _dma_stop_ops(regs: DmaRegs) -> list[Operation]:
    return _store_imm_abs(regs.enable, 0)


def _store_imm_abs(addr: int, value: int) -> list[Operation]:
    addr_op = rv32.LiOp(addr)
    value_op = rv32.LiOp(value)
    return [addr_op, value_op, riscv.SwOp(addr_op.rd, value_op.rd, 0)]


def _store_reg_abs(addr: int, value: SSAValue) -> list[Operation]:
    addr_op = rv32.LiOp(addr)
    return [addr_op, riscv.SwOp(addr_op.rd, value, 0)]


def _mul_const(value: SSAValue, const: int) -> tuple[list[Operation], SSAValue]:
    if const < 0:
        raise ValueError("constant multiply only supports non-negative constants")
    if const == 0:
        zero = rv32.LiOp(0)
        return [zero], zero.rd

    ops: list[Operation] = []
    acc: SSAValue | None = None
    for bit in range(const.bit_length()):
        if not (const & (1 << bit)):
            continue
        if bit == 0:
            term = value
        else:
            shift = riscv.SlliOp(value, bit)
            ops.append(shift)
            term = shift.rd
        if acc is None:
            acc = term
        else:
            add = riscv.AddOp(acc, term)
            ops.append(add)
            acc = add.rd
    assert acc is not None
    return ops, acc


def _compute_flags(
    spec: GemmTargetSpec,
    region: KRegion,
    k_iv: SSAValue,
) -> tuple[list[Operation], SSAValue]:
    first = riscv.SltiuOp(k_iv, 1)
    last_base = rv32.LiOp(spec.k - region.k_count)
    last_delta = riscv.XorOp(k_iv, last_base.rd)
    last_eq = riscv.SltiuOp(last_delta.rd, 1)
    last_flag = riscv.SlliOp(last_eq.rd, 1)
    flags = riscv.AddOp(first.rd, last_flag.rd)
    return [first, last_base, last_delta, last_eq, last_flag, flags], flags.rd


def _log2_power_of_two(value: int) -> int:
    if value <= 0 or value & (value - 1):
        raise ValueError(f"expected power-of-two value, got {value}")
    return value.bit_length() - 1


def _dma_wait_loop(done_addr: int) -> Operation:
    before = Block()
    addr = rv32.LiOp(done_addr)
    status = riscv.LwOp(addr.rd, 0)
    keep_waiting = riscv.SltiuOp(status.rd, 1)
    before.add_ops([addr, status, keep_waiting, riscv_scf.ConditionOp(keep_waiting.rd)])
    after = Block()
    after.add_ops([riscv_scf.YieldOp()])
    return riscv_scf.WhileOp((), (), Region(before), Region(after))


def _dma_drain_fence_ops(status_addr: int) -> list[Operation]:
    """Wait for issued LiteDRAM reads to reach the scratchpad bridge.

    The DMA `done` CSR tracks request issuance, not response consumption. The
    readers have at most 16 outstanding responses each. Re-reading the MMIO
    status register for 64 iterations takes more than 512 CPU clocks and also
    prevents an empty delay loop from being optimized away.
    """
    def poll_body(_iv: SSAValue) -> list[Operation]:
        addr = rv32.LiOp(status_addr)
        return [addr, riscv.LwOp(addr.rd, 0)]

    return _for_ops(0, 64, 1, poll_body)


def _for_ops(
    lower: int,
    upper: int,
    step: int,
    body_builder: Callable[[SSAValue], list[Operation]],
) -> list[Operation]:
    lb = rv32.LiOp(lower)
    ub = rv32.LiOp(upper)
    block = Block(arg_types=[riscv.Registers.UNALLOCATED_INT])
    body_ops = body_builder(block.args[0])
    block.add_ops([*body_ops, riscv_scf.YieldOp()])
    loop = riscv_scf.ForOp(lb.rd, ub.rd, IntegerAttr(step, i32), (), Region(block))
    return [lb, ub, loop]
