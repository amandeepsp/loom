"""Lower loom_riscv CUSTOM0 dialect ops to RV32 instruction words."""

from __future__ import annotations

from xdsl.dialects import riscv as xdsl_riscv
from xdsl.dialects import rv32
from xdsl.ir import Operation

from compiler.dialects.loom_riscv import (
    ComputeStartOp,
    ComputeWaitOp,
    LoomCustom0Op,
    ReadResultOp,
    WriteConfigOp,
    WriteEpilogueParamOp,
)

from . import riscv as rv_words


def encode_custom0_op(op: Operation) -> int:
    """Encode one loom CUSTOM0 dialect operation."""

    if not isinstance(op, LoomCustom0Op):
        raise TypeError(f"unsupported loom CUSTOM0 op: {op.name}")
    funct3, funct7, rd, rs1, rs2 = op.custom0_operands()
    return rv_words.encode_r_type(
        opcode=rv_words.CUSTOM0_OPCODE,
        funct3=funct3,
        funct7=funct7,
        rd=rd,
        rs1=rs1,
        rs2=rs2,
    )


def compute_start_fixed(*, flags_reg: int, k_count_reg: int) -> ComputeStartOp:
    return ComputeStartOp(_fixed_reg(flags_reg), _fixed_reg(k_count_reg))


def compute_wait_fixed() -> ComputeWaitOp:
    return ComputeWaitOp()


def write_epilogue_param_fixed(
    field: int,
    *,
    channel_reg: int,
    value_reg: int,
) -> WriteEpilogueParamOp:
    return WriteEpilogueParamOp(
        _fixed_reg(channel_reg),
        _fixed_reg(value_reg),
        field=field,
    )


def write_config_fixed(register: int, *, value_reg: int) -> WriteConfigOp:
    return WriteConfigOp(_fixed_reg(value_reg), register=register)


def read_result_fixed(*, rd: int, index_reg: int) -> ReadResultOp:
    return ReadResultOp(
        _fixed_reg(index_reg),
        rd=xdsl_riscv.IntRegisterType.from_index(rd),
    )


def _fixed_reg(index: int):
    return rv32.GetRegisterOp(xdsl_riscv.IntRegisterType.from_index(index)).res
