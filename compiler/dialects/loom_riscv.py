"""xDSL dialect for loom RISC-V CUSTOM0 operations.

This is the near-ISA layer for the loom CFU, following the same split used by
Snitch: higher-level scheduling ops lower to target-specific RISC-V extension
ops before final machine-code emission.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from xdsl.dialects import riscv
from xdsl.dialects.builtin import Dialect, IntegerAttr, NoneAttr, i32
from xdsl.ir import Operation, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
)
from xdsl.utils.exceptions import VerifyException


def _i32(value: int) -> IntegerAttr:
    return IntegerAttr(value, i32)


def _int_attr(attr: IntegerAttr) -> int:
    return int(attr.value.data)


def _verify_bits(name: str, attr: IntegerAttr, bits: int) -> None:
    value = _int_attr(attr)
    if not 0 <= value < (1 << bits):
        raise VerifyException(f"{name}={value} does not fit in {bits} bits")


class LoomCustom0Op(IRDLOperation, ABC):
    """Base class for loom CUSTOM0 operations."""

    @abstractmethod
    def custom0_operands(self) -> tuple[int, int, int, int, int]:
        """Return funct3, funct7, rd, rs1, rs2 for R-type CUSTOM0 encoding."""


@irdl_op_definition
class ComputeStartOp(LoomCustom0Op):
    """Start CFU tile compute."""

    name = "loom_riscv.compute_start"

    flags = operand_def(riscv.IntRegisterType)
    k_count = operand_def(riscv.IntRegisterType)

    def __init__(self, flags: SSAValue | Operation, k_count: SSAValue | Operation):
        super().__init__(operands=[flags, k_count])

    def custom0_operands(self) -> tuple[int, int, int, int, int]:
        return (0, 0, 0, _reg_index(self.flags.type), _reg_index(self.k_count.type))


@irdl_op_definition
class ComputeWaitOp(LoomCustom0Op):
    """Wait for CFU tile compute to complete."""

    name = "loom_riscv.compute_wait"

    def __init__(self):
        super().__init__()

    def custom0_operands(self) -> tuple[int, int, int, int, int]:
        return (1, 0, 0, 0, 0)


@irdl_op_definition
class WaitPreviousComputeOp(IRDLOperation):
    """Wait for the previous K tile compute, skipped when k_base is zero."""

    name = "loom_riscv.wait_previous_compute"

    k_base = operand_def(riscv.IntRegisterType)

    def __init__(self, k_base: SSAValue | Operation):
        super().__init__(operands=[k_base])


@irdl_op_definition
class WriteEpilogueParamOp(LoomCustom0Op):
    """Write one epilogue parameter for one output channel."""

    name = "loom_riscv.write_epilogue_param"

    field = prop_def(IntegerAttr)
    channel = operand_def(riscv.IntRegisterType)
    value = operand_def(riscv.IntRegisterType)

    def __init__(
        self,
        channel: SSAValue | Operation,
        value: SSAValue | Operation,
        *,
        field: int,
    ):
        super().__init__(
            operands=[channel, value],
            properties={
                "field": _i32(field),
            },
        )

    def custom0_operands(self) -> tuple[int, int, int, int, int]:
        return (
            2,
            _int_attr(self.field),
            0,
            _reg_index(self.channel.type),
            _reg_index(self.value.type),
        )

    def verify_(self) -> None:
        _verify_bits("field", self.field, 7)


@irdl_op_definition
class WriteConfigOp(LoomCustom0Op):
    """Write one scalar CFU configuration register."""

    name = "loom_riscv.write_config"

    register = prop_def(IntegerAttr)
    value = operand_def(riscv.IntRegisterType)

    def __init__(self, value: SSAValue | Operation, *, register: int):
        super().__init__(
            operands=[value],
            properties={
                "register": _i32(register),
            },
        )

    def custom0_operands(self) -> tuple[int, int, int, int, int]:
        return (3, _int_attr(self.register), 0, 0, _reg_index(self.value.type))

    def verify_(self) -> None:
        _verify_bits("register", self.register, 7)


@irdl_op_definition
class ReadResultOp(LoomCustom0Op):
    """Read one int8 result lane from the CFU result buffer."""

    name = "loom_riscv.read_result"

    index = operand_def(riscv.IntRegisterType)
    rd = result_def(riscv.IntRegisterType)

    def __init__(
        self,
        index: SSAValue | Operation,
        *,
        rd: riscv.IntRegisterType = riscv.Registers.UNALLOCATED_INT,
    ):
        super().__init__(
            operands=[index],
            result_types=[rd],
        )

    def custom0_operands(self) -> tuple[int, int, int, int, int]:
        return (4, 0, _reg_index(self.rd.type), _reg_index(self.index.type), 0)

    def verify_(self) -> None:
        _reg_index(self.rd.type)
        _reg_index(self.index.type)


def _reg_index(reg: riscv.IntRegisterType) -> int:
    if not reg.is_allocated or isinstance(reg.index, NoneAttr):
        raise VerifyException("loom_riscv custom op requires allocated register types before encoding")
    return int(reg.index.data)


LoomRISCV = Dialect(
    "loom_riscv",
    [
        ComputeStartOp,
        ComputeWaitOp,
        WaitPreviousComputeOp,
        WriteEpilogueParamOp,
        WriteConfigOp,
        ReadResultOp,
    ],
    [],
)
