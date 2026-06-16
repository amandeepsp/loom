"""xDSL dialect for loom accelerator plans."""

from __future__ import annotations

from xdsl.dialects.builtin import Dialect, IndexType, IntegerAttr, StringAttr, TensorType, i32
from xdsl.ir import Attribute, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
    traits_def,
)
from xdsl.traits import Pure
from xdsl.utils.exceptions import VerifyException


def _i32(value: int) -> IntegerAttr:
    return IntegerAttr(value, i32)


@irdl_op_definition
class GemmCandidateOp(IRDLOperation):
    """Recognized quantized TOSA GEMM region before memory planning."""

    name = "loom.gemm_candidate"

    accumulator = operand_def(TensorType)
    bias = operand_def(TensorType)
    multiplier = operand_def(TensorType)
    shift = operand_def(TensorType)
    input_zp = operand_def(TensorType)
    output_zp = operand_def(TensorType)
    output = result_def(TensorType)
    kernel_name = prop_def(StringAttr)

    traits = traits_def(Pure())

    def __init__(
        self,
        kernel_name: str,
        accumulator: SSAValue,
        bias: SSAValue,
        multiplier: SSAValue,
        shift: SSAValue,
        input_zp: SSAValue,
        output_zp: SSAValue,
        result_type: TensorType,
    ):
        super().__init__(
            operands=[accumulator, bias, multiplier, shift, input_zp, output_zp],
            result_types=[result_type],
            properties={"kernel_name": StringAttr(kernel_name)},
        )


@irdl_op_definition
class GemmOp(IRDLOperation):
    """A fully planned int8 GEMM for the loom CFU."""

    name = "loom.gemm"

    kernel_name = prop_def(StringAttr)
    m = prop_def(IntegerAttr)
    k = prop_def(IntegerAttr)
    n = prop_def(IntegerAttr)
    padded_m = prop_def(IntegerAttr)
    tile = prop_def(IntegerAttr)
    input_addr = prop_def(IntegerAttr)
    weight_addr = prop_def(IntegerAttr)
    output_addr = prop_def(IntegerAttr)
    bias_addr = prop_def(IntegerAttr)
    multiplier_addr = prop_def(IntegerAttr)
    shift_addr = prop_def(IntegerAttr)
    output_offset = prop_def(IntegerAttr)
    activation_min = prop_def(IntegerAttr)
    activation_max = prop_def(IntegerAttr)

    traits = traits_def(Pure())

    def __init__(
        self,
        kernel_name: str,
        *,
        m: int,
        k: int,
        n: int,
        padded_m: int,
        tile: int,
        input_addr: int,
        weight_addr: int,
        output_addr: int,
        bias_addr: int,
        multiplier_addr: int,
        shift_addr: int,
        output_offset: int,
        activation_min: int = -128,
        activation_max: int = 127,
    ):
        super().__init__(
            properties={
                "kernel_name": StringAttr(kernel_name),
                "m": _i32(m),
                "k": _i32(k),
                "n": _i32(n),
                "padded_m": _i32(padded_m),
                "tile": _i32(tile),
                "input_addr": _i32(input_addr),
                "weight_addr": _i32(weight_addr),
                "output_addr": _i32(output_addr),
                "bias_addr": _i32(bias_addr),
                "multiplier_addr": _i32(multiplier_addr),
                "shift_addr": _i32(shift_addr),
                "output_offset": _i32(output_offset),
                "activation_min": _i32(activation_min),
                "activation_max": _i32(activation_max),
            }
        )

    def verify_(self) -> None:
        for attr_name in ("m", "k", "n", "padded_m", "tile"):
            value = _int_attr(getattr(self, attr_name))
            if value <= 0:
                raise VerifyException(f"loom.gemm {attr_name} must be positive")
        if _int_attr(self.padded_m) < _int_attr(self.m):
            raise VerifyException("loom.gemm padded_m must be >= m")


@irdl_op_definition
class RepackOp(IRDLOperation):
    """A planned row-major activation to loom packed-tile layout transform."""

    name = "loom.repack"

    source = prop_def(StringAttr)
    target = prop_def(StringAttr)
    m = prop_def(IntegerAttr)
    k = prop_def(IntegerAttr)
    tile = prop_def(IntegerAttr)

    traits = traits_def(Pure())

    def __init__(self, source: str, target: str, *, m: int, k: int, tile: int):
        super().__init__(
            properties={
                "source": StringAttr(source),
                "target": StringAttr(target),
                "m": _i32(m),
                "k": _i32(k),
                "tile": _i32(tile),
            }
        )

    def verify_(self) -> None:
        for attr_name in ("m", "k", "tile"):
            value = _int_attr(getattr(self, attr_name))
            if value <= 0:
                raise VerifyException(f"loom.repack {attr_name} must be positive")


@irdl_op_definition
class TileMmaOp(IRDLOperation):
    """One scheduled CFU tile MMA in a tiled affine loop body."""

    name = "loom.tile_mma"

    k_offset = operand_def(IndexType)
    k_count = prop_def(IntegerAttr)

    traits = traits_def(Pure())

    def __init__(self, k_offset: SSAValue, *, k_count: int):
        super().__init__(
            operands=[k_offset],
            properties={"k_count": _i32(k_count)},
        )

    def verify_(self) -> None:
        if _int_attr(self.k_count) <= 0:
            raise VerifyException("loom.tile_mma k_count must be positive")


@irdl_op_definition
class SetEpilogueOp(IRDLOperation):
    """Schedule per-channel epilogue writes for one N tile."""

    name = "loom.set_epilogue"

    n_offset = operand_def(IndexType)
    n_count = prop_def(IntegerAttr)

    traits = traits_def(Pure())

    def __init__(self, n_offset: SSAValue, *, n_count: int):
        super().__init__(operands=[n_offset], properties={"n_count": _i32(n_count)})

    def verify_(self) -> None:
        if _int_attr(self.n_count) <= 0:
            raise VerifyException("loom.set_epilogue n_count must be positive")


@irdl_op_definition
class TileLoadActOp(IRDLOperation):
    """Schedule activation DMA load for one M/K tile."""

    name = "loom.tile_load_act"

    m_offset = operand_def(IndexType)
    k_offset = operand_def(IndexType)
    k_count = prop_def(IntegerAttr)

    traits = traits_def(Pure())

    def __init__(self, m_offset: SSAValue, k_offset: SSAValue, *, k_count: int):
        super().__init__(
            operands=[m_offset, k_offset],
            properties={"k_count": _i32(k_count)},
        )

    def verify_(self) -> None:
        if _int_attr(self.k_count) <= 0:
            raise VerifyException("loom.tile_load_act k_count must be positive")


@irdl_op_definition
class TileLoadWgtOp(IRDLOperation):
    """Schedule weight DMA load for one N/K tile."""

    name = "loom.tile_load_wgt"

    n_offset = operand_def(IndexType)
    k_offset = operand_def(IndexType)
    k_count = prop_def(IntegerAttr)

    traits = traits_def(Pure())

    def __init__(self, n_offset: SSAValue, k_offset: SSAValue, *, k_count: int):
        super().__init__(
            operands=[n_offset, k_offset],
            properties={"k_count": _i32(k_count)},
        )

    def verify_(self) -> None:
        if _int_attr(self.k_count) <= 0:
            raise VerifyException("loom.tile_load_wgt k_count must be positive")


@irdl_op_definition
class TileStoreOp(IRDLOperation):
    """Schedule CFU result readback into output memory for one M/N tile."""

    name = "loom.tile_store"

    m_offset = operand_def(IndexType)
    n_offset = operand_def(IndexType)
    m_count = prop_def(IntegerAttr)
    n_count = prop_def(IntegerAttr)

    traits = traits_def(Pure())

    def __init__(self, m_offset: SSAValue, n_offset: SSAValue, *, m_count: int, n_count: int):
        super().__init__(
            operands=[m_offset, n_offset],
            properties={"m_count": _i32(m_count), "n_count": _i32(n_count)},
        )

    def verify_(self) -> None:
        if _int_attr(self.m_count) <= 0 or _int_attr(self.n_count) <= 0:
            raise VerifyException("loom.tile_store counts must be positive")


def _int_attr(attr: Attribute) -> int:
    if not isinstance(attr, IntegerAttr):
        raise VerifyException(f"expected integer attribute, got {attr}")
    return int(attr.value.data)


Loom = Dialect(
    "loom",
    [
        GemmOp,
        GemmCandidateOp,
        RepackOp,
        SetEpilogueOp,
        TileLoadActOp,
        TileLoadWgtOp,
        TileMmaOp,
        TileStoreOp,
    ],
    [],
)
