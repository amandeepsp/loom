"""
Quant dialect — per-tensor quantized types and cast ops for xDSL.
Doesn't yet support per channel quantization.
"""

from xdsl.traits import AlwaysSpeculatable
from xdsl.dialects.builtin import (
    AnyFloat,
    ArrayAttr,
    Float32Type,
    FloatAttr,
    IntegerAttr,
    IntegerType,
    Signedness,
    TensorType,
)
from xdsl.ir import (
    Attribute,
    Dialect,
    ParametrizedAttribute,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    result_def,
    traits_def,
)
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException

@irdl_attr_definition
class UniformQuantizedType(ParametrizedAttribute, TypeAttribute):
    """A per-layer quantized integer type

    Syntax: !quant.uniform<storageType:expressedType, scale:zeroPoint>
    Example: !quant.uniform<i8:f32, 0.00392:-128>
    """

    name = "quant.uniform"

    storage_type: IntegerType
    expressed_type: AnyFloat
    scale: FloatAttr
    zero_point: IntegerAttr

    def __init__(
        self,
        storage_type: IntegerType,
        expressed_type: AnyFloat,
        scale: FloatAttr,
        zero_point: IntegerAttr,
    ):
        super().__init__(storage_type, expressed_type, scale, zero_point)

    def print_builtin(self, printer: Printer):
        printer.print_string("quant.uniform")
        with printer.in_angle_brackets():
            printer.print_attribute(self.storage_type)
            printer.print_string(":")
            printer.print_attribute(self.expressed_type)
            printer.print_string(", ")
            printer.print_float(self.scale.value.data, self.expressed_type)
            zp = self.zero_point.value.data
            if zp != 0:
                printer.print_string(":")
                printer.print_int(zp, self.storage_type)

    @property
    def scale_val(self) -> float:
        return self.scale.value.data

    @property
    def zero_point_val(self) -> int:
        return self.zero_point.value.data

def _extract_quant_type(ty: Attribute) -> UniformQuantizedType | None:
    if isinstance(ty, UniformQuantizedType):
        return ty
    if isinstance(ty, TensorType):
        elem = ty.get_element_type()
        if isinstance(elem, UniformQuantizedType):
            return elem
    return None


def _float_result_type(input_type: Attribute, qt: UniformQuantizedType) -> Attribute:
    if isinstance(input_type, UniformQuantizedType):
        return qt.expressed_type
    if isinstance(input_type, TensorType):
        return TensorType(qt.expressed_type, input_type.get_shape())
    return qt.expressed_type


def _quant_result_type(input_type: Attribute, qt: UniformQuantizedType) -> Attribute:
    if isinstance(input_type, AnyFloat):
        return qt
    if isinstance(input_type, TensorType):
        return TensorType(qt, input_type.get_shape())
    return qt


def _same_shape(a: Attribute, b: Attribute) -> bool:
    if isinstance(a, TensorType) and isinstance(b, TensorType):
        return a.get_shape() == b.get_shape()
    return not isinstance(a, TensorType) and not isinstance(b, TensorType)

@irdl_op_definition
class QuantizeCastOp(IRDLOperation):
    """
    Quantize cast operation

    Syntax: operation ::= `quant.qcast` $input attr-dict `:` type($input) `to` type($result)
    Convert a floating-point value to a quantized type.
    See: https://mlir.llvm.org/docs/Dialects/QuantDialect/#quantqcast-quantquantizecastop
    """

    name = "quant.qcast"

    input = operand_def(Attribute)
    result = result_def(Attribute)

    traits = traits_def(AlwaysSpeculatable())

    def __init__(self, input: SSAValue, result_type: Attribute):
        super().__init__(operands=[input], result_types=[result_type])

    def verify_(self) -> None:
        out_qt = _extract_quant_type(self.result.type)
        if out_qt is None:
            raise VerifyException(
                f"quant.qcast output must be !quant.uniform, got {self.result.type}"
            )
        in_ty = self.input.type
        if isinstance(in_ty, TensorType):
            in_ty = in_ty.get_element_type()
        if not isinstance(in_ty, AnyFloat):
            raise VerifyException(f"quant.qcast input must be float, got {in_ty}")
        if not _same_shape(self.input.type, self.result.type):
            raise VerifyException("quant.qcast shape mismatch")

@irdl_op_definition
class DequantizeCastOp(IRDLOperation):
    """
    Dequantize cast operation

    Syntax: operation ::= `quant.dcast` $input attr-dict `:` type($input) `to` type($result)
    Convert an input quantized value into its expressed floating-point value.
    See: https://mlir.llvm.org/docs/Dialects/QuantDialect/#quantdcast-quantdequantizecastop
    """

    name = "quant.dcast"

    input = operand_def(Attribute)
    result = result_def(Attribute)

    traits = traits_def(AlwaysSpeculatable())

    def __init__(self, input: SSAValue, result_type: Attribute):
        super().__init__(operands=[input], result_types=[result_type])

    def verify_(self) -> None:
        in_qt = _extract_quant_type(self.input.type)
        if in_qt is None:
            raise VerifyException(
                f"quant.dcast input must be !quant.uniform, got {self.input.type}"
            )
        out_ty = self.result.type
        if isinstance(out_ty, TensorType):
            out_ty = out_ty.get_element_type()
        if not isinstance(out_ty, AnyFloat):
            raise VerifyException(f"quant.dcast output must be float, got {out_ty}")
        if in_qt.expressed_type != out_ty:
            raise VerifyException(
                f"quant.dcast expressed type mismatch: "
                f"{in_qt.expressed_type} vs {out_ty}"
            )
        if not _same_shape(self.input.type, self.result.type):
            raise VerifyException("quant.dcast shape mismatch")


@irdl_op_definition
class StorageCastOp(IRDLOperation):
    """
    Storage cast operation

    Syntax: operation ::= `quant.scast` $input attr-dict `:` type($input) `to` type($result)
    Convert a value from a quantized type to the corresponding signless integer storage
    type, or vice versa. This conversion simply involves a reinterpretation of the input
    bits and does not involve any data manipulation.
    """

    name = "quant.scast"

    input = operand_def(Attribute)
    result = result_def(Attribute)

    traits = traits_def(AlwaysSpeculatable())

    def __init__(self, input: SSAValue, result_type: Attribute):
        super().__init__(operands=[input], result_types=[result_type])

    def verify_(self) -> None:
        in_qt = _extract_quant_type(self.input.type)
        out_qt = _extract_quant_type(self.result.type)
        if (in_qt is None) == (out_qt is None):
            raise VerifyException(
                "quant.scast: exactly one side must be !quant.uniform"
            )
        if not _same_shape(self.input.type, self.result.type):
            raise VerifyException("quant.scast shape mismatch")


Quant = Dialect(
    "quant",
    [QuantizeCastOp, DequantizeCastOp, StorageCastOp],
    [UniformQuantizedType],
)
