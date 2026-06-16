"""Small TOSA op wrappers missing from the installed xDSL dialect."""

from xdsl.dialects.builtin import TensorType
from xdsl.ir import SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    result_def,
    traits_def,
)
from xdsl.traits import Pure


@irdl_op_definition
class CastOp(IRDLOperation):
    """Standard MLIR TOSA cast op.

    xDSL 0.65 does not expose this op, but LLVM MLIR does and lowers it through
    the TOSA conversion passes.
    """

    name = "tosa.cast"

    input = operand_def(TensorType)
    output = result_def(TensorType)

    assembly_format = "$input attr-dict `:` functional-type($input, $output)"
    traits = traits_def(Pure())

    def __init__(self, input: SSAValue, result_type: TensorType):
        super().__init__(operands=[input], result_types=[result_type])


@irdl_op_definition
class MatMulOp(IRDLOperation):
    """Standard MLIR TOSA matmul wrapper.

    The installed xDSL op incorrectly constrains the result element type to
    match the input element type, which rejects quantized i8xi8->i32 matmul.
    """

    name = "tosa.matmul"

    a = operand_def(TensorType)
    b = operand_def(TensorType)
    a_zp = operand_def(TensorType)
    b_zp = operand_def(TensorType)
    output = result_def(TensorType)

    assembly_format = "operands attr-dict `:` functional-type(operands, results)"
    traits = traits_def(Pure())

    def __init__(
        self,
        a: SSAValue,
        b: SSAValue,
        a_zp: SSAValue,
        b_zp: SSAValue,
        result_type: TensorType,
    ):
        super().__init__(
            operands=[a, b, a_zp, b_zp],
            result_types=[result_type],
        )
