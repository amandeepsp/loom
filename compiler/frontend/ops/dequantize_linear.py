"""DequantizeLinear op: ONNX DequantizeLinear → quant.scast + quant.dcast."""

from typing import Sequence

from xdsl.ir import SSAValue
from xdsl.builder import Builder
from xdsl.ir.core import OpResult
from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    DenseIntOrFPElementsAttr,
    TensorType,
    f32,
    i32,
)
from xdsl.dialects.tensor import ReshapeOp
from xdsl.dialects.tosa import ConstOp
from xdsl.dialects.tosa import AddOp, MulOp

import onnx_ir

from ..helpers import i8
from ..tosa_ext import CastOp


def handle(node: onnx_ir.Node, operands: list[SSAValue], builder: Builder) -> Sequence[OpResult]:
    scale = operands[1]
    zero_point = operands[2]

    if not isinstance(scale.owner, ConstOp) or not isinstance(zero_point.owner, ConstOp):
        raise ValueError(
            "DequantizeLinear node must have constant scale and zero_point inputs"
        )

    if len(scale.type.shape) > 0:
        raise ValueError("Only per-layer quantization is supported")

    scale_value = scale.owner.values.get_values()[0]

    if len(zero_point.type.shape) > 0:
        raise ValueError("Only per-layer quantization is supported")

    zero_point_value = zero_point.owner.values.get_values()[0]

    input_type = operands[0].type
    if isinstance(input_type, TensorType):
        rank = len(input_type.get_shape())
        float_type = TensorType(f32, input_type.get_shape())
    else:
        rank = 0
        float_type = f32

    cast_op = builder.insert(CastOp(operands[0], float_type))
    neg_zp = _float_const(builder, -float(zero_point_value), rank)
    shifted = builder.insert(
        AddOp(
            operands=[cast_op.results[0], neg_zp],
            result_types=[float_type],
        )
    ).results[0]

    scale_const = _float_const(builder, float(scale_value), rank)
    shift_const = builder.insert(
        ConstOp(DenseIntOrFPElementsAttr.from_list(TensorType(i8, [1]), [0]))
    ).results[0]
    scaled = builder.insert(
        MulOp(
            operands=[shifted, scale_const, shift_const],
            result_types=[float_type],
        )
    ).results[0]

    result = scaled
    if isinstance(input_type, TensorType) and len(input_type.get_shape()) == 3:
        out_shape = node.outputs[0].shape
        if out_shape is not None and len(out_shape) == 2:
            from ..helpers import tensor_from_onnx_tensor

            final_type = tensor_from_onnx_tensor(node.outputs[0])
            shape_const = builder.insert(
                ConstOp(
                    DenseIntOrFPElementsAttr.from_list(
                        TensorType(i32, [2]),
                        [
                            -1 if dim == DYNAMIC_INDEX else dim
                            for dim in final_type.get_shape()
                        ],
                    )
                )
            )
            result = builder.insert(
                ReshapeOp(result, shape_const.results[0], final_type)
            ).results[0]

    return [result]


def _float_const(builder: Builder, value: float, rank: int) -> SSAValue:
    return builder.insert(
        ConstOp(
            DenseIntOrFPElementsAttr.from_list(
                TensorType(f32, [1] * rank), [value]
            )
        )
    ).results[0]
