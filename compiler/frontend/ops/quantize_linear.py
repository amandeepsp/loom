"""QuantizeLinear op: ONNX QuantizeLinear → quant.qcast + quant.scast."""

from typing import Sequence

from xdsl.ir import SSAValue
from xdsl.builder import Builder
from xdsl.ir.core import OpResult
from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, TensorType, f32
from xdsl.dialects.tosa import AddOp, ConstOp, MulOp

import onnx_ir

from ..helpers import i8
from ..tosa_ext import CastOp


def handle(node: onnx_ir.Node, operands: list[SSAValue], builder: Builder) -> Sequence[OpResult]:
    scale = operands[1]
    zero_point = operands[2]

    if not isinstance(scale.owner, ConstOp) or not isinstance(zero_point.owner, ConstOp):
        raise ValueError(
            "QuantizeLinear node must have constant scale and zero_point inputs"
        )

    if len(scale.type.shape) > 0:
        raise ValueError("Only per-layer quantization is supported")

    scale_value = scale.owner.values.get_values()[0]

    if len(zero_point.type.shape) > 0:
        raise ValueError("Only per-layer quantization is supported")

    zero_point_value = zero_point.owner.values.get_values()[0]

    input_type = operands[0].type
    if isinstance(input_type, TensorType):
        input_shape = input_type.get_shape()
        rank = len(input_shape)
        float_result_type = TensorType(f32, input_shape)
        storage_result_type = TensorType(i8, input_type.get_shape())
    else:
        rank = 0
        float_result_type = f32
        storage_result_type = i8

    scale_op = _float_const(builder, 1.0 / float(scale_value), rank)
    shift_op = _shift_const(builder)
    scaled = builder.insert(
        MulOp(
            operands=[operands[0], scale_op, shift_op],
            result_types=[float_result_type],
        )
    ).results[0]

    zp_op = _float_const(builder, float(zero_point_value), rank)
    shifted = builder.insert(
        AddOp(
            operands=[scaled, zp_op],
            result_types=[float_result_type],
        )
    ).results[0]

    cast_op = builder.insert(CastOp(shifted, storage_result_type))
    return [cast_op.results[0]]


def _float_const(builder: Builder, value: float, rank: int) -> SSAValue:
    shape = [1] * rank
    return builder.insert(
        ConstOp(DenseIntOrFPElementsAttr.from_list(TensorType(f32, shape), [value]))
    ).results[0]


def _shift_const(builder: Builder) -> SSAValue:
    return builder.insert(
        ConstOp(DenseIntOrFPElementsAttr.from_list(TensorType(i8, [1]), [0]))
    ).results[0]
