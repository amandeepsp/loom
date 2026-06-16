"""Flatten op: ONNX Flatten → tensor.collapse_shape."""

from typing import Sequence

from xdsl.ir import SSAValue
from xdsl.builder import Builder
from xdsl.ir.core import OpResult
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, TensorType, i64
from xdsl.dialects.tensor import CollapseShapeOp

import onnx_ir

from ..helpers import product_shape


def handle(node: onnx_ir.Node, operands: list[SSAValue], builder: Builder) -> Sequence[OpResult]:
    if not len(operands) == 1:
        raise ValueError("Flatten node must have exactly one input")

    in_type = operands[0].type
    in_shape = in_type.get_shape()

    axis_attr = node.attributes.get("axis")
    axis = int(axis_attr.value) if axis_attr is not None else 1

    if axis != 1:
        raise ValueError(f"Flatten only supports axis=1, got axis={axis}")

    # Keep the singleton channel dimension for FC/TOSA matmul canonicalization:
    # [B, 1, H, W] -> [B, 1, H*W]. QGemm then rank-lifts to [1, B, K],
    # which is the rank-3 form LLVM's TOSA-to-Linalg path accepts for matmul.
    if len(in_shape) == 4 and in_shape[1] == 1:
        out_type = TensorType(
            in_type.get_element_type(),
            [in_shape[0], 1, product_shape(in_shape[2:])],
        )
        reassoc = ArrayAttr(
            [
                ArrayAttr([IntegerAttr(0, i64)]),
                ArrayAttr([IntegerAttr(1, i64)]),
                ArrayAttr([IntegerAttr(i, i64) for i in range(2, len(in_shape))]),
            ]
        )
    else:
        out_type = TensorType(
            in_type.get_element_type(),
            [in_shape[0], product_shape(in_shape[1:])],
        )
        reassoc = ArrayAttr(
            [
                ArrayAttr([IntegerAttr(0, i64)]),
                ArrayAttr([IntegerAttr(i, i64) for i in range(1, len(in_shape))]),
            ]
        )

    reshape_op = builder.insert(
        CollapseShapeOp(
            operands=operands,
            result_types=[out_type],
            properties={"reassociation": reassoc},
        )
    ).result
    return [reshape_op]
