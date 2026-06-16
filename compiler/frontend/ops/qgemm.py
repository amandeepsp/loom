"""QGemm op: ONNX QGemm → tosa.matmul with 3D reshape.

TOSA matmul requires rank-3 [1, M, K] × [1, K, N]. We reshape the
2D inputs, keeping batch at shape[0] by using concrete leading dim 1.
"""

from typing import Sequence

import numpy as np

from xdsl.ir import SSAValue
from xdsl.builder import Builder
from xdsl.ir.core import OpResult
from xdsl.dialects.builtin import (
    BoolAttr,
    DYNAMIC_INDEX,
    DenseIntOrFPElementsAttr,
    TensorType,
    i32,
)
from xdsl.dialects.tensor import ReshapeOp
from xdsl.dialects.tosa import (
    AddOp,
    ConstOp,
    RescaleOp,
    RoundingMode,
    RoundingModeAttr,
)

import onnx_ir

from ..helpers import i8, compute_rescale_params
from ..tosa_ext import MatMulOp


def _sanitize(name: str) -> str:
    return name.translate(str.maketrans("/:", "__"))


def handle(node: onnx_ir.Node, operands: list[SSAValue], builder: Builder) -> Sequence[OpResult]:
    # operands: a(i8), a_scale(f32), a_zp(i8), b(i8), b_scale(f32), b_zp(i8),
    #            bias(i32), y_scale(f32), y_zp(i8)
    a, a_scale_sv, a_zp_sv = operands[0], operands[1], operands[2]
    b, b_scale_sv, b_zp_sv = operands[3], operands[4], operands[5]
    bias_sv = operands[6]
    y_scale_sv, y_zp_sv = operands[7], operands[8]

    def _get_scalar(sv: SSAValue) -> float | int:
        return sv.owner.values.get_values()[0]

    a_scale_val = _get_scalar(a_scale_sv)
    a_zp_val = int(_get_scalar(a_zp_sv))
    b_scale_val = _get_scalar(b_scale_sv)
    b_zp_val = int(_get_scalar(b_zp_sv))
    y_scale_val = _get_scalar(y_scale_sv)
    y_zp_val = int(_get_scalar(y_zp_sv))

    attrs = {name: attr.value for name, attr in node.attributes.items()}
    if int(attrs.get("transA", 0)) != 0:
        raise ValueError("QGemm transA is not supported")
    if int(attrs.get("transB", 0)) != 1:
        raise ValueError("QGemm currently requires transB=1")
    if float(attrs.get("alpha", 1.0)) != 1.0:
        raise ValueError("QGemm alpha values other than 1.0 are not supported")

    a = _rank3_activation(builder, a)
    a_shape = a.type.get_shape()  # [1, M, K]
    b_shape = b.type.get_shape()  # [N, K]  (stored transposed, transB=1)
    _batch, M, K_a = a_shape
    N, K_b = b_shape[0], b_shape[1]

    if K_a != K_b:
        raise ValueError(f"QGemm K-dim mismatch: A has {K_a}, B has {K_b}")
    K = K_a

    # Transpose and rank-lift weight: [N, K] -> [1, K, N].
    weight_data = b.owner.values
    weight_np = np.array(weight_data.get_values(), dtype=np.int8).reshape(N, K)
    weight_t_np = weight_np.T.copy().reshape(1, K, N)

    weight_t_const = builder.insert(
        ConstOp(
            DenseIntOrFPElementsAttr.from_list(
                TensorType(i8, [1, K, N]), weight_t_np.flatten().tolist()
            )
        )
    )
    b_t = weight_t_const.results[0]

    # tosa.matmul: i8 x i8 -> i32 accumulator.
    a_zp_scalar = _const_1d(builder, a_zp_val, i8)
    b_zp_scalar = _const_1d(builder, b_zp_val, i8)

    matmul_op = builder.insert(
        MatMulOp(
            a,
            b_t,
            a_zp_scalar,
            b_zp_scalar,
            TensorType(i32, [1, M, N]),
        )
    )

    bias = _rank3_bias(builder, bias_sv, N)
    biased = builder.insert(
        AddOp(
            operands=[matmul_op.results[0], bias],
            result_types=[TensorType(i32, [1, M, N])],
        )
    ).results[0]

    multiplier_val, shift_val = compute_rescale_params(
        a_scale_val, b_scale_val, y_scale_val
    )
    epilogue_name = _sanitize(node.outputs[0].name) + "_epilogue"

    mult_op = builder.insert(
        ConstOp(
            DenseIntOrFPElementsAttr.from_list(TensorType(i32, [1]), [multiplier_val])
        )
    )
    mult_op.results[0].name_hint = epilogue_name + "_multiplier"

    shift_op = builder.insert(
        ConstOp(DenseIntOrFPElementsAttr.from_list(TensorType(i8, [1]), [shift_val]))
    )
    shift_op.results[0].name_hint = epilogue_name + "_shift"

    input_zp_op = builder.insert(
        ConstOp(DenseIntOrFPElementsAttr.from_list(TensorType(i32, [1]), [0]))
    )

    out_zp_op = builder.insert(
        ConstOp(DenseIntOrFPElementsAttr.from_list(TensorType(i8, [1]), [y_zp_val]))
    )
    out_zp_op.results[0].name_hint = epilogue_name + "_out_zp"

    rescale = builder.insert(
        RescaleOp(
            operands=[
                biased,
                mult_op.results[0],
                shift_op.results[0],
                input_zp_op.results[0],
                out_zp_op.results[0],
            ],
            result_types=[TensorType(i8, [1, M, N])],
            properties={
                "scale32": BoolAttr.from_bool(True),
                "rounding_mode": RoundingModeAttr(RoundingMode.DOUBLE_ROUND),
                "per_channel": BoolAttr.from_bool(False),
                "input_unsigned": BoolAttr.from_bool(False),
                "output_unsigned": BoolAttr.from_bool(False),
            },
        )
    )

    return [rescale.results[0]]


def _reshape(
    builder: Builder, tensor: SSAValue, new_shape: list[int]
) -> SSAValue:
    """Reshape a tensor to a new shape using tensor.reshape."""
    elem_type = tensor.type.get_element_type()
    shape_const = builder.insert(
        ConstOp(
            DenseIntOrFPElementsAttr.from_list(
                TensorType(i32, [len(new_shape)]),
                [_shape_const_dim(dim) for dim in new_shape],
            )
        )
    )
    return builder.insert(
        ReshapeOp(
            tensor,
            shape_const.results[0],
            TensorType(elem_type, new_shape),
        )
    ).results[0]


def _shape_const_dim(dim: int) -> int:
    return -1 if dim == DYNAMIC_INDEX else dim


def _rank3_activation(builder: Builder, tensor: SSAValue) -> SSAValue:
    """Canonicalize FC activations to [1, M, K] for TOSA matmul."""

    shape = tensor.type.get_shape()
    if len(shape) == 3:
        if shape[0] != 1 and shape[1] == 1:
            return _reshape(builder, tensor, [1, shape[0], shape[2]])
        return tensor
    if len(shape) != 2:
        raise ValueError(f"QGemm input must be rank-2 or rank-3, got {shape}")
    return _reshape(builder, tensor, [1, shape[0], shape[1]])


def _const_1d(builder: Builder, value: int, dtype) -> SSAValue:
    return builder.insert(
        ConstOp(DenseIntOrFPElementsAttr.from_list(TensorType(dtype, [1]), [value]))
    ).results[0]


def _rank3_bias(builder: Builder, bias: SSAValue, n: int) -> SSAValue:
    """Canonicalize bias to [1, 1, N] so TOSA add can broadcast."""

    shape = bias.type.get_shape()
    if shape == (1, 1, n):
        return bias
    if shape != (n,):
        raise ValueError(f"QGemm bias must have shape [{n}], got {shape}")

    if isinstance(bias.owner, ConstOp):
        values = list(bias.owner.values.get_values())
        const = builder.insert(
            ConstOp(
                DenseIntOrFPElementsAttr.from_list(
                    TensorType(i32, [1, 1, n]), values
                )
            )
        )
        const.results[0].name_hint = (bias.name_hint or "bias") + "_rank3"
        return const.results[0]

    return _reshape(builder, bias, [1, 1, n])
