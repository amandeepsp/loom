"""Shared helpers for ONNX → TOSA op handlers."""

import math

from xdsl.ir import SSAValue
from xdsl.builder import Builder
from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    DenseIntOrFPElementsAttr,
    IntegerType,
    TensorType,
    f32,
    i32,
)
from xdsl.dialects.tosa import ConstOp

import onnx_ir

i8 = IntegerType(8)

ONNX_DTYPE_MAP = {
    onnx_ir.DataType.FLOAT: f32,
    onnx_ir.DataType.INT8: i8,
    onnx_ir.DataType.INT32: i32,
}


def tensor_from_onnx_tensor(tensor: onnx_ir.Tensor) -> TensorType:
    if tensor.dtype not in ONNX_DTYPE_MAP:
        raise ValueError(f"Unsupported tensor dtype: {tensor.dtype}")
    tensor_dtype = ONNX_DTYPE_MAP[tensor.dtype]
    tensor_dims = [
        DYNAMIC_INDEX if isinstance(d, onnx_ir.SymbolicDim) else d
        for d in tensor.shape
    ]
    return TensorType(tensor_dtype, tensor_dims)


def compute_rescale_params(
    a_scale: float, b_scale: float, y_scale: float
) -> tuple[int, int]:
    """Compute TOSA rescale multiplier and shift from effective scale.

    Finds (multiplier, shift) such that:
        multiplier * 2^(-shift) ≈ (a_scale * b_scale) / y_scale

    Uses the standard gemmlowp/TFLite double-rounding decomposition
    with a 31-bit mantissa for int32 multiplier.
    """
    effective_scale = (a_scale * b_scale) / y_scale
    if effective_scale == 0.0:
        return 0, 0
    mantissa, exponent = math.frexp(effective_scale)

    multiplier = int(round(mantissa * (1 << 31)))
    if multiplier == (1 << 31):
        multiplier //= 2
        exponent += 1
    shift = 31 - exponent

    if shift < 0:
        multiplier = int(round(effective_scale * (1 << 31)))
        shift = 0

    multiplier = max(1, min(multiplier, (1 << 31) - 1))
    shift = max(0, min(shift, 62))
    return multiplier, shift


def make_scalar_const(builder: Builder, value: int, dtype: IntegerType) -> SSAValue:
    """Create a rank-0 scalar constant tensor."""
    scalar_type = TensorType(dtype, [])
    return builder.insert(
        ConstOp(DenseIntOrFPElementsAttr.from_list(scalar_type, [value]))
    ).results[0]


def product_shape(dims: list[int] | tuple[int, ...]) -> int:
    """Return the product of static dims, or MLIR's dynamic marker."""

    if any(dim == DYNAMIC_INDEX for dim in dims):
        return DYNAMIC_INDEX
    return math.prod(dims)
