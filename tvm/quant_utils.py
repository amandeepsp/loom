"""Quantization utilities for deriving hardware epilogue parameters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class EpilogueParams:
    """Hardware epilogue parameters for a single output channel."""

    bias: np.int32
    multiplier: np.int32
    shift: np.int32


@dataclass
class LayerEpilogueParams:
    """Complete epilogue parameters for a quantized layer."""

    bias: np.ndarray
    multiplier: np.ndarray
    shift: np.ndarray
    output_offset: np.int8
    activation_min: np.int8
    activation_max: np.int8


def quantize_multiplier_less_than_one(
    multiplier: float,
) -> tuple[np.int32, np.int32]:
    """Decompose a float multiplier into multiplier * 2^(-shift).

    This is the standard TFLite/CMSIS-NN approach for computing the
    fixed-point multiplier and shift for requantization.

    The effective multiplier is: multiplier * 2^(-shift)
    which approximates the original float_scale.

    Parameters
    ----------
    multiplier:
        A positive float less than 1.0 representing the requantization scale.

    Returns
    -------
    Tuple of (fixed_point_multiplier, shift)
        - fixed_point_multiplier: int32 in range [0, 2^31)
        - shift: int32 in range [0, 31]
    """
    if multiplier <= 0.0:
        return np.int32(0), np.int32(0)

    significand, exponent = math.frexp(multiplier)
    significand = significand * 2.0

    while significand < 0.5:
        significand *= 2.0
        exponent -= 1

    while significand >= 1.0:
        significand *= 0.5
        exponent += 1

    significand_int = int(round(significand * (1 << 31)))

    if significand_int >= (1 << 31):
        significand_int >>= 1
        exponent += 1

    shift = max(0, -exponent)

    return np.int32(significand_int), np.int32(shift)


def compute_requantization_params(
    input_scale: float,
    input_zero_point: int,
    weight_scale: float,
    weight_zero_point: int,
    output_scale: float,
    output_zero_point: int,
    bias_fp32: np.ndarray,
    has_relu: bool = False,
    activation_is_signed: bool = True,
) -> LayerEpilogueParams:
    """Compute hardware epilogue parameters from quantization constants.

    This function derives the per-channel and per-layer parameters needed
    by the hardware epilogue from the static quantization parameters.

    The full quantization math for QDQ-style quantized matmul is:
        output_fp32 = sum((lhs_q[i] - lhs_zp) * (rhs_q[j] - rhs_zp)) / (lhs_s * rhs_s)
        output_q = round(output_fp32 / output_s) + output_zp

    The hardware epilogue implements:
        result = bias + acc
        result = SRDHM(result, multiplier)
        result = RDBPOT(result, shift)
        result = result + output_offset
        result = clamp(result, act_min, act_max)
        output_q = result

    Parameters
    ----------
    input_scale:
        Scale for dequantizing input activations.
    input_zero_point:
        Zero point for dequantizing input activations.
    weight_scale:
        Scale for dequantizing weights.
    weight_zero_point:
        Zero point for dequantizing weights.
    output_scale:
        Scale for requantizing output.
    output_zero_point:
        Zero point for requantizing output.
    bias_fp32:
        Float32 bias tensor (shape: [N]).
    has_relu:
        Whether the layer has ReLU activation.
    activation_is_signed:
        Whether the quantized activation is signed (int8 vs uint8).

    Returns
    -------
    LayerEpilogueParams with per-channel bias/multiplier/shift and
    per-layer output_offset/activation_min/activation_max.
    """
    n = len(bias_fp32)

    combined_scale = (input_scale * weight_scale) / output_scale

    bias_int32 = np.round(bias_fp32 / output_scale).astype(np.int32)

    multipliers = np.zeros(n, dtype=np.float64)
    shifts = np.zeros(n, dtype=np.int32)

    for i in range(n):
        mult, shift = quantize_multiplier_less_than_one(combined_scale)
        multipliers[i] = mult
        shifts[i] = shift

    output_offset = np.int8(output_zero_point)

    if activation_is_signed:
        act_min = -128
        act_max = 127
    else:
        act_min = 0
        act_max = 255

    if has_relu:
        if activation_is_signed:
            act_min = 0
        else:
            act_min = 0

    return LayerEpilogueParams(
        bias=bias_int32,
        multiplier=multipliers.astype(np.int32),
        shift=shifts,
        output_offset=output_offset,
        activation_min=np.int8(act_min),
        activation_max=np.int8(act_max),
    )


def compute_scale_multiplier_shift(
    input_scale: float,
    weight_scale: float,
    output_scale: float,
) -> tuple[np.int32, np.int32]:
    """Compute a single multiplier/shift for uniform-scale layers.

    For per-tensor quantization where all channels share the same scale.
    """
    combined_scale = (input_scale * weight_scale) / output_scale
    return quantize_multiplier_less_than_one(combined_scale)


def extract_quant_params_from_relax_constant(
    expr: Any,
) -> tuple[float, int] | None:
    """Extract scale and zero_point from a Relax Constant expression.

    Returns (scale, zero_point) if the expression is a scalar Constant,
    None otherwise.
    """
    from tvm import relax

    if isinstance(expr, relax.Constant):
        arr = expr.data.numpy()
        if arr.ndim == 0:
            return float(arr.item()), 0
        elif arr.size == 1:
            return float(arr.flatten()[0]), 0
    return None


def extract_quant_params_from_ndarray(
    data: Any,
) -> tuple[float, int] | None:
    """Extract scale and zero_point from an NDArray or numpy array.

    Returns (scale, zero_point) if the data is a scalar,
    None otherwise.
    """
    if hasattr(data, "numpy"):
        arr = data.numpy()
    elif isinstance(data, np.ndarray):
        arr = data
    else:
        return None

    if arr.ndim == 0:
        return float(arr.item()), 0
    elif arr.size == 1:
        return float(arr.flatten()[0]), 0
    return None
