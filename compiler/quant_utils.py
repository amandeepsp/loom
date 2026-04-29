"""Quantization utilities for deriving hardware epilogue parameters."""

from __future__ import annotations

import numpy as np


def quantize_multiplier_less_than_one(
    multiplier: float,
) -> tuple[np.int32, np.int32]:
    """Decompose a float multiplier into multiplier * 2^(-shift).

    The hardware epilogue computes:
        SRDHM(acc, mult) = round(acc * mult / 2^31)
        RDBPOT(x, shift)  = x >> shift

    Combined: round(acc * mult / 2^(31 + shift))

    This should equal round(acc * scale), so:
        mult / 2^(31 + shift) ≈ scale

    For scale < 1.0, shift=0 and mult = round(scale * 2^31).
    For scale >= 1.0, we reduce mult (right-shift) and increase shift.
    """
    if multiplier <= 0.0:
        return np.int32(0), np.int32(0)

    mult_q31 = int(round(multiplier * (1 << 31)))
    shift = 0

    while mult_q31 >= (1 << 31):
        mult_q31 >>= 1
        shift += 1

    return np.int32(mult_q31), np.int32(shift)
