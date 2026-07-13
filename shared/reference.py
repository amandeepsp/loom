"""Canonical reference implementations matching hardware epilogue pipeline."""

import numpy as np

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


def ref_srdhm(a: int, b: int) -> int:
    """Saturating Rounding Doubling High Multiply.

    Matches gemmlowp: pos_nudge = 1 << 30, neg_nudge = 1 - (1 << 30),
    followed by signed division that truncates toward zero.
    Saturates INT32_MIN * INT32_MIN -> INT32_MAX.
    """
    if a == INT32_MIN and b == INT32_MIN:
        return INT32_MAX
    ab = a * b
    nudge = (1 << 30) if ab >= 0 else (1 - (1 << 30))
    nudged = ab + nudge
    magnitude = abs(nudged) // (1 << 31)
    result = magnitude if nudged >= 0 else -magnitude
    return max(INT32_MIN, min(INT32_MAX, result))


def ref_rdbpot(x: int, exponent: int) -> int:
    """Rounding Divide by Power of Two."""
    if exponent == 0:
        return x
    mask = (1 << exponent) - 1
    remainder = x & mask
    threshold = (mask >> 1) + ((x >> 31) & 1)
    return (x >> exponent) + (1 if remainder > threshold else 0)


def ref_epilogue(acc, bias, multiplier, shift, offset, act_min, act_max) -> int:
    """Single-element epilogue: bias -> SRDHM -> RDBPOT -> offset -> clamp."""
    x = acc + bias
    x = ref_srdhm(x, multiplier)
    x = ref_rdbpot(x, shift)
    x += offset
    return max(act_min, min(act_max, x))


def cpu_requantize(
    acc: np.ndarray,
    bias: np.ndarray,
    multiplier: np.ndarray,
    shift: np.ndarray,
    output_offset: int,
    act_min: int,
    act_max: int,
) -> np.ndarray:
    """Vectorized CPU reference requantization matching the hardware pipeline."""
    x = acc.astype(np.int64) + bias.astype(np.int64)
    out = np.zeros(acc.shape, dtype=np.int8)
    for r in range(acc.shape[0]):
        for c in range(acc.shape[1]):
            val = int(x[r, c])
            val = ref_srdhm(val, int(multiplier[c]))
            val = ref_rdbpot(val, int(shift[c]))
            val += output_offset
            out[r, c] = max(act_min, min(act_max, val))
    return out
