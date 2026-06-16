"""Quantization helpers retained for tests and compiler lowering."""

from __future__ import annotations

from .frontend.helpers import compute_rescale_params


def quantize_multiplier_less_than_one(scale: float) -> tuple[int, int]:
    """Return gemmlowp-style multiplier/shift for a scale in ``[0, 1)``."""

    if not 0 <= scale < 1:
        raise ValueError("scale must be in [0, 1)")
    return compute_rescale_params(scale, 1.0, 1.0)
