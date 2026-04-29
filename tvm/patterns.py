"""Composite matching for the out-of-tree accel backend."""

from __future__ import annotations

from collections.abc import Iterable

import tvm
from tvm import relax
from tvm.relax.dpl import is_op, wildcard
from tvm.relax.transform import FuseOpsByPattern, FusionPattern

ACCEL_CODEGEN_NAME = "accel_cfu"
MATMUL_REQUANT_COMPOSITE_NAME = f"{ACCEL_CODEGEN_NAME}.matmul_requant"
MATMUL_REQUANT_NO_INPUT_Q_COMPOSITE_NAME = f"{ACCEL_CODEGEN_NAME}.matmul_requant_no_input_q"


def make_matmul_requant_pattern(
    *,
    input_is_quantized: bool = True,
    check=None,
) -> FusionPattern:
    """Create a fused quantized matmul pattern for static quantization.

    Parameters
    ----------
    input_is_quantized:
        When ``True`` the LHS is ``dequantize(input_q)`` (first layer).
        When ``False`` the LHS is an un-quantized float32 tensor
        (subsequent layers fed by a previous quantized op).
    check:
        Optional pattern-check callback forwarded to ``FusionPattern``.

    The matched Relax graph is::

        dequantize(quantize(add(matmul(lhs, permute_dims(dequantize(weight))),
                                dequantize(bias))),
                   out_scale, out_zp)

    where ``lhs`` is either ``dequantize(input_q)`` or a ``wildcard()``.
    """

    lhs: tvm.relax.Expr = (
        is_op("relax.dequantize")(wildcard(), wildcard(), wildcard())
        if input_is_quantized
        else wildcard()
    )

    weight_q = wildcard()
    w_scale = wildcard()
    w_zp = wildcard()
    deq_w = is_op("relax.dequantize")(weight_q, w_scale, w_zp)
    perm_w = is_op("relax.permute_dims")(deq_w)

    mm = is_op("relax.matmul")(lhs, perm_w)

    bias_q = wildcard()
    b_scale = wildcard()
    b_zp = wildcard()
    deq_b = is_op("relax.dequantize")(bias_q, b_scale, b_zp)

    biased = is_op("relax.add")(mm, deq_b)

    out_scale = wildcard()
    out_zp = wildcard()
    q_out = is_op("relax.quantize")(biased, out_scale, out_zp)
    output = is_op("relax.dequantize")(q_out, out_scale, out_zp)

    name = (
        MATMUL_REQUANT_COMPOSITE_NAME
        if input_is_quantized
        else MATMUL_REQUANT_NO_INPUT_Q_COMPOSITE_NAME
    )
    return FusionPattern(name=name, pattern=output, check=check)


# Backwards-compatible aliases
def make_matmul_requant_no_input_q_pattern(check=None) -> FusionPattern:
    """Deprecated: use ``make_matmul_requant_pattern(input_is_quantized=False)``."""
    return make_matmul_requant_pattern(input_is_quantized=False, check=check)


def partition_for_accel_cfu(
    mod: tvm.IRModule,
    *,
    patterns: Iterable[FusionPattern] | None = None,
) -> tvm.IRModule:
    """Partition Relax regions intended for the accel backend.

    Parameters
    ----------
    mod:
        The Relax module to partition.
    patterns:
        Optional override for the patterns to use. When omitted, both
        matmul_requant patterns (with and without input quantize) are used.
    """

    if patterns is None:
        pattern_list = [
            make_matmul_requant_pattern(),
            make_matmul_requant_no_input_q_pattern(),
        ]
    else:
        pattern_list = list(patterns)
    if not pattern_list:
        raise ValueError("patterns must not be empty")

    return FuseOpsByPattern(
        pattern_list,
        bind_constants=True,
        annotate_codegen=True,
    )(mod)



