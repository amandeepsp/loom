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


def make_matmul_requant_pattern(check=None) -> FusionPattern:
    """Create a fused quantized matmul pattern for static quantization.

    This pattern matches the Relax IR produced by the ONNX frontend when
    importing a statically-quantized ONNX model (QDQ format):

    `dequantize(quantize(add(matmul(dequantize(input), permute_dims(dequantize(weight))),
                               dequantize(bias))), out_scale, out_zp)`

    The composite captures:
    - input_q: int8 quantized input activations
    - input_scale, input_zp: scale and zero_point for input dequantization
    - weight_q: int8 quantized weights
    - weight_scale, weight_zp: scale and zero_point for weight dequantization
    - bias_q: quantized bias tensor
    - bias_scale, bias_zp: scale and zero_point for bias dequantization
    - output_scale, output_zp: scale and zero_point for requantization

    The output is int8 (after quantize), which matches the hardware contract.
    """

    input_q = wildcard()
    in_scale = wildcard()
    in_zp = wildcard()
    deq_in = is_op("relax.dequantize")(input_q, in_scale, in_zp)

    weight_q = wildcard()
    w_scale = wildcard()
    w_zp = wildcard()
    deq_w = is_op("relax.dequantize")(weight_q, w_scale, w_zp)
    perm_w = is_op("relax.permute_dims")(deq_w)

    mm = is_op("relax.matmul")(deq_in, perm_w)

    bias_q = wildcard()
    b_scale = wildcard()
    b_zp = wildcard()
    deq_b = is_op("relax.dequantize")(bias_q, b_scale, b_zp)

    biased = is_op("relax.add")(mm, deq_b)

    out_scale = wildcard()
    out_zp = wildcard()
    q_out = is_op("relax.quantize")(biased, out_scale, out_zp)
    output = is_op("relax.dequantize")(q_out, out_scale, out_zp)

    return FusionPattern(
        name=MATMUL_REQUANT_COMPOSITE_NAME,
        pattern=output,
        check=check,
    )


def make_matmul_requant_no_input_q_pattern(check=None) -> FusionPattern:
    """Pattern for quantized matmul where input is already float32.

    This matches: `dequantize(quantize(add(matmul(lhs_float32, permute_dims(dequantize(weight))),
                                          dequantize(bias)))), out_scale, out_zp)`

    Used for layers where input comes from a previous quantized layer.
    """

    lhs = wildcard()  # float32 input
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

    return FusionPattern(
        name=MATMUL_REQUANT_NO_INPUT_Q_COMPOSITE_NAME,
        pattern=output,
        check=check,
    )


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



