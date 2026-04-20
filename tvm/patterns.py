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


def extract_partitioned_gemms(mod: tvm.IRModule) -> list[dict[str, int | str]]:
    """Extract logical GEMM shapes from accel-partitioned functions."""

    gemms: list[dict[str, int | str]] = []
    for global_var, func in mod.functions.items():
        if not isinstance(func, relax.Function):
            continue

        attrs = func.attrs
        if not attrs or "Codegen" not in attrs:
            continue
        if str(attrs["Codegen"]) != ACCEL_CODEGEN_NAME:
            continue

        params = list(func.params)
        if len(params) < 2:
            continue

        lhs_sinfo = params[0].struct_info
        rhs_sinfo = params[1].struct_info
        if not isinstance(lhs_sinfo, relax.TensorStructInfo):
            continue
        if not isinstance(rhs_sinfo, relax.TensorStructInfo):
            continue
        if lhs_sinfo.shape is None or rhs_sinfo.shape is None:
            continue

        gemms.append(
            {
                "name": global_var.name_hint,
                "M": int(lhs_sinfo.shape[0]),
                "K": int(lhs_sinfo.shape[1]),
                "N": int(rhs_sinfo.shape[1]),
            }
        )

    return gemms


def extract_composite_params(func: relax.Function) -> dict:
    """Extract quantization parameters from a partitioned composite function.

    Returns a dict with:
    - lhs: int8 input tensor name
    - rhs: int8 weight tensor name
    - bias: bias tensor name
    - input_scale, input_zp: input dequant params
    - weight_scale, weight_zp: weight dequant params
    - output_scale, output_zp: output requant params
    - bias_scale, bias_zp: bias dequant params
    """
    params = {}

    def collect(expr):
        if isinstance(expr, relax.Call):
            op_name = expr.op.name if hasattr(expr.op, "name") else str(expr.op)
            if op_name == "relax.dequantize":
                input_arg = expr.args[0]
                if isinstance(input_arg, relax.Var):
                    sinfo = input_arg.struct_info
                    if hasattr(sinfo, "dtype") and sinfo.dtype in ("int8", "uint8"):
                        name = input_arg.name_hint
                        scale_val = None
                        zp_val = None
                        for i, arg in enumerate(expr.args[1:]):
                            if isinstance(arg, relax.Constant):
                                if i == 0:
                                    scale_val = arg.data.numpy().item()
                                else:
                                    zp_val = arg.data.numpy().item()
                        params[name] = {"scale": scale_val, "zp": zp_val}
            for arg in expr.args:
                collect(arg)
        elif isinstance(expr, relax.Var):
            sinfo = expr.struct_info
            if hasattr(sinfo, "dtype") and sinfo.dtype in ("int8", "uint8"):
                name = expr.name_hint
                if name not in params:
                    params[name] = {"scale": None, "zp": None}

    if hasattr(func, "body"):
        collect(func.body)

    return params
