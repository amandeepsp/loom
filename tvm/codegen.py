"""Lowering from Relax composite regions to accel runtime calls."""

from __future__ import annotations

from typing import Any

import numpy as np
import tvm
from tvm import relax

from runtime import _load_local_module

_patterns = _load_local_module("patterns", "patterns.py")
ACCEL_CODEGEN_NAME = _patterns.ACCEL_CODEGEN_NAME
from shared.ir import ACCEL_EXTERN_PREFIX

COMPOSITE_CONSTANTS: dict[str, dict[str, Any]] = {}


def make_extern_symbol(global_var: relax.GlobalVar) -> str:
    """Return the packed-runtime symbol name for a lowered accel region."""

    name = global_var.name_hint.replace(".", "_")
    return f"{ACCEL_EXTERN_PREFIX}.{name}"


def _check_weight_permute(func: relax.Function) -> bool:
    """Return True if a dequantize(int8) feeds into permute_dims before matmul.

    ONNX stores weight as (out_features, in_features) = [256, 784].  The
    matmul pattern is ``permute_dims(dequantize(weight))``, so the hardware
    needs the transposed weight [784, 256].  We walk the dataflow graph of
    every nested function in the body and check whether any permute_dims
    consumes a dequantize of a 2-D int8 tensor.
    """
    def _check_nested(inner_func: relax.Function) -> bool:
        if not hasattr(inner_func.body, "blocks"):
            return False
        # Build a map from DataflowVar → binding value.
        var_map: dict = {}
        for block in inner_func.body.blocks:
            for binding in block.bindings:
                var_map[binding.var] = binding.value
        # Scan for permute_dims whose input is a dequantize of an int8 weight.
        for binding_val in var_map.values():
            if not isinstance(binding_val, relax.Call):
                continue
            op_name = binding_val.op.name if hasattr(binding_val.op, "name") else str(binding_val.op)
            if op_name != "relax.permute_dims":
                continue
            inp = binding_val.args[0]
            if not isinstance(inp, relax.DataflowVar):
                continue
            dq_expr = var_map.get(inp)
            if not isinstance(dq_expr, relax.Call):
                continue
            dq_op = dq_expr.op.name if hasattr(dq_expr.op, "name") else str(dq_expr.op)
            if dq_op != "relax.dequantize":
                continue
            dq_input = dq_expr.args[0]
            if isinstance(dq_input, relax.Constant):
                dq_arr = dq_input.data.numpy()
                if dq_arr.dtype in (np.int8, np.uint8) and dq_arr.ndim == 2:
                    return True
            elif hasattr(dq_input, "data"):
                dq_arr = dq_input.data.numpy()
                if dq_arr.dtype in (np.int8, np.uint8) and dq_arr.ndim == 2:
                    return True
        return False

    found = [False]
    def find(expr):
        if found[0]:
            return
        if isinstance(expr, relax.SeqExpr):
            for block in expr.blocks:
                for binding in block.bindings:
                    find(binding.value)
        elif isinstance(expr, relax.Function):
            if _check_nested(expr):
                found[0] = True
                return
            if hasattr(expr, "body"):
                find(expr.body)

    if hasattr(func, "body"):
        find(func.body)
    return found[0]


def _extract_composite_constants(func: relax.Function) -> dict[str, Any]:
    """Extract constants from a composite function body.

    Returns a dict with:
    - weight_data: int8 weight tensor (auto-transposed if permute_dims detected)
    - bias_data: int32 bias tensor
    - weight_scale, weight_zp: weight dequantization params
    - bias_scale, bias_zp: bias dequantization params
    - input_scale, input_zp: input dequantization params
    - output_scale, output_zp: output requantization params
    """
    constants: dict[str, Any] = {}

    def visit(expr):
        if isinstance(expr, relax.Function):
            if hasattr(expr, "body"):
                visit(expr.body)
            return

        if isinstance(expr, relax.SeqExpr):
            for block in expr.blocks:
                for binding in block.bindings:
                    visit(binding.value)
            return

        if isinstance(expr, relax.Call):
            op_name = expr.op.name if hasattr(expr.op, "name") else str(expr.op)

            if op_name == "relax.quantize":
                scale = expr.args[1]
                zp = expr.args[2]
                if isinstance(scale, relax.Constant):
                    constants["output_scale"] = scale.data.numpy().item()
                if isinstance(zp, relax.Constant):
                    constants["output_zp"] = int(zp.data.numpy().item())
                for arg in expr.args:
                    visit(arg)
                return

            if op_name == "relax.dequantize":
                data = expr.args[0]
                scale = expr.args[1]
                zp = expr.args[2]

                data_arr = None
                scale_val = None
                zp_val = None

                if isinstance(data, relax.Constant):
                    data_arr = data.data.numpy()
                elif hasattr(data, "data"):
                    data_arr = data.data.numpy()

                if isinstance(scale, relax.Constant):
                    scale_val = scale.data.numpy().item()
                if isinstance(zp, relax.Constant):
                    zp_val = zp.data.numpy().item()

                if data_arr is not None:
                    if data_arr.dtype in (np.int8, np.uint8):
                        constants["weight_data"] = data_arr
                        if scale_val is not None:
                            constants["weight_scale"] = scale_val
                        if zp_val is not None:
                            constants["weight_zp"] = int(zp_val)
                    elif data_arr.dtype == np.int32:
                        constants["bias_data"] = data_arr.astype(np.int32)
                        if scale_val is not None:
                            constants["bias_scale"] = scale_val
                        if zp_val is not None:
                            constants["bias_zp"] = int(zp_val)
                elif scale_val is not None and "input_scale" not in constants:
                    # Dequantize of a Var (not a constant), AND no input already set.
                    # Only the FIRST such dequantize is the real input activation.
                    constants["input_scale"] = scale_val
                    if zp_val is not None:
                        constants["input_zp"] = int(zp_val)

            for arg in expr.args:
                visit(arg)

    if hasattr(func, "body"):
        visit(func.body)

    # Auto-transpose: ONNX stores weight as (out_features, in_features) but
    # the matmul expects (in_features, out_features) via permute_dims.
    if "weight_data" in constants and _check_weight_permute(func):
        constants["weight_data"] = constants["weight_data"].T.copy()

    return constants


@relax.expr_functor.mutator
class _AccelRegionLowerer(relax.PyExprMutator):
    """Rewrite calls to accel-partitioned functions into runtime-call form."""

    def __init__(self, mod: tvm.IRModule) -> None:
        super().__init__(mod)
        self.mod_ = mod

    def transform(self) -> tvm.IRModule:
        """Return a module with accel calls lowered to `call_dps_packed`."""

        for global_var, func in self.mod_.functions.items():
            if not isinstance(func, relax.Function):
                continue
            updated_func = self.visit_expr(func)
            self.builder_.normalize(updated_func)
            self.builder_.update_func(global_var, updated_func)
        return self.builder_.get()

    def visit_call_(self, call: relax.Call) -> relax.Expr:
        call = super().visit_call_(call)

        if not isinstance(call.op, relax.GlobalVar):
            return call

        target = self.mod_[call.op]
        if not isinstance(target, relax.Function):
            return call

        attrs = target.attrs
        if not attrs or "Codegen" not in attrs:
            return call
        if str(attrs["Codegen"]) != ACCEL_CODEGEN_NAME:
            return call

        extern_symbol = make_extern_symbol(call.op)
        out_sinfo = call.struct_info
        if out_sinfo is None:
            raise ValueError(f"missing struct info on call to {call.op.name_hint}")

        # Extract constants from the codegen function body directly.
        # FuseOpsByPattern with bind_constants=True embeds the quantized
        # weights, biases, and scale/zp constants into the codegen-annotated
        # function itself.  LambdaLift later splits out the body into a
        # separate Composite inner function, but at this point the constants
        # are still in the codegen function we're looking at.
        constants = _extract_composite_constants(target)
        COMPOSITE_CONSTANTS[extern_symbol] = constants

        return relax.op.call_dps_packed(
            relax.ExternFunc(extern_symbol),
            relax.Tuple(list(call.args)),
            out_sinfo,
        )


def lower_accel_regions(mod: tvm.IRModule) -> tvm.IRModule:
    """Lower accel-targeted composite regions into runtime-call form.

    The current lowering pass rewrites calls to `Codegen="accel_cfu"` functions
    into `call_dps_packed(ExternFunc(...))` nodes. This gives the out-of-tree
    integration a real runtime-call boundary while the final packed function
    implementation is still being developed.
    """

    return _AccelRegionLowerer(mod).transform()


def get_composite_constants(symbol: str) -> dict[str, Any]:
    """Get the extracted constants for a composite function."""
    return COMPOSITE_CONSTANTS.get(symbol, {})
