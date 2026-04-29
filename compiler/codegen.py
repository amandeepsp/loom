"""Lowering from Relax composite regions to accel runtime calls."""

from __future__ import annotations

from typing import Any

import numpy as np
import tvm
from tvm import relax

from .patterns import ACCEL_CODEGEN_NAME
from shared.ir import ACCEL_EXTERN_PREFIX

COMPOSITE_CONSTANTS: dict[str, dict[str, Any]] = {}


def make_extern_symbol(global_var: relax.GlobalVar) -> str:
    """Return the packed-runtime symbol name for a lowered accel region."""

    name = global_var.name_hint.replace(".", "_")
    return f"{ACCEL_EXTERN_PREFIX}.{name}"


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

    # Detect no_input_q pattern: input is already dequantized outside the composite.
    composite_name = ""
    if hasattr(func, "attrs") and "Composite" in func.attrs:
        composite_name = str(func.attrs["Composite"])
    is_no_input_q = "no_input_q" in composite_name

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
                # Do NOT recurse into args[0] (the data subtree).  The outer
                # dequantize(quantize(...)) pair is the OUTPUT node; everything
                # upstream (input dequantize, weight dequantize, matmul) is
                # already visited via the SeqExpr binding loop.  Reentering
                # args[0] could reach a nested input quantize and overwrite
                # output_scale with input_scale.
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
                elif scale_val is not None and "input_scale" not in constants and not is_no_input_q:
                    # Dequantize of a Var (not a constant), AND no input already set.
                    # Only the FIRST such dequantize is the real input activation.
                    # Skip for no_input_q composites — input dequantize is outside.
                    constants["input_scale"] = scale_val
                    if zp_val is not None:
                        constants["input_zp"] = int(zp_val)

            for arg in expr.args:
                visit(arg)

    if hasattr(func, "body"):
        visit(func.body)

    # Auto-transpose: ONNX stores weight as (out_features, in_features) but
    # the hardware matmul expects (in_features, out_features).  All our patterns
    # include permute_dims, so the transpose is always needed.
    if "weight_data" in constants:
        constants["weight_data"] = constants["weight_data"].T.copy()

    return constants


def _build_var_map(expr: relax.Expr) -> dict[relax.Var, relax.Expr]:
    """Build a map from DataflowVar → binding value in a Relax expression."""
    var_map: dict[relax.Var, relax.Expr] = {}
    if isinstance(expr, relax.SeqExpr):
        for block in expr.blocks:
            for binding in block.bindings:
                var_map[binding.var] = binding.value
    return var_map


@relax.expr_functor.mutator
class _AccelRegionLowerer(relax.PyExprMutator):
    """Rewrite calls to accel-partitioned functions into runtime-call form.

    Uses a two-pass approach:
    1. Pre-extract constants from all codegen functions in the module.
    2. During lowering, trace no_input_q call arguments across function-call
       boundaries to find the input scale (the producer's output scale).
    """

    def __init__(self, mod: tvm.IRModule) -> None:
        super().__init__(mod)
        self.mod_ = mod
        self._var_map: dict[relax.Var, relax.Expr] = {}
        # Pass 1: extract constants from every codegen function once.
        self._codegen_constants: dict[relax.GlobalVar, dict[str, Any]] = {}
        for gv, func in mod.functions.items():
            if (
                isinstance(func, relax.Function)
                and hasattr(func, "attrs")
                and str(func.attrs.get("Codegen", "")) == ACCEL_CODEGEN_NAME
            ):
                self._codegen_constants[gv] = _extract_composite_constants(func)

    def _get_composite_name(self, func: relax.Function) -> str:
        """Return the Composite attribute name from a codegen function's inner function."""
        if not hasattr(func, "body"):
            return ""
        for block in func.body.blocks:
            for binding in block.bindings:
                val = binding.value
                # After FuseOpsByPattern the inner composite function is bound to a local
                # variable (e.g. local_func = @R.function(...)) and then called via that
                # variable.  We look at the Function definition directly.
                if isinstance(val, relax.Function):
                    if hasattr(val, "attrs") and "Composite" in val.attrs:
                        return str(val.attrs["Composite"])
                # Fallback: if the call operator is a Var that references the function,
                # we would need the var_map — but the Function binding above catches it.
        return ""

    def _trace_input_scale(
        self, expr: relax.Expr
    ) -> tuple[float | None, int | None]:
        """Trace an argument back to a dequantize or a codegen call's output scale."""
        current = expr
        for _ in range(10):
            if isinstance(current, (relax.Var, relax.DataflowVar)):
                if current not in self._var_map:
                    break
                current = self._var_map[current]
            elif isinstance(current, relax.Call) and isinstance(current.op, relax.GlobalVar):
                # Output of another codegen function — use its precomputed output scale.
                callee_gv = current.op
                if callee_gv in self._codegen_constants:
                    c = self._codegen_constants[callee_gv]
                    return c.get("output_scale"), c.get("output_zp")
                break
            else:
                break

        if isinstance(current, relax.Call):
            op_name = current.op.name if hasattr(current.op, "name") else str(current.op)
            if op_name == "relax.dequantize":
                scale = current.args[1]
                zp = current.args[2]
                scale_val = (
                    scale.data.numpy().item() if isinstance(scale, relax.Constant) else None
                )
                zp_val = (
                    int(zp.data.numpy().item()) if isinstance(zp, relax.Constant) else None
                )
                return scale_val, zp_val

        return None, None

    def transform(self) -> tvm.IRModule:
        """Return a module with accel calls lowered to `call_dps_packed`."""
        for global_var, func in self.mod_.functions.items():
            if not isinstance(func, relax.Function):
                continue
            self._var_map = _build_var_map(func.body) if hasattr(func, "body") else {}
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

        # Use precomputed constants from Pass 1.
        constants = dict(self._codegen_constants.get(call.op, {}))

        # For no_input_q composites, trace the first argument back across
        # function-call boundaries to find the producer's output scale.
        composite_name = self._get_composite_name(target)
        if "no_input_q" in composite_name and call.args:
            first_arg = call.args[0]
            if isinstance(first_arg, relax.Tuple) and first_arg.fields:
                first_arg = first_arg.fields[0]
            input_scale, input_zp = self._trace_input_scale(first_arg)
            if input_scale is not None:
                constants["input_scale"] = input_scale
                constants["input_zp"] = input_zp

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
