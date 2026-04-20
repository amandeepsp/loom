"""Lowering from Relax composite regions to accel runtime calls."""

from __future__ import annotations

from typing import Any

import numpy as np
import tvm
from tvm import relax

from patterns import ACCEL_CODEGEN_NAME
from shared.ir import ACCEL_EXTERN_PREFIX

COMPOSITE_CONSTANTS: dict[str, dict[str, Any]] = {}


def make_extern_symbol(global_var: relax.GlobalVar) -> str:
    """Return the packed-runtime symbol name for a lowered accel region."""

    name = global_var.name_hint.replace(".", "_")
    return f"{ACCEL_EXTERN_PREFIX}.{name}"


def _extract_composite_constants(func: relax.Function) -> dict[str, Any]:
    """Extract constants from a composite function body.

    Returns a dict with:
    - weights: int8 weight tensor
    - bias: int32 bias tensor
    - input_scale, input_zp: input dequantization params
    - weight_scale, weight_zp: weight dequantization params
    - output_scale, output_zp: output requantization params
    """
    constants: dict[str, Any] = {}

    def visit(expr):
        if isinstance(expr, relax.Call):
            op_name = expr.op.name if hasattr(expr.op, "name") else str(expr.op)

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

            for arg in expr.args:
                visit(arg)

        elif isinstance(expr, relax.Constant):
            arr = expr.data.numpy()
            if arr.ndim == 0 or arr.size == 1:
                val = arr.item() if arr.ndim == 0 else arr.flatten()[0]
                if "input_scale" not in constants:
                    constants["input_scale"] = float(val)
                elif "input_zp" not in constants:
                    constants["input_zp"] = int(val)
                elif "output_scale" not in constants:
                    constants["output_scale"] = float(val)
                elif "output_zp" not in constants:
                    constants["output_zp"] = int(val)

    if hasattr(func, "body"):
        visit(func.body)

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

        composite_attrs = target.attrs
        composite_func = target
        if "Composite" in composite_attrs:
            local_funcs = list(self.builder_.get().functions.values())
            for f in local_funcs:
                if hasattr(f, "attrs") and f.attrs and f.attrs.get("Composite") == composite_attrs.get("Composite"):
                    constants = _extract_composite_constants(f)
                    COMPOSITE_CONSTANTS[extern_symbol] = constants
                    break

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
