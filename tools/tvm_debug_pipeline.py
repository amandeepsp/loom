#!/usr/bin/env -S uv run python
"""Diagnostic script: display ONNX and TVM graphs at every lowering stage.

Usage:
    uv run python tools/tvm_debug_pipeline.py --onnx models/out/mnist_int8.onnx
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from compiler.codegen import COMPOSITE_CONSTANTS, lower_accel_regions
from compiler.patterns import partition_for_accel_cfu


def ascii_onnx(model: onnx.ModelProto, max_nodes: int = 40) -> None:
    """Print ONNX graph as a simple ASCII tree."""
    print("=" * 60)
    print("ONNX Graph")
    print("=" * 60)
    for init in model.graph.initializer:
        arr = onnx.numpy_helper.to_array(init)
        print(f"  init  {init.name:40s}  {arr.shape}  {arr.dtype}")
    for inp in model.graph.input:
        print(f"  input {inp.name}")
    for out in model.graph.output:
        print(f"  output {out.name}")
    print("-" * 60)
    for i, node in enumerate(model.graph.node):
        if i >= max_nodes:
            print(f"  ... ({len(model.graph.node) - max_nodes} more nodes)")
            break
        attrs = ", ".join(f"{a.name}={onnx.helper.get_attribute_value(a)}" for a in node.attribute)
        print(f"  {node.op_type:20s}  {node.name or '':20s}  inputs={list(node.input)}  outputs={list(node.output)}")
        if attrs:
            print(f"    attrs: {attrs}")
    print()


def ascii_relax(mod: tvm.IRModule, title: str) -> None:
    """Print Relax functions in a compact form."""
    print("=" * 60)
    print(title)
    print("=" * 60)
    print(mod.script())
    print()


def show_composite_constants() -> None:
    """Print extracted composite constants."""
    print("=" * 60)
    print("Composite Constants")
    print("=" * 60)
    if not COMPOSITE_CONSTANTS:
        print("  (none)")
    for sym, vals in COMPOSITE_CONSTANTS.items():
        print(f"  {sym}")
        for k, v in vals.items():
            if hasattr(v, "shape"):
                print(f"    {k}: {v.shape} {v.dtype}")
            else:
                print(f"    {k}: {v}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default=str(REPO_ROOT / "models/out/mnist_int8.onnx"))
    args = parser.parse_args()

    onnx_path = Path(args.onnx)
    model_proto = onnx.load(str(onnx_path))
    ascii_onnx(model_proto)

    shape_dict = {"input": [1, 1, 28, 28]}
    mod = from_onnx(model_proto, shape_dict=shape_dict, keep_params_in_input=False)
    ascii_relax(mod, "Stage 0: After ONNX import")

    mod = partition_for_accel_cfu(mod)
    ascii_relax(mod, "Stage 1: After partition_for_accel_cfu")

    mod = lower_accel_regions(mod)
    ascii_relax(mod, "Stage 2: After lower_accel_regions")
    show_composite_constants()

    mod = relax.transform.LambdaLift()(mod)
    ascii_relax(mod, "Stage 3: After LambdaLift")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
