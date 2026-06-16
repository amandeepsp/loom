"""ONNX → TOSA compiler frontend.

Usage:
    from compiler.frontend import load_model, emit_tosa_ir
    model = load_model("models/out/mnist_int8.onnx")
    module = emit_tosa_ir(model)
"""

from xdsl.ir import SSAValue
from xdsl.dialects.builtin import (
    ArrayAttr,
    DenseIntOrFPElementsAttr,
    FunctionType,
    ModuleOp,
)
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.tosa import ConstOp
from xdsl.builder import Builder, InsertPoint

import onnx_ir

from .helpers import tensor_from_onnx_tensor
from .ops import (
    handle_flatten,
    handle_quantize_linear,
    handle_qgemm,
    handle_dequantize_linear,
)

# ── Op registry ────────────────────────────────────────────────────────────
# Add new handlers by: 1) creating compiler/frontend/ops/<name>.py,
# 2) exporting handle(), 3) importing it here and adding to the dict.

OP_HANDLERS: dict = {
    "Flatten": handle_flatten,
    "QuantizeLinear": handle_quantize_linear,
    "QGemm": handle_qgemm,
    "DequantizeLinear": handle_dequantize_linear,
}

# ── Orchestration ──────────────────────────────────────────────────────────


def _sanitize(name: str) -> str:
    """Replace '/' and ':' in ONNX names with '_' for valid MLIR SSA names."""
    return name.translate(str.maketrans("/:", "__"))


def load_model(path: str) -> onnx_ir.Model:
    return onnx_ir.load(path)


def emit_tosa_ir(model: onnx_ir.Model) -> ModuleOp:
    module = ModuleOp([])
    builder = Builder(InsertPoint.at_end(module.body.block))

    ssa_values: dict[str, SSAValue] = {}

    # Build function signature from graph inputs/outputs
    inputs = [tensor_from_onnx_tensor(tensor) for tensor in model.graph.inputs]
    outputs = [tensor_from_onnx_tensor(tensor) for tensor in model.graph.outputs]

    model_func = FuncOp("main", FunctionType(ArrayAttr(inputs), ArrayAttr(outputs)))
    for inp, inp_arg in zip(model.graph.inputs, model_func.args):
        inp_arg.name_hint = _sanitize(inp.name)
        ssa_values[inp.name] = inp_arg

    for outp, outp_arg in zip(model.graph.outputs, model_func.results):
        outp_arg.name_hint = _sanitize(outp.name)
        ssa_values[outp.name] = outp_arg

    builder.insert(model_func)
    builder.insertion_point = InsertPoint.at_end(model_func.body.block)

    # Materialize initializers as tosa.const ops
    for name, init in model.graph.initializers.items():
        data = init.const_value.numpy()
        init_tensor_type = tensor_from_onnx_tensor(init)
        const_op = builder.insert(
            ConstOp(
                DenseIntOrFPElementsAttr.from_list(
                    init_tensor_type, data.flatten().tolist()
                )
            )
        )
        const_op.results[0].name_hint = _sanitize(init.name)
        ssa_values[init.name] = const_op.results[0]

    # Walk the graph in topological order, dispatching each node
    for node in onnx_ir.traversal.RecursiveGraphIterator(model.graph):
        operands = [ssa_values[operand.name] for operand in node.inputs]

        handler = OP_HANDLERS.get(node.op_type)
        if handler is None:
            raise ValueError(f"Unsupported op type: {node.op_type}")

        results = handler(node, operands, builder)
        for output, result in zip(node.outputs, results):
            result.name_hint = _sanitize(output.name)
            ssa_values[output.name] = result

    # Collect graph output SSA values for the return op
    return_operands = [ssa_values[outp.name] for outp in model.graph.outputs]
    builder.insert(ReturnOp(*return_operands))
    return module


if __name__ == "__main__":
    model = load_model("models/out/mnist_int8.onnx")
    module = emit_tosa_ir(model)
    print(module)
