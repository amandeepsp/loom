from __future__ import annotations

import numpy as np
import onnxruntime as ort

from compiler import ModelBinaryArtifact, ModelBinaryRuntime, compile_onnx_model
from compiler.dialects.loom import GemmOp
from compiler.dialects.loom_cpu import Pack2DToTilesOp, PackConv2DToTilesOp
from compiler.frontend.qlinear_mobilenet import parse_qlinear_mobilenet
from shared.reference import cpu_requantize


MODEL = "models/out/mobilenet_v1_025_int8.onnx"


def test_parse_qoperator_mobilenet_v1_025_plan():
    plan = parse_qlinear_mobilenet(MODEL)

    assert plan.input_shape == (1, 3, 32, 32)
    assert plan.output_shape == (1, 10)
    assert len(plan.layers) == 29
    assert sum(layer.kind == "conv2d" for layer in plan.layers) == 14
    assert sum(layer.kind == "depthwise_conv2d" for layer in plan.layers) == 13
    assert plan.layers[-2].kind == "global_average_pool"
    assert plan.layers[-1].kind == "gemm"

    first = plan.layers[0]
    assert (first.m, first.k, first.n) == (1024, 27, 8)
    assert first.packing is not None and first.packing.nchw
    depthwise = plan.layers[1]
    assert (depthwise.m, depthwise.k, depthwise.n) == (1024, 72, 8)
    assert np.count_nonzero(depthwise.weight) <= 8 * 3 * 3


def test_compile_mobilenet_emits_llvm_cpu_packers_and_cfu_gemms():
    compiled = compile_onnx_model(MODEL)

    assert len(compiled.kernels) == 29
    names = [function.name for function in compiled.functions]
    lut_maps = [name for name in names if name.endswith(".lut_map")]
    # pack + GEMM per layer, plus a one-pass LUT pre-map for each 3x3 depthwise.
    # The global average pool runs as a direct CPU kernel (no pack function).
    assert sum(k.kind == "global_average_pool_cpu" for k in compiled.kernels) == 1
    # Pointwise convs get their input packed by the previous GEMM's store.
    fused = [k for k in compiled.kernels if k.packed_by]
    assert len(fused) == 13
    assert len(names) - len(lut_maps) == 58 - 1 - len(fused)
    assert len(lut_maps) == 13
    assert all(function.code and len(function.code) % 4 == 0 for function in compiled.functions)
    ops = tuple(compiled.loom_module.body.block.ops)
    assert len(ops) == 58 - 2 - 13  # pool: no loom pack/GEMM ops; fused: no pack ops
    assert isinstance(ops[0], PackConv2DToTilesOp)
    assert isinstance(ops[1], GemmOp)
    assert isinstance(ops[2], PackConv2DToTilesOp)  # depthwise im2col
    assert isinstance(ops[3], GemmOp)
    assert isinstance(ops[4], GemmOp)  # pointwise: input packed by ops[3]'s store

    contract = compiled.manifest()["runtime_contract"]
    assert contract["input"] == {
        "buffer": "mobilenet.input",
        "dtype": "int8",
        "shape": [1, 3, 32, 32],
        "padded_shape": [1, 3, 32, 32],
        "scale": compiled.runtime_input.scale,
        "zero_point": compiled.runtime_input.zero_point,
        "layout": "NCHW",
    }
    assert contract["output"]["shape"] == [1, 10]
    assert contract["output"]["padded_shape"] == [8, 10]
    assert contract["output"]["layout"] == "NC"
    assert compiled.recommended_model_addr() + len(
        compiled.to_bytes(include_debug_sections=False)
    ) <= 0x40800000


def test_mobilenet_binary_runtime_accepts_nchw_boundary():
    compiled = compile_onnx_model(MODEL)
    artifact = ModelBinaryArtifact.from_bytes(compiled.to_bytes(include_debug_sections=False))
    contract = artifact.manifest["runtime_contract"]
    output_buffer = artifact.buffer(contract["output"]["buffer"])

    class Transport:
        def write_mem(self, _addr, _data):
            pass

        def exec_model(self, _addr, _length):
            return 123

        def read_mem(self, addr, length):
            assert (addr, length) == (output_buffer.addr, output_buffer.size)
            return bytes(length)

    result = ModelBinaryRuntime(artifact, Transport()).run_quantized(
        np.zeros((1, 3, 32, 32), dtype=np.int8)
    )
    assert result.output.shape == (1, 10)


def test_fused_mobilenet_integer_plan_tracks_onnxruntime_prediction():
    plan = parse_qlinear_mobilenet(MODEL)
    rng = np.random.default_rng(42)
    input_float = rng.normal(size=plan.input_shape).astype(np.float32)
    actual = _run_plan(plan, input_float)

    reference_float = ort.InferenceSession(MODEL).run(None, {"input": input_float})[0]
    reference = np.clip(
        np.rint(reference_float / plan.output_scale) + plan.output_zero_point,
        -128,
        127,
    ).astype(np.int8)

    # ReLU6 and the second quantization are fused into the following packer's
    # source load through a 256-entry lookup table, preserving that boundary.
    assert actual.shape == reference.shape == (1, 10)
    assert actual.argmax(axis=1).tolist() == reference.argmax(axis=1).tolist()
    assert np.max(np.abs(actual.astype(np.int16) - reference.astype(np.int16))) <= 24


def _run_plan(plan, input_float: np.ndarray) -> np.ndarray:
    source = np.clip(
        np.rint(input_float / plan.input_scale) + plan.input_zero_point,
        -128,
        127,
    ).astype(np.int8)
    for layer in plan.layers:
        if layer.input_transform is not None:
            source = layer.input_transform[source.astype(np.int16) + 128]
        matrix = _activation_matrix(source, layer)
        accumulator = matrix.astype(np.int32) @ layer.weight.astype(np.int32)
        source = cpu_requantize(
            accumulator,
            layer.bias,
            layer.multiplier,
            layer.shift,
            layer.output_zero_point,
            layer.activation_min,
            layer.activation_max,
        )
    return source[: plan.output_shape[0], : plan.output_shape[1]]


def _activation_matrix(source: np.ndarray, layer) -> np.ndarray:
    packing = layer.packing
    if packing is None:
        return source.reshape(layer.m, layer.k)

    source_view = (
        source[0]
        if packing.nchw
        else source.reshape(packing.input_h, packing.input_w, packing.channels)
    )
    matrix = np.full(
        (layer.m, layer.k), layer.input_zero_point, dtype=np.int8
    )
    for oh in range(packing.output_h):
        for ow in range(packing.output_w):
            row = oh * packing.output_w + ow
            for ky in range(packing.kernel_h):
                ih = oh * packing.stride_h + ky - packing.pad_top
                for kx in range(packing.kernel_w):
                    iw = ow * packing.stride_w + kx - packing.pad_left
                    if not (0 <= ih < packing.input_h and 0 <= iw < packing.input_w):
                        continue
                    values = (
                        source_view[:, ih, iw]
                        if packing.nchw
                        else source_view[ih, iw, :]
                    )
                    offset = (ky * packing.kernel_w + kx) * packing.channels
                    matrix[row, offset : offset + packing.channels] = values
    return matrix
