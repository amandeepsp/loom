"""Canonicalize an INT8 QOperator MobileNet-v1 graph into GEMM layer plans.

The supported graph is the linear MobileNet-v1 shape used by this project:
QLinearConv + ReLU6 blocks, GlobalAveragePool, and a final QGemm.  Spatial
convolutions are represented as im2col GEMMs; depthwise weights become sparse
block-diagonal matrices.  This keeps all arithmetic on the existing CFU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference

from .helpers import compute_rescale_params


@dataclass(frozen=True)
class ConvPacking:
    input_h: int
    input_w: int
    channels: int
    output_h: int
    output_w: int
    kernel_h: int
    kernel_w: int
    stride_h: int
    stride_w: int
    pad_top: int
    pad_left: int
    nchw: bool


@dataclass(frozen=True)
class QuantizedGemmPlan:
    name: str
    kind: str
    m: int
    k: int
    n: int
    weight: np.ndarray
    bias: np.ndarray
    multiplier: np.ndarray
    shift: np.ndarray
    input_zero_point: int
    output_zero_point: int
    activation_min: int
    activation_max: int
    input_transform: np.ndarray | None = None
    packing: ConvPacking | None = None


@dataclass(frozen=True)
class QuantizedMobileNetPlan:
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    input_scale: float
    input_zero_point: int
    output_scale: float
    output_zero_point: int
    layers: tuple[QuantizedGemmPlan, ...]


def is_qlinear_mobilenet(path: str | Path) -> bool:
    model = onnx.load(str(path), load_external_data=False)
    op_types = {node.op_type for node in model.graph.node}
    return "QLinearConv" in op_types and "QGemm" in op_types


def parse_qlinear_mobilenet(path: str | Path) -> QuantizedMobileNetPlan:
    model = shape_inference.infer_shapes(onnx.load(str(path)))
    nodes = list(model.graph.node)
    initializers = {
        tensor.name: np.asarray(numpy_helper.to_array(tensor))
        for tensor in model.graph.initializer
    }
    constants = dict(initializers)
    for node in nodes:
        if node.op_type != "Constant":
            continue
        value = _attribute(node, "value", None)
        if value is not None and node.output:
            constants[node.output[0]] = np.asarray(numpy_helper.to_array(value))

    shapes = {
        value.name: _value_shape(value)
        for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in nodes:
        for operand in node.input:
            consumers.setdefault(operand, []).append(node)

    conv_nodes = [node for node in nodes if node.op_type == "QLinearConv"]
    if len(conv_nodes) != 27:
        raise ValueError(
            f"MobileNet-v1 0.25 requires 27 QLinearConv nodes, found {len(conv_nodes)}"
        )
    qgemms = [node for node in nodes if node.op_type == "QGemm"]
    if len(qgemms) != 1:
        raise ValueError(f"MobileNet-v1 requires one QGemm classifier, found {len(qgemms)}")

    input_quantizers = [
        node
        for node in nodes
        if node.op_type == "QuantizeLinear" and node.input[0] == model.graph.input[0].name
    ]
    if len(input_quantizers) != 1:
        raise ValueError("expected one QuantizeLinear at the MobileNet input")
    input_q = input_quantizers[0]
    input_scale, input_zp = _quant_params(input_q.input[1], input_q.input[2], constants)
    input_shape = _static_shape(shapes[model.graph.input[0].name], "model input")
    if len(input_shape) != 4 or input_shape[0] != 1:
        raise ValueError(f"MobileNet input must be static batch-1 NCHW, got {input_shape}")

    classifier = qgemms[0]
    classifier_input_quant = _producer(nodes, classifier.input[0], "QuantizeLinear")
    pool_scale, pool_zp = _quant_params(
        classifier_input_quant.input[1], classifier_input_quant.input[2], constants
    )

    layers: list[QuantizedGemmPlan] = []
    previous_quantized = input_q.output[0]
    last_conv_shape: tuple[int, int, int, int] | None = None
    last_scale = input_scale
    last_zp = input_zp
    pending_input_transform: np.ndarray | None = None

    for index, node in enumerate(conv_nodes):
        if node.input[0] != previous_quantized:
            raise ValueError(
                f"convolution {index} does not consume the previous fused activation"
            )
        input_scale_i, input_zp_i = _quant_params(node.input[1], node.input[2], constants)
        if index < len(conv_nodes) - 1:
            dequant = _only_consumer(consumers, node.output[0], "DequantizeLinear")
            clip = _only_consumer(consumers, dequant.output[0], "Clip")
            post_quant = _only_consumer(consumers, clip.output[0], "QuantizeLinear")
            output_scale, output_zp = _quant_params(
                post_quant.input[1], post_quant.input[2], constants
            )
            previous_quantized = post_quant.output[0]
        else:
            dequant = _only_consumer(consumers, node.output[0], "DequantizeLinear")
            clip = _only_consumer(consumers, dequant.output[0], "Clip")
            pool = _only_consumer(consumers, clip.output[0], "GlobalAveragePool")
            flatten = _only_consumer(consumers, pool.output[0], "Flatten")
            if classifier_input_quant.input[0] != flatten.output[0]:
                raise ValueError("classifier QuantizeLinear must consume flattened global pool")
            output_scale, output_zp = pool_scale, pool_zp

        clip_min = _scalar(constants, clip.input[1], "Clip minimum")
        clip_max = _scalar(constants, clip.input[2], "Clip maximum")
        if abs(clip_min) > 1e-6 or abs(clip_max - 6.0) > 1e-6:
            raise ValueError(
                f"MobileNet convolution {index} must use ReLU6, got [{clip_min}, {clip_max}]"
            )

        input_shape_i = _static_shape(shapes[node.input[0]], f"convolution {index} input")
        output_shape_i = _static_shape(shapes[node.output[0]], f"convolution {index} output")
        if len(input_shape_i) != 4 or len(output_shape_i) != 4:
            raise ValueError("QLinearConv inputs and outputs must be rank-4 NCHW tensors")
        _, channels, input_h, input_w = input_shape_i
        _, output_channels, output_h, output_w = output_shape_i

        raw_weight = _array(constants, node.input[3], np.int8)
        weight_scale = _array(constants, node.input[4], np.float64).reshape(-1)
        weight_zp = _array(constants, node.input[5], np.int64).reshape(-1)
        if np.any(weight_zp != 0):
            raise ValueError("non-zero MobileNet weight zero-points are unsupported")
        bias = _array(constants, node.input[8], np.int32).reshape(-1)
        group = int(_attribute(node, "group", 1))
        strides = tuple(int(v) for v in _attribute(node, "strides", [1, 1]))
        pads = tuple(int(v) for v in _attribute(node, "pads", [0, 0, 0, 0]))
        kernel = tuple(int(v) for v in _attribute(node, "kernel_shape", raw_weight.shape[2:]))
        if len(strides) != 2 or len(pads) != 4 or len(kernel) != 2:
            raise ValueError("only 2D MobileNet convolutions are supported")

        weight = _conv_weight_matrix(
            raw_weight,
            channels=channels,
            output_channels=output_channels,
            group=group,
        )
        k = weight.shape[0]
        if bias.size != output_channels:
            raise ValueError(f"convolution {index} bias has the wrong channel count")
        bias = bias.astype(np.int64) - input_zp_i * weight.astype(np.int64).sum(axis=0)
        bias = _checked_i32(bias, f"convolution {index} adjusted bias")
        conv_output_scale, conv_output_zp = _quant_params(
            node.input[6], node.input[7], constants
        )
        multiplier, shift = _rescale_vectors(
            input_scale_i, weight_scale, conv_output_scale, output_channels
        )

        packing = None
        if kernel != (1, 1) or strides != (1, 1) or any(pads):
            packing = ConvPacking(
                input_h=input_h,
                input_w=input_w,
                channels=channels,
                output_h=output_h,
                output_w=output_w,
                kernel_h=kernel[0],
                kernel_w=kernel[1],
                stride_h=strides[0],
                stride_w=strides[1],
                pad_top=pads[0],
                pad_left=pads[1],
                nchw=index == 0,
            )
        layers.append(
            QuantizedGemmPlan(
                name=f"mobilenet_conv_{index}",
                kind="depthwise_conv2d" if group != 1 else "conv2d",
                m=output_h * output_w,
                k=k,
                n=output_channels,
                weight=weight,
                bias=bias,
                multiplier=multiplier,
                shift=shift,
                input_zero_point=input_zp_i,
                output_zero_point=conv_output_zp,
                activation_min=-128,
                activation_max=127,
                input_transform=pending_input_transform,
                packing=packing,
            )
        )
        pending_input_transform = _relu6_requant_lut(
            conv_output_scale=conv_output_scale,
            conv_output_zero_point=conv_output_zp,
            output_scale=output_scale,
            output_zero_point=output_zp,
        )
        last_conv_shape = output_shape_i
        last_scale, last_zp = output_scale, output_zp

    assert last_conv_shape is not None
    _, pool_channels, pool_h, pool_w = last_conv_shape
    pool_size = pool_h * pool_w
    pool_weight = np.zeros((pool_size * pool_channels, pool_channels), dtype=np.int8)
    for spatial in range(pool_size):
        channel_index = np.arange(pool_channels)
        pool_weight[spatial * pool_channels + channel_index, channel_index] = 1
    pool_bias = np.full(pool_channels, -last_zp * pool_size, dtype=np.int32)
    pool_multiplier, pool_shift = _rescale_vectors(
        last_scale / pool_size,
        np.ones(1, dtype=np.float64),
        pool_scale,
        pool_channels,
    )
    layers.append(
        QuantizedGemmPlan(
            name="mobilenet_global_average_pool",
            kind="global_average_pool",
            m=1,
            k=pool_size * pool_channels,
            n=pool_channels,
            weight=pool_weight,
            bias=pool_bias,
            multiplier=pool_multiplier,
            shift=pool_shift,
            input_zero_point=last_zp,
            output_zero_point=pool_zp,
            activation_min=-128,
            activation_max=127,
            input_transform=pending_input_transform,
        )
    )

    classifier_scale, classifier_zp = _quant_params(
        classifier.input[1], classifier.input[2], constants
    )
    if abs(classifier_scale - pool_scale) > 1e-9 or classifier_zp != pool_zp:
        raise ValueError("classifier input quantization does not match global pool output")
    classifier_weight_raw = _array(constants, classifier.input[3], np.int8)
    trans_b = int(_attribute(classifier, "transB", 0))
    classifier_weight = classifier_weight_raw.T.copy() if trans_b else classifier_weight_raw.copy()
    if classifier_weight.ndim != 2 or classifier_weight.shape[0] != pool_channels:
        raise ValueError("QGemm classifier weight shape does not match pooled channels")
    classifier_out_scale, classifier_out_zp = _quant_params(
        classifier.input[7], classifier.input[8], constants
    )
    classifier_w_scale = _array(constants, classifier.input[4], np.float64).reshape(-1)
    classifier_w_zp = _array(constants, classifier.input[5], np.int64).reshape(-1)
    if np.any(classifier_w_zp != 0):
        raise ValueError("non-zero classifier weight zero-point is unsupported")
    classifier_bias = _array(constants, classifier.input[6], np.int32).reshape(-1).astype(np.int64)
    classifier_bias -= classifier_zp * classifier_weight.astype(np.int64).sum(axis=0)
    classifier_bias = _checked_i32(classifier_bias, "classifier adjusted bias")
    classifier_multiplier, classifier_shift = _rescale_vectors(
        classifier_scale,
        classifier_w_scale,
        classifier_out_scale,
        classifier_weight.shape[1],
    )
    layers.append(
        QuantizedGemmPlan(
            name="mobilenet_classifier",
            kind="gemm",
            m=1,
            k=classifier_weight.shape[0],
            n=classifier_weight.shape[1],
            weight=classifier_weight,
            bias=classifier_bias,
            multiplier=classifier_multiplier,
            shift=classifier_shift,
            input_zero_point=classifier_zp,
            output_zero_point=classifier_out_zp,
            activation_min=-128,
            activation_max=127,
        )
    )

    output_shape = _static_shape(shapes[model.graph.output[0].name], "model output")
    return QuantizedMobileNetPlan(
        input_shape=input_shape,
        output_shape=output_shape,
        input_scale=input_scale,
        input_zero_point=input_zp,
        output_scale=classifier_out_scale,
        output_zero_point=classifier_out_zp,
        layers=tuple(layers),
    )


def _conv_weight_matrix(
    raw: np.ndarray, *, channels: int, output_channels: int, group: int
) -> np.ndarray:
    if raw.ndim != 4 or raw.shape[0] != output_channels:
        raise ValueError(f"unsupported QLinearConv weight shape: {raw.shape}")
    _, group_channels, kernel_h, kernel_w = raw.shape
    if group == 1 and group_channels == channels:
        return raw.transpose(2, 3, 1, 0).reshape(kernel_h * kernel_w * channels, output_channels)
    if group == channels == output_channels and group_channels == 1:
        matrix = np.zeros((kernel_h * kernel_w * channels, output_channels), dtype=np.int8)
        channel_index = np.arange(channels)
        for ky in range(kernel_h):
            for kx in range(kernel_w):
                base = (ky * kernel_w + kx) * channels
                matrix[base + channel_index, channel_index] = raw[:, 0, ky, kx]
        return matrix
    raise ValueError(
        f"unsupported grouped convolution: group={group}, input={channels}, output={output_channels}"
    )


def _rescale_vectors(
    input_scale: float,
    weight_scale: np.ndarray,
    output_scale: float,
    channels: int,
) -> tuple[np.ndarray, np.ndarray]:
    scales = weight_scale.reshape(-1)
    if scales.size == 1:
        scales = np.full(channels, scales.item(), dtype=np.float64)
    if scales.size != channels:
        raise ValueError(f"weight scale count {scales.size} does not match channels {channels}")
    # TOSA's scale32 shift counts the multiplier's 31 fractional bits. Loom's
    # epilogue performs that high-multiply first, so its following RDBPOT only
    # receives the residual right shift.
    params = [compute_rescale_params(input_scale, float(scale), output_scale) for scale in scales]
    params = [(multiplier, shift - 31) for multiplier, shift in params]
    if any(shift < 0 for _, shift in params):
        raise ValueError("MobileNet requantization scales greater than one are unsupported")
    return (
        np.asarray([item[0] for item in params], dtype=np.int32),
        np.asarray([item[1] for item in params], dtype=np.int32),
    )


def _quant_params(scale_name: str, zp_name: str, constants: dict[str, np.ndarray]) -> tuple[float, int]:
    return float(_scalar(constants, scale_name, "scale")), int(_scalar(constants, zp_name, "zero point"))


def _quantize_bound(value: float, scale: float, zero_point: int) -> int:
    return int(np.clip(np.rint(value / scale) + zero_point, -128, 127))


def _relu6_requant_lut(
    *,
    conv_output_scale: float,
    conv_output_zero_point: int,
    output_scale: float,
    output_zero_point: int,
) -> np.ndarray:
    """Map every QLinearConv i8 result through DQ -> ReLU6 -> Q.

    Keeping this as a lookup preserves ONNX's second round-to-nearest-even
    boundary exactly while allowing the CPU packer to fuse the operation with
    its existing source load.
    """

    values = np.arange(-128, 128, dtype=np.float32)
    real = (values - np.float32(conv_output_zero_point)) * np.float32(
        conv_output_scale
    )
    clipped = np.clip(real, np.float32(0.0), np.float32(6.0))
    return np.clip(
        np.rint(clipped / np.float32(output_scale)) + output_zero_point,
        -128,
        127,
    ).astype(np.int8)


def _array(constants: dict[str, np.ndarray], name: str, dtype: np.dtype) -> np.ndarray:
    if name not in constants:
        raise ValueError(f"expected constant tensor: {name}")
    return np.asarray(constants[name], dtype=dtype)


def _scalar(constants: dict[str, np.ndarray], name: str, label: str) -> float:
    value = _array(constants, name, np.float64).reshape(-1)
    if value.size != 1:
        raise ValueError(f"{label} must be scalar: {name}")
    return float(value[0])


def _checked_i32(values: np.ndarray, label: str) -> np.ndarray:
    if np.any(values < np.iinfo(np.int32).min) or np.any(values > np.iinfo(np.int32).max):
        raise ValueError(f"{label} overflows int32")
    return values.astype(np.int32)


def _attribute(node: onnx.NodeProto, name: str, default):
    for attr in node.attribute:
        if attr.name == name:
            return onnx.helper.get_attribute_value(attr)
    return default


def _value_shape(value: onnx.ValueInfoProto) -> tuple[int | str, ...]:
    return tuple(dim.dim_value if dim.HasField("dim_value") else dim.dim_param for dim in value.type.tensor_type.shape.dim)


def _static_shape(shape: tuple[int | str, ...], label: str) -> tuple[int, ...]:
    result = tuple(1 if dim in ("batch", "") else int(dim) for dim in shape)
    if any(dim <= 0 for dim in result):
        raise ValueError(f"{label} must have a static positive shape, got {shape}")
    return result


def _only_consumer(
    consumers: dict[str, list[onnx.NodeProto]], value: str, op_type: str
) -> onnx.NodeProto:
    matches = [node for node in consumers.get(value, ()) if node.op_type == op_type]
    if len(matches) != 1:
        raise ValueError(f"expected one {op_type} consumer of {value}, found {len(matches)}")
    return matches[0]


def _producer(nodes: list[onnx.NodeProto], value: str, op_type: str) -> onnx.NodeProto:
    matches = [node for node in nodes if value in node.output and node.op_type == op_type]
    if len(matches) != 1:
        raise ValueError(f"expected {value} to be produced by one {op_type}")
    return matches[0]
