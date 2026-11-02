"""IRDL dialect for CPU work scheduled inside a loom model binary."""

from __future__ import annotations

from xdsl.dialects.builtin import Dialect, IntegerAttr, StringAttr, i32
from xdsl.ir import Attribute
from xdsl.irdl import IRDLOperation, irdl_op_definition, prop_def, traits_def
from xdsl.traits import Pure
from xdsl.utils.exceptions import VerifyException


def _i32(value: int) -> IntegerAttr:
    return IntegerAttr(value, i32)


@irdl_op_definition
class Pack2DToTilesOp(IRDLOperation):
    """CPU layout transform: row-major [M,K] i8 -> loom [M/tile,K,tile] i8."""

    name = "loom_cpu.pack_2d_to_tiles"

    function_name = prop_def(StringAttr)
    source = prop_def(StringAttr)
    target = prop_def(StringAttr)
    src_addr = prop_def(IntegerAttr)
    dst_addr = prop_def(IntegerAttr)
    transform_addr = prop_def(IntegerAttr)
    m = prop_def(IntegerAttr)
    k = prop_def(IntegerAttr)
    tile = prop_def(IntegerAttr)

    traits = traits_def(Pure())

    def __init__(
        self,
        function_name: str,
        source: str,
        target: str,
        *,
        src_addr: int,
        dst_addr: int,
        transform_addr: int = 0,
        m: int,
        k: int,
        tile: int,
    ):
        super().__init__(
            properties={
                "function_name": StringAttr(function_name),
                "source": StringAttr(source),
                "target": StringAttr(target),
                "src_addr": _i32(src_addr),
                "dst_addr": _i32(dst_addr),
                "transform_addr": _i32(transform_addr),
                "m": _i32(m),
                "k": _i32(k),
                "tile": _i32(tile),
            }
        )

    def verify_(self) -> None:
        for attr_name in ("src_addr", "dst_addr", "m", "k", "tile"):
            value = _int_attr(getattr(self, attr_name))
            if value <= 0:
                raise VerifyException(f"loom_cpu.pack_2d_to_tiles {attr_name} must be positive")
        if _int_attr(self.transform_addr) < 0:
            raise VerifyException("loom_cpu.pack_2d_to_tiles transform_addr must be non-negative")


@irdl_op_definition
class PackConv2DToTilesOp(IRDLOperation):
    """CPU im2col + pack for an int8 NCHW/NHWC convolution input."""

    name = "loom_cpu.pack_conv2d_to_tiles"

    function_name = prop_def(StringAttr)
    source = prop_def(StringAttr)
    target = prop_def(StringAttr)
    src_addr = prop_def(IntegerAttr)
    dst_addr = prop_def(IntegerAttr)
    transform_addr = prop_def(IntegerAttr)
    input_h = prop_def(IntegerAttr)
    input_w = prop_def(IntegerAttr)
    channels = prop_def(IntegerAttr)
    output_h = prop_def(IntegerAttr)
    output_w = prop_def(IntegerAttr)
    kernel_h = prop_def(IntegerAttr)
    kernel_w = prop_def(IntegerAttr)
    stride_h = prop_def(IntegerAttr)
    stride_w = prop_def(IntegerAttr)
    pad_top = prop_def(IntegerAttr)
    pad_left = prop_def(IntegerAttr)
    input_zero_point = prop_def(IntegerAttr)
    tile = prop_def(IntegerAttr)
    nchw = prop_def(IntegerAttr)
    channel_major = prop_def(IntegerAttr)

    traits = traits_def(Pure())

    def __init__(
        self,
        function_name: str,
        source: str,
        target: str,
        *,
        src_addr: int,
        dst_addr: int,
        transform_addr: int = 0,
        channel_major: bool = False,
        input_h: int,
        input_w: int,
        channels: int,
        output_h: int,
        output_w: int,
        kernel_h: int,
        kernel_w: int,
        stride_h: int,
        stride_w: int,
        pad_top: int,
        pad_left: int,
        input_zero_point: int,
        tile: int,
        nchw: bool,
    ):
        super().__init__(
            properties={
                "function_name": StringAttr(function_name),
                "source": StringAttr(source),
                "target": StringAttr(target),
                "src_addr": _i32(src_addr),
                "dst_addr": _i32(dst_addr),
                "transform_addr": _i32(transform_addr),
                "input_h": _i32(input_h),
                "input_w": _i32(input_w),
                "channels": _i32(channels),
                "output_h": _i32(output_h),
                "output_w": _i32(output_w),
                "kernel_h": _i32(kernel_h),
                "kernel_w": _i32(kernel_w),
                "stride_h": _i32(stride_h),
                "stride_w": _i32(stride_w),
                "pad_top": _i32(pad_top),
                "pad_left": _i32(pad_left),
                "input_zero_point": _i32(input_zero_point),
                "tile": _i32(tile),
                "nchw": _i32(int(nchw)),
                "channel_major": _i32(int(channel_major)),
            }
        )

    def verify_(self) -> None:
        for attr_name in (
            "src_addr",
            "dst_addr",
            "input_h",
            "input_w",
            "channels",
            "output_h",
            "output_w",
            "kernel_h",
            "kernel_w",
            "stride_h",
            "stride_w",
            "tile",
        ):
            if _int_attr(getattr(self, attr_name)) <= 0:
                raise VerifyException(
                    f"loom_cpu.pack_conv2d_to_tiles {attr_name} must be positive"
                )
        for attr_name in ("pad_top", "pad_left"):
            if _int_attr(getattr(self, attr_name)) < 0:
                raise VerifyException(
                    f"loom_cpu.pack_conv2d_to_tiles {attr_name} must be non-negative"
                )
        if _int_attr(self.transform_addr) < 0:
            raise VerifyException(
                "loom_cpu.pack_conv2d_to_tiles transform_addr must be non-negative"
            )
        if _int_attr(self.nchw) not in (0, 1):
            raise VerifyException("loom_cpu.pack_conv2d_to_tiles nchw must be 0 or 1")


def _int_attr(attr: Attribute) -> int:
    if not isinstance(attr, IntegerAttr):
        raise VerifyException(f"expected integer attribute, got {attr}")
    return int(attr.value.data)


LoomCpu = Dialect("loom_cpu", [Pack2DToTilesOp, PackConv2DToTilesOp], [])
