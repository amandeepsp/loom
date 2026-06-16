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


def _int_attr(attr: Attribute) -> int:
    if not isinstance(attr, IntegerAttr):
        raise VerifyException(f"expected integer attribute, got {attr}")
    return int(attr.value.data)


LoomCpu = Dialect("loom_cpu", [Pack2DToTilesOp], [])
