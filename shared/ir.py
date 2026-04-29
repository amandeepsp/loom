"""Accel IR definition — single source of truth for bytecode format."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import ClassVar

KIR_MAGIC = 0x4B495200
KIR_VERSION = 1

ACCEL_EXTERN_PREFIX = "accel"

TILE_LOAD_ACT = 0x01
TILE_LOAD_WGT = 0x02
TILE_MMA = 0x03
TILE_STORE = 0x04
SET_EPILOGUE = 0x05
DONE = 0x06

DTYPE_I8 = 0x0
DTYPE_I32 = 0x1


class _IrType:
    _fmt: ClassVar[str]


@dataclass
class TensorSpec(_IrType):
    base_addr: int
    dim0: int
    dim1: int
    stride: int
    dtype: int
    flags: int = 0
    padding: int = 0
    _struct = struct.Struct("<IHHHBBI")

    def pack(self) -> bytes:
        return self._struct.pack(
            self.base_addr, self.dim0, self.dim1, self.stride,
            self.dtype, self.flags, self.padding,
        )


@dataclass
class TileLoadAct(_IrType):
    opcode: int = TILE_LOAD_ACT
    tensor_id: int = 0
    m_offset: int = 0
    k_offset: int = 0
    k_words: int = 0
    _struct = struct.Struct("<BBHHH")

    def pack(self) -> bytes:
        return self._struct.pack(
            self.opcode, self.tensor_id, self.m_offset, self.k_offset, self.k_words,
        )


@dataclass
class TileLoadWgt(_IrType):
    opcode: int = TILE_LOAD_WGT
    tensor_id: int = 0
    n_offset: int = 0
    k_offset: int = 0
    k_words: int = 0
    _struct = struct.Struct("<BBHHH")

    def pack(self) -> bytes:
        return self._struct.pack(
            self.opcode, self.tensor_id, self.n_offset, self.k_offset, self.k_words,
        )


@dataclass
class TileMma(_IrType):
    opcode: int = TILE_MMA
    flags: int = 0
    k_count: int = 0
    _struct = struct.Struct("<BBH")

    def pack(self) -> bytes:
        return self._struct.pack(self.opcode, self.flags, self.k_count)


@dataclass
class TileStore(_IrType):
    opcode: int = TILE_STORE
    tensor_id: int = 0
    m_offset: int = 0
    n_offset: int = 0
    m_count: int = 0
    n_count: int = 0
    _struct = struct.Struct("<BBHHBB")

    def pack(self) -> bytes:
        return self._struct.pack(
            self.opcode, self.tensor_id, self.m_offset, self.n_offset,
            self.m_count, self.n_count,
        )


@dataclass
class SetEpilogue(_IrType):
    opcode: int = SET_EPILOGUE
    bias_tid: int = 0
    mult_tid: int = 0
    shift_tid: int = 0
    n_offset: int = 0
    n_count: int = 0
    output_offset: int = 0
    act_min: int = 0
    act_max: int = 0
    padding: int = 0
    _struct = struct.Struct("<BBBBHHbbbB")

    def pack(self) -> bytes:
        return self._struct.pack(
            self.opcode, self.bias_tid, self.mult_tid, self.shift_tid,
            self.n_offset, self.n_count, self.output_offset, self.act_min,
            self.act_max, self.padding,
        )


@dataclass
class Done(_IrType):
    opcode: int = DONE
    pad0: int = 0
    pad1: int = 0
    pad2: int = 0
    _struct = struct.Struct("<BBBB")

    def pack(self) -> bytes:
        return self._struct.pack(self.opcode, self.pad0, self.pad1, self.pad2)


_INSTRUCTION_SIZES = {
    TILE_LOAD_ACT: TileLoadAct._struct.size,
    TILE_LOAD_WGT: TileLoadWgt._struct.size,
    TILE_MMA: TileMma._struct.size,
    TILE_STORE: TileStore._struct.size,
    SET_EPILOGUE: SetEpilogue._struct.size,
    DONE: Done._struct.size,
}


def _pack(ir_type: _IrType) -> bytes:
    """Deprecated: use ir_type.pack() directly."""
    return ir_type.pack()


def patch_epilogue(
    program: bytes | bytearray,
    *,
    output_offset: int,
    activation_min: int,
    activation_max: int,
) -> bytes:
    """Rewrite `set_epilogue` immediates in generated KIR bytecode.

    Returns a new bytes object; the input is not modified.
    """
    data = bytearray(program)
    num_tensors = data[5]
    header_size = 8 + num_tensors * struct.calcsize(TensorSpec._fmt)
    cursor = header_size
    patched = 0
    while cursor < len(data):
        opcode = data[cursor]
        if opcode == SET_EPILOGUE:
            data[cursor + 8] = output_offset & 0xFF
            data[cursor + 9] = activation_min & 0xFF
            data[cursor + 10] = activation_max & 0xFF
            patched += 1
        step = _INSTRUCTION_SIZES.get(opcode)
        if step is None:
            raise ValueError(f"unknown opcode while patching program: 0x{opcode:02x}")
        cursor += step
    return bytes(data)


@dataclass
class MemoryLayout:
    bias_addr: int
    mult_addr: int
    shift_addr: int
    weights_addr: int
    input_addr: int
    output_addr: int


class ProgramBuilder:
    def __init__(self):
        self.tensors: list[TensorSpec] = []
        self.code = bytearray()
        self.instruction_count = 0

    def add_tensor(
        self, addr: int, dim0: int, dim1: int, stride: int, dtype: int = DTYPE_I8
    ) -> int:
        tensor_id = len(self.tensors)
        self.tensors.append(TensorSpec(addr, dim0, dim1, stride, dtype))
        return tensor_id

    def tile_load_act(self, tensor_id: int, m_offset: int, k_offset: int, k_words: int) -> None:
        self.code.extend(TileLoadAct(TILE_LOAD_ACT, tensor_id, m_offset, k_offset, k_words).pack())
        self.instruction_count += 1

    def tile_load_wgt(self, tensor_id: int, n_offset: int, k_offset: int, k_words: int) -> None:
        self.code.extend(TileLoadWgt(TILE_LOAD_WGT, tensor_id, n_offset, k_offset, k_words).pack())
        self.instruction_count += 1

    def tile_mma(self, first: bool, last: bool, k_count: int) -> None:
        flags = (1 if first else 0) | (2 if last else 0)
        self.code.extend(TileMma(TILE_MMA, flags, k_count).pack())
        self.instruction_count += 1

    def tile_store(
        self, tensor_id: int, m_offset: int, n_offset: int, m_count: int, n_count: int
    ) -> None:
        self.code.extend(TileStore(TILE_STORE, tensor_id, m_offset, n_offset, m_count, n_count).pack())
        self.instruction_count += 1

    def set_epilogue(
        self,
        bias_tid: int,
        mult_tid: int,
        shift_tid: int,
        n_offset: int,
        n_count: int,
        output_offset: int,
        act_min: int,
        act_max: int,
    ) -> None:
        self.code.extend(
            SetEpilogue(
                SET_EPILOGUE, bias_tid, mult_tid, shift_tid, n_offset, n_count,
                output_offset, act_min, act_max,
            ).pack()
        )
        self.instruction_count += 1

    def done(self) -> None:
        self.code.extend(Done().pack())
        self.instruction_count += 1

    def build(self) -> bytes:
        program = bytearray()
        program.extend(struct.pack("<I", KIR_MAGIC))
        program.append(KIR_VERSION)
        program.append(len(self.tensors))
        program.extend(struct.pack("<H", self.instruction_count))
        for t in self.tensors:
            program.extend(t.pack())
        program.extend(self.code)
        return bytes(program)


def plan_memory(
    input_addr: int,
    weight_addr: int,
    output_addr: int,
    bias_addr: int,
    mult_addr: int,
    shift_addr: int,
) -> MemoryLayout:
    return MemoryLayout(
        bias_addr, mult_addr, shift_addr, weight_addr, input_addr, output_addr
    )


def build_non_pipelined_gemm_program(
    layout: MemoryLayout,
    m: int,
    k: int,
    n: int,
    act_tensor_id: int,
    wgt_tensor_id: int,
    out_tensor_id: int,
    bias_id: int,
    mult_id: int,
    shift_id: int,
    tile: int = 8,
    k_tile: int = 8,
) -> bytes:
    builder = ProgramBuilder()

    builder.add_tensor(layout.input_addr, m, k, k * tile, DTYPE_I8)
    builder.add_tensor(layout.weights_addr, k, n, n, DTYPE_I8)
    builder.add_tensor(layout.output_addr, m, n, n, DTYPE_I8)
    builder.add_tensor(layout.bias_addr, 1, n, n * 4, DTYPE_I32)
    builder.add_tensor(layout.mult_addr, 1, n, n * 4, DTYPE_I32)
    builder.add_tensor(layout.shift_addr, 1, n, n * 4, DTYPE_I32)

    for n_base in range(0, n, tile):
        n_count = min(tile, n - n_base)
        for m_base in range(0, m, tile):
            m_count = min(tile, m - m_base)
            builder.set_epilogue(
                bias_id, mult_id, shift_id, n_base, n_count, 0, -128, 127
            )
            for k_base in range(0, k, k_tile):
                k_count = min(k_tile, k - k_base)
                builder.tile_load_act(act_tensor_id, m_base, k_base * tile, k_count)
                builder.tile_load_wgt(wgt_tensor_id, n_base, k_base, k_count)
                builder.tile_mma(k_base == 0, k_base + k_count == k, k_count)
            builder.tile_store(out_tensor_id, m_base, n_base, m_count, n_count)

    builder.done()
    return builder.build()


def build_pipelined_gemm_program(
    layout: MemoryLayout,
    m: int,
    k: int,
    n: int,
    tile: int,
    act_tensor_id: int,
    wgt_tensor_id: int,
    out_tensor_id: int,
    bias_id: int,
    mult_id: int,
    shift_id: int,
    cfu_word_bits: int = 64,
    cfu_store_depth_words: int = 512,
    k_tile: int | None = None,
) -> bytes:
    if k_tile is None:
        dma_beats_per_line = cfu_word_bits // 32
        k_tile = cfu_store_depth_words // dma_beats_per_line

    builder = ProgramBuilder()

    builder.add_tensor(layout.input_addr, m, k, k * tile, DTYPE_I8)
    builder.add_tensor(layout.weights_addr, k, n, n, DTYPE_I8)
    builder.add_tensor(layout.output_addr, m, n, n, DTYPE_I8)
    builder.add_tensor(layout.bias_addr, 1, n, n * 4, DTYPE_I32)
    builder.add_tensor(layout.mult_addr, 1, n, n * 4, DTYPE_I32)
    builder.add_tensor(layout.shift_addr, 1, n, n * 4, DTYPE_I32)

    for n_base in range(0, n, tile):
        n_count = min(tile, n - n_base)
        for m_base in range(0, m, tile):
            m_count = min(tile, m - m_base)
            builder.set_epilogue(
                bias_id, mult_id, shift_id, n_base, n_count, 0, -128, 127
            )
            for k_base in range(0, k, k_tile):
                k_count = min(k_tile, k - k_base)
                builder.tile_load_act(act_tensor_id, m_base, k_base * tile, k_count)
                builder.tile_load_wgt(wgt_tensor_id, n_base, k_base, k_count)
                builder.tile_mma(k_base == 0, k_base + k_count == k, k_count)
            builder.tile_store(out_tensor_id, m_base, n_base, m_count, n_count)

    builder.done()
    return builder.build()
