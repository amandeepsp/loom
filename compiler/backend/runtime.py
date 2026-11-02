"""Runtime helper for executing compiled loom model binaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Protocol

import numpy as np

from .model import (
    Buffer,
    CompiledModel,
    GemmKernel,
    MEM_END,
    MEM_START,
    SCHED_CALL_RISCV,
    SECTION_CONSTANTS,
    SECTION_RISCV,
    SECTION_SCHEDULE,
    model_binary_manifest,
    model_binary_sections,
)


class ModelTransport(Protocol):
    def write_mem(self, addr: int, data: bytes) -> None: ...

    def read_mem(self, addr: int, length: int) -> bytes: ...

    def exec_model(self, addr: int, length: int) -> int: ...


@dataclass(frozen=True)
class ModelRunResult:
    output: np.ndarray
    cycles: int
    model_addr: int


@dataclass(frozen=True)
class ModelBinaryArtifact:
    """A compiled model binary plus the embedded deployment manifest."""

    blob: bytes
    manifest: dict

    @classmethod
    def from_bytes(cls, blob: bytes) -> "ModelBinaryArtifact":
        blob = bytes(blob)
        sections = model_binary_sections(blob)
        _validate_executable_sections(sections)
        return cls(blob, model_binary_manifest(blob))

    @classmethod
    def from_file(cls, path: str | Path) -> "ModelBinaryArtifact":
        return cls.from_bytes(Path(path).read_bytes())

    @property
    def model_addr(self) -> int:
        deployment = _dict_field(self.manifest, "deployment")
        return _int_field(deployment, "model_addr")

    @property
    def memory_end(self) -> int:
        deployment = _dict_field(self.manifest, "deployment")
        return _int_field(deployment, "memory_end", default=MEM_END)

    def buffer(self, name: str) -> Buffer:
        buffers = self.manifest.get("buffers")
        if not isinstance(buffers, list):
            raise ValueError("model manifest is missing buffers list")
        for item in buffers:
            if not isinstance(item, dict) or item.get("name") != name:
                continue
            return Buffer(
                str(item["name"]),
                _int_field(item, "addr"),
                _int_field(item, "size"),
                str(item.get("kind", "")),
            )
        raise KeyError(f"unknown model buffer: {name}")


class ModelBinaryRuntime:
    """Upload and execute an already-compiled model binary artifact."""

    def __init__(
        self,
        artifact: ModelBinaryArtifact,
        transport: ModelTransport,
        *,
        model_addr: int | None = None,
    ) -> None:
        self.artifact = artifact
        self.transport = transport
        self.model_blob = artifact.blob
        self.model_addr = artifact.model_addr if model_addr is None else model_addr
        self._validate_model_range(self.model_addr, len(self.model_blob))
        # Firmware only reads the blob (constants are copied out of it), so one
        # upload serves every later run on the same transport.
        self._model_uploaded = False

    def run_quantized(self, input_matrix: np.ndarray) -> ModelRunResult:
        contract = _dict_field(self.artifact.manifest, "runtime_contract")
        if contract.get("boundary") != "quantized-int8":
            raise ValueError("model artifact does not declare a quantized-int8 boundary")

        input_contract = _dict_field(contract, "input")
        output_contract = _dict_field(contract, "output")
        input_shape = _shape_field(input_contract, "shape")
        output_shape = _shape_field(output_contract, "shape")
        padded_shape = _shape_field(output_contract, "padded_shape")

        input_arr = np.asarray(input_matrix)
        if input_arr.dtype != np.int8:
            raise TypeError("input_matrix must be dtype int8")
        if input_arr.shape != input_shape:
            raise ValueError(
                f"input shape {input_arr.shape} does not match model shape {input_shape}"
            )

        input_buf = self.artifact.buffer(str(input_contract["buffer"]))
        output_buf = self.artifact.buffer(str(output_contract["buffer"]))
        input_bytes = np.ascontiguousarray(input_arr).tobytes()
        if len(input_bytes) != input_buf.size:
            raise ValueError(
                f"input size {len(input_bytes)} does not match planned input buffer {input_buf.size}"
            )

        self.transport.write_mem(input_buf.addr, input_bytes)
        if not self._model_uploaded:
            self.transport.write_mem(self.model_addr, self.model_blob)
            self._model_uploaded = True
        cycles = self.transport.exec_model(self.model_addr, len(self.model_blob))
        output_bytes = self.transport.read_mem(output_buf.addr, output_buf.size)
        output = np.frombuffer(output_bytes, dtype=np.int8).reshape(padded_shape)
        return ModelRunResult(
            output=output[tuple(slice(0, dim) for dim in output_shape)].copy(),
            cycles=cycles,
            model_addr=self.model_addr,
        )

    def _validate_model_range(self, addr: int, length: int) -> None:
        memory_end = self.artifact.memory_end
        if addr < MEM_START or addr + length > memory_end:
            raise ValueError(
                f"model address range out of runtime RAM: addr=0x{addr:x} len={length}"
            )
        for item in self.artifact.manifest.get("buffers", []):
            if not isinstance(item, dict):
                continue
            buf_addr = _int_field(item, "addr")
            buf_size = _int_field(item, "size")
            buf_name = str(item.get("name", "<unnamed>"))
            if _ranges_overlap(addr, length, buf_addr, buf_size):
                raise ValueError(
                    f"model blob overlaps planned buffer {buf_name}: "
                    f"model=[0x{addr:x},0x{addr + length:x}) "
                    f"buffer=[0x{buf_addr:x},0x{buf_addr + buf_size:x})"
                )


class ModelRuntime:
    """Upload and execute a compiled model binary through the shared transport."""

    def __init__(
        self,
        compiled: CompiledModel,
        transport: ModelTransport,
        *,
        model_addr: int | None = None,
    ) -> None:
        self.compiled = compiled
        self.transport = transport
        self.model_blob = compiled.to_bytes(include_debug_sections=False)
        self.model_addr = (
            self.default_model_addr(compiled, len(self.model_blob))
            if model_addr is None
            else model_addr
        )
        self._validate_model_range(self.model_addr, len(self.model_blob))

    @staticmethod
    def default_model_addr(
        compiled: CompiledModel,
        model_len: int,
        *,
        mem_end: int = MEM_END,
        alignment: int = 32,
    ) -> int:
        """Place the model blob after all planned tensor buffers."""

        addr = compiled.recommended_model_addr(alignment=alignment)
        if addr + model_len > mem_end:
            raise ValueError(
                f"model blob does not fit in runtime RAM: "
                f"addr=0x{addr:x} len={model_len} end=0x{mem_end:x}"
            )
        return addr

    def run_quantized(self, input_matrix: np.ndarray) -> ModelRunResult:
        """Run the model with an already-quantized int8 input matrix."""

        contract = _dict_field(self.compiled.manifest(), "runtime_contract")
        input_contract = _dict_field(contract, "input")
        output_contract = _dict_field(contract, "output")
        input_shape = _shape_field(input_contract, "shape")
        output_shape = _shape_field(output_contract, "shape")
        padded_shape = _shape_field(output_contract, "padded_shape")
        input_arr = np.asarray(input_matrix)
        if input_arr.dtype != np.int8:
            raise TypeError("input_matrix must be dtype int8")
        if input_arr.shape != input_shape:
            raise ValueError(
                f"input shape {input_arr.shape} does not match compiled model shape {input_shape}"
            )

        input_buf = self._buffer(str(input_contract["buffer"]))
        output_buf = self._buffer(str(output_contract["buffer"]))

        input_bytes = np.ascontiguousarray(input_arr).tobytes()
        if len(input_bytes) != input_buf.size:
            raise ValueError(
                f"input size {len(input_bytes)} does not match "
                f"planned input buffer {input_buf.size}"
            )

        self.transport.write_mem(input_buf.addr, input_bytes)
        self.transport.write_mem(self.model_addr, self.model_blob)
        cycles = self.transport.exec_model(self.model_addr, len(self.model_blob))

        output_bytes = self.transport.read_mem(output_buf.addr, output_buf.size)
        output = np.frombuffer(output_bytes, dtype=np.int8).reshape(padded_shape)
        return ModelRunResult(
            output=output[tuple(slice(0, dim) for dim in output_shape)].copy(),
            cycles=cycles,
            model_addr=self.model_addr,
        )

    def _buffer(self, name: str) -> Buffer:
        for buf in self.compiled.buffers:
            if buf.name == name:
                return buf
        raise KeyError(f"unknown model buffer: {name}")

    def _validate_model_range(self, addr: int, length: int) -> None:
        if addr < MEM_START or addr + length > MEM_END:
            raise ValueError(
                f"model address range out of runtime RAM: addr=0x{addr:x} len={length}"
            )
        for buf in self.compiled.buffers:
            if _ranges_overlap(addr, length, buf.addr, buf.size):
                raise ValueError(
                    f"model blob overlaps planned buffer {buf.name}: "
                    f"model=[0x{addr:x},0x{addr + length:x}) "
                    f"buffer=[0x{buf.addr:x},0x{buf.addr + buf.size:x})"
                )


def _ranges_overlap(a_addr: int, a_len: int, b_addr: int, b_len: int) -> bool:
    return a_addr < b_addr + b_len and b_addr < a_addr + a_len


def _validate_executable_sections(sections: dict[int, bytes]) -> None:
    for required in (SECTION_CONSTANTS, SECTION_RISCV, SECTION_SCHEDULE):
        if required not in sections:
            raise ValueError(f"model binary is missing required section {required}")
    function_count = _validate_riscv_section(sections[SECTION_RISCV])
    _validate_schedule_section(sections[SECTION_SCHEDULE], function_count)


def _validate_riscv_section(section: bytes) -> int:
    if len(section) < 4:
        raise ValueError("model RISC-V section is truncated")
    count = struct.unpack_from("<I", section)[0]
    if count == 0:
        raise ValueError("model RISC-V section has no functions")
    cursor = 4
    for index in range(count):
        if cursor + 8 > len(section):
            raise ValueError(f"RISC-V function record {index} is truncated")
        code_len, name_len, _flags = struct.unpack_from("<IHH", section, cursor)
        cursor += 8
        cursor = _align4(cursor + name_len)
        if code_len == 0 or code_len % 4 != 0:
            raise ValueError(f"RISC-V function {index} has invalid code length {code_len}")
        end = cursor + code_len
        if end > len(section):
            raise ValueError(f"RISC-V function {index} code is truncated")
        cursor = _align4(end)
    return count


def _validate_schedule_section(section: bytes, function_count: int) -> None:
    if len(section) < 4:
        raise ValueError("model schedule section is truncated")
    count = struct.unpack_from("<I", section)[0]
    cursor = 4
    for index in range(count):
        if cursor + 8 > len(section):
            raise ValueError(f"schedule command {index} is truncated")
        op, _flags, _reserved, function_index = struct.unpack_from("<BBHI", section, cursor)
        if op != SCHED_CALL_RISCV:
            raise ValueError(f"unsupported schedule command opcode {op}")
        if function_index >= function_count:
            raise ValueError(
                f"schedule command {index} references missing RISC-V function {function_index}"
            )
        cursor += 8


def _align4(value: int) -> int:
    return (value + 3) & ~3


def _dict_field(mapping: dict, key: str) -> dict:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"model manifest field {key} must be an object")
    return value


def _int_field(mapping: dict, key: str, *, default: int | None = None) -> int:
    value = mapping.get(key, default)
    if not isinstance(value, int):
        raise ValueError(f"model manifest field {key} must be an integer")
    return value


def _shape_field(mapping: dict, key: str) -> tuple[int, ...]:
    value = mapping.get(key)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(dim, int) and dim > 0 for dim in value)
    ):
        raise ValueError(f"model manifest field {key} must be a positive integer shape")
    return tuple(value)
