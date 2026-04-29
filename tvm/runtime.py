"""Host runtime bridge for the out-of-tree accel integration."""

from __future__ import annotations

import ctypes as ct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tvm
from tvm import relax
from tvm.relax.expr_functor import PyExprVisitor, visitor

import importlib.util
import sys

from shared.ir import build_pipelined_gemm_program, plan_memory

TENSOR_POOL_BASE = 0x40010000
MEM_ALIGN = 32


def _load_local_module(name: str, filename: str) -> Any:
    path = Path(__file__).with_name(filename)
    module_name = f"accel_local_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load local module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


quant_utils = _load_local_module("quant_utils", "quant_utils.py")
compute_requantization_params = quant_utils.compute_requantization_params
LayerEpilogueParams = quant_utils.LayerEpilogueParams
quantize_multiplier_less_than_one = quant_utils.quantize_multiplier_less_than_one

codegen = _load_local_module("codegen", "codegen.py")
get_composite_constants = codegen.get_composite_constants


class AccelRuntimeError(RuntimeError):
    """Raised when the host runtime cannot complete an accelerator operation."""


@dataclass(frozen=True)
class RuntimeConfig:
    """Configuration for the host-side accelerator runtime bridge."""

    port: str = "/dev/ttyUSB1"
    baud_rate: int = 115200
    lib_path: str = "zig-out/lib/libaccel.so"
    tile: int = 8
    cfu_word_bits: int = 64
    cfu_store_depth_words: int = 512
    tensor_pool_base: int = TENSOR_POOL_BASE
    mem_align: int = MEM_ALIGN


# ---------------------------------------------------------------------------
# Transport layer — decouples AccelRuntime from wire protocol
# ---------------------------------------------------------------------------


class TcpTransport:
    """Transport that speaks the accel wire protocol over a raw TCP socket.

    Used for Verilator simulation (the sim exposes a virtual UART as a TCP
    server).  This class is intentionally standalone so it can also be reused
    by test harnesses that don't need the full TVM stack.
    """

    MAGIC_REQ = 0xCF
    MAGIC_RESP = 0xFC
    OP_READ = 0x10
    OP_WRITE = 0x11
    OP_EXEC = 0x12
    STATUS_OK = 0x00

    def __init__(self, endpoint: str, timeout_s: float = 300):
        import socket

        host, _, port = endpoint.removeprefix("tcp://").rpartition(":")
        self._sock = socket.create_connection((host, int(port)), timeout=timeout_s)
        self._sock.settimeout(timeout_s)
        self._seq_id = 0

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def _read_exact(self, length: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < length:
            chunk = self._sock.recv(length - len(chunks))
            if not chunk:
                raise ConnectionError("TCP transport connection closed")
            chunks.extend(chunk)
        return bytes(chunks)

    def _request(self, op: int, payload: bytes, *, expected_len: int | None = None) -> bytes:
        import struct

        seq_id = self._seq_id
        self._seq_id = (self._seq_id + 1) & 0xFFFF
        self._sock.sendall(struct.pack(
            "<BBHHH", self.MAGIC_REQ, op, len(payload), seq_id, 0))
        self._sock.sendall(payload)

        # Skip firmware startup banner
        while self._read_exact(1)[0] != self.MAGIC_RESP:
            pass
        status, payload_len, resp_seq, _cycles = struct.unpack(
            "<BHHH", self._read_exact(7))
        data = self._read_exact(payload_len)
        if resp_seq != seq_id:
            raise RuntimeError(
                f"bad response seq: expected {seq_id}, got {resp_seq}")
        if status != self.STATUS_OK:
            raise RuntimeError(
                f"device error status=0x{status:02x} debug={list(data)}")
        if expected_len is not None and payload_len != expected_len:
            raise RuntimeError(
                f"bad response length: expected {expected_len}, got {payload_len}")
        return data

    def write_mem(self, addr: int, data: bytes) -> None:
        import struct

        chunk_max = 0xFFFF - 4
        for offset in range(0, len(data), chunk_max):
            chunk = data[offset : offset + chunk_max]
            self._request(self.OP_WRITE,
                          struct.pack("<I", addr + offset) + chunk,
                          expected_len=0)

    def read_mem(self, addr: int, length: int) -> bytes:
        import struct

        out = bytearray()
        chunk_max = 0xFFFF
        for offset in range(0, length, chunk_max):
            chunk_len = min(chunk_max, length - offset)
            payload = struct.pack("<II", addr + offset, chunk_len)
            out.extend(self._request(self.OP_READ, payload,
                                     expected_len=chunk_len))
        return bytes(out)

    def exec_program(self, program: bytes) -> int:
        import struct

        data = self._request(self.OP_EXEC, program, expected_len=4)
        return struct.unpack("<I", data)[0]


# ---------------------------------------------------------------------------
# Serial transport (libaccel.so via ctypes)
# ---------------------------------------------------------------------------


class _AccelHandle(ct.Structure):
    """Opaque C handle for the accelerator driver."""


class SerialTransport:
    """Transport over a physical serial port via ``libaccel.so``.

    Supports an extra ``ping`` method not available on the TCP transport.
    """

    def __init__(self, port: str, baud_rate: int, lib_path: str):
        lib = ct.CDLL(str(lib_path))
        self._configure(lib)
        self._lib = lib
        self._handle: ct.POINTER(_AccelHandle) | None = None
        self._port = port
        self._baud_rate = baud_rate

    @staticmethod
    def _configure(lib) -> None:
        lib.accel_open.argtypes = [
            ct.c_char_p, ct.c_uint32,
            ct.POINTER(ct.POINTER(_AccelHandle)),
        ]
        lib.accel_open.restype = ct.c_int

        lib.accel_close.argtypes = [ct.POINTER(_AccelHandle)]
        lib.accel_close.restype = None

        lib.accel_ping.argtypes = [ct.POINTER(_AccelHandle)]
        lib.accel_ping.restype = ct.c_int

        lib.accel_last_cycles.argtypes = [ct.POINTER(_AccelHandle)]
        lib.accel_last_cycles.restype = ct.c_uint16

        lib.accel_status_string.argtypes = [ct.c_int]
        lib.accel_status_string.restype = ct.c_char_p

        lib.accel_write_mem.argtypes = [
            ct.POINTER(_AccelHandle), ct.c_uint32,
            ct.POINTER(ct.c_uint8), ct.c_size_t,
        ]
        lib.accel_write_mem.restype = ct.c_int

        lib.accel_read_mem.argtypes = [
            ct.POINTER(_AccelHandle), ct.c_uint32,
            ct.POINTER(ct.c_uint8), ct.c_size_t,
        ]
        lib.accel_read_mem.restype = ct.c_int

        lib.accel_exec.argtypes = [
            ct.POINTER(_AccelHandle),
            ct.POINTER(ct.c_uint8), ct.c_size_t,
            ct.POINTER(ct.c_uint32),
        ]
        lib.accel_exec.restype = ct.c_int

    def open(self) -> None:
        if self._handle is not None:
            return
        handle = ct.POINTER(_AccelHandle)()
        status = self._lib.accel_open(
            self._port.encode("utf-8"), self._baud_rate, ct.byref(handle))
        self._check_status(status, "accel_open")
        self._handle = handle

    def close(self) -> None:
        if self._handle is None:
            return
        self._lib.accel_close(self._handle)
        self._handle = None

    def ping(self) -> None:
        status = self._lib.accel_ping(self._handle)
        self._check_status(status, "accel_ping")

    def last_cycles(self) -> int:
        return int(self._lib.accel_last_cycles(self._handle))

    def write_mem(self, addr: int, data: bytes) -> None:
        if not data:
            return
        c_buf = (ct.c_uint8 * len(data)).from_buffer_copy(data)
        status = self._lib.accel_write_mem(self._handle, addr, c_buf, len(data))
        self._check_status(status, "accel_write_mem")

    def read_mem(self, addr: int, length: int) -> bytes:
        if length == 0:
            return b""
        buf = (ct.c_uint8 * length)()
        status = self._lib.accel_read_mem(self._handle, addr, buf, length)
        self._check_status(status, "accel_read_mem")
        return bytes(buf)

    def exec_program(self, program: bytes) -> int:
        c_buf = (ct.c_uint8 * len(program)).from_buffer_copy(program)
        cycles = ct.c_uint32()
        status = self._lib.accel_exec(self._handle, c_buf, len(program), ct.byref(cycles))
        self._check_status(status, "accel_exec")
        return int(cycles.value)

    def status_string(self, code: int) -> str:
        raw = self._lib.accel_status_string(code)
        if raw is None:
            return f"status {code}"
        return raw.decode("utf-8", errors="replace")

    def _check_status(self, status: int, opname: str) -> None:
        if status != 0:
            detail = self.status_string(status)
            raise AccelRuntimeError(f"{opname} failed: {detail}")


@visitor
class _ExternSymbolCollector(PyExprVisitor):
    """Collect packed extern symbols referenced by `call_dps_packed`."""

    def __init__(self) -> None:
        self.symbols: set[str] = set()

    def visit_call_(self, call: relax.Call) -> None:
        if call.op == tvm.ir.Op.get("relax.call_dps_packed"):
            packed_func = call.args[0]
            if isinstance(packed_func, relax.ExternFunc):
                self.symbols.add(str(packed_func.global_symbol))
        super().visit_call_(call)


def align_up(value: int, alignment: int = MEM_ALIGN) -> int:
    """Round `value` up to `alignment` bytes."""

    return (value + alignment - 1) & -alignment


def pack_i8(arr: np.ndarray) -> bytes:
    """Pack an array as contiguous signed-int8 bytes."""

    return np.ascontiguousarray(arr, dtype=np.int8).tobytes()


def pack_i32(arr: np.ndarray) -> bytes:
    """Pack an array as contiguous signed-int32 bytes."""

    return np.ascontiguousarray(arr, dtype=np.int32).tobytes()


def pack_input_tiles(matrix: np.ndarray, tile: int) -> bytes:
    """Pack `[M, K]` activations into the `[K, tile]` layout expected by HW."""

    m, _k = matrix.shape
    chunks: list[np.ndarray] = []
    for m_base in range(0, m, tile):
        tile_slice = matrix[m_base : m_base + tile, :].T
        if tile_slice.shape[1] < tile:
            pad = tile - tile_slice.shape[1]
            tile_slice = np.pad(tile_slice, ((0, 0), (0, pad)), mode="constant")
        chunks.append(np.ascontiguousarray(tile_slice))
    return np.concatenate(chunks).astype(np.int8).tobytes()


def pack_weight_rows(matrix: np.ndarray, tile: int = 8) -> bytes:
    """Pack weights as [K, tile] words expected by the hardware scratchpad."""
    k, n = matrix.shape
    words = []
    for t in range(0, n, tile):
        for kk in range(k):
            vals = matrix[kk, t:t+tile]
            if vals.shape[0] < tile:
                vals = np.pad(vals, (0, tile - vals.shape[0]), mode="constant")
            words.append(np.ascontiguousarray(vals, dtype=np.int8).tobytes())
    return b"".join(words)


def _as_i8_matrix(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int8)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be rank-2, got shape {arr.shape}")
    return np.ascontiguousarray(arr)


def _as_i32_vector(value: Any, *, name: str, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int32)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be rank-1, got shape {arr.shape}")
    if arr.shape[0] != length:
        raise ValueError(f"{name} must have length {length}, got {arr.shape[0]}")
    return np.ascontiguousarray(arr)


def compute_reference_matmul(lhs: Any, rhs: Any) -> np.ndarray:
    """Compute the current `matmul_integer` composite semantics on CPU."""

    lhs_arr = np.asarray(lhs, dtype=np.int32)
    rhs_arr = np.asarray(rhs, dtype=np.int32)
    if lhs_arr.ndim != 2 or rhs_arr.ndim != 2:
        raise ValueError(
            f"reference matmul expects rank-2 inputs, got {lhs_arr.shape} and {rhs_arr.shape}"
        )
    return np.ascontiguousarray(lhs_arr @ rhs_arr, dtype=np.int32)


# Transport union: either TCP or serial
AccelTransport = TcpTransport | SerialTransport


class AccelRuntime:
    """Runtime for tile-oriented GEMM execution on the accelerator.

    Requires a *transport* — ``TcpTransport`` for Verilator simulation or
    ``SerialTransport`` for real hardware via ``libaccel.so``.
    """

    def __init__(self, transport: AccelTransport,
                 config: RuntimeConfig | None = None):
        self.transport = transport
        self.config = config or RuntimeConfig()

    def open(self) -> None:
        if isinstance(self.transport, SerialTransport):
            self.transport.open()

    def close(self) -> None:
        self.transport.close()

    def __enter__(self) -> "AccelRuntime":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def ping(self) -> None:
        if not isinstance(self.transport, SerialTransport):
            raise AccelRuntimeError("ping only available on serial transport")
        self.transport.ping()

    def last_cycles(self) -> int:
        if not isinstance(self.transport, SerialTransport):
            raise AccelRuntimeError("last_cycles only available on serial transport")
        return self.transport.last_cycles()

    def write_mem(self, addr: int, data: bytes | bytearray | memoryview) -> None:
        blob = bytes(data) if not isinstance(data, bytes) else data
        if blob:
            self.transport.write_mem(addr, blob)

    def read_mem(self, addr: int, length: int) -> bytes:
        if length < 0:
            raise ValueError("length must be non-negative")
        if length == 0:
            return b""
        return self.transport.read_mem(addr, length)

    def exec_program(self, program: bytes | bytearray | memoryview) -> int:
        blob = bytes(program) if not isinstance(program, bytes) else program
        if not blob:
            raise ValueError("program must not be empty")
        return self.transport.exec_program(blob)

    def execute_tile(
        self,
        lhs: Any,
        rhs: Any,
        *,
        bias: Any,
        multiplier: Any,
        shift: Any,
        output_offset: int,
        activation_min: int,
        activation_max: int,
    ) -> np.ndarray:
        """Execute one logical output tile and return the final int8 result.

        `lhs` must be shaped `[M, K]`, `rhs` must be `[K, N]`, and the current
        runtime only supports `M <= tile` and `N <= tile`. This matches the
        intended compiler contract where `M/N` tiling is made explicit above the
        runtime boundary.
        """

        lhs_arr = _as_i8_matrix(lhs, name="lhs")
        rhs_arr = _as_i8_matrix(rhs, name="rhs")
        m, k = lhs_arr.shape
        rk, n = rhs_arr.shape
        if k != rk:
            raise ValueError(f"incompatible matmul shapes: {lhs_arr.shape} x {rhs_arr.shape}")


        bias_arr = _as_i32_vector(bias, name="bias", length=n)
        mult_arr = _as_i32_vector(multiplier, name="multiplier", length=n)
        shift_arr = _as_i32_vector(shift, name="shift", length=n)

        if not -128 <= output_offset <= 127:
            raise ValueError("output_offset must fit in signed int8")
        if not -128 <= activation_min <= 127:
            raise ValueError("activation_min must fit in signed int8")
        if not -128 <= activation_max <= 127:
            raise ValueError("activation_max must fit in signed int8")

        input_data = pack_input_tiles(lhs_arr, self.config.tile)
        weight_data = pack_weight_rows(rhs_arr)
        bias_data = pack_i32(bias_arr)
        mult_data = pack_i32(mult_arr)
        shift_data = pack_i32(shift_arr)
        output_size = m * n

        input_addr = self.config.tensor_pool_base + 256
        weight_addr = align_up(input_addr + len(input_data), self.config.mem_align)
        output_addr = align_up(weight_addr + len(weight_data), self.config.mem_align)
        bias_addr = align_up(output_addr + output_size, self.config.mem_align)
        mult_addr = align_up(bias_addr + len(bias_data), self.config.mem_align)
        shift_addr = align_up(mult_addr + len(mult_data), self.config.mem_align)

        layout = plan_memory(
            input_addr=input_addr,
            weight_addr=weight_addr,
            output_addr=output_addr,
            bias_addr=bias_addr,
            mult_addr=mult_addr,
            shift_addr=shift_addr,
        )

        program = build_pipelined_gemm_program(
            layout=layout,
            m=m,
            k=k,
            n=n,
            tile=self.config.tile,
            act_tensor_id=0,
            wgt_tensor_id=1,
            out_tensor_id=2,
            bias_id=3,
            mult_id=4,
            shift_id=5,
            cfu_word_bits=self.config.cfu_word_bits,
            cfu_store_depth_words=self.config.cfu_store_depth_words,
        )

        self.open()
        self.write_mem(input_addr, input_data)
        self.write_mem(weight_addr, weight_data)
        self.write_mem(bias_addr, bias_data)
        self.write_mem(mult_addr, mult_data)
        self.write_mem(shift_addr, shift_data)

        # Patch the program epilogue fields for this tile-level invocation.
        # `shared.ir` emits the same structure as the firmware expects, so it is
        # safe to rewrite the signed byte fields in place.
        patched_program = bytearray(program)
        self._patch_epilogue_bytes(
            patched_program,
            output_offset=output_offset,
            activation_min=activation_min,
            activation_max=activation_max,
        )
        self.exec_program(patched_program)

        output = self.read_mem(output_addr, output_size)
        return np.frombuffer(output, dtype=np.int8).reshape(m, n).copy()

    def _patch_epilogue_bytes(
        self,
        program: bytearray,
        *,
        output_offset: int,
        activation_min: int,
        activation_max: int,
    ) -> None:
        """Rewrite `set_epilogue` immediates in the generated KIR bytecode."""

        set_epilogue_opcode = 0x05
        tile_load_act_opcode = 0x01
        tile_load_wgt_opcode = 0x02
        tile_mma_opcode = 0x03
        tile_store_opcode = 0x04
        done_opcode = 0x06

        num_tensors = program[5]
        cursor = 8 + num_tensors * 16
        while cursor < len(program):
            opcode = program[cursor]
            if opcode == set_epilogue_opcode:
                program[cursor + 8] = output_offset & 0xFF
                program[cursor + 9] = activation_min & 0xFF
                program[cursor + 10] = activation_max & 0xFF
                cursor += 12
            elif opcode in (tile_load_act_opcode, tile_load_wgt_opcode):
                cursor += 8
            elif opcode in (tile_mma_opcode, done_opcode):
                cursor += 4
            elif opcode == tile_store_opcode:
                cursor += 8
            else:
                raise AccelRuntimeError(f"unknown opcode while patching program: 0x{opcode:02x}")


def collect_extern_symbols(mod: tvm.IRModule) -> list[str]:
    """Collect `call_dps_packed` extern symbol names from a Relax module."""

    collector = _ExternSymbolCollector()
    for func in mod.functions.values():
        if isinstance(func, relax.Function):
            collector.visit_expr(func)
    return sorted(collector.symbols)


def _copy_result_to_output(out: Any, result: np.ndarray) -> None:
    """Copy a NumPy result array into a TVM output tensor."""

    if hasattr(out, "copyfrom"):
        out.copyfrom(np.ascontiguousarray(result))
        return
    raise TypeError(f"unsupported DPS output object: {type(out)!r}")


def _make_cpu_packed(symbol: str):
    """Create a CPU packed function for the current matmul-core boundary."""

    def _packed(*args):
        if len(args) not in (2, 3):
            raise ValueError(f"{symbol} expects 2 or 3 arguments, got {len(args)}")

        lhs, rhs = args[0], args[1]
        result = compute_reference_matmul(lhs.numpy(), rhs.numpy())

        if len(args) == 2:
            return tvm.runtime.tensor(result)

        out = args[2]
        _copy_result_to_output(out, result)
        return None

    return _packed


def _make_accel_packed(symbol: str, runtime: AccelRuntime):
    """Create an accelerator-backed packed function.

    This function looks up the composite constants for the symbol and
    uses them to derive hardware epilogue parameters for the accelerator.
    """
    constants = {}
    if get_composite_constants:
        constants = get_composite_constants(symbol)

    def _packed(*args):
        if len(args) < 1:
            raise ValueError(f"{symbol} expects at least 1 argument (input tensor)")

        lhs = args[0]

        if not hasattr(lhs, "numpy"):
            raise TypeError(f"{symbol} first argument must be a TVM tensor")

        lhs_arr = lhs.numpy()

        weight_data = constants.get("weight_data")
        bias_data = constants.get("bias_data")
        weight_scale = constants.get("weight_scale", 1.0)
        weight_zp = constants.get("weight_zp", 0)
        input_scale = constants.get("input_scale", 1.0)
        input_zp = constants.get("input_zp", 0)
        output_scale = constants.get("output_scale", 1.0)
        output_zp = constants.get("output_zp", 0)

        if lhs_arr.dtype != np.int8:
            # Input is not int8 — re-quantize from float32 using input_scale/zp.
            if lhs_arr.dtype == np.float32:
                lhs_arr = np.clip(
                    np.round(lhs_arr / float(input_scale) + float(input_zp)),
                    -128, 127,
                ).astype(np.int8)
            else:
                raise TypeError(
                    f"{symbol}: expected int8 or float32 input, got {lhs_arr.dtype}"
                )

        if weight_data is None:
            raise ValueError(f"{symbol}: missing weight data in constants")

        if bias_data is None:
            n = weight_data.shape[0]
            bias_data = np.zeros(n, dtype=np.int32)

        weight_arr = np.ascontiguousarray(weight_data, dtype=np.int8)
        bias_arr = np.ascontiguousarray(bias_data, dtype=np.int32)

        m, k = lhs_arr.shape
        rk, n = weight_arr.shape
        if k != rk:
            raise ValueError(f"{symbol}: incompatible shapes lhs {lhs_arr.shape} x weight {weight_arr.shape}")


        if compute_requantization_params and LayerEpilogueParams:
            epi_params = compute_requantization_params(
                input_scale=input_scale,
                input_zero_point=input_zp,
                weight_scale=weight_scale,
                weight_zero_point=weight_zp,
                output_scale=output_scale,
                output_zero_point=output_zp,
                bias_fp32=bias_arr.astype(np.float32) * float(constants.get("bias_scale", 1.0)),
                has_relu=False,
                activation_is_signed=True,
            )
            bias = epi_params.bias
            multiplier = epi_params.multiplier
            shift = epi_params.shift
            output_offset = epi_params.output_offset
            act_min = epi_params.activation_min
            act_max = epi_params.activation_max
        else:
            combined_scale = (input_scale * weight_scale) / output_scale
            mult, shft = quantize_multiplier_less_than_one(combined_scale)
            mult = int(mult)
            shft = int(shft)
            bias = np.full(n, 0, dtype=np.int32)
            multiplier = np.full(n, mult, dtype=np.int32)
            shift = np.full(n, shft, dtype=np.int32)
            output_offset = np.int8(output_zp)
            act_min = np.int8(-128)
            act_max = np.int8(127)

        result = runtime.execute_tile(
            lhs=lhs_arr,
            rhs=weight_arr,
            bias=bias,
            multiplier=multiplier,
            shift=shift,
            output_offset=int(output_offset),
            activation_min=int(act_min),
            activation_max=int(act_max),
        )

        # Explicit contiguous copy to ensure proper memory lifetime.
        result_copy = np.ascontiguousarray(result.astype(np.float32))

        if len(args) >= 2:
            out = args[1]
            _copy_result_to_output(out, result_copy)
            return None

        print(f"[_packed] {symbol[-40:]}: M={m}K={k}N={n} result=[{result.min()},{result.max()}] epi(off={output_offset},min={act_min},max={act_max}) mult=[{multiplier.min()},{multiplier.max()}] shift=[{shift.min()},{shift.max()}] bias=[{bias.min()},{bias.max()}]", file=sys.stderr)
        return tvm.runtime.tensor(result_copy)

    return _packed
def register_runtime_functions(
    mod: tvm.IRModule,
    *,
    mode: str = "cpu",
    runtime: AccelRuntime | None = None,
    override: bool = True,
) -> list[str]:
    """Register packed runtime functions referenced by a lowered Relax module.

    Parameters
    ----------
    mod:
        Lowered Relax module that contains `call_dps_packed` extern calls.
    mode:
        Registration mode. `cpu` registers correct reference implementations for
        the current matmul-core boundary. `accel` installs fast-fail stubs until
        the composite boundary includes hardware epilogue semantics.
    runtime:
        Optional accelerator runtime instance for future accelerator-backed
        registration. Required when `mode="accel"`.
    override:
        Passed through to `tvm.register_global_func`.
    """

    symbols = collect_extern_symbols(mod)
    if mode not in {"cpu", "accel"}:
        raise ValueError(f"unsupported runtime mode: {mode}")
    if mode == "accel" and runtime is None:
        runtime = create_runtime()

    for symbol in symbols:
        if mode == "cpu":
            packed = _make_cpu_packed(symbol)
        else:
            assert runtime is not None
            packed = _make_accel_packed(symbol, runtime)
        tvm.register_global_func(symbol, packed, override=override)

    return symbols


def create_runtime(
    *,
    port: str = "/dev/ttyUSB1",
    baud_rate: int = 115200,
    lib_path: str = "zig-out/lib/libaccel.so",
) -> AccelRuntime:
    """Create a configured runtime instance with a serial transport."""

    transport = SerialTransport(port=port, baud_rate=baud_rate, lib_path=lib_path)
    return AccelRuntime(transport, RuntimeConfig(port=port, baud_rate=baud_rate, lib_path=lib_path))
