"""Host runtime bridge for the out-of-tree accel integration."""

from __future__ import annotations

import ctypes as ct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tvm
from tvm import relax
from tvm.relax.expr_functor import PyExprVisitor, visitor

from shared.ir import build_gemm_program, patch_epilogue, plan_memory

from .codegen import get_composite_constants
from .quant_utils import quantize_multiplier_less_than_one

TENSOR_POOL_BASE = 0x40010000
MEM_ALIGN = 32


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

# Re-export shared transport classes for backward compatibility.
from shared.protocol import TcpTransport

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

        program = build_gemm_program(
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
        program = patch_epilogue(
            program,
            output_offset=output_offset,
            activation_min=activation_min,
            activation_max=activation_max,
        )
        self.exec_program(program)

        output = self.read_mem(output_addr, output_size)
        return np.frombuffer(output, dtype=np.int8).reshape(m, n).copy()


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

        # Compute epilogue params directly for our hardware.
        #
        # The hardware computes pure integer matmul without subtracting zero points:
        #     acc_hw[m,n] = sum_k x_q[m,k] * w_q[k,n]
        # but the QDQ math requires:
        #     acc_true[m,n] = sum_k (x_q[m,k] - x_zp) * (w_q[k,n] - w_zp)
        # With w_zp = 0 (typical for symmetric per-channel weights):
        #     acc_true = acc_hw - x_zp * sum_k w_q[k,n]
        # Fold this correction into the bias (constant per output channel):
        #     bias_hw[n] = bias_onnx[n] - x_zp * sum_k w_q[k,n]
        #
        # ONNX bias is already quantized as round(bias_fp32 / (input_scale * weight_scale)),
        # which is exactly the scale the hardware epilogue expects.
        if int(weight_zp) != 0:
            raise NotImplementedError(
                f"{symbol}: nonzero weight_zp ({weight_zp}) not yet supported"
            )
        combined_scale = (input_scale * weight_scale) / output_scale
        mult, shft = quantize_multiplier_less_than_one(combined_scale)
        mult = int(mult)
        shft = int(shft)

        sum_w = weight_arr.astype(np.int32).sum(axis=0)  # shape [N]
        bias = (bias_arr - int(input_zp) * sum_w).astype(np.int32)
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

        # The accel returns int8, but the Relax composite expects float32
        # because it ends with a dequantize node.  Dequantize before returning.
        result_float = (result.astype(np.float32) - float(output_zp)) * float(output_scale)
        result_copy = np.ascontiguousarray(result_float)

        if len(args) >= 2:
            out = args[1]
            _copy_result_to_output(out, result_copy)
            return None

        return tvm.runtime.tensor(result_copy)

    return _packed
def register_runtime_functions(
    mod: tvm.IRModule,
    *,
    runtime: AccelRuntime | None = None,
    override: bool = True,
) -> list[str]:
    """Register accelerator-backed packed functions referenced by a lowered Relax module.

    Parameters
    ----------
    mod:
        Lowered Relax module that contains `call_dps_packed` extern calls.
    runtime:
        Accelerator runtime instance. Created automatically when omitted.
    override:
        Passed through to `tvm.register_global_func`.
    """

    symbols = collect_extern_symbols(mod)
    if runtime is None:
        runtime = create_runtime()

    for symbol in symbols:
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
