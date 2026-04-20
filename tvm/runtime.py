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
from pathlib import Path

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


class _AccelHandle(ct.Structure):
    """Opaque C handle for the accelerator driver."""


class _AccelApi:
    """Typed wrapper around the exported `libaccel.so` symbols."""

    def __init__(self, lib_path: Path):
        self.lib = ct.CDLL(str(lib_path))
        self._configure()

    def _configure(self) -> None:
        self.lib.accel_open.argtypes = [
            ct.c_char_p,
            ct.c_uint32,
            ct.POINTER(ct.POINTER(_AccelHandle)),
        ]
        self.lib.accel_open.restype = ct.c_int

        self.lib.accel_close.argtypes = [ct.POINTER(_AccelHandle)]
        self.lib.accel_close.restype = None

        self.lib.accel_ping.argtypes = [ct.POINTER(_AccelHandle)]
        self.lib.accel_ping.restype = ct.c_int

        self.lib.accel_last_cycles.argtypes = [ct.POINTER(_AccelHandle)]
        self.lib.accel_last_cycles.restype = ct.c_uint16

        self.lib.accel_status_string.argtypes = [ct.c_int]
        self.lib.accel_status_string.restype = ct.c_char_p

        self.lib.accel_write_mem.argtypes = [
            ct.POINTER(_AccelHandle),
            ct.c_uint32,
            ct.POINTER(ct.c_uint8),
            ct.c_size_t,
        ]
        self.lib.accel_write_mem.restype = ct.c_int

        self.lib.accel_read_mem.argtypes = [
            ct.POINTER(_AccelHandle),
            ct.c_uint32,
            ct.POINTER(ct.c_uint8),
            ct.c_size_t,
        ]
        self.lib.accel_read_mem.restype = ct.c_int

        self.lib.accel_exec.argtypes = [
            ct.POINTER(_AccelHandle),
            ct.POINTER(ct.c_uint8),
            ct.c_size_t,
            ct.POINTER(ct.c_uint32),
        ]
        self.lib.accel_exec.restype = ct.c_int

    def status_string(self, code: int) -> str:
        raw = self.lib.accel_status_string(code)
        if raw is None:
            return f"status {code}"
        return raw.decode("utf-8", errors="replace")


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


def pack_weight_rows(matrix: np.ndarray) -> bytes:
    """Pack weights as row-major int8 bytes."""

    return np.ascontiguousarray(matrix, dtype=np.int8).tobytes()


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


class AccelRuntime:
    """Runtime wrapper around `libaccel.so` for tile-oriented execution."""

    def __init__(self, config: RuntimeConfig | None = None):
        self.config = config or RuntimeConfig()
        self._api: _AccelApi | None = None
        self._handle: ct.POINTER(_AccelHandle) | None = None

    def library_exists(self) -> bool:
        """Return whether the configured shared library path exists."""

        return Path(self.config.lib_path).exists()

    def is_open(self) -> bool:
        """Return whether the runtime currently holds an open device handle."""

        return self._handle is not None

    def open(self) -> None:
        """Open the accelerator device connection."""

        if self._handle is not None:
            return

        lib_path = Path(self.config.lib_path)
        if not lib_path.exists():
            raise FileNotFoundError(f"missing accelerator library: {lib_path}")

        self._api = _AccelApi(lib_path)
        handle = ct.POINTER(_AccelHandle)()
        status = self._api.lib.accel_open(
            self.config.port.encode("utf-8"),
            self.config.baud_rate,
            ct.byref(handle),
        )
        self._check_status(status, "accel_open")
        self._handle = handle

    def close(self) -> None:
        """Close the accelerator device connection."""

        if self._handle is None or self._api is None:
            return
        self._api.lib.accel_close(self._handle)
        self._handle = None

    def __enter__(self) -> "AccelRuntime":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def ping(self) -> None:
        """Issue a ping command to the device."""

        api = self._require_api()
        handle = self._require_handle()
        status = api.lib.accel_ping(handle)
        self._check_status(status, "accel_ping")

    def last_cycles(self) -> int:
        """Return the most recent cycle count recorded by the driver."""

        api = self._require_api()
        handle = self._require_handle()
        return int(api.lib.accel_last_cycles(handle))

    def write_mem(self, addr: int, data: bytes | bytearray | memoryview) -> None:
        """Write a raw byte buffer into accelerator-visible memory."""

        api = self._require_api()
        handle = self._require_handle()
        blob = bytes(data)
        if not blob:
            return
        array_type = ct.c_uint8 * len(blob)
        buf = array_type.from_buffer_copy(blob)
        status = api.lib.accel_write_mem(handle, addr, buf, len(blob))
        self._check_status(status, "accel_write_mem")

    def read_mem(self, addr: int, length: int) -> bytes:
        """Read a raw byte buffer from accelerator-visible memory."""

        api = self._require_api()
        handle = self._require_handle()
        if length < 0:
            raise ValueError("length must be non-negative")
        if length == 0:
            return b""
        array_type = ct.c_uint8 * length
        buf = array_type()
        status = api.lib.accel_read_mem(handle, addr, buf, length)
        self._check_status(status, "accel_read_mem")
        return bytes(buf)

    def exec_program(self, program: bytes | bytearray | memoryview) -> int:
        """Execute a KIR program and return the reported cycle count."""

        api = self._require_api()
        handle = self._require_handle()
        blob = bytes(program)
        if not blob:
            raise ValueError("program must not be empty")
        array_type = ct.c_uint8 * len(blob)
        program_buf = array_type.from_buffer_copy(blob)
        cycles = ct.c_uint32()
        status = api.lib.accel_exec(handle, program_buf, len(blob), ct.byref(cycles))
        self._check_status(status, "accel_exec")
        return int(cycles.value)

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
        if m > self.config.tile or n > self.config.tile:
            raise ValueError(
                f"tile execution only supports M,N <= {self.config.tile}, got M={m} N={n}"
            )

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
        cursor = 8 + num_tensors * 12
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

    def _require_api(self) -> _AccelApi:
        api = self._api
        if api is None:
            lib_path = Path(self.config.lib_path)
            if not lib_path.exists():
                raise FileNotFoundError(f"missing accelerator library: {lib_path}")
            api = _AccelApi(lib_path)
            self._api = api
        return api

    def _require_handle(self) -> ct.POINTER(_AccelHandle):
        handle = self._handle
        if handle is None:
            raise AccelRuntimeError("accelerator runtime is not open")
        return handle

    def _check_status(self, status: int, opname: str) -> None:
        if status == 0:
            return
        api = self._api
        detail = f"status {status}"
        if api is not None:
            detail = api.status_string(status)
        raise AccelRuntimeError(f"{opname} failed: {detail}")


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

        if lhs_arr.dtype != np.int8:
            raise TypeError(f"{symbol} expects int8 input, got {lhs_arr.dtype}")

        weight_data = constants.get("weight_data")
        bias_data = constants.get("bias_data")
        weight_scale = constants.get("weight_scale", 1.0)
        weight_zp = constants.get("weight_zp", 0)
        input_scale = constants.get("input_scale", 1.0)
        input_zp = constants.get("input_zp", 0)
        output_scale = constants.get("output_scale", 1.0)
        output_zp = constants.get("output_zp", 0)

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

        if m > runtime.config.tile or n > runtime.config.tile:
            raise ValueError(
                f"{symbol}: tile execution only supports M,N <= {runtime.config.tile}, got M={m} N={n}"
            )

        if compute_requantization_params and LayerEpilogueParams:
            epi_params = compute_requantization_params(
                input_scale=input_scale,
                input_zero_point=input_zp,
                weight_scale=weight_scale,
                weight_zero_point=weight_zp,
                output_scale=output_scale,
                output_zero_point=output_zp,
                bias_fp32=bias_arr.astype(np.float32),
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

        if len(args) >= 3:
            out = args[2]
            _copy_result_to_output(out, result.astype(np.float32))
            return None

        return tvm.runtime.tensor(result.astype(np.float32))


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
    """Create a configured runtime instance."""

    return AccelRuntime(
        RuntimeConfig(
            port=port,
            baud_rate=baud_rate,
            lib_path=lib_path,
        )
    )
