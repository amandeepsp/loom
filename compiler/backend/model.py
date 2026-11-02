"""Compile supported xDSL/TOSA GEMM regions into loom model binaries.

The supported accelerator pattern is:

    tosa.matmul -> tosa.add(bias) -> tosa.rescale

Each matched region is lowered through a tiled affine schedule into compact
RV32I + CUSTOM0 code.  The model-binary runtime loads constants, executes the
RISC-V code sections, and represents activation layout packing as plain RV32I
functions in the same model binary.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np

from xdsl.dialects import tensor
from xdsl.dialects.builtin import ModuleOp, TensorType
from xdsl.dialects.func import ReturnOp
from xdsl.dialects.tosa import AddOp, ConstOp, MulOp, RescaleOp
from xdsl.ir import SSAValue

from compiler.frontend import emit_tosa_ir, load_model
from compiler.frontend.qlinear_mobilenet import (
    QuantizedGemmPlan,
    QuantizedMobileNetPlan,
    is_qlinear_mobilenet,
    parse_qlinear_mobilenet,
)
from compiler.frontend.tosa_ext import CastOp, MatMulOp
from compiler.dialects.loom import (
    GemmOp,
    GemmCandidateOp,
    RepackOp,
)
from compiler.dialects.loom_cpu import Pack2DToTilesOp, PackConv2DToTilesOp
from shared.layout import align_up, pack_weight_rows

from .cpu import compile_c_kernel, emit_lut_map, emit_pack_2d_to_tiles_op, emit_pack_conv2d_to_tiles_op
from .riscv_ir import RiscvLoweringContext, lower_affine_kernel_to_riscv_scf
from .riscv_ir_encode import encode_riscv_module
from .target import CsrMap, GemmTargetSpec
from .rewrites import (
    PackFunctionBinding,
    PlannedGemmReplacement,
    compute_k_tile,
    lower_gemm_candidates_to_loom,
    lower_gemm_to_affine,
    lower_repack_to_cpu,
    lower_tosa_gemm_to_candidates,
)

MODEL_MAGIC = b"LMB0"
MODEL_VERSION = 1
SECTION_MANIFEST = 1
SECTION_CONSTANTS = 2
SECTION_ASM = 4
SECTION_RISCV = 5
SECTION_LOOM_IR = 6
SECTION_AFFINE_IR = 7
SECTION_SCHEDULE = 8
SECTION_RISCV_IR = 9
DEFAULT_CSR_JSON = Path("build/sipeed_tang_nano_20k/csr.json")
MEM_START = 0x40010000
MEM_END = 0x40800000

SCHED_CALL_RISCV = 1


@dataclass(frozen=True)
class Buffer:
    name: str
    addr: int
    size: int
    kind: str
    data: bytes = b""

    @property
    def is_constant(self) -> bool:
        return bool(self.data)


@dataclass(frozen=True)
class GemmKernel:
    name: str
    m: int
    k: int
    n: int
    padded_m: int
    tile: int
    input_buffer: str
    packed_input_buffer: str
    output_buffer: str
    riscv_code: bytes
    riscv_asm: str
    requires_repack_from: str | None = None
    kind: str = "gemm"
    # Name of the kernel whose result store writes this kernel's packed input
    # directly (fused pack); such kernels have no separate pack function.
    packed_by: str | None = None


@dataclass(frozen=True)
class RiscvFunction:
    name: str
    code: bytes
    asm: str = ""


@dataclass(frozen=True)
class RuntimeTensor:
    buffer: str
    shape: tuple[int, ...]
    padded_shape: tuple[int, ...]
    scale: float | None = None
    zero_point: int | None = None
    layout: str | None = None


@dataclass(frozen=True)
class CompiledModel:
    buffers: tuple[Buffer, ...]
    kernels: tuple[GemmKernel, ...]
    functions: tuple[RiscvFunction, ...]
    memory_base: int
    entry: str
    loom_module: ModuleOp
    affine_module: ModuleOp
    riscv_module: ModuleOp
    runtime_input: RuntimeTensor | None = None
    runtime_output: RuntimeTensor | None = None
    source: dict | None = None

    def recommended_model_addr(self, *, alignment: int = 32) -> int:
        """Return a non-overlapping runtime RAM address for this model blob."""

        high_water = max(
            (buf.addr + buf.size for buf in self.buffers),
            default=self.memory_base,
        )
        return align_up(high_water, alignment)

    def manifest(self) -> dict:
        first = self.kernels[0]
        final = self.kernels[-1]
        if self.runtime_input is None:
            input_contract = {
                "buffer": first.input_buffer,
                "dtype": "int8",
                "shape": [first.m, first.k],
            }
        else:
            input_contract = _runtime_tensor_manifest(self.runtime_input)
        if self.runtime_output is None:
            output_contract = {
                "buffer": final.output_buffer,
                "dtype": "int8",
                "shape": [final.m, final.n],
                "padded_shape": [final.padded_m, final.n],
            }
        else:
            output_contract = _runtime_tensor_manifest(self.runtime_output)
        return {
            "format": "loom-model",
            "version": MODEL_VERSION,
            "memory_base": self.memory_base,
            "entry": self.entry,
            "deployment": {
                "model_addr": self.recommended_model_addr(),
                "alignment": 32,
                "memory_end": MEM_END,
            },
            "runtime_contract": {
                "boundary": "quantized-int8",
                "input": input_contract,
                "output": output_contract,
                "omitted_boundary_ops": ["input_quantize", "output_dequantize"],
            },
            "buffers": [
                {
                    "name": b.name,
                    "addr": b.addr,
                    "size": b.size,
                    "kind": b.kind,
                    "constant": b.is_constant,
                }
                for b in self.buffers
            ],
            "kernels": [
                {
                    "name": k.name,
                    "m": k.m,
                    "k": k.k,
                    "n": k.n,
                    "padded_m": k.padded_m,
                    "tile": k.tile,
                    "input_buffer": k.input_buffer,
                    "packed_input_buffer": k.packed_input_buffer,
                    "output_buffer": k.output_buffer,
                    "riscv_code_size": len(k.riscv_code),
                    "requires_repack_from": k.requires_repack_from,
                    "kind": k.kind,
                    "packed_by": k.packed_by,
                }
                for k in self.kernels
            ],
            "riscv_functions": [
                {
                    "name": fn.name,
                    "riscv_code_size": len(fn.code),
                }
                for fn in self.functions
            ],
            **({"source": self.source} if self.source else {}),
        }

    def to_bytes(self, *, include_debug_sections: bool = True) -> bytes:
        manifest = json.dumps(self.manifest(), sort_keys=True).encode()
        constants = _pack_constant_section(b for b in self.buffers if b.is_constant)
        riscv_code = _pack_riscv_section(self.functions)
        schedule = _pack_schedule_section(self.functions)
        # Largest, most stable section first: when only code or the manifest
        # changes, the constant bytes keep their offsets, so a chunked
        # checksum diff re-uploads just the tail (see ModelBinaryRuntime).
        sections = [
            (SECTION_CONSTANTS, constants),
            (SECTION_RISCV, riscv_code),
            (SECTION_SCHEDULE, schedule),
            (SECTION_MANIFEST, manifest),
        ]
        if include_debug_sections:
            sections.extend(
                (
                    (SECTION_ASM, "\n\n".join(fn.asm for fn in self.functions if fn.asm).encode()),
                    (SECTION_LOOM_IR, str(self.loom_module).encode()),
                    (SECTION_AFFINE_IR, str(self.affine_module).encode()),
                    (SECTION_RISCV_IR, str(self.riscv_module).encode()),
                )
            )
        return _pack_sections(tuple(sections))


def model_binary_sections(blob: bytes) -> dict[int, bytes]:
    """Return model-binary sections keyed by section type."""

    if len(blob) < 12:
        raise ValueError("model binary is too small")
    magic, version, _flags, section_count = struct.unpack_from("<4sHHI", blob)
    if magic != MODEL_MAGIC:
        raise ValueError("bad model binary magic")
    if version != MODEL_VERSION:
        raise ValueError(f"unsupported model binary version: {version}")
    if section_count > 64:
        raise ValueError(f"model binary has too many sections: {section_count}")

    cursor = 12
    sections: dict[int, bytes] = {}
    for index in range(section_count):
        if cursor + 12 > len(blob):
            raise ValueError(f"section table entry {index} is truncated")
        section_type, _section_flags, offset, size = struct.unpack_from("<HHII", blob, cursor)
        cursor += 12
        end = offset + size
        if end < offset or end > len(blob):
            raise ValueError(f"section {section_type} range is outside model binary")
        sections[section_type] = blob[offset:end]
    return sections


def model_binary_manifest(blob: bytes) -> dict:
    """Extract and decode the JSON manifest embedded in a model binary."""

    sections = model_binary_sections(blob)
    manifest_data = sections.get(SECTION_MANIFEST)
    if manifest_data is None:
        raise ValueError("model binary is missing manifest section")
    manifest = json.loads(manifest_data.decode())
    if not isinstance(manifest, dict):
        raise ValueError("model manifest must be a JSON object")
    return manifest


@dataclass(frozen=True)
class _LoweredGemm:
    name: str
    input: SSAValue
    output: SSAValue
    m: int
    k: int
    n: int
    weight: np.ndarray
    bias: np.ndarray
    multiplier: np.ndarray
    shift: np.ndarray
    output_zp: int
    activation_min: int = -128
    activation_max: int = 127


class _Planner:
    def __init__(self, base: int, alignment: int):
        self.next_addr = align_up(base, alignment)
        self.alignment = alignment
        self.buffers: list[Buffer] = []

    def add(self, name: str, size: int, kind: str, data: bytes = b"") -> Buffer:
        if size < 0:
            raise ValueError(f"negative buffer size for {name}")
        addr = self.next_addr
        self.next_addr = align_up(addr + size, self.alignment)
        buf = Buffer(name, addr, size, kind, data)
        self.buffers.append(buf)
        return buf


def compile_onnx_model(
    path: str,
    *,
    batch_size: int = 1,
    memory_base: int = 0x40010100,
    tile: int = 8,
    cfu_word_bits: int = 64,
    cfu_store_depth_words: int = 512,
    csr_json: str | Path | None = DEFAULT_CSR_JSON,
) -> CompiledModel:
    """Load ONNX, emit xDSL TOSA, then compile supported GEMM regions."""

    compiled = _compile_onnx_model(
        path,
        batch_size=batch_size,
        memory_base=memory_base,
        tile=tile,
        cfu_word_bits=cfu_word_bits,
        cfu_store_depth_words=cfu_store_depth_words,
        csr_json=csr_json,
    )
    # Record provenance so runners can recompute fresh references from the ONNX.
    onnx_path = Path(path).resolve()
    source = {
        "onnx": str(onnx_path),
        "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
    }
    return replace(compiled, source=source)


def block_diagonal_taps(layer, tile: int = 8) -> int | None:
    """Kernel taps if `layer` is a block-diagonal conv GEMM (depthwise, pooling).

    Such layers are lowered as dense GEMMs over K = taps * C whose weight is
    nonzero only where (kk % C) == output channel. With a channel-major K
    order, each tile of output channels needs one contiguous K slice of
    taps * tile rows, so the zero blocks never reach the array.
    """
    pk = getattr(layer, "packing", None)
    if pk is None or layer.n != pk.channels or layer.n % tile:
        return None
    taps = pk.kernel_h * pk.kernel_w
    if layer.k != taps * pk.channels:
        return None
    kk_channel = np.arange(layer.k)[:, None] % pk.channels
    off_diagonal = kk_channel != np.arange(layer.n)[None, :]
    if np.any(np.asarray(layer.weight)[off_diagonal]):
        return None
    return taps


def block_diagonal_2d_taps(layer) -> int | None:
    """Taps if a 2-D-packed layer (no conv packing) is block-diagonal, tap-major.

    E.g. global average pooling: K = taps * C, weight nonzero only where
    k % C == column. Such layers run as a small direct CPU kernel.
    """
    if getattr(layer, "packing", None) is not None or layer.k % layer.n or layer.k == layer.n:
        return None
    kk_channel = np.arange(layer.k)[:, None] % layer.n
    if np.any(np.asarray(layer.weight)[kk_channel != np.arange(layer.n)[None, :]]):
        return None
    return layer.k // layer.n


def block_diagonal_weight(layer, taps: int, tile: int = 8) -> np.ndarray:
    """[taps * tile, C] stacked diagonal blocks in channel-major K order."""
    channels = layer.n
    weight = np.asarray(layer.weight, dtype=np.int8)
    out = np.zeros((taps * tile, channels), dtype=np.int8)
    for c in range(channels):
        local = (c % tile) * taps
        out[local : local + taps, c] = weight[[t * channels + c for t in range(taps)], c]
    return out


def compile_gemm(
    weight: np.ndarray,
    *,
    m: int,
    bias: np.ndarray,
    multiplier: np.ndarray,
    shift: np.ndarray,
    output_zero_point: int = 0,
    activation_min: int = -128,
    activation_max: int = 127,
    memory_base: int = 0x40010100,
    tile: int = 8,
    cfu_word_bits: int = 64,
    cfu_store_depth_words: int = 512,
    csr_json: str | Path | None = DEFAULT_CSR_JSON,
) -> CompiledModel:
    """Compile one int8 GEMM `[m, k] @ weight[k, n]` with explicit epilogue params.

    `shift` is loom's residual right shift (applied after SRDHM), not TOSA's
    scale32 shift. The result is an ordinary `.lmb`: run it with
    ModelBinaryRuntime like any model (used by GEMM tests and benchmarks).
    """

    weight = np.ascontiguousarray(weight, dtype=np.int8)
    if weight.ndim != 2:
        raise ValueError("weight must be [k, n]")
    k, n = weight.shape

    def vec(values: np.ndarray, label: str) -> np.ndarray:
        arr = np.asarray(values, dtype=np.int32).reshape(-1)
        if arr.size != n:
            raise ValueError(f"{label} has {arr.size} entries, expected N={n}")
        return arr

    layer = QuantizedGemmPlan(
        name="gemm",
        kind="gemm",
        m=m,
        k=k,
        n=n,
        weight=weight,
        bias=vec(bias, "bias"),
        multiplier=vec(multiplier, "multiplier"),
        shift=vec(shift, "shift"),
        input_zero_point=0,
        output_zero_point=output_zero_point,
        activation_min=activation_min,
        activation_max=activation_max,
    )
    plan = QuantizedMobileNetPlan(
        input_shape=(m, k),
        output_shape=(m, n),
        input_scale=1.0,
        input_zero_point=0,
        output_scale=1.0,
        output_zero_point=output_zero_point,
        layers=(layer,),
    )
    return _compile_quantized_mobilenet(
        plan,
        memory_base=memory_base,
        tile=tile,
        cfu_word_bits=cfu_word_bits,
        cfu_store_depth_words=cfu_store_depth_words,
        csr_json=csr_json,
    )


def _compile_onnx_model(
    path: str,
    *,
    batch_size: int,
    memory_base: int,
    tile: int,
    cfu_word_bits: int,
    cfu_store_depth_words: int,
    csr_json: str | Path | None,
) -> CompiledModel:
    if is_qlinear_mobilenet(path):
        return _compile_quantized_mobilenet(
            parse_qlinear_mobilenet(path),
            memory_base=memory_base,
            tile=tile,
            cfu_word_bits=cfu_word_bits,
            cfu_store_depth_words=cfu_store_depth_words,
            csr_json=csr_json,
        )

    return compile_tosa_module(
        emit_tosa_ir(load_model(path)),
        batch_size=batch_size,
        memory_base=memory_base,
        tile=tile,
        cfu_word_bits=cfu_word_bits,
        cfu_store_depth_words=cfu_store_depth_words,
        csr_json=csr_json,
    )


def compile_tosa_module(
    module: ModuleOp,
    *,
    batch_size: int = 1,
    memory_base: int = 0x40010100,
    tile: int = 8,
    cfu_word_bits: int = 64,
    cfu_store_depth_words: int = 512,
    csr_json: str | Path | None = DEFAULT_CSR_JSON,
) -> CompiledModel:
    """Lower every supported TOSA GEMM region to planned executable payloads."""

    return _compile_tosa_module(
        module,
        batch_size=batch_size,
        memory_base=memory_base,
        tile=tile,
        cfu_word_bits=cfu_word_bits,
        cfu_store_depth_words=cfu_store_depth_words,
        csr_json=csr_json,
        emit_code=True,
    )


def _compile_quantized_mobilenet(
    plan: QuantizedMobileNetPlan,
    *,
    memory_base: int,
    tile: int,
    cfu_word_bits: int,
    cfu_store_depth_words: int,
    csr_json: str | Path | None,
) -> CompiledModel:
    """Compile a canonical QOperator MobileNet plan to CPU-pack + CFU GEMMs."""

    _validate_target_params(
        tile=tile,
        cfu_word_bits=cfu_word_bits,
        cfu_store_depth_words=cfu_store_depth_words,
    )
    csr_map = _load_csr_map(csr_json)
    if csr_map is None:
        raise FileNotFoundError(f"csr.json is required to emit executable GEMM code: {csr_json}")

    planner = _Planner(memory_base, 32)
    input_size = int(np.prod(plan.input_shape))
    source = planner.add("mobilenet.input", input_size, "input")
    input_buffer = source
    kernels: list[GemmKernel] = []
    functions: list[RiscvFunction] = []
    loom_ops = []
    affine_ops = []
    riscv_ir_ops = []

    fused_packs: dict[int, Buffer] = {}  # layer index -> packed buffer its producer fills
    for index, layer in enumerate(plan.layers):
        transform_addr = 0
        if layer.input_transform is not None:
            transform_data = np.ascontiguousarray(
                layer.input_transform, dtype=np.int8
            ).tobytes()
            transform_buffer = planner.add(
                _buffer_name(index, "input_transform"),
                len(transform_data),
                "constant.requant_lut",
                transform_data,
            )
            transform_addr = transform_buffer.addr
        padded_m = align_up(layer.m, tile)
        packed_input_size = padded_m * layer.k
        prepacked = fused_packs.pop(index, None)
        packed_input = prepacked or planner.add(
            _buffer_name(index, "input_packed"),
            packed_input_size,
            "activation_packed",
        )
        pack_name = _buffer_name(index, "pack_input")
        taps = block_diagonal_taps(layer, tile)
        cpu_taps = block_diagonal_2d_taps(layer)
        if layer.packing is None:
            pack_op = Pack2DToTilesOp(
                pack_name,
                source.name,
                packed_input.name,
                src_addr=source.addr,
                dst_addr=packed_input.addr,
                transform_addr=transform_addr,
                m=layer.m,
                k=layer.k,
                tile=tile,
            )
            pack_code = emit_pack_2d_to_tiles_op(pack_op)
        else:
            packing = layer.packing
            pack_src_addr = source.addr
            pack_transform_addr = transform_addr
            lut_bytes = packing.input_h * packing.input_w * packing.channels
            if (transform_addr and packing.kernel_h * packing.kernel_w > 1
                    and not packing.nchw and lut_bytes % 4 == 0):
                # Each source byte feeds up to kernel_h*kernel_w taps; apply
                # the requant LUT once into a scratch copy, pack without it.
                lut_scratch = planner.add(_buffer_name(index, "input_lut_mapped"),
                                          lut_bytes, "activation")
                lut_name = _buffer_name(index, "lut_map")
                functions.append(RiscvFunction(lut_name, emit_lut_map(
                    lut_name, src_addr=source.addr, dst_addr=lut_scratch.addr,
                    n_bytes=lut_bytes, lut_addr=transform_addr)))
                pack_src_addr = lut_scratch.addr
                pack_transform_addr = 0
            pack_op = PackConv2DToTilesOp(
                pack_name,
                source.name,
                packed_input.name,
                src_addr=pack_src_addr,
                dst_addr=packed_input.addr,
                transform_addr=pack_transform_addr,
                channel_major=taps is not None,
                input_h=packing.input_h,
                input_w=packing.input_w,
                channels=packing.channels,
                output_h=packing.output_h,
                output_w=packing.output_w,
                kernel_h=packing.kernel_h,
                kernel_w=packing.kernel_w,
                stride_h=packing.stride_h,
                stride_w=packing.stride_w,
                pad_top=packing.pad_top,
                pad_left=packing.pad_left,
                input_zero_point=layer.input_zero_point,
                tile=tile,
                nchw=packing.nchw,
            )
            pack_code = emit_pack_conv2d_to_tiles_op(pack_op)
        if cpu_taps is None and prepacked is None:
            loom_ops.append(pack_op)
            functions.append(RiscvFunction(pack_name, pack_code))

        gemm_k = layer.k if taps is None else taps * tile
        if cpu_taps is not None:
            # [taps][C]: the diagonal of each tap block, for the CPU kernel.
            w = np.asarray(layer.weight, dtype=np.int8)
            t_idx, c_idx = np.meshgrid(np.arange(cpu_taps), np.arange(layer.n), indexing="ij")
            weight_data = np.ascontiguousarray(w[t_idx * layer.n + c_idx, c_idx]).tobytes()
        else:
            gemm_weight = layer.weight if taps is None else block_diagonal_weight(layer, taps, tile)
            weight_data = pack_weight_rows(gemm_weight, tile)
        weight_buffer = planner.add(
            _buffer_name(index, "weights"),
            0,
            "constant.weights",
            weight_data,
        )
        weight_buffer = _replace_size(planner, weight_buffer, len(weight_buffer.data))
        output = planner.add(
            _buffer_name(index, "output"),
            padded_m * layer.n,
            "output" if index == len(plan.layers) - 1 else "activation",
        )
        bias_buffer = planner.add(
            _buffer_name(index, "bias"),
            layer.n * 4,
            "constant.bias",
            layer.bias.astype("<i4", copy=False).tobytes(),
        )
        multiplier_buffer = planner.add(
            _buffer_name(index, "multiplier"),
            layer.n * 4,
            "constant.multiplier",
            layer.multiplier.astype("<i4", copy=False).tobytes(),
        )
        shift_buffer = planner.add(
            _buffer_name(index, "shift"),
            layer.n * 4,
            "constant.shift",
            layer.shift.astype("<i4", copy=False).tobytes(),
        )
        if cpu_taps is not None:
            code = compile_c_kernel(layer.name, "blockdiag.c", {
                "SRC_ADDR": source.addr, "DST_ADDR": output.addr,
                "WGT_ADDR": weight_buffer.addr, "BIAS_ADDR": bias_buffer.addr,
                "MULT_ADDR": multiplier_buffer.addr, "SHIFT_ADDR": shift_buffer.addr,
                "LUT_ADDR": transform_addr, "M": layer.m, "C": layer.n, "TAPS": cpu_taps,
                "OUT_ZP": layer.output_zero_point,
                "ACT_MIN": layer.activation_min, "ACT_MAX": layer.activation_max,
            })
            functions.append(RiscvFunction(layer.name, code, ""))
            kernels.append(GemmKernel(
                name=layer.name, m=layer.m, k=layer.k, n=layer.n, padded_m=padded_m,
                tile=tile, input_buffer=source.name, packed_input_buffer=packed_input.name,
                output_buffer=output.name, riscv_code=code, riscv_asm="",
                requires_repack_from=source.name if index else None,
                kind=layer.kind + "_cpu",
            ))
            source = output
            continue
        gemm_op = GemmOp(
            layer.name,
            m=layer.m,
            k=gemm_k,
            n=layer.n,
            padded_m=padded_m,
            tile=tile,
            input_addr=packed_input.addr,
            weight_addr=weight_buffer.addr,
            output_addr=output.addr,
            bias_addr=bias_buffer.addr,
            multiplier_addr=multiplier_buffer.addr,
            shift_addr=shift_buffer.addr,
            output_offset=layer.output_zero_point,
            activation_min=layer.activation_min,
            activation_max=layer.activation_max,
        )
        loom_ops.append(gemm_op)
        kernel_affine_module = lower_gemm_to_affine(
            ModuleOp([gemm_op.clone()]),
            cfu_word_bits=cfu_word_bits,
            cfu_store_depth_words=cfu_store_depth_words,
        )
        kernel_affine_ops = tuple(op.clone() for op in kernel_affine_module.body.block.ops)
        affine_ops.extend(op.clone() for op in kernel_affine_ops)
        # F1: if the next layer is a pointwise conv fed by a 2-D pack + LUT,
        # store this layer's results straight into its packed layout.
        fused_addr = fused_lut_addr = 0
        nxt = plan.layers[index + 1] if index + 1 < len(plan.layers) else None
        if (nxt is not None and nxt.packing is None and nxt.kind == "conv2d"
                and nxt.input_transform is not None and nxt.k == layer.n and nxt.m == layer.m
                and block_diagonal_2d_taps(nxt) is None):
            fused = planner.add(_buffer_name(index + 1, "input_packed"),
                                padded_m * nxt.k, "activation_packed")
            fused_packs[index + 1] = fused
            lut = np.asarray(nxt.input_transform, dtype=np.int8)
            offsets = np.arange(-128, 256)
            rebased = lut[np.where(offsets < 128, offsets, offsets - 256) + 128]
            store_lut = planner.add(_buffer_name(index + 1, "store_lut"), len(rebased),
                                    "constant.requant_lut", rebased.astype(np.int8).tobytes())
            fused_addr, fused_lut_addr = fused.addr, store_lut.addr
        target_spec = GemmTargetSpec(
            packed_output_addr=fused_addr,
            store_lut_addr=fused_lut_addr,
            m=padded_m,
            k=gemm_k,
            n=layer.n,
            tile=tile,
            input_addr=packed_input.addr,
            weight_addr=weight_buffer.addr,
            output_addr=output.addr,
            bias_addr=bias_buffer.addr,
            mult_addr=multiplier_buffer.addr,
            shift_addr=shift_buffer.addr,
            output_offset=layer.output_zero_point,
            activation_min=layer.activation_min,
            activation_max=layer.activation_max,
            cfu_word_bits=cfu_word_bits,
            cfu_store_depth_words=cfu_store_depth_words,
            input_k_stride=layer.k if taps is not None else 0,
            input_n_k_offset=taps or 0,
        )
        kernel_riscv_module = lower_affine_kernel_to_riscv_scf(
            RiscvLoweringContext(spec=target_spec, csr=csr_map),
            kernel_affine_ops,
        )
        riscv_ir_ops.extend(op.clone() for op in kernel_riscv_module.body.block.ops)
        riscv_code = encode_riscv_module(kernel_riscv_module)
        asm = _emit_custom0_asm(layer, padded_m)
        functions.append(RiscvFunction(layer.name, riscv_code, asm))
        kernels.append(
            GemmKernel(
                name=layer.name,
                m=layer.m,
                k=layer.k,
                n=layer.n,
                padded_m=padded_m,
                tile=tile,
                input_buffer=source.name,
                packed_input_buffer=packed_input.name,
                output_buffer=output.name,
                riscv_code=riscv_code,
                riscv_asm=asm,
                requires_repack_from=source.name if index else None,
                kind=layer.kind,
                packed_by=plan.layers[index - 1].name if prepacked is not None else None,
            )
        )
        source = output

    final = kernels[-1]
    compiled = CompiledModel(
        buffers=tuple(planner.buffers),
        kernels=tuple(kernels),
        functions=tuple(functions),
        memory_base=memory_base,
        entry=kernels[0].name,
        loom_module=ModuleOp([op.clone() for op in loom_ops]),
        affine_module=ModuleOp(affine_ops),
        riscv_module=ModuleOp(riscv_ir_ops),
        runtime_input=RuntimeTensor(
            input_buffer.name,
            plan.input_shape,
            plan.input_shape,
            scale=plan.input_scale,
            zero_point=plan.input_zero_point,
            layout="NCHW" if len(plan.input_shape) == 4 else "NC",
        ),
        runtime_output=RuntimeTensor(
            final.output_buffer,
            plan.output_shape,
            (final.padded_m, final.n),
            scale=plan.output_scale,
            zero_point=plan.output_zero_point,
            layout="NC",
        ),
    )
    _validate_executable_model(compiled)
    return compiled


def _compile_tosa_module(
    module: ModuleOp,
    *,
    batch_size: int,
    memory_base: int,
    tile: int,
    cfu_word_bits: int,
    cfu_store_depth_words: int,
    csr_json: str | Path | None,
    emit_code: bool,
) -> CompiledModel:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    _validate_target_params(
        tile=tile,
        cfu_word_bits=cfu_word_bits,
        cfu_store_depth_words=cfu_store_depth_words,
    )

    module = lower_tosa_gemm_to_candidates(module)
    candidates = tuple(op for op in module.walk() if isinstance(op, GemmCandidateOp))
    _validate_supported_quantized_chain(module, candidates)
    planner = _Planner(memory_base, 32)
    kernels: list[GemmKernel] = []
    affine_ops = []
    riscv_ir_ops = []
    pack_bindings: list[PackFunctionBinding] = []
    gemm_functions: dict[str, RiscvFunction] = {}
    value_buffers: dict[SSAValue, Buffer] = {}
    csr_map = _load_csr_map(csr_json) if emit_code else None
    if emit_code and csr_map is None:
        raise FileNotFoundError(
            f"csr.json is required to emit executable GEMM code: {csr_json}"
        )
    candidate_index = 0

    def plan_candidate(candidate: GemmCandidateOp) -> PlannedGemmReplacement:
        nonlocal candidate_index
        index = candidate_index
        candidate_index += 1

        gemm = _lower_gemm_candidate(candidate, batch_size=batch_size)
        padded_m = align_up(gemm.m, tile)
        packed_input_size = (padded_m // tile) * gemm.k * tile
        output_size = padded_m * gemm.n

        input_source = value_buffers.get(gemm.input)
        if input_source is None:
            raw_input_buf = planner.add(
                _buffer_name(index, "input"),
                gemm.m * gemm.k,
                "input" if index == 0 else "activation",
            )
            input_buf = planner.add(
                _buffer_name(index, "input_packed"),
                packed_input_size,
                "activation_packed",
            )
            pack_from = raw_input_buf.name
        else:
            input_buf = planner.add(
                _buffer_name(index, "input_packed"),
                packed_input_size,
                "activation_packed",
            )
            pack_from = input_source.name

        pack_function_name = _buffer_name(index, "pack_input")
        pack_bindings.append(
            PackFunctionBinding(
                source=pack_from,
                target=input_buf.name,
                function_name=pack_function_name,
            )
        )

        weight_buf = planner.add(
            _buffer_name(index, "weights"),
            0,
            "constant.weights",
            pack_weight_rows(gemm.weight, tile),
        )
        weight_buf = _replace_size(planner, weight_buf, len(weight_buf.data))

        output_buf = planner.add(_buffer_name(index, "output"), output_size, "activation")
        bias_buf = planner.add(
            _buffer_name(index, "bias"),
            gemm.n * 4,
            "constant.bias",
            gemm.bias.astype("<i4", copy=False).tobytes(),
        )
        mult_buf = planner.add(
            _buffer_name(index, "multiplier"),
            gemm.n * 4,
            "constant.multiplier",
            gemm.multiplier.astype("<i4", copy=False).tobytes(),
        )
        shift_buf = planner.add(
            _buffer_name(index, "shift"),
            gemm.n * 4,
            "constant.shift",
            gemm.shift.astype("<i4", copy=False).tobytes(),
        )

        kernel_affine_module = lower_gemm_to_affine(
            ModuleOp([
                GemmOp(
                    gemm.name,
                    m=gemm.m,
                    k=gemm.k,
                    n=gemm.n,
                    padded_m=padded_m,
                    tile=tile,
                    input_addr=input_buf.addr,
                    weight_addr=weight_buf.addr,
                    output_addr=output_buf.addr,
                    bias_addr=bias_buf.addr,
                    multiplier_addr=mult_buf.addr,
                    shift_addr=shift_buf.addr,
                    output_offset=gemm.output_zp,
                    activation_min=gemm.activation_min,
                    activation_max=gemm.activation_max,
                )
            ]),
            cfu_word_bits=cfu_word_bits,
            cfu_store_depth_words=cfu_store_depth_words,
        )
        kernel_affine_ops = tuple(op.clone() for op in kernel_affine_module.body.block.ops)
        target_spec = GemmTargetSpec(
            m=padded_m,
            k=gemm.k,
            n=gemm.n,
            tile=tile,
            input_addr=input_buf.addr,
            weight_addr=weight_buf.addr,
            output_addr=output_buf.addr,
            bias_addr=bias_buf.addr,
            mult_addr=mult_buf.addr,
            shift_addr=shift_buf.addr,
            output_offset=gemm.output_zp,
            activation_min=gemm.activation_min,
            activation_max=gemm.activation_max,
            cfu_word_bits=cfu_word_bits,
            cfu_store_depth_words=cfu_store_depth_words,
        )
        riscv_code = b""
        if emit_code and csr_map is not None:
            kernel_riscv_module = lower_affine_kernel_to_riscv_scf(
                RiscvLoweringContext(spec=target_spec, csr=csr_map),
                kernel_affine_ops,
            )
            riscv_ir_ops.extend(op.clone() for op in kernel_riscv_module.body.block.ops)
            riscv_code = encode_riscv_module(kernel_riscv_module)
        gemm_fn = RiscvFunction(
            name=gemm.name,
            code=riscv_code,
            asm=_emit_custom0_asm(gemm, padded_m),
        )
        gemm_functions[gemm.name] = gemm_fn

        affine_ops.extend(op.clone() for op in kernel_affine_ops)

        kernel = GemmKernel(
            name=gemm.name,
            m=gemm.m,
            k=gemm.k,
            n=gemm.n,
            padded_m=padded_m,
            tile=tile,
            input_buffer=pack_from,
            packed_input_buffer=input_buf.name,
            output_buffer=output_buf.name,
            riscv_code=riscv_code,
            riscv_asm=_emit_custom0_asm(gemm, padded_m),
            requires_repack_from=pack_from if input_source is not None else None,
        )
        kernels.append(kernel)
        value_buffers[gemm.output] = output_buf

        return PlannedGemmReplacement(
            RepackOp(
                pack_from,
                input_buf.name,
                m=gemm.m,
                k=gemm.k,
                tile=tile,
            ),
            GemmOp(
                gemm.name,
                m=gemm.m,
                k=gemm.k,
                n=gemm.n,
                padded_m=padded_m,
                tile=tile,
                input_addr=input_buf.addr,
                weight_addr=weight_buf.addr,
                output_addr=output_buf.addr,
                bias_addr=bias_buf.addr,
                multiplier_addr=mult_buf.addr,
                shift_addr=shift_buf.addr,
                output_offset=gemm.output_zp,
                activation_min=gemm.activation_min,
                activation_max=gemm.activation_max,
            ),
        )

    lower_gemm_candidates_to_loom(module, plan_candidate)
    if not kernels:
        raise ValueError("no supported tosa.matmul -> tosa.add -> tosa.rescale regions found")

    loom_ops = [
        op.clone()
        for op in module.walk()
        if isinstance(op, RepackOp | GemmOp)
    ]
    loom_module = ModuleOp(loom_ops)
    lower_repack_to_cpu(loom_module, planner.buffers, pack_bindings)
    functions = (
        tuple(_functions_from_rewritten_loom(loom_module, gemm_functions))
        if emit_code
        else ()
    )

    compiled = CompiledModel(
        buffers=tuple(planner.buffers),
        kernels=tuple(kernels),
        functions=functions,
        memory_base=memory_base,
        entry=kernels[0].name,
        loom_module=loom_module,
        affine_module=ModuleOp(affine_ops),
        riscv_module=ModuleOp(riscv_ir_ops),
    )
    if emit_code:
        _validate_executable_model(compiled)
    return compiled


def lower_tosa_to_loom(
    module: ModuleOp,
    *,
    batch_size: int = 1,
    memory_base: int = 0x40010100,
    tile: int = 8,
    cfu_word_bits: int = 64,
    cfu_store_depth_words: int = 512,
    csr_json: str | Path | None = DEFAULT_CSR_JSON,
) -> ModuleOp:
    """Return the IRDL-declared loom plan produced from supported TOSA regions."""

    return _compile_tosa_module(
        module,
        batch_size=batch_size,
        memory_base=memory_base,
        tile=tile,
        cfu_word_bits=cfu_word_bits,
        cfu_store_depth_words=cfu_store_depth_words,
        csr_json=csr_json,
        emit_code=False,
    ).loom_module


def lower_tosa_to_affine(
    module: ModuleOp,
    *,
    batch_size: int = 1,
    memory_base: int = 0x40010100,
    tile: int = 8,
    cfu_word_bits: int = 64,
    cfu_store_depth_words: int = 512,
    csr_json: str | Path | None = DEFAULT_CSR_JSON,
) -> ModuleOp:
    """Return the tiled affine loop plan produced from supported TOSA regions."""

    return _compile_tosa_module(
        module,
        batch_size=batch_size,
        memory_base=memory_base,
        tile=tile,
        cfu_word_bits=cfu_word_bits,
        cfu_store_depth_words=cfu_store_depth_words,
        csr_json=csr_json,
        emit_code=False,
    ).affine_module


def lower_tosa_to_riscv(
    module: ModuleOp,
    *,
    batch_size: int = 1,
    memory_base: int = 0x40010100,
    tile: int = 8,
    cfu_word_bits: int = 64,
    cfu_store_depth_words: int = 512,
    csr_json: str | Path | None = DEFAULT_CSR_JSON,
) -> ModuleOp:
    """Return the structured xDSL RISC-V target IR for supported TOSA regions."""

    return _compile_tosa_module(
        module,
        batch_size=batch_size,
        memory_base=memory_base,
        tile=tile,
        cfu_word_bits=cfu_word_bits,
        cfu_store_depth_words=cfu_store_depth_words,
        csr_json=csr_json,
        emit_code=True,
    ).riscv_module


def _replace_size(planner: _Planner, old: Buffer, size: int) -> Buffer:
    new = Buffer(old.name, old.addr, size, old.kind, old.data)
    planner.buffers[-1] = new
    planner.next_addr = align_up(old.addr + size, planner.alignment)
    return new


def _validate_target_params(
    *,
    tile: int,
    cfu_word_bits: int,
    cfu_store_depth_words: int,
) -> None:
    if tile <= 0:
        raise ValueError("tile must be positive")
    if cfu_word_bits % 8 != 0:
        raise ValueError("cfu_word_bits must be a multiple of 8")
    if tile != cfu_word_bits // 8:
        raise ValueError(
            f"tile must match CFU line width: tile={tile}, cfu_word_bits={cfu_word_bits}"
        )
    if not _is_power_of_two(tile):
        raise ValueError("tile must be a power of two for RV32I-only CPU packing")
    compute_k_tile(
        cfu_word_bits=cfu_word_bits,
        cfu_store_depth_words=cfu_store_depth_words,
    )


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _validate_supported_quantized_chain(
    module: ModuleOp,
    candidates: tuple[GemmCandidateOp, ...],
) -> None:
    if not candidates:
        raise ValueError("no supported tosa.matmul -> tosa.add -> tosa.rescale regions found")

    matched_matmuls = {candidate.accumulator.owner for candidate in candidates}
    for op in module.walk():
        if isinstance(op, MatMulOp) and op not in matched_matmuls:
            raise ValueError(f"unsupported unmatched TOSA matmul in model: {op.name}")
        if isinstance(op, RescaleOp):
            raise ValueError("unsupported unmatched TOSA rescale in model")

    for previous, current in zip(candidates, candidates[1:]):
        current_matmul = current.accumulator.owner
        if not isinstance(current_matmul, MatMulOp):
            raise ValueError("loom.gemm_candidate accumulator must be produced by tosa.matmul")
        if current_matmul.a is not previous.output:
            raise ValueError(
                "unsupported non-linear quantized graph: GEMM inputs must form one chain"
            )

    returns = [op for op in module.walk() if isinstance(op, ReturnOp)]
    if len(returns) != 1 or len(returns[0].operands) != 1:
        raise ValueError("supported quantized model must have exactly one returned tensor")
    if not _return_reaches_candidate(returns[0].operands[0], candidates[-1]):
        raise ValueError(
            "model return is not the final supported quantized GEMM output; "
            "only boundary dequantize/reshape scaffolding may be omitted"
        )


def _return_reaches_candidate(value: SSAValue, candidate: GemmCandidateOp) -> bool:
    allowed_boundary_ops = (
        AddOp,
        MulOp,
        tensor.ReshapeOp,
        tensor.CollapseShapeOp,
    )
    while True:
        owner = value.owner
        if owner is candidate:
            return value is candidate.output
        if isinstance(owner, allowed_boundary_ops):
            value = owner.operands[0]
            continue
        if isinstance(owner, CastOp):
            value = owner.operands[0]
            continue
        return False


def _functions_from_rewritten_loom(
    module: ModuleOp,
    gemm_functions: dict[str, RiscvFunction],
) -> Iterable[RiscvFunction]:
    for op in module.walk():
        if isinstance(op, Pack2DToTilesOp):
            yield RiscvFunction(
                name=op.function_name.data,
                code=emit_pack_2d_to_tiles_op(op),
            )
        elif isinstance(op, PackConv2DToTilesOp):
            yield RiscvFunction(
                name=op.function_name.data,
                code=emit_pack_conv2d_to_tiles_op(op),
            )
        elif isinstance(op, GemmOp):
            yield gemm_functions[op.kernel_name.data]


def _validate_executable_model(compiled: CompiledModel) -> None:
    expected_functions = []
    for index, kernel in enumerate(compiled.kernels):
        # CPU kernels read their input directly; fused kernels get it packed
        # by their producer's result store.
        if not kernel.kind.endswith("_cpu") and kernel.packed_by is None:
            expected_functions.append(_buffer_name(index, "pack_input"))
        expected_functions.append(kernel.name)
    names = [fn.name for fn in compiled.functions]
    # Optional "<gemmN>.lut_map" pre-pass must directly precede "<gemmN>.pack_input".
    for i, name in enumerate(names):
        if name.endswith(".lut_map"):
            want = name.removesuffix(".lut_map") + ".pack_input"
            if i + 1 >= len(names) or names[i + 1] != want:
                raise ValueError(f"{name} must directly precede {want}")
    actual_functions = [n for n in names if not n.endswith(".lut_map")]
    if actual_functions != expected_functions:
        raise ValueError(
            "model schedule functions do not match planned kernels: "
            f"expected {expected_functions}, got {actual_functions}"
        )

    seen_names: set[str] = set()
    for function in compiled.functions:
        if function.name in seen_names:
            raise ValueError(f"duplicate RISC-V function: {function.name}")
        seen_names.add(function.name)
        if not function.code:
            raise ValueError(f"RISC-V function {function.name} has no code")
        if len(function.code) % 4 != 0:
            raise ValueError(f"RISC-V function {function.name} is not 4-byte aligned")

    for kernel in compiled.kernels:
        if not kernel.riscv_code:
            raise ValueError(f"GEMM kernel {kernel.name} has no executable code")
        if len(kernel.riscv_code) % 4 != 0:
            raise ValueError(f"GEMM kernel {kernel.name} code is not 4-byte aligned")

    for buf in compiled.buffers:
        if not _range_valid(buf.addr, buf.size):
            raise ValueError(
                f"planned buffer {buf.name} is outside runtime RAM: "
                f"addr=0x{buf.addr:x} size={buf.size}"
            )

    blob_len = len(compiled.to_bytes(include_debug_sections=False))
    model_addr = compiled.recommended_model_addr()
    if not _range_valid(model_addr, blob_len):
        raise ValueError(
            f"model blob does not fit in runtime RAM: "
            f"addr=0x{model_addr:x} len={blob_len} end=0x{MEM_END:x}"
        )


def _range_valid(addr: int, length: int) -> bool:
    return MEM_START <= addr < MEM_END and addr + length <= MEM_END


def _load_csr_map(path: str | Path | None) -> CsrMap | None:
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    return CsrMap.from_json_dict(json.loads(path.read_text()))


def _lower_gemm_candidate(candidate: GemmCandidateOp, *, batch_size: int) -> _LoweredGemm:
    matmul = candidate.accumulator.owner
    if not isinstance(matmul, MatMulOp):
        raise ValueError("loom.gemm_candidate accumulator must be produced by tosa.matmul")
    bias_value = candidate.bias
    name = candidate.kernel_name.data

    a_shape = _shape(matmul.a.type)
    b_shape = _shape(matmul.b.type)
    if len(a_shape) != 3 or len(b_shape) != 3 or a_shape[0] != 1 or b_shape[0] != 1:
        raise ValueError(f"{name}: expected rank-3 TOSA matmul operands")

    m = batch_size if a_shape[1] < 0 else a_shape[1]
    k = a_shape[2]
    if k != b_shape[1]:
        raise ValueError(f"{name}: K mismatch between activation and weight")
    n = b_shape[2]

    weight = _const_array(matmul.b, np.int8).reshape(k, n)
    a_zp = int(_const_scalar(matmul.a_zp))
    b_zp = int(_const_scalar(matmul.b_zp))
    if b_zp != 0:
        raise ValueError(f"{name}: non-zero weight zero-point needs a row-sum correction kernel")

    raw_bias = _const_array(bias_value, np.int32).reshape(-1)
    if raw_bias.size != n:
        raise ValueError(f"{name}: bias length {raw_bias.size} does not match N={n}")
    bias = raw_bias - a_zp * weight.astype(np.int32).sum(axis=0)

    multiplier = _const_array(candidate.multiplier, np.int32).reshape(-1)
    shift = _const_array(candidate.shift, np.int32).reshape(-1)
    multiplier = _broadcast_param(multiplier, n, f"{name}: multiplier")
    shift = _broadcast_param(shift, n, f"{name}: shift")
    # TOSA's scale32 shift counts the multiplier's 31 fractional bits. Loom's
    # epilogue does that high-multiply (SRDHM) first, so RDBPOT only gets the
    # residual right shift. Same conversion as the MobileNet frontend.
    shift = shift - 31
    if np.any(shift < 0):
        raise ValueError(f"{name}: requantization scales >= 1 are unsupported")

    input_zp = int(_const_scalar(candidate.input_zp))
    if input_zp != 0:
        raise ValueError(f"{name}: rescale input_zp must be zero")

    return _LoweredGemm(
        name=name,
        input=matmul.a,
        output=candidate.output,
        m=m,
        k=k,
        n=n,
        weight=weight,
        bias=bias.astype(np.int32),
        multiplier=multiplier.astype(np.int32),
        shift=shift.astype(np.int32),
        output_zp=int(_const_scalar(candidate.output_zp)),
    )


def _shape(typ) -> tuple[int, ...]:
    if not isinstance(typ, TensorType):
        raise ValueError(f"expected tensor type, got {typ}")
    return tuple(int(dim) for dim in typ.get_shape())


def _runtime_tensor_manifest(tensor: RuntimeTensor) -> dict:
    result = {
        "buffer": tensor.buffer,
        "dtype": "int8",
        "shape": list(tensor.shape),
        "padded_shape": list(tensor.padded_shape),
    }
    if tensor.scale is not None:
        result["scale"] = tensor.scale
    if tensor.zero_point is not None:
        result["zero_point"] = tensor.zero_point
    if tensor.layout is not None:
        result["layout"] = tensor.layout
    return result


def _const_array(value: SSAValue, dtype: np.dtype) -> np.ndarray:
    owner = value.owner
    if not isinstance(owner, ConstOp):
        raise ValueError(f"expected constant operand, got {owner.name}")
    return np.array(tuple(owner.values.get_values()), dtype=dtype)


def _const_scalar(value: SSAValue) -> int | float:
    data = _const_array(value, np.int64)
    if data.size != 1:
        raise ValueError("expected scalar constant")
    return data.item()


def _broadcast_param(values: np.ndarray, n: int, label: str) -> np.ndarray:
    if values.size == 1:
        return np.full(n, int(values[0]), dtype=np.int32)
    if values.size == n:
        return values.astype(np.int32)
    raise ValueError(f"{label} length {values.size} cannot broadcast to N={n}")


def _buffer_name(index: int, suffix: str) -> str:
    return f"gemm{index}.{suffix}"


def _emit_custom0_asm(gemm: _LoweredGemm, padded_m: int) -> str:
    lines = [
        f"# {gemm.name}: GEMM M={gemm.m} padded_M={padded_m} K={gemm.k} N={gemm.n}",
        "# Executable code is emitted from xDSL RISC-V target IR into SECTION_RISCV.",
        "# See SECTION_RISCV_IR for the structured loop/custom-op lowering trace.",
        "# No assembly template or hardcoded kernel body is used for execution.",
    ]
    return "\n".join(lines)


def _pack_sections(sections: tuple[tuple[int, bytes], ...]) -> bytes:
    header_size = 12
    entry_size = 12
    cursor = header_size + len(sections) * entry_size
    entries: list[tuple[int, int, int, int]] = []
    payload = bytearray()
    for section_type, data in sections:
        cursor = align_up(cursor, 16)
        pad = cursor - (header_size + len(sections) * entry_size + len(payload))
        payload.extend(b"\0" * pad)
        offset = cursor
        payload.extend(data)
        entries.append((section_type, 0, offset, len(data)))
        cursor += len(data)

    out = bytearray()
    out.extend(struct.pack("<4sHHI", MODEL_MAGIC, MODEL_VERSION, 0, len(sections)))
    for entry in entries:
        out.extend(struct.pack("<HHII", *entry))
    out.extend(payload)
    return bytes(out)


def _pack_constant_section(buffers: Iterable[Buffer]) -> bytes:
    buffers = tuple(buffers)
    out = bytearray(struct.pack("<I", len(buffers)))
    for buf in buffers:
        name = buf.name.encode()
        out.extend(struct.pack("<IIHH", buf.addr, len(buf.data), len(name), 0))
        out.extend(name)
        out.extend(b"\0" * ((4 - len(out) % 4) % 4))
        out.extend(buf.data)
        out.extend(b"\0" * ((4 - len(out) % 4) % 4))
    return bytes(out)


def _pack_riscv_section(functions: Iterable[RiscvFunction]) -> bytes:
    functions = tuple(functions)
    out = bytearray(struct.pack("<I", len(functions)))
    for function in functions:
        name = function.name.encode()
        out.extend(struct.pack("<IHH", len(function.code), len(name), 0))
        out.extend(name)
        out.extend(b"\0" * ((4 - len(out) % 4) % 4))
        out.extend(function.code)
        out.extend(b"\0" * ((4 - len(out) % 4) % 4))
    return bytes(out)


def _pack_schedule_section(functions: Iterable[RiscvFunction]) -> bytes:
    functions = tuple(functions)
    commands = [
        struct.pack("<BBHI", SCHED_CALL_RISCV, 0, 0, index)
        for index, _function in enumerate(functions)
    ]

    out = bytearray(struct.pack("<I", len(commands)))
    for command in commands:
        out.extend(command)
    return bytes(out)
