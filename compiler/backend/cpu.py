"""CPU-region lowering through MLIR and LLVM command-line tools."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from xdsl.dialects import arith, func, llvm, scf
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, i32, i8, IndexType
from xdsl.ir import Block, Region

from compiler.dialects.loom_cpu import Pack2DToTilesOp, PackConv2DToTilesOp


REPO_ROOT = Path(__file__).resolve().parents[2]
LLVM_PROJECT_BIN = REPO_ROOT.parent / "llvm-project" / "build" / "bin"


def emit_pack_2d_to_tiles_op(op: Pack2DToTilesOp) -> bytes:
    """Lower a rewritten loom_cpu pack operation to RV32I code."""

    return lower_mlir_cpu_region_to_rv32i(
        op.function_name.data,
        str(pack_2d_to_tiles_module(
            src_addr=int(op.src_addr.value.data),
            dst_addr=int(op.dst_addr.value.data),
            transform_addr=int(op.transform_addr.value.data),
            m=int(op.m.value.data),
            k=int(op.k.value.data),
            tile=int(op.tile.value.data),
        )),
    )


def emit_pack_conv2d_to_tiles_op(op: PackConv2DToTilesOp) -> bytes:
    """Lower fused convolution im2col/packing to RV32I code."""

    return lower_mlir_cpu_region_to_rv32i(
        op.function_name.data,
        str(
            pack_conv2d_to_tiles_module(
                src_addr=int(op.src_addr.value.data),
                dst_addr=int(op.dst_addr.value.data),
                transform_addr=int(op.transform_addr.value.data),
                input_h=int(op.input_h.value.data),
                input_w=int(op.input_w.value.data),
                channels=int(op.channels.value.data),
                output_h=int(op.output_h.value.data),
                output_w=int(op.output_w.value.data),
                kernel_h=int(op.kernel_h.value.data),
                kernel_w=int(op.kernel_w.value.data),
                stride_h=int(op.stride_h.value.data),
                stride_w=int(op.stride_w.value.data),
                pad_top=int(op.pad_top.value.data),
                pad_left=int(op.pad_left.value.data),
                input_zero_point=int(op.input_zero_point.value.data),
                tile=int(op.tile.value.data),
                nchw=bool(op.nchw.value.data),
            )
        ),
    )


def pack_2d_to_tiles_module(
    *,
    src_addr: int,
    dst_addr: int,
    transform_addr: int = 0,
    m: int,
    k: int,
    tile: int,
) -> ModuleOp:
    """Build structured xDSL IR for the activation packing CPU region."""

    if tile <= 0:
        raise ValueError("tile must be positive")
    if not _is_power_of_two(tile):
        raise ValueError("tile must be a power of two for RV32I-only CPU packing")

    src_i = _i32_const(src_addr)
    dst_i = _i32_const(dst_addr)
    src0 = llvm.IntToPtrOp(src_i.result)
    dst0 = llvm.IntToPtrOp(dst_i.result)
    transform_i = _i32_const(transform_addr)
    transform0 = llvm.IntToPtrOp(transform_i.result)
    c0 = _index_const(0)
    c1 = _index_const(1)
    m_const = _index_const(m)
    k_const = _index_const(k)
    tile_const = _index_const(tile)

    row_block = Block(arg_types=[IndexType()])
    row = row_block.args[0]

    # These are filled after the parent loop blocks exist.
    k_block = Block(arg_types=[IndexType()])
    m_block = Block(arg_types=[IndexType()])
    m_base = m_block.args[0]
    kk = k_block.args[0]

    src_row = arith.AddiOp(m_base, row)
    in_bounds = arith.CmpiOp(src_row.result, m_const.result, "ult")
    src_linear0 = arith.MuliOp(src_row.result, k_const.result)
    src_linear = arith.AddiOp(src_linear0.result, kk)
    m_tile = arith.DivUIOp(m_base, tile_const.result)
    dst_tile0 = arith.MuliOp(m_tile.result, k_const.result)
    dst_linear0 = arith.AddiOp(dst_tile0.result, kk)
    dst_linear1 = arith.MuliOp(dst_linear0.result, tile_const.result)
    dst_linear = arith.AddiOp(dst_linear1.result, row)
    zero = _i8_const(0)

    true_block = Block()
    src_off = arith.IndexCastOp(src_linear.result, i32)
    src_ptr = llvm.GEPOp(src0.output, [llvm.GEP_USE_SSA_VAL], i8, ssa_indices=[src_off.result])
    loaded = llvm.LoadOp(src_ptr.result, i8)
    true_ops = [src_off, src_ptr, loaded]
    loaded_value = loaded.results[0]
    if transform_addr:
        c128 = _i32_const(128)
        extended = arith.ExtSIOp(loaded_value, i32)
        lut_index = arith.AddiOp(extended.result, c128.result)
        lut_ptr = llvm.GEPOp(
            transform0.output,
            [llvm.GEP_USE_SSA_VAL],
            i8,
            ssa_indices=[lut_index.result],
        )
        mapped = llvm.LoadOp(lut_ptr.result, i8)
        true_ops.extend([c128, extended, lut_index, lut_ptr, mapped])
        loaded_value = mapped.results[0]
    true_block.add_ops([*true_ops, scf.YieldOp(loaded_value)])

    false_block = Block()
    false_block.add_op(scf.YieldOp(zero.result))

    value_if = scf.IfOp(
        in_bounds.result,
        [i8],
        Region(true_block),
        Region(false_block),
    )
    dst_off = arith.IndexCastOp(dst_linear.result, i32)
    dst_ptr = llvm.GEPOp(dst0.output, [llvm.GEP_USE_SSA_VAL], i8, ssa_indices=[dst_off.result])
    llvm_store = llvm.StoreOp(value_if.results[0], dst_ptr.result)
    row_block.add_ops(
        [
            src_row,
            in_bounds,
            src_linear0,
            src_linear,
            m_tile,
            dst_tile0,
            dst_linear0,
            dst_linear1,
            dst_linear,
            zero,
            value_if,
            dst_off,
            dst_ptr,
            llvm_store,
            scf.YieldOp(),
        ]
    )

    row_loop = scf.ForOp(c0.result, tile_const.result, c1.result, [], Region(row_block))
    k_block.add_ops([row_loop, scf.YieldOp()])
    k_loop = scf.ForOp(c0.result, k_const.result, c1.result, [], Region(k_block))
    m_block.add_ops([k_loop, scf.YieldOp()])
    m_loop = scf.ForOp(c0.result, m_const.result, tile_const.result, [], Region(m_block))

    pack_func = func.FuncOp("loom_pack_2d_to_tiles", ([], []))
    pack_func.body.block.add_ops(
        [
            src_i,
            dst_i,
            src0,
            dst0,
            transform_i,
            transform0,
            c0,
            c1,
            m_const,
            k_const,
            tile_const,
            m_loop,
            func.ReturnOp(),
        ]
    )
    return ModuleOp([pack_func])


def pack_conv2d_to_tiles_module(
    *,
    src_addr: int,
    dst_addr: int,
    transform_addr: int = 0,
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
) -> ModuleOp:
    """Build RV32I-friendly im2col directly into loom's packed tile layout."""

    if tile <= 0 or not _is_power_of_two(tile):
        raise ValueError("tile must be a positive power of two")

    src_i = _i32_const(src_addr)
    dst_i = _i32_const(dst_addr)
    src0 = llvm.IntToPtrOp(src_i.result)
    dst0 = llvm.IntToPtrOp(dst_i.result)
    transform_i = _i32_const(transform_addr)
    transform0 = llvm.IntToPtrOp(transform_i.result)
    c0 = _index_const(0)
    c1 = _index_const(1)
    m_const = _index_const(output_h * output_w)
    k_const = _index_const(kernel_h * kernel_w * channels)
    tile_const = _index_const(tile)
    output_w_const = _index_const(output_w)
    kernel_h_const = _index_const(kernel_h)
    kernel_w_const = _index_const(kernel_w)
    channels_const = _index_const(channels)
    input_h_const = _index_const(input_h)
    input_w_const = _index_const(input_w)
    stride_h_const = _index_const(stride_h)
    stride_w_const = _index_const(stride_w)
    pad_top_const = _index_const(pad_top)
    pad_left_const = _index_const(pad_left)

    # Loop nest: m_tile -> row -> ky -> kx -> channel. Bounds checks and the
    # source/destination bases depend only on (row, ky, kx), so they are
    # computed once per kernel tap and the innermost channel loop is a plain
    # strided copy (or zero-point fill for padding taps).
    m_block = Block(arg_types=[IndexType()])
    row_block = Block(arg_types=[IndexType()])
    ky_block = Block(arg_types=[IndexType()])
    kx_block = Block(arg_types=[IndexType()])
    copy_block = Block(arg_types=[IndexType()])
    fill_block = Block(arg_types=[IndexType()])
    m_base = m_block.args[0]
    row = row_block.args[0]
    ky = ky_block.args[0]
    kx = kx_block.args[0]

    # Per (m_base, row): output pixel and destination tile base.
    out_index = arith.AddiOp(m_base, row)
    row_in_bounds = arith.CmpiOp(out_index.result, m_const.result, "ult")
    oh = arith.DivUIOp(out_index.result, output_w_const.result)
    ow = arith.RemUIOp(out_index.result, output_w_const.result)
    ih_base0 = arith.MuliOp(oh.result, stride_h_const.result)
    ih_base = arith.SubiOp(ih_base0.result, pad_top_const.result)
    iw_base0 = arith.MuliOp(ow.result, stride_w_const.result)
    iw_base = arith.SubiOp(iw_base0.result, pad_left_const.result)
    m_tile = arith.DivUIOp(m_base, tile_const.result)
    dst_tile0 = arith.MuliOp(m_tile.result, k_const.result)
    dst_tile1 = arith.MuliOp(dst_tile0.result, tile_const.result)
    dst_row = arith.AddiOp(dst_tile1.result, row)
    row_ops = [out_index, row_in_bounds, oh, ow, ih_base0, ih_base, iw_base0, iw_base,
               m_tile, dst_tile0, dst_tile1, dst_row]

    # Per kernel tap (ky, kx): bounds, source base, channel stride, dst base.
    ih = arith.AddiOp(ih_base.result, ky)
    iw = arith.AddiOp(iw_base.result, kx)
    ih_in_bounds = arith.CmpiOp(ih.result, input_h_const.result, "ult")
    iw_in_bounds = arith.CmpiOp(iw.result, input_w_const.result, "ult")
    spatial_in_bounds = arith.AndIOp(ih_in_bounds.result, iw_in_bounds.result)
    in_bounds = arith.AndIOp(row_in_bounds.result, spatial_in_bounds.result)
    kernel_pos0 = arith.MuliOp(ky, kernel_w_const.result)
    kernel_pos = arith.AddiOp(kernel_pos0.result, kx)
    kk0 = arith.MuliOp(kernel_pos.result, channels_const.result)
    dst_kk = arith.MuliOp(kk0.result, tile_const.result)
    dst_tap = arith.AddiOp(dst_row.result, dst_kk.result)
    if nchw:
        src_tap0 = arith.MuliOp(ih.result, input_w_const.result)
        src_tap = arith.AddiOp(src_tap0.result, iw.result)
        plane = arith.MuliOp(input_h_const.result, input_w_const.result)
        src_cstride = plane
        src_tap_ops = [src_tap0, src_tap, plane]
    else:
        src_tap0 = arith.MuliOp(ih.result, input_w_const.result)
        src_tap1 = arith.AddiOp(src_tap0.result, iw.result)
        src_tap = arith.MuliOp(src_tap1.result, channels_const.result)
        src_cstride = c1
        src_tap_ops = [src_tap0, src_tap1, src_tap]
    zero_point = _i8_const(input_zero_point)

    # Innermost: copy one tap's channels (with optional requant LUT).
    channel = copy_block.args[0]
    src_c = arith.MuliOp(channel, src_cstride.result)
    src_linear = arith.AddiOp(src_tap.result, src_c.result)
    dst_c = arith.MuliOp(channel, tile_const.result)
    dst_linear = arith.AddiOp(dst_tap.result, dst_c.result)
    src_off = arith.IndexCastOp(src_linear.result, i32)
    src_ptr = llvm.GEPOp(src0.output, [llvm.GEP_USE_SSA_VAL], i8, ssa_indices=[src_off.result])
    loaded = llvm.LoadOp(src_ptr.result, i8)
    copy_ops = [src_c, src_linear, dst_c, dst_linear, src_off, src_ptr, loaded]
    value = loaded.results[0]
    if transform_addr:
        c128 = _i32_const(128)
        extended = arith.ExtSIOp(value, i32)
        lut_index = arith.AddiOp(extended.result, c128.result)
        lut_ptr = llvm.GEPOp(
            transform0.output,
            [llvm.GEP_USE_SSA_VAL],
            i8,
            ssa_indices=[lut_index.result],
        )
        mapped = llvm.LoadOp(lut_ptr.result, i8)
        copy_ops.extend([c128, extended, lut_index, lut_ptr, mapped])
        value = mapped.results[0]
    dst_off = arith.IndexCastOp(dst_linear.result, i32)
    dst_ptr = llvm.GEPOp(dst0.output, [llvm.GEP_USE_SSA_VAL], i8, ssa_indices=[dst_off.result])
    copy_block.add_ops([*copy_ops, dst_off, dst_ptr, llvm.StoreOp(value, dst_ptr.result),
                        scf.YieldOp()])

    # Padding tap: fill the zero point.
    fchannel = fill_block.args[0]
    fdst_c = arith.MuliOp(fchannel, tile_const.result)
    fdst_linear = arith.AddiOp(dst_tap.result, fdst_c.result)
    fdst_off = arith.IndexCastOp(fdst_linear.result, i32)
    fdst_ptr = llvm.GEPOp(dst0.output, [llvm.GEP_USE_SSA_VAL], i8, ssa_indices=[fdst_off.result])
    fill_block.add_ops([fdst_c, fdst_linear, fdst_off, fdst_ptr,
                        llvm.StoreOp(zero_point.result, fdst_ptr.result), scf.YieldOp()])

    copy_loop_block = Block()
    copy_loop_block.add_ops([
        scf.ForOp(c0.result, channels_const.result, c1.result, [], Region(copy_block)),
        scf.YieldOp(),
    ])
    fill_loop_block = Block()
    fill_loop_block.add_ops([
        scf.ForOp(c0.result, channels_const.result, c1.result, [], Region(fill_block)),
        scf.YieldOp(),
    ])
    tap_if = scf.IfOp(in_bounds.result, [], Region(copy_loop_block), Region(fill_loop_block))
    kx_block.add_ops([
        ih, iw, ih_in_bounds, iw_in_bounds, spatial_in_bounds, in_bounds,
        kernel_pos0, kernel_pos, kk0, dst_kk, dst_tap, *src_tap_ops, zero_point,
        tap_if, scf.YieldOp(),
    ])
    kx_loop = scf.ForOp(c0.result, kernel_w_const.result, c1.result, [], Region(kx_block))
    ky_block.add_ops([kx_loop, scf.YieldOp()])
    ky_loop = scf.ForOp(c0.result, kernel_h_const.result, c1.result, [], Region(ky_block))
    row_block.add_ops([*row_ops, ky_loop, scf.YieldOp()])
    row_loop = scf.ForOp(c0.result, tile_const.result, c1.result, [], Region(row_block))
    m_block.add_ops([row_loop, scf.YieldOp()])
    m_loop = scf.ForOp(c0.result, m_const.result, tile_const.result, [], Region(m_block))

    pack_func = func.FuncOp("loom_pack_conv2d_to_tiles", ([], []))
    pack_func.body.block.add_ops(
        [
            src_i,
            dst_i,
            src0,
            dst0,
            transform_i,
            transform0,
            c0,
            c1,
            m_const,
            k_const,
            tile_const,
            output_w_const,
            kernel_h_const,
            kernel_w_const,
            channels_const,
            input_h_const,
            input_w_const,
            stride_h_const,
            stride_w_const,
            pad_top_const,
            pad_left_const,
            m_loop,
            func.ReturnOp(),
        ]
    )
    return ModuleOp([pack_func])


def _i32_const(value: int) -> arith.ConstantOp:
    return arith.ConstantOp(IntegerAttr(value, i32))


def _i8_const(value: int) -> arith.ConstantOp:
    return arith.ConstantOp(IntegerAttr(value, i8))


def _index_const(value: int) -> arith.ConstantOp:
    return arith.ConstantOp(IntegerAttr(value, IndexType()))


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def lower_mlir_cpu_region_to_rv32i(name: str, mlir_module: str) -> bytes:
    """Run MLIR -> LLVM IR -> RV32I object -> raw .text for a CPU region."""

    mlir_opt = _find_tool("mlir-opt", prefer_project=True)
    mlir_translate = _find_tool("mlir-translate", prefer_project=True)
    llc = _find_riscv_llc()
    objcopy = _find_tool("llvm-objcopy", prefer_project=False)
    readobj = _find_tool("llvm-readobj", prefer_project=False)

    with tempfile.TemporaryDirectory(prefix="loom-mlir-cpu-", dir="/tmp") as tmp:
        tmpdir = Path(tmp)
        src_mlir = tmpdir / f"{name}.mlir"
        lowered_mlir = tmpdir / f"{name}.llvm.mlir"
        llvm_ir = tmpdir / f"{name}.ll"
        obj = tmpdir / f"{name}.o"
        text = tmpdir / f"{name}.text"
        src_mlir.write_text(mlir_module)

        _run(
            [
                str(mlir_opt),
                str(src_mlir),
                "--lower-affine",
                "--convert-scf-to-cf",
                "--convert-cf-to-llvm=index-bitwidth=32",
                "--convert-arith-to-llvm=index-bitwidth=32",
                "--convert-index-to-llvm=index-bitwidth=32",
                "--convert-func-to-llvm=index-bitwidth=32",
                "--reconcile-unrealized-casts",
                "-o",
                str(lowered_mlir),
            ]
        )
        _run([str(mlir_translate), "--mlir-to-llvmir", str(lowered_mlir), "-o", str(llvm_ir)])
        _run(
            [
                str(llc),
                "-mtriple=riscv32",
                "-target-abi=ilp32",
                "-mattr=-m,-a,-f,-d,-c",
                "-filetype=obj",
                str(llvm_ir),
                "-o",
                str(obj),
            ]
        )
        relocations = _run_output([str(readobj), "--relocations", str(obj)])
        if ".rela.text" in relocations:
            raise RuntimeError(
                f"MLIR CPU lowering for {name} emitted unresolved RISC-V relocations; "
                "the model runtime loads raw .text and cannot resolve helper calls\n"
                f"{relocations.strip()}"
            )
        _run([str(objcopy), "-O", "binary", "-j", ".text", str(obj), str(text)])
        code = text.read_bytes()
    if not code or len(code) % 4 != 0:
        raise RuntimeError(f"MLIR CPU lowering produced invalid RV32I text for {name}: {len(code)} bytes")
    return code


def _find_tool(name: str, *, prefer_project: bool) -> Path:
    candidates: list[Path] = []
    env_bin = os.environ.get("LOOM_LLVM_BIN")
    if env_bin:
        candidates.append(Path(env_bin) / name)
    if prefer_project:
        candidates.append(LLVM_PROJECT_BIN / name)
    path_tool = shutil.which(name)
    if path_tool:
        candidates.append(Path(path_tool))
    if not prefer_project:
        candidates.append(LLVM_PROJECT_BIN / name)
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise FileNotFoundError(f"could not find LLVM tool {name}")


def _find_riscv_llc() -> Path:
    for candidate in _tool_candidates("llc"):
        result = subprocess.run(
            [str(candidate), "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
        if "riscv32" in result.stdout:
            return candidate
    raise RuntimeError("could not find an llc with the riscv32 target enabled")


def _tool_candidates(name: str) -> list[Path]:
    candidates: list[Path] = []
    env_bin = os.environ.get("LOOM_LLVM_BIN")
    if env_bin:
        candidates.append(Path(env_bin) / name)
    candidates.append(LLVM_PROJECT_BIN / name)
    path_tool = shutil.which(name)
    if path_tool:
        candidates.append(Path(path_tool))

    out: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.exists() or not os.access(candidate, os.X_OK):
            continue
        seen.add(candidate)
        out.append(candidate)
    return out


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{detail}")


def _run_output(cmd: list[str]) -> str:
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{detail}")
    return result.stdout
