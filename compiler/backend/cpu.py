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
                channel_major=bool(op.channel_major.value.data),
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
    channel_major: bool = False,
) -> ModuleOp:
    """Build RV32I-friendly im2col directly into loom's packed tile layout.

    K order is tap-major (kk = tap * C + c) by default. `channel_major` uses
    kk = c * taps + tap instead, which makes each group of 8 output channels
    of a block-diagonal (depthwise / pooling) weight read one contiguous K
    slice, so the GEMM can skip the all-zero blocks.
    """

    if tile <= 0 or not _is_power_of_two(tile):
        raise ValueError("tile must be a positive power of two")
    if tile == 8:
        return _pack_conv2d_rowwords_module(
            src_addr=src_addr, dst_addr=dst_addr, transform_addr=transform_addr,
            input_h=input_h, input_w=input_w, channels=channels,
            output_h=output_h, output_w=output_w, kernel_h=kernel_h, kernel_w=kernel_w,
            stride_h=stride_h, stride_w=stride_w, pad_top=pad_top, pad_left=pad_left,
            input_zero_point=input_zero_point, tile=tile, nchw=nchw,
            channel_major=channel_major,
        )
    if channel_major:
        return _pack_conv2d_channel_major_module(
            src_addr=src_addr, dst_addr=dst_addr, transform_addr=transform_addr,
            input_h=input_h, input_w=input_w, channels=channels,
            output_h=output_h, output_w=output_w, kernel_h=kernel_h, kernel_w=kernel_w,
            stride_h=stride_h, stride_w=stride_w, pad_top=pad_top, pad_left=pad_left,
            input_zero_point=input_zero_point, tile=tile, nchw=nchw,
        )

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
    if channel_major:
        kk0 = arith.AddiOp(kernel_pos.result, c0.result)  # kk = c * taps + tap
    else:
        kk0 = arith.MuliOp(kernel_pos.result, channels_const.result)  # kk = tap * C + c
    dst_kk = arith.MuliOp(kk0.result, tile_const.result)
    dst_cstride = _index_const(
        (kernel_h * kernel_w * tile) if channel_major else tile
    )
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
    dst_c = arith.MuliOp(channel, dst_cstride.result)
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
    fdst_c = arith.MuliOp(fchannel, dst_cstride.result)
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
        kernel_pos0, kernel_pos, kk0, dst_kk, dst_cstride, dst_tap, *src_tap_ops, zero_point,
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


def _pack_conv2d_channel_major_module(
    *,
    src_addr: int,
    dst_addr: int,
    transform_addr: int,
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
    """im2col with channel-major K (kk = c * taps + tap), taps unrolled.

    Loop nest m_tile -> row -> channel, with the kernel taps unrolled at
    compile time: per row, each tap's source offset and in-bounds flag are
    computed once; per channel, the taps write consecutive K rows (8 bytes
    apart), keeping the same store locality as the tap-major layout.
    Padding taps select the zero point; their (possibly out-of-range) source
    address is still read, which is harmless in this SoC's flat RAM.
    """
    taps = kernel_h * kernel_w
    k_total = taps * channels
    src0 = llvm.IntToPtrOp(_i32_const(src_addr).result)
    src_i = src0.operands[0].owner
    dst_i = _i32_const(dst_addr)
    dst0 = llvm.IntToPtrOp(dst_i.result)
    transform_i = _i32_const(transform_addr)
    transform0 = llvm.IntToPtrOp(transform_i.result)
    c0 = _index_const(0)
    c1 = _index_const(1)
    consts = {
        "m": _index_const(output_h * output_w),
        "tile": _index_const(tile),
        "ow": _index_const(output_w),
        "ih": _index_const(input_h),
        "iw": _index_const(input_w),
        "c": _index_const(channels),
        "sh": _index_const(stride_h),
        "sw": _index_const(stride_w),
        "pt": _index_const(pad_top),
        "pl": _index_const(pad_left),
        "dst_tile": _index_const(k_total * tile),
        "dst_c": _index_const(taps * tile),
        "src_c": _index_const(input_h * input_w if nchw else 1),
    }
    zero_point = _i8_const(input_zero_point)
    c128 = _i32_const(128)

    m_block = Block(arg_types=[IndexType()])
    row_block = Block(arg_types=[IndexType()])
    ch_block = Block(arg_types=[IndexType()])
    m_base, row, channel = m_block.args[0], row_block.args[0], ch_block.args[0]

    row_ops: list = []

    def emit(op):
        row_ops.append(op)
        return op

    out_index = emit(arith.AddiOp(m_base, row))
    row_ok = emit(arith.CmpiOp(out_index.result, consts["m"].result, "ult"))
    oh = emit(arith.DivUIOp(out_index.result, consts["ow"].result))
    ow = emit(arith.RemUIOp(out_index.result, consts["ow"].result))
    ih_b = emit(arith.SubiOp(emit(arith.MuliOp(oh.result, consts["sh"].result)).result, consts["pt"].result))
    iw_b = emit(arith.SubiOp(emit(arith.MuliOp(ow.result, consts["sw"].result)).result, consts["pl"].result))
    m_tile = emit(arith.DivUIOp(m_base, consts["tile"].result))
    dst_row = emit(arith.AddiOp(emit(arith.MuliOp(m_tile.result, consts["dst_tile"].result)).result, row))

    tap_src: list = []
    tap_ok: list = []
    for ky in range(kernel_h):
        for kx in range(kernel_w):
            ih = emit(arith.AddiOp(ih_b.result, _emit_index(emit, ky)))
            iw = emit(arith.AddiOp(iw_b.result, _emit_index(emit, kx)))
            ok_h = emit(arith.CmpiOp(ih.result, consts["ih"].result, "ult"))
            ok_w = emit(arith.CmpiOp(iw.result, consts["iw"].result, "ult"))
            ok = emit(arith.AndIOp(row_ok.result, emit(arith.AndIOp(ok_h.result, ok_w.result)).result))
            pix = emit(arith.AddiOp(emit(arith.MuliOp(ih.result, consts["iw"].result)).result, iw.result))
            base = pix if nchw else emit(arith.MuliOp(pix.result, consts["c"].result))
            tap_src.append(base)
            tap_ok.append(ok)

    ch_ops: list = []
    src_c = arith.MuliOp(channel, consts["src_c"].result)
    dst_c0 = arith.MuliOp(channel, consts["dst_c"].result)
    dst_c = arith.AddiOp(dst_row.result, dst_c0.result)
    ch_ops += [src_c, dst_c0, dst_c]
    for t in range(taps):
        src_lin = arith.AddiOp(tap_src[t].result, src_c.result)
        src_off = arith.IndexCastOp(src_lin.result, i32)
        src_ptr = llvm.GEPOp(src0.output, [llvm.GEP_USE_SSA_VAL], i8, ssa_indices=[src_off.result])
        loaded = llvm.LoadOp(src_ptr.result, i8)
        ch_ops += [src_lin, src_off, src_ptr, loaded]
        value = loaded.results[0]
        if transform_addr:
            ext = arith.ExtSIOp(value, i32)
            idx = arith.AddiOp(ext.result, c128.result)
            lut_ptr = llvm.GEPOp(transform0.output, [llvm.GEP_USE_SSA_VAL], i8, ssa_indices=[idx.result])
            mapped = llvm.LoadOp(lut_ptr.result, i8)
            ch_ops += [ext, idx, lut_ptr, mapped]
            value = mapped.results[0]
        chosen = arith.SelectOp(tap_ok[t].result, value, zero_point.result)
        dst_off0 = arith.IndexCastOp(dst_c.result, i32)
        dst_off = arith.AddiOp(dst_off0.result, _i32_const_op(ch_ops, t * tile).result)
        dst_ptr = llvm.GEPOp(dst0.output, [llvm.GEP_USE_SSA_VAL], i8, ssa_indices=[dst_off.result])
        ch_ops += [chosen, dst_off0, dst_off, dst_ptr, llvm.StoreOp(chosen.result, dst_ptr.result)]
    ch_block.add_ops([*ch_ops, scf.YieldOp()])

    ch_loop = scf.ForOp(c0.result, consts["c"].result, c1.result, [], Region(ch_block))
    row_block.add_ops([*row_ops, ch_loop, scf.YieldOp()])
    row_loop = scf.ForOp(c0.result, consts["tile"].result, c1.result, [], Region(row_block))
    m_block.add_ops([row_loop, scf.YieldOp()])
    m_loop = scf.ForOp(c0.result, consts["m"].result, consts["tile"].result, [], Region(m_block))

    pack_func = func.FuncOp("loom_pack_conv2d_to_tiles", ([], []))
    pack_func.body.block.add_ops(
        [src_i, src0, dst_i, dst0, transform_i, transform0, c0, c1, *consts.values(),
         zero_point, c128, m_loop, func.ReturnOp()]
    )
    return ModuleOp([pack_func])


def _pack_conv2d_rowwords_module(
    *,
    src_addr: int,
    dst_addr: int,
    transform_addr: int,
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
    channel_major: bool,
) -> ModuleOp:
    """im2col that writes each (channel, tap) row group as two 32-bit words.

    In loom's packed layout the `tile` rows of one K entry are contiguous
    bytes, in both tap-major and channel-major K order. Loop m_tile -> tap
    (unrolled) -> channel, with the 8 rows unrolled: gather 8 source bytes,
    apply the optional requant LUT, select the zero point for padding, pack
    into two words, and store with `sw`. The VexRiscv dcache is write-through,
    so this cuts SDRAM write transactions 4x versus byte stores. Per (tap,
    row) source offsets and in-bounds flags are computed once per m_tile.
    """
    assert tile == 8
    taps = kernel_h * kernel_w
    k_total = taps * channels
    src_i = _i32_const(src_addr)
    src0 = llvm.IntToPtrOp(src_i.result)
    dst_i = _i32_const(dst_addr)
    dst0 = llvm.IntToPtrOp(dst_i.result)
    transform_i = _i32_const(transform_addr)
    transform0 = llvm.IntToPtrOp(transform_i.result)
    c0 = _index_const(0)
    c1 = _index_const(1)
    k = {
        "m": _index_const(output_h * output_w),
        "tile": _index_const(tile),
        "ow": _index_const(output_w),
        "ih": _index_const(input_h),
        "iw": _index_const(input_w),
        "c": _index_const(channels),
        "sh": _index_const(stride_h),
        "sw": _index_const(stride_w),
        "pt": _index_const(pad_top),
        "pl": _index_const(pad_left),
        "dst_tile": _index_const(k_total * tile),
        # K-entry stride between consecutive channels, in bytes.
        "dst_c": _index_const((taps if channel_major else 1) * tile),
        "src_c": _index_const(input_h * input_w if nchw else 1),
    }
    zero_point = _i8_const(input_zero_point)
    c128 = _i32_const(128)
    shifts = {r: _i32_const(8 * (r % 4)) for r in range(1, 8) if r % 4}

    m_block = Block(arg_types=[IndexType()])
    m_base = m_block.args[0]
    m_ops: list = []

    def emit(op):
        m_ops.append(op)
        return op

    m_tile = emit(arith.DivUIOp(m_base, k["tile"].result))
    dst_mt = emit(arith.MuliOp(m_tile.result, k["dst_tile"].result))
    # Per row: output pixel -> input origin; shared by all taps.
    rows = []
    for r in range(tile):
        out_index = emit(arith.AddiOp(m_base, _emit_index(emit, r)))
        row_ok = emit(arith.CmpiOp(out_index.result, k["m"].result, "ult"))
        oh = emit(arith.DivUIOp(out_index.result, k["ow"].result))
        ow = emit(arith.RemUIOp(out_index.result, k["ow"].result))
        ih_b = emit(arith.SubiOp(emit(arith.MuliOp(oh.result, k["sh"].result)).result, k["pt"].result))
        iw_b = emit(arith.SubiOp(emit(arith.MuliOp(ow.result, k["sw"].result)).result, k["pl"].result))
        rows.append((row_ok, ih_b, iw_b))

    # Tap loops (ky, kx) are real loops so the function stays well under the
    # 4 KiB direct-mapped icache; only the 8 rows are unrolled.
    kx_block = Block(arg_types=[IndexType()])
    ky_block = Block(arg_types=[IndexType()])
    ky, kx = ky_block.args[0], kx_block.args[0]
    tap_ops: list = []

    def temit(op):
        tap_ops.append(op)
        return op

    row_src, row_ok = [], []
    for ok_row, ih_b, iw_b in rows:
        ih = temit(arith.AddiOp(ih_b.result, ky))
        iw = temit(arith.AddiOp(iw_b.result, kx))
        ok = temit(arith.AndIOp(
            ok_row.result,
            temit(arith.AndIOp(
                temit(arith.CmpiOp(ih.result, k["ih"].result, "ult")).result,
                temit(arith.CmpiOp(iw.result, k["iw"].result, "ult")).result,
            )).result,
        ))
        pix = temit(arith.AddiOp(temit(arith.MuliOp(ih.result, k["iw"].result)).result, iw.result))
        base = pix if nchw else temit(arith.MuliOp(pix.result, k["c"].result))
        row_src.append(base)
        row_ok.append(ok)
    kw_const = temit(_index_const(kernel_w))
    tap = temit(arith.AddiOp(temit(arith.MuliOp(ky, kw_const.result)).result, kx))
    kk_scale = temit(_index_const((1 if channel_major else channels) * tile))
    dst_tap = temit(arith.AddiOp(dst_mt.result, temit(arith.MuliOp(tap.result, kk_scale.result)).result))

    ch_block = Block(arg_types=[IndexType()])
    channel = ch_block.args[0]
    ops: list = []
    src_c = arith.MuliOp(channel, k["src_c"].result)
    dst_c0 = arith.MuliOp(channel, k["dst_c"].result)
    dst_c = arith.AddiOp(dst_tap.result, dst_c0.result)
    dst_c32 = arith.IndexCastOp(dst_c.result, i32)
    ops += [src_c, dst_c0, dst_c, dst_c32]
    words = [None, None]
    for r in range(tile):
        lin = arith.AddiOp(row_src[r].result, src_c.result)
        off = arith.IndexCastOp(lin.result, i32)
        ptr = llvm.GEPOp(src0.output, [llvm.GEP_USE_SSA_VAL], i8, ssa_indices=[off.result])
        val = llvm.LoadOp(ptr.result, i8)
        ops += [lin, off, ptr, val]
        v = val.results[0]
        if transform_addr:
            ext = arith.ExtSIOp(v, i32)
            idx = arith.AddiOp(ext.result, c128.result)
            lptr = llvm.GEPOp(transform0.output, [llvm.GEP_USE_SSA_VAL], i8, ssa_indices=[idx.result])
            mapped = llvm.LoadOp(lptr.result, i8)
            ops += [ext, idx, lptr, mapped]
            v = mapped.results[0]
        chosen = arith.SelectOp(row_ok[r].result, v, zero_point.result)
        wide = arith.ExtUIOp(chosen.result, i32)
        ops += [chosen, wide]
        part = wide.result
        if r % 4:
            sh = arith.ShLIOp(part, shifts[r].result)
            ops.append(sh)
            part = sh.result
        w = r // 4
        if words[w] is None:
            words[w] = part
        else:
            acc = arith.OrIOp(words[w], part)
            ops.append(acc)
            words[w] = acc.result
    for w in range(2):
        if w:
            off = arith.AddiOp(dst_c32.result, _i32_const_op(ops, 4).result)
            ops.append(off)
        else:
            off = dst_c32
        ptr = llvm.GEPOp(dst0.output, [llvm.GEP_USE_SSA_VAL], i8, ssa_indices=[off.result])
        ops += [ptr, llvm.StoreOp(words[w], ptr.result, alignment=4)]
    ch_block.add_ops([*ops, scf.YieldOp()])
    ch_loop = scf.ForOp(c0.result, k["c"].result, c1.result, [], Region(ch_block))
    kx_block.add_ops([*tap_ops, ch_loop, scf.YieldOp()])
    kh_const = emit(_index_const(kernel_h))
    kwl_const = emit(_index_const(kernel_w))
    kx_loop = scf.ForOp(c0.result, kwl_const.result, c1.result, [], Region(kx_block))
    ky_block.add_ops([kx_loop, scf.YieldOp()])
    tap_loops = [scf.ForOp(c0.result, kh_const.result, c1.result, [], Region(ky_block))]

    m_block.add_ops([*m_ops, *tap_loops, scf.YieldOp()])
    m_loop = scf.ForOp(c0.result, k["m"].result, k["tile"].result, [], Region(m_block))
    pack_func = func.FuncOp("loom_pack_conv2d_to_tiles", ([], []))
    pack_func.body.block.add_ops(
        [src_i, src0, dst_i, dst0, transform_i, transform0, c0, c1, *k.values(),
         zero_point, c128, *shifts.values(), m_loop, func.ReturnOp()]
    )
    return ModuleOp([pack_func])


def _emit_index(emit, value: int):
    return emit(_index_const(value)).result


def _i32_const_op(ops: list, value: int) -> arith.ConstantOp:
    op = _i32_const(value)
    ops.append(op)
    return op


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
