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

from compiler.dialects.loom_cpu import Pack2DToTilesOp


REPO_ROOT = Path(__file__).resolve().parents[2]
LLVM_PROJECT_BIN = REPO_ROOT.parent / "llvm-project" / "build" / "bin"


def emit_pack_2d_to_tiles_op(op: Pack2DToTilesOp) -> bytes:
    """Lower a rewritten loom_cpu pack operation to RV32I code."""

    return lower_mlir_cpu_region_to_rv32i(
        op.function_name.data,
        str(pack_2d_to_tiles_module(
            src_addr=int(op.src_addr.value.data),
            dst_addr=int(op.dst_addr.value.data),
            m=int(op.m.value.data),
            k=int(op.k.value.data),
            tile=int(op.tile.value.data),
        )),
    )


def pack_2d_to_tiles_module(*, src_addr: int, dst_addr: int, m: int, k: int, tile: int) -> ModuleOp:
    """Build structured xDSL IR for the activation packing CPU region."""

    if tile <= 0:
        raise ValueError("tile must be positive")
    if not _is_power_of_two(tile):
        raise ValueError("tile must be a power of two for RV32I-only CPU packing")

    src_i = _i32_const(src_addr)
    dst_i = _i32_const(dst_addr)
    src0 = llvm.IntToPtrOp(src_i.result)
    dst0 = llvm.IntToPtrOp(dst_i.result)
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
    true_block.add_ops([src_off, src_ptr, loaded, scf.YieldOp(loaded.results[0])])

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
