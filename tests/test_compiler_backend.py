import struct

import numpy as np
import pytest

from xdsl.context import Context
from xdsl.dialects import affine, arith
from xdsl.dialects import riscv as xdsl_riscv
from xdsl.dialects import riscv_scf
from xdsl.dialects import rv32 as xdsl_rv32
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.func import ReturnOp

import compiler.backend.__main__ as backend_cli
from compiler.backend import (
    ModelBinaryArtifact,
    compile_onnx_model,
    compile_tosa_module,
    lower_tosa_to_affine,
    lower_tosa_to_loom,
    lower_tosa_to_riscv,
)
from compiler.backend.cpu import emit_pack_2d_to_tiles_op, pack_2d_to_tiles_module
from compiler.backend.loom_custom0 import (
    compute_start_fixed,
    compute_wait_fixed,
    encode_custom0_op,
    read_result_fixed,
    write_config_fixed,
    write_epilogue_param_fixed,
)
import compiler.backend.model as backend_model
from compiler.backend.model import (
    MODEL_MAGIC,
    SCHED_CALL_RISCV,
    SECTION_AFFINE_IR,
    SECTION_ASM,
    SECTION_CONSTANTS,
    SECTION_LOOM_IR,
    SECTION_MANIFEST,
    SECTION_RISCV,
    SECTION_RISCV_IR,
    SECTION_SCHEDULE,
)
from compiler.backend.riscv_ir import (
    LowerAffineKernelToRiscvScfPass,
    RiscvLoweringContext,
    lower_affine_kernel_to_riscv_scf,
)
from compiler.backend.riscv_ir_encode import encode_riscv_module
from compiler.backend.riscv import compute_start, compute_wait, encode_r_type, read_result, write_config, write_epilogue_param
from compiler.backend.target import CsrMap, GemmTargetSpec, DmaRegs
from compiler.backend.rewrites import (
    LowerGemmCandidatesToLoomPlanPass,
    LowerGemmToAffinePass,
    LowerRepackToCpuPackPass,
    LowerTosaGemmToCandidatesPass,
    PackFunctionBinding,
    PlannedGemmReplacement,
    lower_gemm_candidates_to_loom,
    lower_gemm_to_affine,
    lower_repack_to_cpu,
    lower_tosa_gemm_to_candidates,
)
from compiler.dialects.loom import (
    GemmOp,
    GemmCandidateOp,
    RepackOp,
    TileLoadActOp,
    TileLoadWgtOp,
    TileMmaOp,
    TileStoreOp,
)
from compiler.dialects.loom_cpu import Pack2DToTilesOp
from compiler.dialects.loom_riscv import (
    ComputeStartOp,
    ComputeWaitOp,
    ReadResultOp,
    WaitPreviousComputeOp,
    WriteConfigOp,
)
from compiler.frontend import emit_tosa_ir, load_model
from compiler.frontend.tosa_ext import MatMulOp


def test_compile_mnist_model_binary_extracts_gemm_kernels():
    compiled = compile_onnx_model("models/out/mnist_int8.onnx")

    assert len(compiled.kernels) == 2
    assert compiled.kernels[0].k == 784
    assert compiled.kernels[0].n == 256
    assert compiled.kernels[1].k == 256
    assert compiled.kernels[1].n == 10
    assert compiled.kernels[1].requires_repack_from == compiled.kernels[0].output_buffer

    blob = compiled.to_bytes()
    magic, version, _flags, section_count = struct.unpack_from("<4sHHI", blob)
    assert magic == MODEL_MAGIC
    assert version == 1
    assert section_count == 8
    assert b"loom_riscv.compute_start" in _sections(blob)[SECTION_RISCV_IR]

    for kernel in compiled.kernels:
        assert len(kernel.riscv_code) > 0
        assert "SECTION_RISCV" in kernel.riscv_asm
        assert ".insn r CUSTOM_0" not in kernel.riscv_asm
        assert ".word 0x" not in kernel.riscv_asm
    assert [fn.name for fn in compiled.functions] == [
        "gemm0.pack_input",
        compiled.kernels[0].name,
        "gemm1.pack_input",
        compiled.kernels[1].name,
    ]
    assert all(fn.code for fn in compiled.functions)
    _assert_cpu_pack_is_plain_rv32i(compiled.functions[0].code)
    assert _contains_opcode(compiled.functions[1].code, 0x0B)
    _assert_cpu_pack_is_plain_rv32i(compiled.functions[2].code)
    assert _contains_opcode(compiled.functions[3].code, 0x0B)

    loom_ops = tuple(compiled.loom_module.walk())
    assert sum(isinstance(op, GemmOp) for op in loom_ops) == 2
    assert sum(isinstance(op, RepackOp) for op in loom_ops) == 0
    assert sum(isinstance(op, Pack2DToTilesOp) for op in loom_ops) == 2
    assert any(isinstance(op, ComputeStartOp) for op in compiled.riscv_module.walk())


def test_lower_tosa_to_loom_uses_irdl_dialect_ops():
    module = emit_tosa_ir(load_model("models/out/mnist_int8.onnx"))
    loom_module = lower_tosa_to_loom(module)

    gemms = [op for op in loom_module.walk() if isinstance(op, GemmOp)]
    repacks = [op for op in loom_module.walk() if isinstance(op, RepackOp)]
    packs = [op for op in loom_module.walk() if isinstance(op, Pack2DToTilesOp)]
    assert len(gemms) == 2
    assert len(repacks) == 0
    assert len(packs) == 2
    assert packs[0].function_name.data == "gemm0.pack_input"
    assert int(gemms[0].k.value.data) == 784
    assert int(gemms[1].n.value.data) == 10


def test_tosa_gemm_region_recognition_is_pattern_rewrite():
    module = lower_tosa_gemm_to_candidates(emit_tosa_ir(load_model("models/out/mnist_int8.onnx")))

    candidates = [op for op in module.walk() if isinstance(op, GemmCandidateOp)]
    assert len(candidates) == 2
    assert candidates[0].kernel_name.data == "_net_net.1_Gemm_output_0_quantized"
    assert candidates[1].kernel_name.data == "output_quantized"
    assert all(isinstance(op.accumulator.owner, MatMulOp) for op in candidates)


def test_tosa_gemm_region_recognition_is_a_module_pass():
    module = emit_tosa_ir(load_model("models/out/mnist_int8.onnx"))

    LowerTosaGemmToCandidatesPass().apply(Context(), module)

    candidates = [op for op in module.walk() if isinstance(op, GemmCandidateOp)]
    assert [candidate.kernel_name.data for candidate in candidates] == [
        "_net_net.1_Gemm_output_0_quantized",
        "output_quantized",
    ]


def test_gemm_candidate_to_loom_plan_is_pattern_rewrite():
    module = lower_tosa_gemm_to_candidates(emit_tosa_ir(load_model("models/out/mnist_int8.onnx")))
    planned_names = []

    def plan(candidate: GemmCandidateOp) -> PlannedGemmReplacement:
        index = len(planned_names)
        planned_names.append(candidate.kernel_name.data)
        return PlannedGemmReplacement(
            RepackOp(f"src{index}", f"dst{index}", m=1, k=8, tile=8),
            GemmOp(
                candidate.kernel_name.data,
                m=1,
                k=8,
                n=8,
                padded_m=8,
                tile=8,
                input_addr=0x1000,
                weight_addr=0x2000,
                output_addr=0x3000,
                bias_addr=0x4000,
                multiplier_addr=0x5000,
                shift_addr=0x6000,
                output_offset=0,
            ),
        )

    lower_gemm_candidates_to_loom(module, plan)

    ops = tuple(module.walk())
    assert planned_names == ["_net_net.1_Gemm_output_0_quantized", "output_quantized"]
    assert sum(isinstance(op, RepackOp) for op in ops) == 2
    assert sum(isinstance(op, GemmOp) for op in ops) == 2
    assert sum(isinstance(op, GemmCandidateOp) for op in ops) == 2


def test_gemm_candidate_to_loom_plan_is_a_module_pass():
    module = lower_tosa_gemm_to_candidates(emit_tosa_ir(load_model("models/out/mnist_int8.onnx")))
    planned_names = []

    def plan(candidate: GemmCandidateOp) -> PlannedGemmReplacement:
        index = len(planned_names)
        planned_names.append(candidate.kernel_name.data)
        return PlannedGemmReplacement(
            RepackOp(f"src{index}", f"dst{index}", m=1, k=8, tile=8),
            GemmOp(
                candidate.kernel_name.data,
                m=1,
                k=8,
                n=8,
                padded_m=8,
                tile=8,
                input_addr=0x1000,
                weight_addr=0x2000,
                output_addr=0x3000,
                bias_addr=0x4000,
                multiplier_addr=0x5000,
                shift_addr=0x6000,
                output_offset=0,
            ),
        )

    LowerGemmCandidatesToLoomPlanPass(plan).apply(Context(), module)

    ops = tuple(module.walk())
    assert planned_names == ["_net_net.1_Gemm_output_0_quantized", "output_quantized"]
    assert sum(isinstance(op, RepackOp) for op in ops) == 2
    assert sum(isinstance(op, GemmOp) for op in ops) == 2


def test_quantized_model_legality_requires_linear_gemm_chain():
    module = lower_tosa_gemm_to_candidates(emit_tosa_ir(load_model("models/out/mnist_int8.onnx")))
    candidates = tuple(op for op in module.walk() if isinstance(op, GemmCandidateOp))

    with pytest.raises(ValueError, match="one chain"):
        backend_model._validate_supported_quantized_chain(module, tuple(reversed(candidates)))


def test_quantized_model_legality_requires_return_from_final_gemm():
    module = lower_tosa_gemm_to_candidates(emit_tosa_ir(load_model("models/out/mnist_int8.onnx")))
    candidates = tuple(op for op in module.walk() if isinstance(op, GemmCandidateOp))
    ret = next(op for op in module.walk() if isinstance(op, ReturnOp))

    assert not backend_model._return_reaches_candidate(ret.operands[0], candidates[0])
    assert backend_model._return_reaches_candidate(ret.operands[0], candidates[-1])


def test_repack_to_cpu_layout_is_pattern_rewrite():
    buffers = [
        backend_model.Buffer("src", 0x40010100, 16, "activation"),
        backend_model.Buffer("dst", 0x40010200, 32, "activation_packed"),
    ]
    module = lower_repack_to_cpu(
        ModuleOp([RepackOp("src", "dst", m=2, k=8, tile=8)]),
        buffers,
        [PackFunctionBinding("src", "dst", "pack0")],
    )

    ops = tuple(module.walk())
    assert not any(isinstance(op, RepackOp) for op in ops)
    pack = next(op for op in ops if isinstance(op, Pack2DToTilesOp))
    assert pack.function_name.data == "pack0"
    assert int(pack.src_addr.value.data) == 0x40010100
    assert int(pack.dst_addr.value.data) == 0x40010200
    assert int(pack.k.value.data) == 8


def test_repack_to_cpu_lowering_is_a_module_pass():
    buffers = (
        backend_model.Buffer("src", 0x40010100, 16, "activation"),
        backend_model.Buffer("dst", 0x40010200, 32, "activation_packed"),
    )
    bindings = (PackFunctionBinding("src", "dst", "pack0"),)
    module = ModuleOp([RepackOp("src", "dst", m=2, k=8, tile=8)])

    LowerRepackToCpuPackPass(buffers, bindings).apply(Context(), module)

    ops = tuple(module.walk())
    assert not any(isinstance(op, RepackOp) for op in ops)
    assert any(isinstance(op, Pack2DToTilesOp) for op in ops)


def test_cpu_pack_layout_uses_m_tile_index_for_multi_tile_batches():
    module = pack_2d_to_tiles_module(
        src_addr=0x40010100,
        dst_addr=0x40010200,
        m=16,
        k=4,
        tile=8,
    )

    assert any(isinstance(op, arith.DivUIOp) for op in module.walk())

    code = emit_pack_2d_to_tiles_op(
        Pack2DToTilesOp(
            "pack16",
            "src",
            "dst",
            src_addr=0x40010100,
            dst_addr=0x40010200,
            m=16,
            k=4,
            tile=8,
        )
    )
    assert code
    _assert_cpu_pack_is_plain_rv32i(code)


def test_planned_gemm_to_affine_schedule_is_pattern_rewrite():
    module = lower_gemm_to_affine(
        ModuleOp([
            GemmOp(
                "gemm0",
                m=1,
                k=272,
                n=10,
                padded_m=8,
                tile=8,
                input_addr=0x40010100,
                weight_addr=0x40011000,
                output_addr=0x40012000,
                bias_addr=0x40013000,
                multiplier_addr=0x40014000,
                shift_addr=0x40015000,
                output_offset=0,
            )
        ]),
        cfu_word_bits=64,
        cfu_store_depth_words=512,
    )

    ops = tuple(module.walk())
    assert not any(isinstance(op, GemmOp) for op in ops)
    loops = [op for op in ops if isinstance(op, affine.ForOp)]
    tile_ops = [op for op in ops if isinstance(op, TileMmaOp)]
    assert len(loops) >= 3
    assert [int(op.k_count.value.data) for op in tile_ops] == [256, 16, 256, 16]


def test_riscv_target_pass_rejects_affine_plan_missing_tail_region():
    module = lower_gemm_to_affine(
        ModuleOp([
            GemmOp(
                "gemm0",
                m=1,
                k=272,
                n=10,
                padded_m=8,
                tile=8,
                input_addr=0x40010100,
                weight_addr=0x40011000,
                output_addr=0x40012000,
                bias_addr=0x40013000,
                multiplier_addr=0x40014000,
                shift_addr=0x40015000,
                output_offset=0,
            )
        ]),
        cfu_word_bits=64,
        cfu_store_depth_words=512,
    )
    full_n_only = tuple(module.body.block.ops)[:1]

    spec = GemmTargetSpec(
        m=8,
        k=272,
        n=10,
        tile=8,
        input_addr=0x40010100,
        weight_addr=0x40011000,
        output_addr=0x40012000,
        bias_addr=0x40013000,
        mult_addr=0x40014000,
        shift_addr=0x40015000,
        output_offset=0,
        cfu_word_bits=64,
        cfu_store_depth_words=512,
    )

    with pytest.raises(ValueError, match="N regions do not cover"):
        lower_affine_kernel_to_riscv_scf(
            RiscvLoweringContext(spec=spec, csr=_fake_csr_map()),
            full_n_only,
        )


def test_riscv_target_pass_rejects_affine_plan_with_wrong_m_region():
    module = lower_gemm_to_affine(
        ModuleOp([
            GemmOp(
                "gemm0",
                m=1,
                k=272,
                n=8,
                padded_m=8,
                tile=8,
                input_addr=0x40010100,
                weight_addr=0x40011000,
                output_addr=0x40012000,
                bias_addr=0x40013000,
                multiplier_addr=0x40014000,
                shift_addr=0x40015000,
                output_offset=0,
            )
        ]),
        cfu_word_bits=64,
        cfu_store_depth_words=512,
    )
    spec = GemmTargetSpec(
        m=16,
        k=272,
        n=8,
        tile=8,
        input_addr=0x40010100,
        weight_addr=0x40011000,
        output_addr=0x40012000,
        bias_addr=0x40013000,
        mult_addr=0x40014000,
        shift_addr=0x40015000,
        output_offset=0,
        cfu_word_bits=64,
        cfu_store_depth_words=512,
    )

    with pytest.raises(ValueError, match="M region does not cover"):
        lower_affine_kernel_to_riscv_scf(
            RiscvLoweringContext(spec=spec, csr=_fake_csr_map()),
            tuple(module.body.block.ops),
        )


def test_lower_tosa_to_affine_splits_tiled_k_reduction():
    module = emit_tosa_ir(load_model("models/out/mnist_int8.onnx"))
    affine_module = lower_tosa_to_affine(module)

    loops = [op for op in affine_module.walk() if isinstance(op, affine.ForOp)]
    tile_ops = [op for op in affine_module.walk() if isinstance(op, TileMmaOp)]
    act_loads = [op for op in affine_module.walk() if isinstance(op, TileLoadActOp)]
    wgt_loads = [op for op in affine_module.walk() if isinstance(op, TileLoadWgtOp)]
    stores = [op for op in affine_module.walk() if isinstance(op, TileStoreOp)]

    assert len(loops) >= 6
    assert [int(op.k_count.value.data) for op in tile_ops[:2]] == [256, 16]
    assert len(act_loads) == len(wgt_loads) == len(tile_ops)
    assert [int(op.n_count.value.data) for op in stores] == [8, 8, 2]


def test_gemm_to_affine_lowering_is_a_module_pass():
    module = ModuleOp([
        GemmOp(
            "gemm0",
            m=1,
            k=272,
            n=10,
            padded_m=8,
            tile=8,
            input_addr=0x40010100,
            weight_addr=0x40011000,
            output_addr=0x40012000,
            bias_addr=0x40013000,
            multiplier_addr=0x40014000,
            shift_addr=0x40015000,
            output_offset=0,
        )
    ])

    LowerGemmToAffinePass(cfu_word_bits=64, cfu_store_depth_words=512).apply(Context(), module)

    ops = tuple(module.walk())
    assert not any(isinstance(op, GemmOp) for op in ops)
    assert any(isinstance(op, affine.ForOp) for op in ops)
    assert any(isinstance(op, TileMmaOp) for op in ops)


def test_lower_tosa_to_riscv_exposes_structured_target_ir():
    module = emit_tosa_ir(load_model("models/out/mnist_int8.onnx"))
    riscv_module = lower_tosa_to_riscv(module)

    ops = tuple(riscv_module.walk())
    assert any(isinstance(op, riscv_scf.ForOp) for op in ops)
    assert any(isinstance(op, riscv_scf.WhileOp) for op in ops)
    assert any(isinstance(op, ComputeStartOp) for op in ops)
    assert any(isinstance(op, WaitPreviousComputeOp) for op in ops)
    assert not any(isinstance(op, affine.ForOp) for op in ops)
    assert not any(isinstance(op, (TileLoadActOp, TileLoadWgtOp, TileMmaOp, TileStoreOp)) for op in ops)
    assert "loom_riscv.kernel" not in str(riscv_module)


def test_ir_only_lowerings_do_not_emit_riscv_code(monkeypatch):
    def fail_csr_load(*_args, **_kwargs):
        raise AssertionError("CSR loading should not run for IR-only lowering")

    def fail_riscv_encoder(*_args, **_kwargs):
        raise AssertionError("RISC-V encoder should not run for IR-only lowering")

    def fail_pack(*_args, **_kwargs):
        raise AssertionError("CPU pack emitter should not run for IR-only lowering")

    monkeypatch.setattr(backend_model, "_load_csr_map", fail_csr_load)
    monkeypatch.setattr(backend_model, "encode_riscv_module", fail_riscv_encoder)
    monkeypatch.setattr(backend_model, "emit_pack_2d_to_tiles_op", fail_pack)

    loom_module = lower_tosa_to_loom(emit_tosa_ir(load_model("models/out/mnist_int8.onnx")))
    affine_module = lower_tosa_to_affine(emit_tosa_ir(load_model("models/out/mnist_int8.onnx")))

    assert any(isinstance(op, Pack2DToTilesOp) for op in loom_module.walk())
    assert any(isinstance(op, GemmOp) for op in loom_module.walk())
    assert any(isinstance(op, affine.ForOp) for op in affine_module.walk())
    assert any(isinstance(op, TileMmaOp) for op in affine_module.walk())


def test_executable_compile_requires_csr_json(tmp_path):
    module = emit_tosa_ir(load_model("models/out/mnist_int8.onnx"))
    missing = tmp_path / "missing-csr.json"

    with pytest.raises(FileNotFoundError, match="csr.json is required"):
        compile_tosa_module(module, csr_json=missing)


def test_compile_encodes_structured_riscv_target_ir(monkeypatch):
    module = emit_tosa_ir(load_model("models/out/mnist_int8.onnx"))
    seen = []

    def fake_encode(module):
        ops = tuple(module.walk())
        seen.append(ops)
        assert any(isinstance(op, riscv_scf.ForOp) for op in ops)
        assert any(isinstance(op, ComputeStartOp) for op in ops)
        assert not any(isinstance(op, affine.ForOp) for op in ops)
        assert not any(isinstance(op, TileMmaOp) for op in ops)
        return b"\x13\x00\x00\x00"

    monkeypatch.setattr(backend_model, "encode_riscv_module", fake_encode)
    compiled = compile_tosa_module(module)

    assert len(seen) == 2
    assert compiled.kernels[0].riscv_code == b"\x13\x00\x00\x00"


def test_gemm_kernel_schedule_remains_affine_loop_roots():
    compiled = compile_onnx_model("models/out/mnist_int8.onnx")
    affine_roots = tuple(compiled.affine_module.body.block.ops)

    assert affine_roots
    assert all(isinstance(op, affine.ForOp) for op in affine_roots)
    assert "loom_riscv.kernel" not in str(compiled.affine_module)


def test_affine_loop_kernel_lowers_to_structured_riscv_target_ir():
    affine_module = lower_gemm_to_affine(
        ModuleOp([
            GemmOp(
                "gemm0",
                m=1,
                k=272,
                n=10,
                padded_m=8,
                tile=8,
                input_addr=0x40010100,
                weight_addr=0x40011000,
                output_addr=0x40012000,
                bias_addr=0x40013000,
                multiplier_addr=0x40014000,
                shift_addr=0x40015000,
                output_offset=0,
            )
        ]),
        cfu_word_bits=64,
        cfu_store_depth_words=512,
    )
    spec = GemmTargetSpec(
        m=8,
        k=272,
        n=10,
        tile=8,
        input_addr=0x40010100,
        weight_addr=0x40011000,
        output_addr=0x40012000,
        bias_addr=0x40013000,
        mult_addr=0x40014000,
        shift_addr=0x40015000,
        output_offset=0,
        cfu_word_bits=64,
        cfu_store_depth_words=512,
    )

    csr = _fake_csr_map()
    target = lower_affine_kernel_to_riscv_scf(
        RiscvLoweringContext(spec=spec, csr=csr),
        tuple(affine_module.body.block.ops),
    )
    ops = tuple(target.walk())
    immediates = [
        int(op.immediate.value.data)
        for op in ops
        if isinstance(op, xdsl_rv32.LiOp)
    ]

    assert any(isinstance(op, riscv_scf.ForOp) for op in ops)
    assert any(isinstance(op, riscv_scf.WhileOp) for op in ops)
    assert any(isinstance(op, ComputeStartOp) for op in ops)
    assert any(isinstance(op, ComputeWaitOp) for op in ops)
    assert any(isinstance(op, WaitPreviousComputeOp) for op in ops)
    assert any(isinstance(op, WriteConfigOp) for op in ops)
    assert any(isinstance(op, ReadResultOp) for op in ops)
    assert any(isinstance(op, xdsl_riscv.SwOp) for op in ops)
    assert any(isinstance(op, xdsl_riscv.SbOp) for op in ops)
    assert any(isinstance(op, xdsl_riscv.LwOp) for op in ops)
    assert not any(isinstance(op, xdsl_riscv.MulOp) for op in ops)
    for addr in (
        spec.input_addr,
        spec.weight_addr,
        spec.output_addr,
        spec.bias_addr,
        spec.mult_addr,
        spec.shift_addr,
        csr.act_dma.base,
        csr.act_dma.length,
        csr.act_dma.enable,
        csr.act_dma.done,
        csr.act_dma.loop,
        csr.wgt_dma.base,
        csr.wgt_dma.length,
        csr.wgt_dma.enable,
        csr.wgt_dma.done,
        csr.wgt_dma.loop,
    ):
        assert addr in immediates
    assert not any(isinstance(op, affine.ForOp) for op in ops)
    assert not any(isinstance(op, (TileLoadActOp, TileLoadWgtOp, TileMmaOp, TileStoreOp)) for op in ops)
    assert "loom_riscv.kernel" not in str(target)


def test_structured_k_loop_overlaps_next_dma_with_previous_compute():
    affine_module = lower_gemm_to_affine(
        ModuleOp([
            GemmOp(
                "gemm0",
                m=1,
                k=272,
                n=8,
                padded_m=8,
                tile=8,
                input_addr=0x40010100,
                weight_addr=0x40011000,
                output_addr=0x40012000,
                bias_addr=0x40013000,
                multiplier_addr=0x40014000,
                shift_addr=0x40015000,
                output_offset=0,
            )
        ]),
        cfu_word_bits=64,
        cfu_store_depth_words=512,
    )
    spec = GemmTargetSpec(
        m=8,
        k=272,
        n=8,
        tile=8,
        input_addr=0x40010100,
        weight_addr=0x40011000,
        output_addr=0x40012000,
        bias_addr=0x40013000,
        mult_addr=0x40014000,
        shift_addr=0x40015000,
        output_offset=0,
        cfu_word_bits=64,
        cfu_store_depth_words=512,
    )

    target = lower_affine_kernel_to_riscv_scf(
        RiscvLoweringContext(spec=spec, csr=_fake_csr_map()),
        tuple(affine_module.body.block.ops),
    )
    k_body = next(
        op.body.block
        for op in target.walk()
        if isinstance(op, riscv_scf.ForOp)
        and any(isinstance(nested, WaitPreviousComputeOp) for nested in op.body.block.ops)
    )
    body_ops = tuple(k_body.ops)
    wait_idx = next(i for i, op in enumerate(body_ops) if isinstance(op, WaitPreviousComputeOp))
    dma_wait_indices = [i for i, op in enumerate(body_ops) if isinstance(op, riscv_scf.WhileOp)]
    start_idx = next(i for i, op in enumerate(body_ops) if isinstance(op, ComputeStartOp))
    flag_idx = next(i for i, op in enumerate(body_ops) if isinstance(op, xdsl_riscv.SltiuOp))

    assert dma_wait_indices
    assert any(
        isinstance(op, xdsl_riscv.SwOp)
        and int(op.rs1.owner.immediate.value.data) == _fake_csr_map().act_dma.enable
        and int(op.rs2.owner.immediate.value.data) == 1
        for op in body_ops[:wait_idx]
    )
    assert any(
        isinstance(op, xdsl_riscv.SwOp)
        and int(op.rs1.owner.immediate.value.data) == _fake_csr_map().wgt_dma.enable
        and int(op.rs2.owner.immediate.value.data) == 1
        for op in body_ops[:wait_idx]
    )
    assert wait_idx < min(dma_wait_indices)
    assert max(dma_wait_indices) < flag_idx < start_idx
    post_wait_ops = body_ops[max(dma_wait_indices) + 1 : flag_idx]
    assert any(
        isinstance(op, xdsl_riscv.SwOp)
        and int(op.rs1.owner.immediate.value.data) == _fake_csr_map().act_dma.enable
        and int(op.rs2.owner.immediate.value.data) == 0
        for op in post_wait_ops
    )
    assert any(
        isinstance(op, xdsl_riscv.SwOp)
        and int(op.rs1.owner.immediate.value.data) == _fake_csr_map().wgt_dma.enable
        and int(op.rs2.owner.immediate.value.data) == 0
        for op in post_wait_ops
    )
    assert any(isinstance(op, xdsl_riscv.XorOp) for op in body_ops)
    assert not any(isinstance(op, xdsl_rv32.LiOp) and int(op.immediate.value.data) in (1, 2, 3) for op in body_ops[flag_idx:start_idx])


def test_affine_loop_kernel_lowering_is_a_module_pass():
    affine_module = lower_gemm_to_affine(
        ModuleOp([
            GemmOp(
                "gemm0",
                m=1,
                k=16,
                n=8,
                padded_m=8,
                tile=8,
                input_addr=0x40010100,
                weight_addr=0x40011000,
                output_addr=0x40012000,
                bias_addr=0x40013000,
                multiplier_addr=0x40014000,
                shift_addr=0x40015000,
                output_offset=0,
            )
        ]),
        cfu_word_bits=64,
        cfu_store_depth_words=512,
    )
    spec = GemmTargetSpec(
        m=8,
        k=16,
        n=8,
        tile=8,
        input_addr=0x40010100,
        weight_addr=0x40011000,
        output_addr=0x40012000,
        bias_addr=0x40013000,
        mult_addr=0x40014000,
        shift_addr=0x40015000,
        output_offset=0,
        cfu_word_bits=64,
        cfu_store_depth_words=512,
    )

    LowerAffineKernelToRiscvScfPass(
        RiscvLoweringContext(spec=spec, csr=_fake_csr_map())
    ).apply(Context(), affine_module)
    ops = tuple(affine_module.walk())

    assert any(isinstance(op, riscv_scf.ForOp) for op in ops)
    assert any(isinstance(op, ComputeStartOp) for op in ops)
    assert not any(isinstance(op, affine.ForOp) for op in ops)
    assert not any(isinstance(op, (TileLoadActOp, TileLoadWgtOp, TileMmaOp, TileStoreOp)) for op in ops)
    assert "loom_riscv.kernel" not in str(affine_module)


def test_structured_riscv_target_ir_encodes_to_rv32i_custom0_bytes():
    affine_module = lower_gemm_to_affine(
        ModuleOp([
            GemmOp(
                "gemm0",
                m=1,
                k=16,
                n=8,
                padded_m=8,
                tile=8,
                input_addr=0x40010100,
                weight_addr=0x40011000,
                output_addr=0x40012000,
                bias_addr=0x40013000,
                multiplier_addr=0x40014000,
                shift_addr=0x40015000,
                output_offset=0,
            )
        ]),
        cfu_word_bits=64,
        cfu_store_depth_words=512,
    )
    spec = GemmTargetSpec(
        m=8,
        k=16,
        n=8,
        tile=8,
        input_addr=0x40010100,
        weight_addr=0x40011000,
        output_addr=0x40012000,
        bias_addr=0x40013000,
        mult_addr=0x40014000,
        shift_addr=0x40015000,
        output_offset=0,
        cfu_word_bits=64,
        cfu_store_depth_words=512,
    )
    target = lower_affine_kernel_to_riscv_scf(
        RiscvLoweringContext(spec=spec, csr=_fake_csr_map()),
        tuple(affine_module.body.block.ops),
    )

    code = encode_riscv_module(target)

    assert code
    assert len(code) % 4 == 0
    assert _contains_opcode(code, 0x63)
    assert _contains_opcode(code, 0x03)
    assert _contains_opcode(code, 0x23)
    assert _contains_opcode(code, 0x0B)
    assert not _contains_rv32m(code)


def test_encoded_k_loop_preserves_wait_skip_and_dynamic_flags():
    affine_module = lower_gemm_to_affine(
        ModuleOp([
            GemmOp(
                "gemm0",
                m=1,
                k=272,
                n=8,
                padded_m=8,
                tile=8,
                input_addr=0x40010100,
                weight_addr=0x40011000,
                output_addr=0x40012000,
                bias_addr=0x40013000,
                multiplier_addr=0x40014000,
                shift_addr=0x40015000,
                output_offset=0,
            )
        ]),
        cfu_word_bits=64,
        cfu_store_depth_words=512,
    )
    spec = GemmTargetSpec(
        m=8,
        k=272,
        n=8,
        tile=8,
        input_addr=0x40010100,
        weight_addr=0x40011000,
        output_addr=0x40012000,
        bias_addr=0x40013000,
        mult_addr=0x40014000,
        shift_addr=0x40015000,
        output_offset=0,
        cfu_word_bits=64,
        cfu_store_depth_words=512,
    )
    target = lower_affine_kernel_to_riscv_scf(
        RiscvLoweringContext(spec=spec, csr=_fake_csr_map()),
        tuple(affine_module.body.block.ops),
    )

    words = _code_words(encode_riscv_module(target))

    assert any(
        _is_beq_to_next_plus_one(word)
        and next_word == compute_wait().encode()
        for word, next_word in zip(words, words[1:])
    )
    assert any(_is_riscv_xor(word) for word in words)
    assert any(_is_sltiu_imm_one(word) for word in words)
    assert not _contains_rv32m(b"".join(word.to_bytes(4, "little") for word in words))


def test_generated_riscv_kernels_are_looped_not_unrolled_blobs():
    compiled = compile_onnx_model("models/out/mnist_int8.onnx")

    code_sizes = [len(kernel.riscv_code) for kernel in compiled.kernels]
    assert code_sizes[0] < 2048
    assert code_sizes[1] < 2048

    for kernel in compiled.kernels:
        words = [
            struct.unpack_from("<I", kernel.riscv_code, offset)[0]
            for offset in range(0, len(kernel.riscv_code), 4)
        ]
        branch_count = sum(1 for word in words if (word & 0x7F) == 0x63)
        custom0_count = sum(1 for word in words if (word & 0x7F) == 0x0B)
        assert branch_count >= 10
        assert custom0_count >= 6


def test_model_binary_schedule_and_riscv_code_are_aligned():
    compiled = compile_onnx_model("models/out/mnist_int8.onnx")
    blob = compiled.to_bytes()
    sections = _sections(blob)

    riscv = sections[SECTION_RISCV]
    functions = _riscv_function_records(riscv)
    count = len(functions)
    assert count == 4
    for _name, code_len, code_offset in functions:
        assert code_offset % 4 == 0
        assert code_len > 0

    schedule = sections[SECTION_SCHEDULE]
    cmd_count = struct.unpack_from("<I", schedule)[0]
    assert cmd_count == 4
    calls = [struct.unpack_from("<BBHI", schedule, 4 + index * 8) for index in range(cmd_count)]
    assert calls == [
        (SCHED_CALL_RISCV, 0, 0, 0),
        (SCHED_CALL_RISCV, 0, 0, 1),
        (SCHED_CALL_RISCV, 0, 0, 2),
        (SCHED_CALL_RISCV, 0, 0, 3),
    ]

    manifest = sections[SECTION_MANIFEST]
    assert b"program_size" not in manifest
    assert b"riscv_code_size" in manifest
    assert b"riscv_functions" in manifest


def test_model_binary_deploy_mode_contains_only_executable_sections():
    compiled = compile_onnx_model("models/out/mnist_int8.onnx")

    debug_blob = compiled.to_bytes()
    deploy_blob = compiled.to_bytes(include_debug_sections=False)
    debug_sections = _sections(debug_blob)
    deploy_sections = _sections(deploy_blob)

    assert set(deploy_sections) == {
        SECTION_MANIFEST,
        SECTION_CONSTANTS,
        SECTION_RISCV,
        SECTION_SCHEDULE,
    }
    assert SECTION_ASM in debug_sections
    assert SECTION_LOOM_IR in debug_sections
    assert SECTION_AFFINE_IR in debug_sections
    assert SECTION_RISCV_IR in debug_sections
    assert len(deploy_blob) < len(debug_blob)

    artifact = ModelBinaryArtifact.from_bytes(deploy_blob)
    assert artifact.blob == deploy_blob
    assert artifact.manifest["entry"] == compiled.entry


def test_model_binary_schedule_calls_rewritten_loom_plan_order():
    compiled = compile_onnx_model("models/out/mnist_int8.onnx")
    blob = compiled.to_bytes()
    sections = _sections(blob)
    functions = _riscv_function_records(sections[SECTION_RISCV])
    schedule = sections[SECTION_SCHEDULE]
    calls = [
        struct.unpack_from("<BBHI", schedule, 4 + index * 8)
        for index in range(struct.unpack_from("<I", schedule)[0])
    ]

    planned_names = [
        op.function_name.data if isinstance(op, Pack2DToTilesOp) else op.kernel_name.data
        for op in compiled.loom_module.body.block.ops
        if isinstance(op, (Pack2DToTilesOp, GemmOp))
    ]
    scheduled_names = [
        functions[index][0]
        for op, _flags, _reserved, index in calls
        if op == SCHED_CALL_RISCV
    ]

    assert scheduled_names == planned_names == [fn.name for fn in compiled.functions]


def test_model_manifest_declares_quantized_runtime_boundary():
    compiled = compile_onnx_model("models/out/mnist_int8.onnx")
    manifest = compiled.manifest()

    assert manifest["runtime_contract"] == {
        "boundary": "quantized-int8",
        "input": {
            "buffer": compiled.kernels[0].input_buffer,
            "dtype": "int8",
            "shape": [compiled.kernels[0].m, compiled.kernels[0].k],
        },
        "output": {
            "buffer": compiled.kernels[-1].output_buffer,
            "dtype": "int8",
            "shape": [compiled.kernels[-1].m, compiled.kernels[-1].n],
            "padded_shape": [compiled.kernels[-1].padded_m, compiled.kernels[-1].n],
        },
        "omitted_boundary_ops": ["input_quantize", "output_dequantize"],
    }


def test_model_manifest_declares_deployment_load_address():
    compiled = compile_onnx_model("models/out/mnist_int8.onnx")
    manifest = compiled.manifest()
    deployment = manifest["deployment"]

    assert deployment == {
        "model_addr": compiled.recommended_model_addr(),
        "alignment": 32,
        "memory_end": backend_model.MEM_END,
    }
    assert deployment["model_addr"] % deployment["alignment"] == 0
    for buf in compiled.buffers:
        assert deployment["model_addr"] >= buf.addr + buf.size


def test_executable_model_validation_rejects_empty_function_code():
    kernel = backend_model.GemmKernel(
        name="gemm0",
        m=1,
        k=8,
        n=8,
        padded_m=8,
        tile=8,
        input_buffer="gemm0.input",
        packed_input_buffer="gemm0.input_packed",
        output_buffer="gemm0.output",
        riscv_code=b"\x13\x00\x00\x00",
        riscv_asm="",
    )
    compiled = backend_model.CompiledModel(
        buffers=(
            backend_model.Buffer("gemm0.input", backend_model.MEM_START + 0x100, 8, "input"),
            backend_model.Buffer("gemm0.input_packed", backend_model.MEM_START + 0x120, 64, "activation_packed"),
            backend_model.Buffer("gemm0.output", backend_model.MEM_START + 0x180, 64, "activation"),
        ),
        kernels=(kernel,),
        functions=(
            backend_model.RiscvFunction("gemm0.pack_input", b""),
            backend_model.RiscvFunction("gemm0", b"\x13\x00\x00\x00"),
        ),
        memory_base=backend_model.MEM_START + 0x100,
        entry="gemm0",
        loom_module=ModuleOp([]),
        affine_module=ModuleOp([]),
        riscv_module=ModuleOp([]),
    )

    with pytest.raises(ValueError, match="has no code"):
        backend_model._validate_executable_model(compiled)


def test_compile_rejects_memory_plan_outside_fpga_ram():
    with pytest.raises(ValueError, match="runtime RAM"):
        compile_onnx_model("models/out/mnist_int8.onnx", memory_base=backend_model.MEM_END - 16)


def test_compile_rejects_target_tile_width_mismatch():
    with pytest.raises(ValueError, match="tile must match CFU line width"):
        compile_onnx_model("models/out/mnist_int8.onnx", tile=8, cfu_word_bits=32)


def test_compile_rejects_non_power_of_two_tile_width():
    with pytest.raises(ValueError, match="power of two"):
        compile_onnx_model("models/out/mnist_int8.onnx", tile=12, cfu_word_bits=96)


def test_cpu_pack_rejects_non_power_of_two_tile_width():
    with pytest.raises(ValueError, match="power of two"):
        pack_2d_to_tiles_module(
            src_addr=0x40010100,
            dst_addr=0x40010200,
            m=12,
            k=4,
            tile=12,
        )


def test_affine_lowering_rejects_invalid_k_tile_target():
    module = ModuleOp([
        GemmOp(
            "gemm0",
            m=1,
            k=8,
            n=8,
            padded_m=8,
            tile=8,
            input_addr=0x40010100,
            weight_addr=0x40011000,
            output_addr=0x40012000,
            bias_addr=0x40013000,
            multiplier_addr=0x40014000,
            shift_addr=0x40015000,
            output_offset=0,
        )
    ])

    with pytest.raises(ValueError, match="too small"):
        lower_gemm_to_affine(
            module,
            cfu_word_bits=64,
            cfu_store_depth_words=1,
        )


def test_compiler_adjusts_bias_for_activation_zero_point():
    module = emit_tosa_ir(load_model("models/out/mnist_int8.onnx"))
    matmul = next(op for op in module.walk() if isinstance(op, MatMulOp))
    weight = np.array(tuple(matmul.b.owner.values.get_values()), dtype=np.int8).reshape(784, 256)

    compiled = compile_onnx_model("models/out/mnist_int8.onnx")
    bias_buf = next(buf for buf in compiled.buffers if buf.name == "gemm0.bias")
    lowered_bias = np.frombuffer(bias_buf.data, dtype="<i4")

    # The ONNX QGemm input zero-point is -128 and hardware accumulates raw
    # signed int8 products, so lowering folds -zp * sum(weight) into bias.
    raw_bias = np.array(tuple(next(
        op for op in module.walk()
        if op.results
        and getattr(op.results[0], "name_hint", None) == "net.1.bias_quantized_rank3"
    ).values.get_values()), dtype=np.int32).reshape(-1)
    np.testing.assert_array_equal(lowered_bias, raw_bias + 128 * weight.astype(np.int32).sum(axis=0))


def test_custom0_r_type_encoding_matches_riscv_layout():
    assert encode_r_type(opcode=0x0B, funct3=0, funct7=0, rd=0, rs1=10, rs2=11) == 0x00B5000B
    assert compute_start(flags_reg=10, k_count_reg=11).to_bytes() == bytes.fromhex("0b00b500")
    assert compute_wait().encode() == 0x0000100B


def test_loom_riscv_custom0_ops_encode_like_legacy_helpers():
    start = compute_start_fixed(flags_reg=10, k_count_reg=11)
    assert isinstance(start, ComputeStartOp)
    assert encode_custom0_op(start) == compute_start(flags_reg=10, k_count_reg=11).encode()
    assert encode_custom0_op(compute_wait_fixed()) == compute_wait().encode()
    assert encode_custom0_op(write_epilogue_param_fixed(1, channel_reg=12, value_reg=13)) == write_epilogue_param(1, channel_reg=12, value_reg=13).encode()
    assert encode_custom0_op(write_config_fixed(2, value_reg=13)) == write_config(2, value_reg=13).encode()
    assert encode_custom0_op(read_result_fixed(rd=6, index_reg=7)) == read_result(rd=6, index_reg=7).encode()


def test_compile_cli_forwards_target_parameters(monkeypatch, tmp_path, capsys):
    seen = {}
    output = tmp_path / "model.lmb"

    class FakeCompiled:
        def to_bytes(self, *, include_debug_sections: bool = True) -> bytes:
            seen["include_debug_sections"] = include_debug_sections
            return b"LMB"

        def manifest(self) -> dict:
            return {
                "ok": True,
                "deployment": {
                    "model_addr": 0x40050000,
                    "alignment": 32,
                    "memory_end": backend_model.MEM_END,
                },
            }

    def fake_compile(path, **kwargs):
        seen["path"] = path
        seen.update(kwargs)
        return FakeCompiled()

    monkeypatch.setattr(backend_cli, "compile_onnx_model", fake_compile)
    monkeypatch.setattr(
        "sys.argv",
        [
            "loom-compile-model",
            "model.onnx",
            "-o",
            str(output),
            "--batch-size",
            "2",
            "--memory-base",
            "0x40020000",
            "--csr-json",
            "build/sim/csr.json",
            "--tile",
            "4",
            "--cfu-word-bits",
            "32",
            "--cfu-store-depth-words",
            "128",
            "--manifest",
        ],
    )

    assert backend_cli.main() == 0
    assert output.read_bytes() == b"LMB"
    assert seen == {
        "path": "model.onnx",
        "batch_size": 2,
        "memory_base": 0x40020000,
        "tile": 4,
        "cfu_word_bits": 32,
        "cfu_store_depth_words": 128,
        "csr_json": "build/sim/csr.json",
        "include_debug_sections": False,
    }
    out = capsys.readouterr().out
    assert '"ok": true' in out
    assert '"model_addr": 1074069504' in out


def test_compile_cli_can_emit_debug_sections(monkeypatch, tmp_path):
    seen = {}
    output = tmp_path / "model.lmb"

    class FakeCompiled:
        def to_bytes(self, *, include_debug_sections: bool = True) -> bytes:
            seen["include_debug_sections"] = include_debug_sections
            return b"LMB"

        def manifest(self) -> dict:
            return {}

    monkeypatch.setattr(backend_cli, "compile_onnx_model", lambda _path, **_kwargs: FakeCompiled())
    monkeypatch.setattr(
        "sys.argv",
        [
            "loom-compile-model",
            "model.onnx",
            "-o",
            str(output),
            "--debug-sections",
        ],
    )

    assert backend_cli.main() == 0
    assert output.read_bytes() == b"LMB"
    assert seen == {"include_debug_sections": True}


def _sections(blob: bytes) -> dict[int, bytes]:
    magic, _version, _flags, section_count = struct.unpack_from("<4sHHI", blob)
    assert magic == MODEL_MAGIC
    out = {}
    cursor = 12
    for _ in range(section_count):
        typ, _flags, off, size = struct.unpack_from("<HHII", blob, cursor)
        cursor += 12
        out[typ] = blob[off: off + size]
    return out


def _riscv_function_records(section: bytes) -> list[tuple[str, int, int]]:
    count = struct.unpack_from("<I", section)[0]
    cursor = 4
    functions: list[tuple[str, int, int]] = []
    for _ in range(count):
        code_len, name_len, _flags = struct.unpack_from("<IHH", section, cursor)
        cursor += 8
        name = section[cursor: cursor + name_len].decode()
        cursor += name_len
        cursor = (cursor + 3) & ~3
        code_offset = cursor
        functions.append((name, code_len, code_offset))
        cursor += code_len
        cursor = (cursor + 3) & ~3
    assert cursor == len(section)
    return functions


def _contains_opcode(code: bytes, opcode: int) -> bool:
    return any(
        (struct.unpack_from("<I", code, offset)[0] & 0x7F) == opcode
        for offset in range(0, len(code), 4)
    )


def _assert_cpu_pack_is_plain_rv32i(code: bytes) -> None:
    allowed_opcodes = {0x03, 0x13, 0x23, 0x33, 0x37, 0x63, 0x67, 0x6F}
    words = _code_words(code)
    assert words
    for word in words:
        assert (word & 0x3) == 0x3
        opcode = word & 0x7F
        assert opcode in allowed_opcodes
        assert opcode not in {0x0B, 0x2F, 0x73}
        assert opcode not in {0x07, 0x27, 0x43, 0x47, 0x4B, 0x4F, 0x53}
        if opcode == 0x33:
            assert ((word >> 25) & 0x7F) != 0x01
        if opcode == 0x6F:
            assert ((word >> 7) & 0x1F) == 0
        if opcode == 0x67:
            rd = (word >> 7) & 0x1F
            rs1 = (word >> 15) & 0x1F
            imm = (word >> 20) & 0xFFF
            assert (rd, rs1, imm) == (0, 1, 0)


def _code_words(code: bytes) -> list[int]:
    assert len(code) % 4 == 0
    return [
        struct.unpack_from("<I", code, offset)[0]
        for offset in range(0, len(code), 4)
    ]


def _is_beq_to_next_plus_one(word: int) -> bool:
    return (
        (word & 0x7F) == 0x63
        and ((word >> 12) & 0x7) == 0
        and ((word >> 20) & 0x1F) == 0
        and _branch_offset(word) == 8
    )


def _branch_offset(word: int) -> int:
    imm = (
        ((word >> 31) & 0x1) << 12
        | ((word >> 7) & 0x1) << 11
        | ((word >> 25) & 0x3F) << 5
        | ((word >> 8) & 0xF) << 1
    )
    if imm & (1 << 12):
        imm -= 1 << 13
    return imm


def _is_riscv_xor(word: int) -> bool:
    return (
        (word & 0x7F) == 0x33
        and ((word >> 12) & 0x7) == 4
        and ((word >> 25) & 0x7F) == 0
    )


def _is_sltiu_imm_one(word: int) -> bool:
    return (
        (word & 0x7F) == 0x13
        and ((word >> 12) & 0x7) == 3
        and ((word >> 20) & 0xFFF) == 1
    )


def _contains_rv32m(code: bytes) -> bool:
    return any(
        (word & 0x7F) == 0x33 and ((word >> 25) & 0x7F) == 0x01
        for word in (
            struct.unpack_from("<I", code, offset)[0]
            for offset in range(0, len(code), 4)
        )
    )


def _fake_csr_map() -> CsrMap:
    return CsrMap(
        act_dma=DmaRegs(
            base=0x1000,
            length=0x1004,
            enable=0x1008,
            done=0x100C,
            loop=0x1010,
            offset=0x1014,
        ),
        wgt_dma=DmaRegs(
            base=0x1020,
            length=0x1024,
            enable=0x1028,
            done=0x102C,
            loop=0x1030,
            offset=0x1034,
        ),
    )
