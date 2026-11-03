"""CFU top-level integration tests.

Tests drive the datapath via CFU instruction interface.
Verifies: DMA fill → sequencer → array → epilogue → results.
"""

import numpy as np
import pytest
from amaranth.back.verilog import convert
from amaranth.sim import Simulator

from hardware.testing import cfu_op, dma_fill, pack_int8, run_tile, to_signed8
from hardware.top import Top, TopConfig
from shared.reference import INT32_MAX, INT32_MIN, ref_epilogue


async def write_per_channel_params(ctx, dut, biases, mults, shifts):
    """Load per-channel epilogue params via EPI_PARAM (funct3=2)."""
    for ch in range(len(biases)):
        await cfu_op(ctx, dut, funct3=2, funct7=0, in0=ch, in1=biases[ch])
        await cfu_op(ctx, dut, funct3=2, funct7=1, in0=ch, in1=mults[ch])
        await cfu_op(ctx, dut, funct3=2, funct7=2, in0=ch, in1=shifts[ch])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCfuTop:
    def test_submodules_exist(self):
        """All datapath submodules are instantiated."""
        dut = Top(TopConfig(rows=4, cols=4))
        Simulator(dut)
        assert dut.act_scratch is not None
        assert dut.wgt_scratch is not None
        assert dut.array is not None
        assert dut.seq is not None
        assert dut.epi is not None
        assert dut.params is not None

    def test_debug_ports_present_in_verilog(self):
        """Generated Verilog keeps the board-facing debug ports stable."""
        dut = Top(TopConfig(rows=4, cols=4))
        v = convert(dut, name="Cfu", ports=dut.ports)
        for name in [
            "cfu_state_debug",
            "cfu_instr_debug",
            "cfu_busy_debug",
            "seq_state_debug",
            "seq_busy_debug",
            "error_warn_debug",
            "dma_act_busy",
            "dma_wgt_busy",
        ]:
            assert name in v, f"missing port: {name}"

    def test_performance_counters(self):
        """DMA activity/traffic counters are readable and resettable."""

        async def testbench(ctx):
            await cfu_op(ctx, dut, funct3=5, funct7=0, in0=0, in1=0)

            ctx.set(dut.dma_act_busy, 1)
            ctx.set(dut.dma_wgt_busy, 1)
            for cycle in range(5):
                ctx.set(dut.dma_act.en, cycle < 3)
                ctx.set(dut.dma_wgt.en, cycle < 2)
                await ctx.tick()
            ctx.set(dut.dma_act_busy, 0)
            ctx.set(dut.dma_wgt_busy, 0)
            ctx.set(dut.dma_act.en, 0)
            ctx.set(dut.dma_wgt.en, 0)
            await ctx.tick()

            assert await cfu_op(ctx, dut, funct3=5, funct7=2, in0=0, in1=0) == 5

            await cfu_op(ctx, dut, funct3=5, funct7=0, in0=0, in1=0)
            assert await cfu_op(ctx, dut, funct3=5, funct7=2, in0=0, in1=0) == 0

        dut = Top(TopConfig(rows=4, cols=4))
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()


class TestIntegration:
    """End-to-end datapath test: DMA → CFU instructions → array →
    epilogue → INT8 results. Drives all datapath operations through
    the CFU instruction interface."""

    def test_4x4_matmul_k2(self):
        """4×4 matmul with K=2, full epilogue pipeline.

        A = [[1, 2],     B = [[1, 0, 1, 0],
             [3, 4],          [0, 1, 0, 1]]
             [5, 6],
             [7, 8]]

        C = A @ B = [[1, 2, 1, 2],
                     [3, 4, 3, 4],
                     [5, 6, 5, 6],
                     [7, 8, 7, 8]]
        """
        ROWS, COLS, K = 4, 4, 2

        A = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int8)
        B = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.int8)
        C = A.astype(np.int32) @ B.astype(np.int32)  # INT32 accumulators

        # Epilogue params: near-identity quantization
        MULT = 0x7FFFFFFF
        SHIFT = 0
        BIAS = 0
        OFFSET = 0
        ACT_MIN, ACT_MAX = -128, 127

        # Expected INT8 outputs (row-major: chan = r*COLS + c)
        expected = []
        for r in range(ROWS):
            for c in range(COLS):
                expected.append(
                    ref_epilogue(
                        int(C[r, c]), BIAS, MULT, SHIFT, OFFSET, ACT_MIN, ACT_MAX
                    )
                )

        # Pack scratchpad words
        # act[k] = pack(A[0,k], A[1,k], A[2,k], A[3,k])
        act_words = [pack_int8([int(A[r, k]) for r in range(ROWS)]) for k in range(K)]
        # wgt[k] = pack(B[k,0], B[k,1], B[k,2], B[k,3])
        wgt_words = [pack_int8([int(B[k, c]) for c in range(COLS)]) for k in range(K)]

        dut = Top(TopConfig(rows=4, cols=4))

        async def testbench(ctx):
            num_ch = ROWS * COLS

            # --- Step 1: DMA fill scratchpads (data goes to fill bank) ---
            await dma_fill(ctx, dut.dma_act, act_words)
            await dma_fill(ctx, dut.dma_wgt, wgt_words)

            # --- Step 2: Hardware auto-swaps on computeStart(first=True) ---
            # No longer needed: manual dummy tile for swap

            # --- Step 3: Load epilogue params via EPI_PARAM ---
            biases = [BIAS] * num_ch
            mults = [MULT] * num_ch
            shifts = [SHIFT] * num_ch
            await write_per_channel_params(ctx, dut, biases, mults, shifts)

            # --- Step 4: Set global config via CONFIG ---
            await cfu_op(ctx, dut, funct3=3, funct7=0, in0=0, in1=OFFSET)
            await cfu_op(ctx, dut, funct3=3, funct7=1, in0=0, in1=ACT_MIN & 0xFF)
            await cfu_op(ctx, dut, funct3=3, funct7=2, in0=0, in1=ACT_MAX & 0xFF)

            # --- Step 5: Run real compute tile ---
            await run_tile(ctx, dut, k=K, first=True, last=True)

            # --- Step 6: Read and verify results via READ_RESULT ---
            for i in range(num_ch):
                got = await cfu_op(ctx, dut, funct3=4, funct7=0, in0=i, in1=0)
                got = to_signed8(got)
                assert got == expected[i], (
                    f"result[{i}] (r={i // COLS}, c={i % COLS}): got {got}, expected {expected[i]}"
                )

        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_integration_4x4.vcd"):
            sim.run()

    def test_4x4_matmul_no_swap(self):
        """Same as test_4x4_matmul_k2 but WITHOUT dummy tile for swap.

        This tests the real-world scenario where we don't run a dummy tile.
        Should fail without proper swap because data is in wrong bank.
        """
        ROWS, COLS, K = 4, 4, 2

        A = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int8)
        B = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.int8)
        C = A.astype(np.int32) @ B.astype(np.int32)

        MULT = 0x7FFFFFFF
        SHIFT = 0
        BIAS = 0
        OFFSET = 0
        ACT_MIN, ACT_MAX = -128, 127

        expected = []
        for r in range(ROWS):
            for c in range(COLS):
                expected.append(
                    ref_epilogue(
                        int(C[r, c]), BIAS, MULT, SHIFT, OFFSET, ACT_MIN, ACT_MAX
                    )
                )

        act_words = [pack_int8([int(A[r, k]) for r in range(ROWS)]) for k in range(K)]
        wgt_words = [pack_int8([int(B[k, c]) for c in range(COLS)]) for k in range(K)]

        dut = Top(TopConfig(rows=4, cols=4))

        async def testbench(ctx):
            num_ch = ROWS * COLS

            # --- Step 1: DMA fill scratchpads ---
            await dma_fill(ctx, dut.dma_act, act_words)
            await dma_fill(ctx, dut.dma_wgt, wgt_words)

            # --- Step 2: NO SWAP (skip dummy tile!) ---
            # This is the bug: without swap, we're reading from wrong bank

            # --- Step 3: Load epilogue params ---
            biases = [BIAS] * num_ch
            mults = [MULT] * num_ch
            shifts = [SHIFT] * num_ch
            await write_per_channel_params(ctx, dut, biases, mults, shifts)

            # --- Step 4: Set global config ---
            await cfu_op(ctx, dut, funct3=3, funct7=0, in0=0, in1=OFFSET)
            await cfu_op(ctx, dut, funct3=3, funct7=1, in0=0, in1=ACT_MIN & 0xFF)
            await cfu_op(ctx, dut, funct3=3, funct7=2, in0=0, in1=ACT_MAX & 0xFF)

            # --- Step 5: Run compute tile ---
            await run_tile(ctx, dut, k=K, first=True, last=True)

            # --- Step 6: Read results ---
            for i in range(num_ch):
                got = await cfu_op(ctx, dut, funct3=4, funct7=0, in0=i, in1=0)
                got = to_signed8(got)
                assert got == expected[i], (
                    f"result[{i}]: got {got}, expected {expected[i]}"
                )

        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_no_swap.vcd"):
            sim.run()

    def test_4x4_matmul_with_quant(self):
        """Same matmul but with non-trivial quantization params.

        Uses per-channel bias, multiplier ≈ 0.5, shift=2, offset=10.
        Verifies the full requantization pipeline matches Python reference.
        """
        ROWS, COLS, K = 4, 4, 2

        A = np.array([[10, 20], [30, 40], [50, 60], [70, 80]], dtype=np.int8)
        B = np.array([[1, 2, 1, 2], [2, 1, 2, 1]], dtype=np.int8)
        C = A.astype(np.int32) @ B.astype(np.int32)

        MULT = 0x40000000  # ~0.5
        SHIFT = 2
        OFFSET = 5
        ACT_MIN, ACT_MAX = -128, 127

        # Per-channel biases: alternate 10 / -10
        num_ch = ROWS * COLS
        biases = [(10 if i % 2 == 0 else -10) for i in range(num_ch)]
        mults = [MULT] * num_ch
        shifts = [SHIFT] * num_ch

        expected = []
        for r in range(ROWS):
            for c in range(COLS):
                ch = r * COLS + c
                expected.append(
                    ref_epilogue(
                        int(C[r, c]), biases[ch], MULT, SHIFT, OFFSET, ACT_MIN, ACT_MAX
                    )
                )

        act_words = [pack_int8([int(A[r, k]) for r in range(ROWS)]) for k in range(K)]
        wgt_words = [pack_int8([int(B[k, c]) for c in range(COLS)]) for k in range(K)]

        dut = Top(TopConfig(rows=4, cols=4))

        async def testbench(ctx):
            await dma_fill(ctx, dut.dma_act, act_words)
            await dma_fill(ctx, dut.dma_wgt, wgt_words)

            # Hardware auto-swaps on computeStart(first=True)

            await write_per_channel_params(ctx, dut, biases, mults, shifts)

            await cfu_op(ctx, dut, funct3=3, funct7=0, in0=0, in1=OFFSET)
            await cfu_op(ctx, dut, funct3=3, funct7=1, in0=0, in1=ACT_MIN & 0xFF)
            await cfu_op(ctx, dut, funct3=3, funct7=2, in0=0, in1=ACT_MAX & 0xFF)

            await run_tile(ctx, dut, k=K, first=True, last=True)

            for i in range(num_ch):
                got = await cfu_op(ctx, dut, funct3=4, funct7=0, in0=i, in1=0)
                got = to_signed8(got)
                assert got == expected[i], (
                    f"ch[{i}]: got {got}, expected {expected[i]} (acc={int(C[i // COLS, i % COLS])})"
                )

        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_integration_quant.vcd"):
            sim.run()

    def test_e2e_like_quant_boundary(self):
        """Test with E2E-like random data and mult=256 to catch SRDHM boundary cases.

        This test uses the same random seed and multiplier as E2E, exposing the
        SRDHM boundary case that causes ±1 differences. The 11 boundary errors are
        mathematically expected, not a hardware bug.

        This test verifies the simulation matches hardware for these boundary cases.
        """
        # Use same random seeds as E2E test
        np.random.seed(42)
        A = np.random.randint(-128, 128, size=(4, 4), dtype=np.int8)
        np.random.seed(137)
        B = np.random.randint(-128, 128, size=(4, 4), dtype=np.int8)

        # Use same multiplier as E2E (not the identity mult used in other tests)
        C = A.astype(np.int32) @ B.astype(np.int32)
        MULT = 256  # Same as E2E
        SHIFT = 0
        BIAS = 0
        OFFSET = 0
        ACT_MIN, ACT_MAX = -128, 127

        ROWS, COLS = 4, 4

        expected = []
        for r in range(ROWS):
            for c in range(COLS):
                expected.append(
                    ref_epilogue(
                        int(C[r, c]), BIAS, MULT, SHIFT, OFFSET, ACT_MIN, ACT_MAX
                    )
                )

        act_words = [pack_int8([int(A[r, k]) for r in range(ROWS)]) for k in range(4)]
        wgt_words = [pack_int8([int(B[k, c]) for c in range(COLS)]) for k in range(4)]

        dut = Top(TopConfig(rows=4, cols=4))

        async def testbench(ctx):
            num_ch = ROWS * COLS

            await dma_fill(ctx, dut.dma_act, act_words)
            await dma_fill(ctx, dut.dma_wgt, wgt_words)

            # Epilogue params (same as E2E)
            biases = [BIAS] * num_ch
            mults = [MULT] * num_ch
            shifts = [SHIFT] * num_ch
            await write_per_channel_params(ctx, dut, biases, mults, shifts)

            await cfu_op(ctx, dut, funct3=3, funct7=0, in0=0, in1=OFFSET)
            await cfu_op(ctx, dut, funct3=3, funct7=1, in0=0, in1=ACT_MIN & 0xFF)
            await cfu_op(ctx, dut, funct3=3, funct7=2, in0=0, in1=ACT_MAX & 0xFF)

            await run_tile(ctx, dut, k=4, first=True, last=True)

            # Allow tolerance for boundary cases
            for i in range(num_ch):
                got = await cfu_op(ctx, dut, funct3=4, funct7=0, in0=i, in1=0)
                got = to_signed8(got)
                assert got == expected[i], (
                    f"ch[{i}]: got {got}, expected {expected[i]} "
                    f"(acc={int(C[i // COLS, i % COLS])})"
                )

        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_e2e_like.vcd"):
            sim.run()


class TestBatchA:
    """keep_wgt (compute flag bit 2). (Packed READ_RESULT funct7=1 was
    removed: unused, and its muxes cost placement headroom.)"""

    def test_packed_read_and_weight_reuse(self):
        ROWS, COLS, K = 4, 4, 2
        rng = np.random.default_rng(1)
        A1 = rng.integers(-20, 20, size=(ROWS, K), dtype=np.int8)
        A2 = rng.integers(-20, 20, size=(ROWS, K), dtype=np.int8)
        B = rng.integers(-20, 20, size=(K, COLS), dtype=np.int8)

        def expected(A):
            C = A.astype(np.int32) @ B.astype(np.int32)
            return [ref_epilogue(int(C[r, c]), 0, 0x7FFFFFFF, 0, 0, -128, 127)
                    for r in range(ROWS) for c in range(COLS)]

        act = lambda A: [pack_int8([int(A[r, k]) for r in range(ROWS)]) for k in range(K)]
        wgt = [pack_int8([int(B[k, c]) for c in range(COLS)]) for k in range(K)]
        dut = Top(TopConfig(rows=4, cols=4))

        async def testbench(ctx):
            n = ROWS * COLS
            await write_per_channel_params(ctx, dut, [0] * n, [0x7FFFFFFF] * n, [0] * n)
            await cfu_op(ctx, dut, funct3=3, funct7=1, in0=0, in1=(-128) & 0xFF)
            await cfu_op(ctx, dut, funct3=3, funct7=2, in0=0, in1=127)
            # Tile 1: fill both banks, swap both.
            await dma_fill(ctx, dut.dma_act, act(A1))
            await dma_fill(ctx, dut.dma_wgt, wgt)
            await run_tile(ctx, dut, k=K, first=True, last=True)
            exp = expected(A1)
            for i in range(n):
                got = to_signed8(await cfu_op(ctx, dut, funct3=4, funct7=0, in0=i, in1=0))
                assert got == exp[i], (i, got, exp[i])
            # Tile 2: new activations only; keep_wgt reuses the weight bank.
            await dma_fill(ctx, dut.dma_act, act(A2))
            await cfu_op(ctx, dut, funct3=0, funct7=0, in0=0b111, in1=K)
            await cfu_op(ctx, dut, funct3=1, funct7=0, in0=0, in1=0)
            exp2 = expected(A2)
            for i in range(n):
                got = to_signed8(await cfu_op(ctx, dut, funct3=4, funct7=0, in0=i, in1=0))
                assert got == exp2[i], (i, got, exp2[i])

        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()


class TestKeepAct:
    def test_activation_reuse_across_n_tiles(self):
        ROWS, COLS, K = 4, 4, 2
        rng = np.random.default_rng(2)
        A = rng.integers(-20, 20, size=(ROWS, K), dtype=np.int8)
        B1 = rng.integers(-20, 20, size=(K, COLS), dtype=np.int8)
        B2 = rng.integers(-20, 20, size=(K, COLS), dtype=np.int8)

        def expected(B):
            C = A.astype(np.int32) @ B.astype(np.int32)
            return [ref_epilogue(int(C[r, c]), 0, 0x7FFFFFFF, 0, 0, -128, 127)
                    for r in range(ROWS) for c in range(COLS)]

        act = [pack_int8([int(A[r, k]) for r in range(ROWS)]) for k in range(K)]
        wgt = lambda B: [pack_int8([int(B[k, c]) for c in range(COLS)]) for k in range(K)]
        dut = Top(TopConfig(rows=4, cols=4))

        async def testbench(ctx):
            n = ROWS * COLS
            await write_per_channel_params(ctx, dut, [0] * n, [0x7FFFFFFF] * n, [0] * n)
            await cfu_op(ctx, dut, funct3=3, funct7=1, in0=0, in1=(-128) & 0xFF)
            await cfu_op(ctx, dut, funct3=3, funct7=2, in0=0, in1=127)
            await dma_fill(ctx, dut.dma_act, act)
            await dma_fill(ctx, dut.dma_wgt, wgt(B1))
            await run_tile(ctx, dut, k=K, first=True, last=True)
            for i, e in enumerate(expected(B1)):
                assert to_signed8(await cfu_op(ctx, dut, funct3=4, funct7=0, in0=i, in1=0)) == e
            # New weights only; keep_act (bit 3) reuses the activation bank.
            await dma_fill(ctx, dut.dma_wgt, wgt(B2))
            await cfu_op(ctx, dut, funct3=0, funct7=0, in0=0b1011, in1=K)
            await cfu_op(ctx, dut, funct3=1, funct7=0, in0=0, in1=0)
            for i, e in enumerate(expected(B2)):
                assert to_signed8(await cfu_op(ctx, dut, funct3=4, funct7=0, in0=i, in1=0)) == e

        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()


class TestResultLutAndColumnRead:
    def test_lut_mapped_column_reads(self):
        ROWS, COLS, K = 8, 8, 3
        rng = np.random.default_rng(5)
        A = rng.integers(-20, 20, size=(ROWS, K), dtype=np.int8)
        B = rng.integers(-20, 20, size=(K, COLS), dtype=np.int8)
        lut = rng.integers(-128, 128, size=256, dtype=np.int16)  # lut[byte(x)]
        C = A.astype(np.int32) @ B.astype(np.int32)
        plain = [[ref_epilogue(int(C[r, c]), 0, 0x7FFFFFFF, 0, 0, -128, 127)
                  for c in range(COLS)] for r in range(ROWS)]
        mapped = [[int(lut[v & 0xFF]) for v in row] for row in plain]
        words = [int.from_bytes(bytes(int(x) & 0xFF for x in lut[4 * i:4 * i + 4]), "little")
                 for i in range(64)]
        act = [pack_int8([int(A[r, k]) for r in range(ROWS)]) for k in range(K)]
        wgt = [pack_int8([int(B[k, c]) for c in range(COLS)]) for k in range(K)]
        dut = Top(TopConfig(rows=ROWS, cols=COLS))

        def col_word(table, r0, c):
            return int.from_bytes(bytes(table[r0 + i][c] & 0xFF for i in range(4)), "little")

        async def testbench(ctx):
            n = ROWS * COLS
            await write_per_channel_params(ctx, dut, [0] * n, [0x7FFFFFFF] * n, [0] * n)
            await cfu_op(ctx, dut, funct3=3, funct7=1, in0=0, in1=(-128) & 0xFF)
            await cfu_op(ctx, dut, funct3=3, funct7=2, in0=0, in1=127)
            for i, w in enumerate(words):
                await cfu_op(ctx, dut, funct3=3, funct7=3, in0=i, in1=w)
            for lut_on, table in ((0, plain), (1, mapped)):
                await cfu_op(ctx, dut, funct3=3, funct7=4, in0=0, in1=lut_on)
                await dma_fill(ctx, dut.dma_act, act)
                await dma_fill(ctx, dut.dma_wgt, wgt)
                await run_tile(ctx, dut, k=K, first=True, last=True)
                for r in range(ROWS):
                    for c in range(COLS):
                        got = to_signed8(await cfu_op(ctx, dut, funct3=4, funct7=0, in0=r * COLS + c, in1=0))
                        assert got == table[r][c], (lut_on, r, c)
                for r0 in (0, 4):
                    for c in range(COLS):
                        got = await cfu_op(ctx, dut, funct3=4, funct7=2, in0=r0 * COLS + c, in1=0)
                        assert (got & 0xFFFFFFFF) == col_word(table, r0, c), (lut_on, r0, c)
            await cfu_op(ctx, dut, funct3=3, funct7=4, in0=0, in1=0)

        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()


class TestWriteAct:
    def test_cpu_written_activations(self):
        ROWS, COLS, K = 8, 8, 5
        rng = np.random.default_rng(9)
        A = rng.integers(-20, 20, size=(ROWS, K), dtype=np.int8)
        B = rng.integers(-20, 20, size=(K, COLS), dtype=np.int8)
        C = A.astype(np.int32) @ B.astype(np.int32)
        expected = [ref_epilogue(int(C[r, c]), 0, 0x7FFFFFFF, 0, 0, -128, 127)
                    for r in range(ROWS) for c in range(COLS)]
        lines = [pack_int8([int(A[r, k]) for r in range(ROWS)]) for k in range(K)]
        wgt = [pack_int8([int(B[k, c]) for c in range(COLS)]) for k in range(K)]
        dut = Top(TopConfig(rows=ROWS, cols=COLS))

        async def testbench(ctx):
            n = ROWS * COLS
            await write_per_channel_params(ctx, dut, [0] * n, [0x7FFFFFFF] * n, [0] * n)
            await cfu_op(ctx, dut, funct3=3, funct7=1, in0=0, in1=(-128) & 0xFF)
            await cfu_op(ctx, dut, funct3=3, funct7=2, in0=0, in1=127)
            await cfu_op(ctx, dut, funct3=6, funct7=1, in0=0, in1=0)
            for line in lines:
                await cfu_op(ctx, dut, funct3=6, funct7=0,
                             in0=line & 0xFFFFFFFF, in1=(line >> 32) & 0xFFFFFFFF)
            await dma_fill(ctx, dut.dma_wgt, wgt)
            await run_tile(ctx, dut, k=K, first=True, last=True)
            for i, e in enumerate(expected):
                assert to_signed8(await cfu_op(ctx, dut, funct3=4, funct7=0, in0=i, in1=0)) == e, i

        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()


class TestColumnParams:
    def test_row0_params_apply_to_every_row(self):
        ROWS, COLS, K = 8, 8, 4
        rng = np.random.default_rng(11)
        A = rng.integers(-20, 20, size=(ROWS, K), dtype=np.int8)
        B = rng.integers(-20, 20, size=(K, COLS), dtype=np.int8)
        bias = [int(x) for x in rng.integers(-500, 500, size=COLS)]
        mult = [int(x) for x in rng.integers(0x30000000, 0x7FFFFFFF, size=COLS)]
        shift = [int(x) for x in rng.integers(0, 4, size=COLS)]
        C = A.astype(np.int32) @ B.astype(np.int32)
        expected = [ref_epilogue(int(C[r, c]), bias[c], mult[c], shift[c], 0, -128, 127)
                    for r in range(ROWS) for c in range(COLS)]
        act = [pack_int8([int(A[r, k]) for r in range(ROWS)]) for k in range(K)]
        wgt = [pack_int8([int(B[k, c]) for c in range(COLS)]) for k in range(K)]
        dut = Top(TopConfig(rows=ROWS, cols=COLS))

        async def testbench(ctx):
            # Only slots 0..COLS-1 (row 0) are written.
            await write_per_channel_params(ctx, dut, bias, mult, shift)
            await cfu_op(ctx, dut, funct3=3, funct7=1, in0=0, in1=(-128) & 0xFF)
            await cfu_op(ctx, dut, funct3=3, funct7=2, in0=0, in1=127)
            await dma_fill(ctx, dut.dma_act, act)
            await dma_fill(ctx, dut.dma_wgt, wgt)
            await run_tile(ctx, dut, k=K, first=True, last=True)
            for i, e in enumerate(expected):
                assert to_signed8(await cfu_op(ctx, dut, funct3=4, funct7=0, in0=i, in1=0)) == e, i

        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()


class TestLineGenTop:
    def test_hardware_im2col_depthwise_tile(self):
        """Plane-buffer DMA + GEN + compute = depthwise 3x3 on 8 channels."""
        from hardware.memory.test_linegen import reference_lines
        IH = IW = 8
        rng = np.random.default_rng(21)
        planes = rng.integers(-20, 20, size=(8, IH, IW), dtype=np.int8)
        w = rng.integers(-20, 20, size=(8, 9), dtype=np.int8)
        zp = -3
        buf = planes.reshape(-1).view(np.uint8)
        words = [int.from_bytes(buf[i:i + 8].tobytes(), "little") for i in range(0, len(buf), 8)]
        # Block-diagonal weights: line c*9 + tap carries w[c, tap] in column c.
        wgt = [pack_int8([int(w[c, line % 9]) if c == line // 9 else 0 for c in range(8)])
               for line in range(72)]
        dut = Top(TopConfig(rows=8, cols=8))

        async def testbench(ctx):
            n = 64
            await write_per_channel_params(ctx, dut, [0] * n, [0x7FFFFFFF] * n, [0] * n)
            await cfu_op(ctx, dut, funct3=3, funct7=1, in0=0, in1=(-128) & 0xFF)
            await cfu_op(ctx, dut, funct3=3, funct7=2, in0=0, in1=127)
            await cfu_op(ctx, dut, funct3=7, funct7=0, in0=IW | (IH << 10), in1=IH * IW)
            await cfu_op(ctx, dut, funct3=7, funct7=1, in0=1, in1=zp & 0xFF)
            await dma_fill(ctx, dut.dma_act, words)  # -> plane buffer
            await cfu_op(ctx, dut, funct3=7, funct7=1, in0=0, in1=zp & 0xFF)
            await dma_fill(ctx, dut.dma_wgt, wgt)
            for oy in (0, 4, 7):
                iy0 = oy - 1
                await cfu_op(ctx, dut, funct3=7, funct7=2,
                             in0=((iy0 * IW) & 0xFFFF) | ((iy0 & 0x7FF) << 16), in1=(-1) & 0xFFF)
                await cfu_op(ctx, dut, funct3=7, funct7=3, in0=0, in1=0)
                # Keep the weight bank after the first tile (flag bit 2).
                await cfu_op(ctx, dut, funct3=0, funct7=0, in0=3 | (4 if oy else 0), in1=72)
                await cfu_op(ctx, dut, funct3=1, funct7=0, in0=0, in1=0)
                lines = reference_lines(planes, oy, 0, 1, 1, 1, zp)
                act = np.array([[to_signed8((lines[l] >> (8 * r)) & 0xFF) for l in range(72)]
                                for r in range(8)], dtype=np.int32)
                C = act @ np.array([[int(w[c, l % 9]) if c == l // 9 else 0 for c in range(8)]
                                    for l in range(72)], dtype=np.int32)
                for r in range(8):
                    for c in range(8):
                        e = ref_epilogue(int(C[r, c]), 0, 0x7FFFFFFF, 0, 0, -128, 127)
                        got = to_signed8(await cfu_op(ctx, dut, funct3=4, funct7=0, in0=r * 8 + c, in1=0))
                        assert got == e, (oy, r, c)

        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()


class TestLineGenGroups:
    def test_two_groups_pipelined_like_dwgemm(self):
        """dwgemm's order: per group plane DMA; GEN(t+1) during compute(t);
        next group's weights prefetched into the fill bank after the first
        compute of a group."""
        from hardware.memory.test_linegen import reference_lines
        IH = IW = 8
        G, T = 3, 8
        rng = np.random.default_rng(23)
        planes = rng.integers(-20, 20, size=(8 * G, IH, IW), dtype=np.int8)
        w = rng.integers(-20, 20, size=(8 * G, 9), dtype=np.int8)
        zp = -3

        def plane_words(g):
            buf = planes[8 * g:8 * g + 8].reshape(-1).view(np.uint8)
            return [int.from_bytes(buf[i:i + 8].tobytes(), "little") for i in range(0, len(buf), 8)]

        def wgt_words(g):
            return [pack_int8([int(w[8 * g + c, line % 9]) if c == line // 9 else 0 for c in range(8)])
                    for line in range(72)]

        def expected(g, oy):
            lines = reference_lines(planes[8 * g:8 * g + 8], oy, 0, 1, 1, 1, zp)
            act = np.array([[to_signed8((lines[l] >> (8 * r)) & 0xFF) for l in range(72)]
                            for r in range(8)], dtype=np.int32)
            W = np.array([[int(w[8 * g + c, l % 9]) if c == l // 9 else 0 for c in range(8)]
                          for l in range(72)], dtype=np.int32)
            C = act @ W
            return [[ref_epilogue(int(C[r, c]), 0, 0x7FFFFFFF, 0, 0, -128, 127) for c in range(8)]
                    for r in range(8)]

        dut = Top(TopConfig(rows=8, cols=8))

        async def gen(ctx, oy):
            iy0 = oy - 1
            await cfu_op(ctx, dut, funct3=7, funct7=2,
                         in0=((iy0 * IW) & 0xFFFF) | ((iy0 & 0x7FF) << 16), in1=(-1) & 0xFFF)

        async def testbench(ctx):
            await write_per_channel_params(ctx, dut, [0] * 64, [0x7FFFFFFF] * 64, [0] * 64)
            await cfu_op(ctx, dut, funct3=3, funct7=1, in0=0, in1=(-128) & 0xFF)
            await cfu_op(ctx, dut, funct3=3, funct7=2, in0=0, in1=127)
            await cfu_op(ctx, dut, funct3=7, funct7=0, in0=IW | (IH << 10), in1=IH * IW)
            await cfu_op(ctx, dut, funct3=7, funct7=1, in0=1, in1=zp & 0xFF)
            await dma_fill(ctx, dut.dma_wgt, wgt_words(0))
            for g in range(G):
                # Per-group param upload (row-0 slots), as dwgemm does.
                for col in range(8):
                    for field, val in ((0, 0), (1, 0x7FFFFFFF), (2, 0)):
                        await cfu_op(ctx, dut, funct3=2, funct7=field, in0=col, in1=val)
                # Plane buffer, bursty like the LiteDRAM reader stream.
                for i, word in enumerate(plane_words(g)):
                    ctx.set(dut.dma_act.addr, i)
                    ctx.set(dut.dma_act.data, word)
                    ctx.set(dut.dma_act.en, 1)
                    await ctx.tick()
                    ctx.set(dut.dma_act.en, 0)
                    for _ in range(int(rng.integers(0, 3))):
                        await ctx.tick()
                await gen(ctx, 0)
                for t in range(1, T + 1):
                    await cfu_op(ctx, dut, funct3=7, funct7=3, in0=0, in1=0)  # wait GEN(t-1)
                    await cfu_op(ctx, dut, funct3=0, funct7=0, in0=3 | (4 if t > 1 else 0), in1=72)
                    if t == 1 and g + 1 < G:
                        for _ in range(20):  # a real DMA lands well after PRIME
                            await ctx.tick()
                        await dma_fill(ctx, dut.dma_wgt, wgt_words(g + 1))  # prefetch
                    if t < T:
                        await gen(ctx, t)
                    await cfu_op(ctx, dut, funct3=1, funct7=0, in0=0, in1=0)
                    exp = expected(g, t - 1)
                    for r in range(8):
                        for c in range(8):
                            got = to_signed8(await cfu_op(ctx, dut, funct3=4, funct7=0,
                                                          in0=r * 8 + c, in1=0))
                            assert got == exp[r][c], (g, t - 1, r, c, got, exp[r][c])

        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()


class TestResultWriteback:
    def test_writeback_beats_and_wait(self):
        ROWS, COLS, K = 8, 8, 5
        rng = np.random.default_rng(31)
        A = rng.integers(-20, 20, size=(ROWS, K), dtype=np.int8)
        B = rng.integers(-20, 20, size=(K, COLS), dtype=np.int8)
        C = A.astype(np.int32) @ B.astype(np.int32)
        res = [[ref_epilogue(int(C[r, c]), 0, 0x7FFFFFFF, 0, 0, -128, 127) for c in range(COLS)]
               for r in range(ROWS)]
        act = [pack_int8([int(A[r, k]) for r in range(ROWS)]) for k in range(K)]
        wgt = [pack_int8([int(B[k, c]) for c in range(COLS)]) for k in range(K)]
        BASE, STRIDE = 0x40123400, 1088
        dut = Top(TopConfig(rows=ROWS, cols=COLS))
        beats = []

        async def collector(ctx):
            # Stall the writer for a while, then accept every beat.
            ctx.set(dut.wb_ready, 0)
            stalled = 0
            while True:
                await ctx.tick()
                if ctx.get(dut.wb_valid):
                    if stalled < 30:
                        stalled += 1
                        continue
                    ctx.set(dut.wb_ready, 1)
                    beats.append((ctx.get(dut.wb_addr), ctx.get(dut.wb_data)))

        async def testbench(ctx):
            await write_per_channel_params(ctx, dut, [0] * 8, [0x7FFFFFFF] * 8, [0] * 8)
            await cfu_op(ctx, dut, funct3=3, funct7=1, in0=0, in1=(-128) & 0xFF)
            await cfu_op(ctx, dut, funct3=3, funct7=2, in0=0, in1=127)
            await cfu_op(ctx, dut, funct3=3, funct7=6, in0=1, in1=STRIDE)
            await cfu_op(ctx, dut, funct3=3, funct7=5, in0=0, in1=BASE)
            await dma_fill(ctx, dut.dma_act, act)
            await dma_fill(ctx, dut.dma_wgt, wgt)
            await cfu_op(ctx, dut, funct3=0, funct7=0, in0=3, in1=K)
            await cfu_op(ctx, dut, funct3=1, funct7=0, in0=0, in1=0)
            # COMPUTE_WAIT returned: every beat must have been accepted.
            assert len(beats) == 8, len(beats)
            for c, (addr, data) in enumerate(beats):
                assert addr == BASE + c * STRIDE, (c, hex(addr))
                want = int.from_bytes(bytes(res[r][c] & 0xFF for r in range(ROWS)), "little")
                assert data == want, (c, hex(data), hex(want))
            await cfu_op(ctx, dut, funct3=3, funct7=6, in0=0, in1=0)

        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(collector, background=True)
        sim.add_testbench(testbench)
        sim.run()
