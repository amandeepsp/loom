"""CFU top-level integration tests.

Tests drive the datapath via CFU instruction interface.
Verifies: DMA fill → sequencer → array → epilogue → results.
"""

import numpy as np
import pytest
from amaranth.back.verilog import convert
from amaranth.sim import Simulator

from hardware.top import Top, TopConfig

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


# ---------------------------------------------------------------------------
# Reference functions (match hardware pipeline exactly)
# ---------------------------------------------------------------------------

def ref_srdhm(a, b):
    if a == INT32_MIN and b == INT32_MIN:
        return INT32_MAX
    return int(((a * b) + (1 << 30)) >> 31)


def ref_rdbpot(x, exponent):
    if exponent == 0:
        return x
    mask = (1 << exponent) - 1
    remainder = x & mask
    threshold = (mask >> 1) + ((x >> 31) & 1)
    return int((x >> exponent) + (1 if remainder > threshold else 0))


def ref_epilogue(acc, bias, multiplier, shift, offset, act_min, act_max):
    x = acc + bias
    x = ref_srdhm(x, multiplier)
    x = ref_rdbpot(x, shift)
    x += offset
    return max(act_min, min(act_max, x))


def pack_int8(vals):
    """Pack list of int8 values into a little-endian word."""
    word = 0
    for i, v in enumerate(vals):
        word |= (v & 0xFF) << (8 * i)
    return word


def to_signed8(val):
    val = val & 0xFF
    return val - 256 if val >= 128 else val


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def dma_fill(ctx, dma_port, words):
    """Write a list of 32-bit words via a DMA port."""
    for addr, word in enumerate(words):
        ctx.set(dma_port.addr, addr)
        ctx.set(dma_port.data, word)
        ctx.set(dma_port.en, 1)
        await ctx.tick()
    ctx.set(dma_port.en, 0)
    await ctx.tick()


async def cfu_op(ctx, dut, funct3, funct7, in0, in1, max_cycles=500):
    """Issue a CFU command and wait for the response."""
    ctx.set(dut.cmd_valid, 1)
    ctx.set(dut.cmd_function_id, {"funct3": funct3, "funct7": funct7})
    ctx.set(dut.cmd_in0, in0)
    ctx.set(dut.cmd_in1, in1)
    ctx.set(dut.rsp_ready, 1)
    await ctx.tick()

    for _ in range(max_cycles):
        if ctx.get(dut.rsp_valid):
            result = ctx.get(dut.rsp_out)
            ctx.set(dut.cmd_valid, 0)
            await ctx.tick()
            return result
        await ctx.tick()
    raise TimeoutError(f"CFU did not respond within {max_cycles} cycles")


async def run_tile(ctx, dut, k, *, first, last):
    """Issue COMPUTE_START then COMPUTE_WAIT via CFU instructions."""
    flags = int(first) | (int(last) << 1)
    await cfu_op(ctx, dut, funct3=0, funct7=0, in0=flags, in1=k)
    await cfu_op(ctx, dut, funct3=1, funct7=0, in0=0, in1=0)


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
    def test_fallback_instruction(self):
        """Unused funct3 slots return in0 (fallback behavior)."""

        async def testbench(ctx):
            result = await cfu_op(ctx, dut, funct3=7, funct7=0,
                                  in0=0xDEADBEEF, in1=0)
            assert result == 0xDEADBEEF

        dut = Top(TopConfig(rows=4, cols=4))
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_cfu_fallback.vcd"):
            sim.run()

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
        ]:
            assert name in v, f"missing port: {name}"


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
        C = (A.astype(np.int32) @ B.astype(np.int32))  # INT32 accumulators

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
                    ref_epilogue(int(C[r, c]), BIAS, MULT, SHIFT, OFFSET, ACT_MIN, ACT_MAX)
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

            # --- Step 2: Swap banks via dummy tile ---
            # Data is in fill bank. Run a no-op tile so sequencer's DONE
            # state triggers swap, moving data to compute bank.
            await run_tile(ctx, dut, k=1, first=False, last=False)

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
                assert got == expected[i], \
                    f"result[{i}] (r={i // COLS}, c={i % COLS}): got {got}, expected {expected[i]}"

        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_integration_4x4.vcd"):
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
                    ref_epilogue(int(C[r, c]), biases[ch], MULT, SHIFT,
                                 OFFSET, ACT_MIN, ACT_MAX)
                )

        act_words = [pack_int8([int(A[r, k]) for r in range(ROWS)]) for k in range(K)]
        wgt_words = [pack_int8([int(B[k, c]) for c in range(COLS)]) for k in range(K)]

        dut = Top(TopConfig(rows=4, cols=4))

        async def testbench(ctx):
            await dma_fill(ctx, dut.dma_act, act_words)
            await dma_fill(ctx, dut.dma_wgt, wgt_words)

            await run_tile(ctx, dut, k=1, first=False, last=False)

            await write_per_channel_params(ctx, dut, biases, mults, shifts)

            await cfu_op(ctx, dut, funct3=3, funct7=0, in0=0, in1=OFFSET)
            await cfu_op(ctx, dut, funct3=3, funct7=1, in0=0, in1=ACT_MIN & 0xFF)
            await cfu_op(ctx, dut, funct3=3, funct7=2, in0=0, in1=ACT_MAX & 0xFF)

            await run_tile(ctx, dut, k=K, first=True, last=True)

            for i in range(num_ch):
                got = await cfu_op(ctx, dut, funct3=4, funct7=0, in0=i, in1=0)
                got = to_signed8(got)
                assert got == expected[i], \
                    f"ch[{i}]: got {got}, expected {expected[i]} (acc={int(C[i // COLS, i % COLS])})"

        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_integration_quant.vcd"):
            sim.run()
