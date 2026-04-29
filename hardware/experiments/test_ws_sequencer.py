from amaranth import Module, signed
from amaranth.sim import Simulator
import numpy as np
import pytest

from hardware.control.ws_sequencer import WSSequencer
from hardware.systolic.ws_pe_array import WeightStationaryPEArray


def pack_act(vals):
    """Build a dict for the act_rd_data struct port."""
    return {f"r{i}": v for i, v in enumerate(vals)}


def pack_wgt(vals):
    """Build a dict for the wgt_rd_data struct port."""
    return {f"c{i}": v for i, v in enumerate(vals)}


def build_ws_top(rows, cols, in_width=8, acc_width=32, wide=1):
    """Build a top module with WSSequencer + WeightStationaryPEArray wired together."""
    m = Module()
    seq = WSSequencer(rows=rows, cols=cols, in_width=in_width, acc_width=acc_width,
                      wide=wide)
    array = WeightStationaryPEArray(rows, cols, in_width=in_width, acc_width=acc_width)
    m.submodules.seq = seq
    m.submodules.array = array

    # Sequencer → Array: activation inputs
    for r in range(rows):
        m.d.comb += getattr(array, f"act_in_{r}").eq(getattr(seq.arr_act_in, f"r{r}"))
    # Sequencer → Array: weight inputs
    for c in range(cols):
        m.d.comb += getattr(array, f"w_in_{c}").eq(getattr(seq.arr_w_in, f"c{c}"))
    # Sequencer → Array: weight load
    m.d.comb += array.w_load.eq(seq.arr_w_load)

    # Array → Sequencer: bottom-row psum outputs
    for c in range(cols):
        m.d.comb += getattr(seq, f"arr_psum_out_{c}").eq(getattr(array, f"psum_out_{c}"))

    return m, seq


async def run_ws_tile(ctx, seq, A, B, *, rows, cols, first=True, last=True,
                      timeout=500):
    """Drive one WS tile through the sequencer, return (results_dict, cycle_count).

    A: np.ndarray [M=rows, K=rows]  — activation matrix
    B: np.ndarray [K=rows, C=cols]  — weight matrix
    """
    K = A.shape[1]
    assert K == rows, "K must equal rows for single-tile WS"

    ctx.set(seq.k_count, K)
    ctx.set(seq.first, int(first))
    ctx.set(seq.last, int(last))
    ctx.set(seq.start, 1)
    await ctx.tick()
    ctx.set(seq.start, 0)
    cycle_count = 1

    # Provide scratchpad data and wait for done
    results = {}
    epi_first_seen = False
    epi_last_seen = False

    while True:
        # Respond to scratchpad reads
        act_addr = ctx.get(seq.act_rd_addr)
        wgt_addr = ctx.get(seq.wgt_rd_addr)
        if act_addr < rows:
            ctx.set(seq.act_rd_data, pack_act([int(A[act_addr, r]) for r in range(rows)]))
        else:
            ctx.set(seq.act_rd_data, pack_act([0] * rows))
        if wgt_addr < K:
            ctx.set(seq.wgt_rd_data, pack_wgt([int(B[wgt_addr, c]) for c in range(cols)]))
        else:
            ctx.set(seq.wgt_rd_data, pack_wgt([0] * cols))

        # Capture epilogue outputs (check combinationally before tick)
        await ctx.delay(1e-7)
        state = ctx.get(seq.state_debug)
        # Look for epilogue markers using epi_first/epi_last
        if ctx.get(seq.epi_first):
            epi_first_seen = True
        if ctx.get(seq.epi_last):
            epi_last_seen = True
        # Capture any epilogue data when epi_first or epi_last are set or
        # when we're in an epilogue-like state.  Safest: always capture if
        # epi_index looks valid and we haven't stored this index yet.
        idx = ctx.get(seq.epi_index)
        if epi_first_seen and not epi_last_seen or ctx.get(seq.epi_first) or ctx.get(seq.epi_last):
            results[idx] = ctx.get(seq.epi_data)

        await ctx.tick()
        cycle_count += 1

        if ctx.get(seq.done):
            break
        if cycle_count > timeout:
            raise TimeoutError(f"WSSequencer did not complete within {timeout} cycles")

    if last:
        # Signal epi_done so sequencer transitions out of EPILOGUE_WAIT → DONE
        # (already done if done is already asserted)
        pass

    return results, cycle_count, epi_first_seen, epi_last_seen


async def run_ws_tile_with_epilogue(ctx, seq, A, B, *, rows, cols, wide=1,
                                     first=True, last=True, timeout=500):
    """Run a WS tile and handle the epilogue handshake. Returns (results, cycle_count)."""
    K = A.shape[1]
    assert K == rows

    ctx.set(seq.k_count, K)
    ctx.set(seq.first, int(first))
    ctx.set(seq.last, int(last))
    ctx.set(seq.start, 1)
    await ctx.tick()
    ctx.set(seq.start, 0)
    cycle_count = 1

    results = {}
    epi_firsts = []
    epi_lasts = []
    epi_indices = []
    in_epilogue = False
    done = False
    has_wide = hasattr(seq, "epi_data_1")

    while not done:
        # Respond to scratchpad reads
        act_addr = ctx.get(seq.act_rd_addr)
        wgt_addr = ctx.get(seq.wgt_rd_addr)
        if act_addr < rows:
            ctx.set(seq.act_rd_data, pack_act([int(A[act_addr, r]) for r in range(rows)]))
        else:
            ctx.set(seq.act_rd_data, pack_act([0] * rows))
        if wgt_addr < K:
            ctx.set(seq.wgt_rd_data, pack_wgt([int(B[wgt_addr, c]) for c in range(cols)]))
        else:
            ctx.set(seq.wgt_rd_data, pack_wgt([0] * cols))

        await ctx.delay(1e-7)

        # Detect epilogue phase via epi_first
        if ctx.get(seq.epi_first):
            in_epilogue = True

        if in_epilogue:
            idx = ctx.get(seq.epi_index)
            val = ctx.get(seq.epi_data)
            results[idx] = val
            if has_wide:
                results[idx + 1] = ctx.get(seq.epi_data_1)
            epi_firsts.append(ctx.get(seq.epi_first))
            epi_lasts.append(ctx.get(seq.epi_last))
            epi_indices.append(idx)
            if ctx.get(seq.epi_last):
                in_epilogue = False

        await ctx.tick()
        cycle_count += 1

        # After epi_last, we should be in EPILOGUE_WAIT — send epi_done
        if last and epi_lasts and epi_lasts[-1] == 1:
            ctx.set(seq.epi_done, 1)
            await ctx.tick()
            cycle_count += 1
            ctx.set(seq.epi_done, 0)

            if ctx.get(seq.done):
                done = True
                break

        if ctx.get(seq.done):
            done = True
            break

        if cycle_count > timeout:
            raise TimeoutError(f"WSSequencer did not complete within {timeout} cycles")

    return results, cycle_count, epi_firsts, epi_lasts, epi_indices


class TestWSSequencer:
    def test_2x2_single_tile(self):
        """2×2 WS GEMM: verify results against NumPy and cycle count."""
        R, C = 2, 2
        A = np.array([[1, 2], [3, 4]], dtype=np.int8)
        B = np.array([[5, 6], [7, 8]], dtype=np.int8)
        expected = A.astype(np.int32) @ B.astype(np.int32)

        top, seq = build_ws_top(R, C)

        async def testbench(ctx):
            results, cycles, epi_firsts, epi_lasts, _ = await run_ws_tile_with_epilogue(
                ctx, seq, A, B, rows=R, cols=C, first=True, last=True,
            )
            assert ctx.get(seq.done) == 1

            # Verify against NumPy
            for r in range(R):
                for c in range(C):
                    idx = r * C + c
                    assert results[idx] == int(expected[r, c]), \
                        f"result[{r}][{c}] = {results[idx]}, expected {int(expected[r, c])}"

            # Verify cycle count: 3R + C - 1 + R*C = 6 + 2 - 1 + 4 = 11
            expected_compute_drain = 3 * R + C - 1 + R * C
            # cycles includes start tick + epilogue_wait handshake overhead;
            # allow small overhead for FSM transitions
            assert cycles <= expected_compute_drain + 4, \
                f"cycle count {cycles} exceeds expected {expected_compute_drain} + overhead"

        sim = Simulator(top)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_ws_seq_2x2.vcd"):
            sim.run()

    def test_4x4_single_tile(self):
        """4×4 WS GEMM with random data: verify against NumPy, assert cycle count."""
        R, C = 4, 4
        rng = np.random.default_rng(42)
        A = rng.integers(-10, 10, size=(R, R), dtype=np.int8)
        B = rng.integers(-10, 10, size=(R, C), dtype=np.int8)
        expected = A.astype(np.int32) @ B.astype(np.int32)

        top, seq = build_ws_top(R, C)

        async def testbench(ctx):
            results, cycles, _, _, _ = await run_ws_tile_with_epilogue(
                ctx, seq, A, B, rows=R, cols=C, first=True, last=True,
            )
            assert ctx.get(seq.done) == 1

            for r in range(R):
                for c in range(C):
                    idx = r * C + c
                    assert results[idx] == int(expected[r, c]), \
                        f"result[{r}][{c}] = {results[idx]}, expected {int(expected[r, c])}"

            # 3*4 + 4 - 1 + 16 = 31
            expected_compute_drain = 3 * R + C - 1 + R * C
            assert expected_compute_drain == 31
            assert cycles <= expected_compute_drain + 4, \
                f"cycle count {cycles} exceeds expected {expected_compute_drain} + overhead"

        sim = Simulator(top)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_ws_seq_4x4.vcd"):
            sim.run()

    def test_4x4_with_negatives(self):
        """4×4 WS GEMM with signed values matching the OS test data."""
        R, C = 4, 4
        A = np.array([[ 1, 0, 2,-1],
                       [ 0, 3,-1, 0],
                       [ 2, 1, 0, 4],
                       [-1, 0, 1, 2]], dtype=np.int8)
        B = np.array([[ 1, 2, 0,-1],
                       [ 0, 1, 3, 2],
                       [-1, 0, 1, 0],
                       [ 2,-1, 0, 3]], dtype=np.int8)
        expected = A.astype(np.int32) @ B.astype(np.int32)

        top, seq = build_ws_top(R, C)

        async def testbench(ctx):
            results, _, _, _, _ = await run_ws_tile_with_epilogue(
                ctx, seq, A, B, rows=R, cols=C, first=True, last=True,
            )
            assert ctx.get(seq.done) == 1

            for r in range(R):
                for c in range(C):
                    idx = r * C + c
                    assert results[idx] == int(expected[r, c]), \
                        f"result[{r}][{c}] = {results[idx]}, expected {int(expected[r, c])}"

        sim = Simulator(top)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_ws_seq_4x4_neg.vcd"):
            sim.run()

    def test_epilogue_handshake(self):
        """Verify epi_first, epi_last, epi_index markers during DRAIN."""
        R, C = 2, 2
        A = np.array([[1, 2], [3, 4]], dtype=np.int8)
        B = np.array([[5, 6], [7, 8]], dtype=np.int8)
        num_results = R * C

        top, seq = build_ws_top(R, C)

        async def testbench(ctx):
            results, _, epi_firsts, epi_lasts, epi_indices = \
                await run_ws_tile_with_epilogue(
                    ctx, seq, A, B, rows=R, cols=C, first=True, last=True,
                )

            # epi_first should be 1 only on the first epilogue cycle
            assert epi_firsts[0] == 1, "epi_first not set on first epilogue output"
            for i in range(1, len(epi_firsts)):
                assert epi_firsts[i] == 0, f"epi_first unexpectedly set at index {i}"

            # epi_last should be 1 only on the last epilogue cycle
            for i in range(len(epi_lasts) - 1):
                assert epi_lasts[i] == 0, f"epi_last unexpectedly set at index {i}"
            assert epi_lasts[-1] == 1, "epi_last not set on final epilogue output"

            # All R*C indices should have been emitted
            assert len(epi_indices) == num_results, \
                f"expected {num_results} epilogue outputs, got {len(epi_indices)}"

            # Indices should be sequential 0..R*C-1
            assert epi_indices == list(range(num_results)), \
                f"epilogue indices {epi_indices} not sequential"

        sim = Simulator(top)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_ws_seq_epi.vcd"):
            sim.run()

    def test_first_last_control(self):
        """Verify first=1 triggers LOAD_WEIGHTS, first=0 skips it."""
        R, C = 2, 2
        A = np.array([[1, 0], [0, 1]], dtype=np.int8)
        B = np.array([[2, 0], [0, 3]], dtype=np.int8)

        top, seq = build_ws_top(R, C)

        async def testbench(ctx):
            # Tile 0: first=1, last=0 — should load weights, no epilogue
            ctx.set(seq.k_count, R)
            ctx.set(seq.first, 1)
            ctx.set(seq.last, 0)
            ctx.set(seq.start, 1)
            await ctx.tick()
            ctx.set(seq.start, 0)

            # Wait for done, providing scratchpad data
            for _ in range(100):
                act_addr = ctx.get(seq.act_rd_addr)
                wgt_addr = ctx.get(seq.wgt_rd_addr)
                if act_addr < R:
                    ctx.set(seq.act_rd_data, pack_act([int(A[act_addr, r]) for r in range(R)]))
                if wgt_addr < R:
                    ctx.set(seq.wgt_rd_data, pack_wgt([int(B[wgt_addr, c]) for c in range(C)]))
                await ctx.tick()
                if ctx.get(seq.done):
                    break

            assert ctx.get(seq.done) == 1
            # Swap is asserted in STREAM_ROWS done (one cycle before DONE)

            await ctx.tick()  # return to IDLE

            # Tile 1: first=0, last=1 — should skip weight load, run epilogue
            cycle_before = 0
            ctx.set(seq.k_count, R)
            ctx.set(seq.first, 0)
            ctx.set(seq.last, 1)
            ctx.set(seq.start, 1)
            await ctx.tick()
            ctx.set(seq.start, 0)
            cycle_before += 1

            # Should NOT enter LOAD_WEIGHTS — arr_w_load should stay 0
            await ctx.delay(1e-7)
            assert ctx.get(seq.arr_w_load) == 0, \
                "arr_w_load asserted with first=0, should skip weight load"

            # Wait for done
            for _ in range(100):
                act_addr = ctx.get(seq.act_rd_addr)
                if act_addr < R:
                    ctx.set(seq.act_rd_data, pack_act([int(A[act_addr, r]) for r in range(R)]))
                await ctx.tick()
                cycle_before += 1

                # Handle epilogue_wait
                if ctx.get(seq.epi_last):
                    await ctx.tick()
                    cycle_before += 1
                    ctx.set(seq.epi_done, 1)
                    await ctx.tick()
                    cycle_before += 1
                    ctx.set(seq.epi_done, 0)

                if ctx.get(seq.done):
                    break

            assert ctx.get(seq.done) == 1

        sim = Simulator(top)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_ws_seq_first_last.vcd"):
            sim.run()

    @pytest.mark.parametrize("R,C", [(2, 2), (4, 4)])
    def test_cycle_count_formula(self, R, C):
        """Verify total compute+drain cycles = 3R + C - 1 + R×C."""
        rng = np.random.default_rng(123 + R * 10 + C)
        A = rng.integers(-5, 5, size=(R, R), dtype=np.int8)
        B = rng.integers(-5, 5, size=(R, C), dtype=np.int8)
        expected = A.astype(np.int32) @ B.astype(np.int32)
        expected_total = 3 * R + C - 1 + R * C

        top, seq = build_ws_top(R, C)

        async def testbench(ctx):
            ctx.set(seq.k_count, R)
            ctx.set(seq.first, 1)
            ctx.set(seq.last, 1)
            ctx.set(seq.start, 1)
            await ctx.tick()
            ctx.set(seq.start, 0)

            # Count cycles from after start pulse to done, excluding
            # the epilogue_wait handshake.
            compute_drain_cycles = 0
            results = {}
            epi_last_seen = False

            while True:
                act_addr = ctx.get(seq.act_rd_addr)
                wgt_addr = ctx.get(seq.wgt_rd_addr)
                if act_addr < R:
                    ctx.set(seq.act_rd_data, pack_act([int(A[act_addr, r]) for r in range(R)]))
                else:
                    ctx.set(seq.act_rd_data, pack_act([0] * R))
                if wgt_addr < R:
                    ctx.set(seq.wgt_rd_data, pack_wgt([int(B[wgt_addr, c]) for c in range(C)]))
                else:
                    ctx.set(seq.wgt_rd_data, pack_wgt([0] * C))

                await ctx.delay(1e-7)

                # Capture epilogue results
                if ctx.get(seq.epi_first) or (results and not epi_last_seen):
                    idx = ctx.get(seq.epi_index)
                    results[idx] = ctx.get(seq.epi_data)
                if ctx.get(seq.epi_last):
                    epi_last_seen = True

                await ctx.tick()
                compute_drain_cycles += 1

                if epi_last_seen:
                    # Now in EPILOGUE_WAIT — send epi_done
                    ctx.set(seq.epi_done, 1)
                    await ctx.tick()
                    ctx.set(seq.epi_done, 0)
                    break

                if compute_drain_cycles > 500:
                    raise TimeoutError("Sequencer did not reach epi_last")

            # Verify cycle count matches formula
            assert compute_drain_cycles == expected_total, \
                f"R={R}, C={C}: got {compute_drain_cycles} compute+drain cycles, " \
                f"expected 3R+C-1+R*C = {expected_total}"

            # Verify correctness
            for r in range(R):
                for c in range(C):
                    idx = r * C + c
                    assert results[idx] == int(expected[r, c]), \
                        f"result[{r}][{c}] = {results[idx]}, expected {int(expected[r, c])}"

        sim = Simulator(top)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd(f"waves/test_ws_seq_cycles_{R}x{C}.vcd"):
            sim.run()

    def test_wide2_single_buffer_drain(self):
        """Verify 2-wide drain with one result buffer.

        For 4x4, wide=1 drains 16 cycles; wide=2 drains 8 cycles.
        This keeps the simple single-FSM sequencer but preserves the main WS
        cycle-count advantage over OS for K=R.
        """
        R, C = 4, 4
        rng = np.random.default_rng(999)
        A = rng.integers(-5, 5, size=(R, R), dtype=np.int8)
        B = rng.integers(-5, 5, size=(R, C), dtype=np.int8)
        expected = A.astype(np.int32) @ B.astype(np.int32)

        top, seq = build_ws_top(R, C, wide=2)

        async def testbench(ctx):
            results, cycles, _, _, epi_indices = await run_ws_tile_with_epilogue(
                ctx, seq, A, B, rows=R, cols=C, wide=2, first=True, last=True
            )

            assert epi_indices == list(range(0, R * C, 2))
            assert cycles <= (3 * R + C - 1 + (R * C // 2) + 4), \
                f"wide=2 drain took too long: {cycles} cycles"

            for r in range(R):
                for c in range(C):
                    idx = r * C + c
                    assert results[idx] == int(expected[r, c]), \
                        f"result[{r}][{c}] mismatch"

        sim = Simulator(top)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_ws_seq_wide2.vcd"):
            sim.run()
