from amaranth.sim import Simulator

from hardware.control.os_sequencer import OSSequencer


def pack_act(vals):
    """Build a dict for the act_rd_data struct port."""
    return {f"r{i}": v for i, v in enumerate(vals)}


def pack_wgt(vals):
    """Build a dict for the wgt_rd_data struct port."""
    return {f"c{i}": v for i, v in enumerate(vals)}


async def run_tile(ctx, dut, k, *, first=True, last=True, rows=2, cols=2):
    """Run one tile: set first/last, start, wait through FEED+FLUSH+epilogue."""
    ctx.set(dut.k_count, k)
    ctx.set(dut.first, int(first))
    ctx.set(dut.last, int(last))
    ctx.set(dut.start, 1)
    await ctx.tick()
    ctx.set(dut.start, 0)

    if first:
        # PRIME
        await ctx.tick()

    # FEED
    for _ in range(k):
        ctx.set(dut.act_rd_data, pack_act([0] * rows))
        ctx.set(dut.wgt_rd_data, pack_wgt([0] * cols))
        await ctx.tick()

    # FLUSH
    for _ in range(rows + cols - 2):
        await ctx.tick()

    if last:
        # EPILOGUE: sequencer walks R*C psums
        num_results = rows * cols
        for _ in range(num_results):
            await ctx.tick()
        # EPILOGUE_WAIT: assert epi_done so sequencer transitions to DONE
        ctx.set(dut.epi_done, 1)
        await ctx.tick()
        ctx.set(dut.epi_done, 0)

    # DONE
    assert ctx.get(dut.done) == 1


class TestSequencer:
    def test_fsm_cycle_count(self):
        """Verify FSM transitions with first=1, last=1 (single tile)."""
        ROWS, COLS, K = 2, 2, 4

        async def testbench(ctx):
            await run_tile(ctx, dut, K, first=True, last=True, rows=ROWS, cols=COLS)
            assert ctx.get(dut.done) == 1
            assert ctx.get(dut.act_swap) == 1
            assert ctx.get(dut.wgt_swap) == 1

        dut = OSSequencer(rows=ROWS, cols=COLS)
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_seq_fsm.vcd"):
            sim.run()

    def test_scratchpad_addr_sequence(self):
        """Verify scratchpad addresses during FEED."""
        ROWS, COLS, K = 2, 2, 4

        async def testbench(ctx):
            ctx.set(dut.k_count, K)
            ctx.set(dut.first, 1)
            ctx.set(dut.last, 0)
            ctx.set(dut.start, 1)
            await ctx.tick()
            ctx.set(dut.start, 0)

            # PRIME: addr should be 0
            assert ctx.get(dut.act_rd_addr) == 0
            assert ctx.get(dut.wgt_rd_addr) == 0
            await ctx.tick()

            # FEED: addr should advance 1, 2, 3, ...
            for i in range(K):
                expected_addr = i + 1 if i < K - 1 else i + 1
                assert ctx.get(dut.act_rd_addr) == expected_addr, \
                    f"cycle {i}: act_rd_addr={ctx.get(dut.act_rd_addr)}, expected {expected_addr}"
                ctx.set(dut.act_rd_data, pack_act([0] * ROWS))
                ctx.set(dut.wgt_rd_data, pack_wgt([0] * COLS))
                await ctx.tick()

        dut = OSSequencer(rows=ROWS, cols=COLS)
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_seq_addr.vcd"):
            sim.run()

    def test_data_passthrough(self):
        """Verify act/wgt data is forwarded to array inputs during FEED."""
        ROWS, COLS, K = 2, 2, 2

        async def testbench(ctx):
            ctx.set(dut.k_count, K)
            ctx.set(dut.first, 1)
            ctx.set(dut.last, 0)
            ctx.set(dut.start, 1)
            await ctx.tick()
            ctx.set(dut.start, 0)

            # PRIME
            await ctx.tick()

            # FEED cycle 0
            ctx.set(dut.act_rd_data, pack_act([3, -7]))
            ctx.set(dut.wgt_rd_data, pack_wgt([5, -2]))
            await ctx.tick()

            # FEED cycle 1: check comb passthrough
            ctx.set(dut.act_rd_data, pack_act([10, -20]))
            ctx.set(dut.wgt_rd_data, pack_wgt([15, -30]))
            await ctx.delay(1e-7)
            assert ctx.get(dut.arr_act_in.r0) == 10
            assert ctx.get(dut.arr_act_in.r1) == -20
            assert ctx.get(dut.arr_w_in.c0) == 15
            assert ctx.get(dut.arr_w_in.c1) == -30

        dut = OSSequencer(rows=ROWS, cols=COLS)
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_seq_data.vcd"):
            sim.run()

    def test_epilogue_handshake(self):
        """Verify EPILOGUE emits psums with first/last markers."""
        ROWS, COLS, K = 2, 2, 1
        NUM = ROWS * COLS

        async def testbench(ctx):
            ctx.set(dut.k_count, K)
            ctx.set(dut.first, 1)
            ctx.set(dut.last, 1)
            ctx.set(dut.start, 1)
            await ctx.tick()
            ctx.set(dut.start, 0)

            # PRIME
            await ctx.tick()

            # FEED
            ctx.set(dut.act_rd_data, pack_act([0] * ROWS))
            ctx.set(dut.wgt_rd_data, pack_wgt([0] * COLS))
            await ctx.tick()

            # FLUSH
            for _ in range(ROWS + COLS - 2):
                await ctx.tick()

            # Set known psum values
            ctx.set(dut.arr_psum_out_0_0, 100)
            ctx.set(dut.arr_psum_out_0_1, 200)
            ctx.set(dut.arr_psum_out_1_0, 300)
            ctx.set(dut.arr_psum_out_1_1, 400)

            # EPILOGUE: check first/last markers and data
            expected = [100, 200, 300, 400]
            for i in range(NUM):
                await ctx.delay(1e-7)
                assert ctx.get(dut.epi_first) == (1 if i == 0 else 0), f"epi_first wrong at {i}"
                assert ctx.get(dut.epi_last) == (1 if i == NUM - 1 else 0), f"epi_last wrong at {i}"
                assert ctx.get(dut.epi_data) == expected[i], \
                    f"epi_data={ctx.get(dut.epi_data)}, expected {expected[i]} at index {i}"
                await ctx.tick()

            # EPILOGUE_WAIT
            ctx.set(dut.epi_done, 1)
            await ctx.tick()
            ctx.set(dut.epi_done, 0)

            # DONE
            assert ctx.get(dut.done) == 1

        dut = OSSequencer(rows=ROWS, cols=COLS)
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_seq_epilogue.vcd"):
            sim.run()

    def test_first_last_k_tiling(self):
        """Verify first/last control: first=1 runs PRIME, last=1 runs EPILOGUE."""
        ROWS, COLS, K = 2, 2, 1

        async def testbench(ctx):
            # Tile 0: first=1, last=0 → PRIME + FEED + FLUSH + DONE (no EPILOGUE)
            await run_tile(ctx, dut, K, first=True, last=False, rows=ROWS, cols=COLS)
            assert ctx.get(dut.done) == 1
            await ctx.tick()  # return to IDLE

            # Tile 1: first=0, last=1 → FEED + FLUSH + EPILOGUE + DONE (no PRIME)
            # PE accumulators should persist from tile 0
            await run_tile(ctx, dut, K, first=False, last=True, rows=ROWS, cols=COLS)
            assert ctx.get(dut.done) == 1

        dut = OSSequencer(rows=ROWS, cols=COLS)
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_seq_ktile.vcd"):
            sim.run()

    def test_swap_single_pulse(self):
        """Verify swap signals pulse once in DONE, not continuously."""
        ROWS, COLS, K = 2, 2, 1

        async def testbench(ctx):
            await run_tile(ctx, dut, K, first=True, last=True, rows=ROWS, cols=COLS)

            # DONE: swap asserted
            assert ctx.get(dut.act_swap) == 1
            assert ctx.get(dut.wgt_swap) == 1

            await ctx.tick()

            # After DONE → IDLE: swap deasserted
            assert ctx.get(dut.act_swap) == 0
            assert ctx.get(dut.wgt_swap) == 0

        dut = OSSequencer(rows=ROWS, cols=COLS)
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_seq_swap.vcd"):
            sim.run()

    def test_idle_after_done(self):
        """Verify FSM returns to IDLE and can restart."""
        ROWS, COLS, K = 2, 2, 1

        async def testbench(ctx):
            await run_tile(ctx, dut, K, first=True, last=True, rows=ROWS, cols=COLS)
            assert ctx.get(dut.done) == 1
            await ctx.tick()
            assert ctx.get(dut.done) == 0

            # Can start again — first=1 should hit PRIME
            ctx.set(dut.k_count, K)
            ctx.set(dut.first, 1)
            ctx.set(dut.last, 0)
            ctx.set(dut.start, 1)
            await ctx.tick()
            ctx.set(dut.start, 0)
            assert ctx.get(dut.arr_psum_load) == 1

        dut = OSSequencer(rows=ROWS, cols=COLS)
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_seq_idle.vcd"):
            sim.run()
