from amaranth.sim import Simulator

from scratchpad import DoubleScratchpad


class TestDoubleScratchpad:
    def test_write_swap_read(self):
        """Write to fill bank, swap, read from compute bank."""

        async def testbench(ctx):
            # Write 0xDEAD to addr 0, 0xBEEF to addr 1 (fill bank = A)
            ctx.set(dut.wr.addr, 0)
            ctx.set(dut.wr.data, 0xDEAD)
            ctx.set(dut.wr.en, 1)
            await ctx.tick()
            ctx.set(dut.wr.addr, 1)
            ctx.set(dut.wr.data, 0xBEEF)
            await ctx.tick()
            ctx.set(dut.wr.en, 0)

            # Swap: A becomes compute, B becomes fill
            ctx.set(dut.swap, 1)
            await ctx.tick()
            ctx.set(dut.swap, 0)

            # Read addr 0 from compute bank (now A)
            ctx.set(dut.rd_addr, 0)
            await ctx.tick()  # 1-cycle read latency
            assert ctx.get(dut.rd_data) == 0xDEAD

            ctx.set(dut.rd_addr, 1)
            await ctx.tick()
            assert ctx.get(dut.rd_data) == 0xBEEF

        dut = DoubleScratchpad(depth=16)
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_scratchpad_basic.vcd"):
            sim.run()

    def test_isolation(self):
        """Writing to fill bank does not affect compute bank reads."""

        async def testbench(ctx):
            # Fill bank A with data
            ctx.set(dut.wr.addr, 0)
            ctx.set(dut.wr.data, 100)
            ctx.set(dut.wr.en, 1)
            await ctx.tick()
            ctx.set(dut.wr.en, 0)

            # Swap: A=compute, B=fill
            ctx.set(dut.swap, 1)
            await ctx.tick()
            ctx.set(dut.swap, 0)

            # Write different value to same addr in fill bank (B)
            ctx.set(dut.wr.addr, 0)
            ctx.set(dut.wr.data, 999)
            ctx.set(dut.wr.en, 1)
            await ctx.tick()
            ctx.set(dut.wr.en, 0)

            # Read from compute bank (A) — should still be 100
            ctx.set(dut.rd_addr, 0)
            await ctx.tick()
            assert ctx.get(dut.rd_data) == 100

            # Swap again: B=compute — now reads 999
            ctx.set(dut.swap, 1)
            await ctx.tick()
            ctx.set(dut.swap, 0)

            ctx.set(dut.rd_addr, 0)
            await ctx.tick()
            assert ctx.get(dut.rd_data) == 999

        dut = DoubleScratchpad(depth=16)
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_scratchpad_isolation.vcd"):
            sim.run()

    def test_concurrent_rw(self):
        """Read from compute bank while writing to fill bank simultaneously."""

        async def testbench(ctx):
            # Fill bank A
            for i in range(4):
                ctx.set(dut.wr.addr, i)
                ctx.set(dut.wr.data, (i + 1) * 10)
                ctx.set(dut.wr.en, 1)
                await ctx.tick()
            ctx.set(dut.wr.en, 0)

            # Swap: A=compute, B=fill
            ctx.set(dut.swap, 1)
            await ctx.tick()
            ctx.set(dut.swap, 0)

            # Simultaneously: read A (compute) and write B (fill)
            for i in range(4):
                ctx.set(dut.rd_addr, i)       # read from A
                ctx.set(dut.wr.addr, i)       # write to B
                ctx.set(dut.wr.data, (i + 1) * 100)
                ctx.set(dut.wr.en, 1)
                await ctx.tick()
                assert ctx.get(dut.rd_data) == (i + 1) * 10  # A unchanged
            ctx.set(dut.wr.en, 0)

            # Swap: B=compute — verify B has the new data
            ctx.set(dut.swap, 1)
            await ctx.tick()
            ctx.set(dut.swap, 0)

            for i in range(4):
                ctx.set(dut.rd_addr, i)
                await ctx.tick()
                assert ctx.get(dut.rd_data) == (i + 1) * 100

        dut = DoubleScratchpad(depth=16)
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_scratchpad_concurrent.vcd"):
            sim.run()
