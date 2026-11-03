from amaranth.sim import Simulator

from hardware.systolic.os_pe import OutputStationaryPE


class TestOutputStationaryPE:
    def test_accumulate(self):
        """prod <- act*w, then acc += prod: each product lands one cycle later."""

        async def testbench(ctx):
            ctx.set(dut.act_in, 3)
            ctx.set(dut.w_in, 4)
            await ctx.tick()
            assert ctx.get(dut.psum_out) == 0  # product registered, not yet added

            ctx.set(dut.act_in, 2)
            ctx.set(dut.w_in, 5)
            await ctx.tick()
            assert ctx.get(dut.psum_out) == 12  # 3*4

            ctx.set(dut.act_in, 0)
            ctx.set(dut.w_in, 0)
            await ctx.tick()
            assert ctx.get(dut.psum_out) == 22  # 12 + 2*5

        dut = OutputStationaryPE()
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_os_pe.vcd"):
            sim.run()

    def test_registered_passthrough(self):
        """act_out and w_out are registered (1-cycle delay)."""

        async def testbench(ctx):
            ctx.set(dut.act_in, 7)
            ctx.set(dut.w_in, -3)
            await ctx.tick()
            assert ctx.get(dut.act_out) == 7
            assert ctx.get(dut.w_out) == -3

            ctx.set(dut.act_in, 0)
            ctx.set(dut.w_in, 0)
            await ctx.tick()
            assert ctx.get(dut.act_out) == 0
            assert ctx.get(dut.w_out) == 0

        dut = OutputStationaryPE()
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_os_pe_pass.vcd"):
            sim.run()

    def test_psum_load(self):
        """psum_load resets the accumulator to 0 (idle inputs keep prod at 0)."""

        async def testbench(ctx):
            ctx.set(dut.act_in, 3)
            ctx.set(dut.w_in, 4)
            await ctx.tick()
            ctx.set(dut.act_in, 0)
            ctx.set(dut.w_in, 0)
            await ctx.tick()
            assert ctx.get(dut.psum_out) == 12

            await ctx.tick()  # prod drains to 0
            ctx.set(dut.psum_load, 1)
            await ctx.tick()
            ctx.set(dut.psum_load, 0)
            assert ctx.get(dut.psum_out) == 0

            ctx.set(dut.act_in, 5)
            ctx.set(dut.w_in, 2)
            await ctx.tick()
            ctx.set(dut.act_in, 0)
            ctx.set(dut.w_in, 0)
            await ctx.tick()
            assert ctx.get(dut.psum_out) == 10  # fresh: 5*2

        dut = OutputStationaryPE()
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_os_pe_load.vcd"):
            sim.run()
