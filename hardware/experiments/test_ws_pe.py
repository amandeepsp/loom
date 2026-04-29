from amaranth.sim import Simulator

from hardware.systolic.ws_pe import WeightStationaryPE


class TestWeightStationaryPE:
    def test_psum_flow(self):
        """psum_out ← psum_in + act_in * w_reg, registered one cycle."""

        async def testbench(ctx):
            ctx.set(dut.w_in, 4)
            ctx.set(dut.w_load, 1)
            await ctx.tick()
            ctx.set(dut.w_load, 0)

            # psum_out = 10 + 3*4 = 22
            ctx.set(dut.act_in, 3)
            ctx.set(dut.psum_in, 10)
            await ctx.tick()
            assert ctx.get(dut.psum_out) == 22

            # Pipeline clears: psum_out = 0 + 0*4 = 0
            ctx.set(dut.act_in, 0)
            ctx.set(dut.psum_in, 0)
            await ctx.tick()
            assert ctx.get(dut.psum_out) == 0

        dut = WeightStationaryPE()
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_pe_psum.vcd"):
            sim.run()

    def test_act_passthrough(self):
        """act_out is a registered copy of act_in."""

        async def testbench(ctx):
            ctx.set(dut.act_in, 7)
            await ctx.tick()
            assert ctx.get(dut.act_out) == 7

            ctx.set(dut.act_in, -3)
            await ctx.tick()
            assert ctx.get(dut.act_out) == -3

        dut = WeightStationaryPE()
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_pe_act.vcd"):
            sim.run()

    def test_weight_stationary(self):
        """Weight stays loaded across multiple cycles."""

        async def testbench(ctx):
            ctx.set(dut.w_in, 5)
            ctx.set(dut.w_load, 1)
            await ctx.tick()
            ctx.set(dut.w_load, 0)

            ctx.set(dut.act_in, 3)
            await ctx.tick()
            assert ctx.get(dut.psum_out) == 15  # 0 + 3*5

            ctx.set(dut.act_in, 7)
            await ctx.tick()
            assert ctx.get(dut.psum_out) == 35  # 0 + 7*5 (no accumulation, fresh each cycle)

        dut = WeightStationaryPE()
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_pe_stationary.vcd"):
            sim.run()
