from amaranth.sim import Simulator

from hardware.systolic.skew import SkewBuffer


class TestSkewBuffer:
    def test_4wide_skew(self):
        """
        Input skew: row i delayed by i cycles.
        Feed [10, 20, 30, 40] simultaneously → outputs stagger out.
          out_0: immediate (combinational)
          out_1: 1 cycle later
          out_2: 2 cycles later
          out_3: 3 cycles later
        """

        async def testbench(ctx):
            ctx.set(dut.in_0, 10)
            ctx.set(dut.in_1, 20)
            ctx.set(dut.in_2, 30)
            ctx.set(dut.in_3, 40)

            # Row 0: combinational, no delay
            await ctx.delay(1e-7)
            assert ctx.get(dut.out_0) == 10

            await ctx.tick()
            # Clear inputs to observe the delay chain
            ctx.set(dut.in_0, 0)
            ctx.set(dut.in_1, 0)
            ctx.set(dut.in_2, 0)
            ctx.set(dut.in_3, 0)
            assert ctx.get(dut.out_0) == 0   # combinational, follows input
            assert ctx.get(dut.out_1) == 20   # 1 cycle delay

            await ctx.tick()
            assert ctx.get(dut.out_2) == 30   # 2 cycle delay

            await ctx.tick()
            assert ctx.get(dut.out_3) == 40   # 3 cycle delay

        dut = SkewBuffer(4, 8)
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_skew_4.vcd"):
            sim.run()

    def test_4wide_deskew(self):
        """
        Output de-skew (reverse=True): row i delayed by (depth-1-i) cycles.
        Feed staggered inputs (simulating skewed array output) → aligned output.
          in_0 at tick 0 (3 delay), in_1 at tick 1 (2 delay),
          in_2 at tick 2 (1 delay), in_3 at tick 3 (0 delay, comb).
          All outputs align after tick 2 / at tick 3.
        """

        async def testbench(ctx):
            # Staggered input: one row per cycle (simulates array output skew)
            ctx.set(dut.in_0, 10)
            await ctx.tick()
            ctx.set(dut.in_0, 0)
            ctx.set(dut.in_1, 20)
            await ctx.tick()
            ctx.set(dut.in_1, 0)
            ctx.set(dut.in_2, 30)
            await ctx.tick()
            ctx.set(dut.in_2, 0)
            ctx.set(dut.in_3, 40)

            # After 3 ticks: registered outputs aligned, row 3 combinational
            await ctx.delay(1e-7)
            assert ctx.get(dut.out_0) == 10   # 3 delays from tick 0
            assert ctx.get(dut.out_1) == 20   # 2 delays from tick 1
            assert ctx.get(dut.out_2) == 30   # 1 delay from tick 2
            assert ctx.get(dut.out_3) == 40   # combinational

        dut = SkewBuffer(4, 8, reverse=True)
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_deskew_4.vcd"):
            sim.run()

    def test_2wide_skew(self):
        """Minimal case: 1 register total."""

        async def testbench(ctx):
            ctx.set(dut.in_0, 5)
            ctx.set(dut.in_1, 7)

            await ctx.delay(1e-7)
            assert ctx.get(dut.out_0) == 5  # immediate

            await ctx.tick()
            ctx.set(dut.in_0, 0)
            ctx.set(dut.in_1, 0)
            assert ctx.get(dut.out_1) == 7  # 1 cycle delay

        dut = SkewBuffer(2, 8)
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_skew_2.vcd"):
            sim.run()

    def test_streaming(self):
        """
        Continuous stream through a 3-wide skew buffer.
        Feed a new vector each cycle → outputs form a diagonal wavefront.
        """

        async def testbench(ctx):
            # Cycle 0: feed [1, 2, 3]
            ctx.set(dut.in_0, 1)
            ctx.set(dut.in_1, 2)
            ctx.set(dut.in_2, 3)
            await ctx.delay(1e-7)
            assert ctx.get(dut.out_0) == 1  # immediate

            await ctx.tick()
            # Cycle 1: feed [4, 5, 6]
            ctx.set(dut.in_0, 4)
            ctx.set(dut.in_1, 5)
            ctx.set(dut.in_2, 6)
            await ctx.delay(1e-7)
            assert ctx.get(dut.out_0) == 4  # immediate
            assert ctx.get(dut.out_1) == 2  # from cycle 0, 1 delay

            await ctx.tick()
            # Cycle 2: feed [7, 8, 9]
            ctx.set(dut.in_0, 7)
            ctx.set(dut.in_1, 8)
            ctx.set(dut.in_2, 9)
            await ctx.delay(1e-7)
            assert ctx.get(dut.out_0) == 7  # immediate
            assert ctx.get(dut.out_1) == 5  # from cycle 1, 1 delay
            assert ctx.get(dut.out_2) == 3  # from cycle 0, 2 delay

        dut = SkewBuffer(3, 8)
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_skew_stream.vcd"):
            sim.run()
