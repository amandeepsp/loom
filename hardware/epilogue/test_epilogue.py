import pytest
from amaranth.sim import Simulator

from epilogue import Epilogue, PerChannelStore, PerChannelWriteSelect

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


def ref_srdhm(a, b):
    if a == INT32_MIN and b == INT32_MIN:
        return INT32_MAX
    return ((a * b) + (1 << 30)) >> 31


def ref_rdbpot(x, exponent):
    if exponent == 0:
        return x
    mask = (1 << exponent) - 1
    remainder = x & mask
    threshold = (mask >> 1) + ((x >> 31) & 1)
    return (x >> exponent) + (1 if remainder > threshold else 0)


def ref_epilogue(acc, bias, multiplier, shift, output_offset, act_min, act_max):
    """Full Python reference for one element through the pipeline."""
    x = acc + bias
    x = ref_srdhm(x, multiplier)
    x = ref_rdbpot(x, shift)
    x += output_offset
    return max(act_min, min(act_max, x))


def to_signed8(val):
    if val >= 128:
        val -= 256
    return val


class TestEpilogue:
    def test_single_result(self):
        """One result through the pipeline (first=1, last=1 same cycle)."""
        ACC = 100000
        BIAS = 50
        MULT = 0x40000000  # ~0.5 in fixed-point
        SHIFT = 2
        OFFSET = 3
        ACT_MIN, ACT_MAX = -128, 127

        expected = ref_epilogue(ACC, BIAS, MULT, SHIFT, OFFSET, ACT_MIN, ACT_MAX)

        async def testbench(ctx):
            # Configure per-layer params
            ctx.set(dut.output_offset, OFFSET)
            ctx.set(dut.activation_min, ACT_MIN)
            ctx.set(dut.activation_max, ACT_MAX)

            # Feed one result: first and last on same cycle
            ctx.set(dut.data_in, ACC)
            ctx.set(dut.bias, BIAS)
            ctx.set(dut.multiplier, MULT)
            ctx.set(dut.shift, SHIFT)
            ctx.set(dut.first_in, 1)
            ctx.set(dut.last_in, 1)
            await ctx.tick()

            # Deassert
            ctx.set(dut.first_in, 0)
            ctx.set(dut.last_in, 0)

            # Wait for pipeline (1 cycle SRDHM + 1 cycle done registration)
            await ctx.tick()
            assert ctx.get(dut.done) == 1

            # Read result
            ctx.set(dut.out_addr, 0)
            await ctx.delay(1e-7)
            result = to_signed8(ctx.get(dut.out_data))
            assert result == expected, f"got {result}, expected {expected}"

        dut = Epilogue(num_results=4)
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_epi_single.vcd"):
            sim.run()

    def test_four_results(self):
        """Four results with different per-channel params."""
        accs = [10000, -20000, 50000, -80000]
        biases = [10, -5, 20, 0]
        mults = [0x50000000, 0x60000000, 0x40000000, 0x70000000]
        shifts = [3, 2, 4, 1]
        OFFSET = -1
        ACT_MIN, ACT_MAX = -128, 127

        expected = [
            ref_epilogue(a, b, m, s, OFFSET, ACT_MIN, ACT_MAX)
            for a, b, m, s in zip(accs, biases, mults, shifts)
        ]

        async def testbench(ctx):
            ctx.set(dut.output_offset, OFFSET)
            ctx.set(dut.activation_min, ACT_MIN)
            ctx.set(dut.activation_max, ACT_MAX)

            # Feed 4 results with first/last markers
            for i in range(4):
                ctx.set(dut.data_in, accs[i])
                ctx.set(dut.bias, biases[i])
                ctx.set(dut.multiplier, mults[i])
                ctx.set(dut.shift, shifts[i])
                ctx.set(dut.first_in, int(i == 0))
                ctx.set(dut.last_in, int(i == 3))
                await ctx.tick()

            # Deassert
            ctx.set(dut.first_in, 0)
            ctx.set(dut.last_in, 0)

            # Wait for done (pipeline drains last result)
            await ctx.tick()
            assert ctx.get(dut.done) == 1

            # Read all results
            for i in range(4):
                ctx.set(dut.out_addr, i)
                await ctx.delay(1e-7)
                result = to_signed8(ctx.get(dut.out_data))
                assert result == expected[i], \
                    f"result[{i}]: got {result}, expected {expected[i]}"

        dut = Epilogue(num_results=4)
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_epi_four.vcd"):
            sim.run()

    def test_clamp_to_range(self):
        """Results outside [min, max] are clamped."""
        # Large positive accumulator should clamp to activation_max
        ACC_HIGH = 500000000
        # Large negative should clamp to activation_min
        ACC_LOW = -500000000
        MULT = 0x7FFFFFFF  # max multiplier
        SHIFT = 2
        OFFSET = 0
        ACT_MIN, ACT_MAX = -100, 100

        expected_high = ref_epilogue(ACC_HIGH, 0, MULT, SHIFT, OFFSET, ACT_MIN, ACT_MAX)
        expected_low = ref_epilogue(ACC_LOW, 0, MULT, SHIFT, OFFSET, ACT_MIN, ACT_MAX)
        assert expected_high == ACT_MAX
        assert expected_low == ACT_MIN

        async def testbench(ctx):
            ctx.set(dut.output_offset, OFFSET)
            ctx.set(dut.activation_min, ACT_MIN)
            ctx.set(dut.activation_max, ACT_MAX)
            ctx.set(dut.multiplier, MULT)
            ctx.set(dut.shift, SHIFT)
            ctx.set(dut.bias, 0)

            # High
            ctx.set(dut.data_in, ACC_HIGH)
            ctx.set(dut.first_in, 1)
            ctx.set(dut.last_in, 0)
            await ctx.tick()

            # Low
            ctx.set(dut.data_in, ACC_LOW)
            ctx.set(dut.first_in, 0)
            ctx.set(dut.last_in, 1)
            await ctx.tick()

            ctx.set(dut.last_in, 0)
            await ctx.tick()
            assert ctx.get(dut.done) == 1

            ctx.set(dut.out_addr, 0)
            await ctx.delay(1e-7)
            assert to_signed8(ctx.get(dut.out_data)) == ACT_MAX

            ctx.set(dut.out_addr, 1)
            await ctx.delay(1e-7)
            assert to_signed8(ctx.get(dut.out_data)) == ACT_MIN

        dut = Epilogue(num_results=4)
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_epi_clamp.vcd"):
            sim.run()

    def test_done_pulses_once(self):
        """done asserts for exactly one cycle after last result exits pipeline."""

        async def testbench(ctx):
            ctx.set(dut.output_offset, 0)
            ctx.set(dut.activation_min, -128)
            ctx.set(dut.activation_max, 127)
            ctx.set(dut.multiplier, 0x40000000)
            ctx.set(dut.shift, 2)
            ctx.set(dut.bias, 0)

            # Single result
            ctx.set(dut.data_in, 1000)
            ctx.set(dut.first_in, 1)
            ctx.set(dut.last_in, 1)
            await ctx.tick()
            ctx.set(dut.first_in, 0)
            ctx.set(dut.last_in, 0)

            # Pipeline draining
            assert ctx.get(dut.done) == 0
            await ctx.tick()
            assert ctx.get(dut.done) == 1  # fires here

            await ctx.tick()
            assert ctx.get(dut.done) == 0  # cleared next cycle

        dut = Epilogue(num_results=4)
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_epi_done.vcd"):
            sim.run()

    def test_bias_applied(self):
        """Bias is added before SRDHM."""
        # Zero accumulator + non-zero bias should produce a non-zero result
        BIAS = 10000
        MULT = 0x40000000
        SHIFT = 2

        expected = ref_epilogue(0, BIAS, MULT, SHIFT, 0, -128, 127)
        assert expected != 0  # sanity: bias actually has effect

        async def testbench(ctx):
            ctx.set(dut.output_offset, 0)
            ctx.set(dut.activation_min, -128)
            ctx.set(dut.activation_max, 127)
            ctx.set(dut.multiplier, MULT)
            ctx.set(dut.shift, SHIFT)
            ctx.set(dut.bias, BIAS)

            ctx.set(dut.data_in, 0)
            ctx.set(dut.first_in, 1)
            ctx.set(dut.last_in, 1)
            await ctx.tick()
            ctx.set(dut.first_in, 0)
            ctx.set(dut.last_in, 0)

            await ctx.tick()
            assert ctx.get(dut.done) == 1

            ctx.set(dut.out_addr, 0)
            await ctx.delay(1e-7)
            assert to_signed8(ctx.get(dut.out_data)) == expected

        dut = Epilogue(num_results=4)
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_epi_bias.vcd"):
            sim.run()


class TestPerChannelStore:
    def test_write_and_read(self):
        """Write bias/mult/shift for two channels, read them back."""
        DEPTH = 4

        async def testbench(ctx):
            # Write channel 0: bias=100, mult=0x40000000, shift=3
            for sel, val in [
                (PerChannelWriteSelect.BIAS, 100),
                (PerChannelWriteSelect.MULT, 0x40000000),
                (PerChannelWriteSelect.SHIFT, 3),
            ]:
                ctx.set(dut.wr_addr, 0)
                ctx.set(dut.wr_sel, sel)
                ctx.set(dut.wr_data, val)
                ctx.set(dut.wr_en, 1)
                await ctx.tick()

            # Write channel 1: bias=-50, mult=0x60000000, shift=5
            for sel, val in [
                (PerChannelWriteSelect.BIAS, -50),
                (PerChannelWriteSelect.MULT, 0x60000000),
                (PerChannelWriteSelect.SHIFT, 5),
            ]:
                ctx.set(dut.wr_addr, 1)
                ctx.set(dut.wr_sel, sel)
                ctx.set(dut.wr_data, val)
                ctx.set(dut.wr_en, 1)
                await ctx.tick()

            ctx.set(dut.wr_en, 0)

            # Read channel 0 (sync read: set addr, wait 1 cycle)
            ctx.set(dut.rd_addr, 0)
            await ctx.tick()
            assert ctx.get(dut.bias) == 100
            assert ctx.get(dut.multiplier) == 0x40000000
            assert ctx.get(dut.shift) == 3

            # Read channel 1
            ctx.set(dut.rd_addr, 1)
            await ctx.tick()
            assert ctx.get(dut.bias) == -50
            assert ctx.get(dut.multiplier) == 0x60000000
            assert ctx.get(dut.shift) == 5

        dut = PerChannelStore(depth=DEPTH)
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_param_store.vcd"):
            sim.run()

    def test_sequential_readout(self):
        """Simulate sequencer drain: walk rd_addr 0..N-1, data arrives pipelined."""
        DEPTH = 4
        biases = [10, -20, 30, -40]

        async def testbench(ctx):
            # Load biases
            for i, b in enumerate(biases):
                ctx.set(dut.wr_addr, i)
                ctx.set(dut.wr_sel, PerChannelWriteSelect.BIAS)
                ctx.set(dut.wr_data, b)
                ctx.set(dut.wr_en, 1)
                await ctx.tick()
            ctx.set(dut.wr_en, 0)

            # Sequential read — addr presented cycle N, data valid cycle N+1
            for i in range(DEPTH):
                ctx.set(dut.rd_addr, i)
                await ctx.tick()
                assert ctx.get(dut.bias) == biases[i], \
                    f"chan {i}: got {ctx.get(dut.bias)}, expected {biases[i]}"

        dut = PerChannelStore(depth=DEPTH)
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_param_seq.vcd"):
            sim.run()

    def test_write_does_not_bleed(self):
        """Writing one field doesn't corrupt other fields at the same address."""
        DEPTH = 4

        async def testbench(ctx):
            # Write all three fields for channel 0
            ctx.set(dut.wr_addr, 0)
            ctx.set(dut.wr_sel, PerChannelWriteSelect.BIAS)
            ctx.set(dut.wr_data, 42)
            ctx.set(dut.wr_en, 1)
            await ctx.tick()

            ctx.set(dut.wr_sel, PerChannelWriteSelect.MULT)
            ctx.set(dut.wr_data, 0x12345678)
            await ctx.tick()

            ctx.set(dut.wr_sel, PerChannelWriteSelect.SHIFT)
            ctx.set(dut.wr_data, 7)
            await ctx.tick()

            # Overwrite just bias — mult and shift should survive
            ctx.set(dut.wr_sel, PerChannelWriteSelect.BIAS)
            ctx.set(dut.wr_data, 99)
            await ctx.tick()
            ctx.set(dut.wr_en, 0)

            ctx.set(dut.rd_addr, 0)
            await ctx.tick()
            assert ctx.get(dut.bias) == 99
            assert ctx.get(dut.multiplier) == 0x12345678
            assert ctx.get(dut.shift) == 7

        dut = PerChannelStore(depth=DEPTH)
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_param_bleed.vcd"):
            sim.run()
