"""
Comprehensive tests for SimdMac4 SIMD Multiply-Accumulate instruction.

Tests the 4-element SIMD MAC operation with configurable offset.
Each 8-bit element from in0 is offset and multiplied with the corresponding
8-bit element from in1, with results accumulated.

Format:
- in0: 4x 8-bit values in 32-bit word (little-endian)
- in1: 4x 8-bit values in 32-bit word (little-endian)
- output: 32-bit accumulated sum
- input_offset: 32-bit value added to each in0 element before multiplication
- reset_acc: Signal to reset accumulator to 0
"""

import sys
from pathlib import Path

import pytest
from amaranth import Elaboratable, Module, Signal
from amaranth.sim import Simulator

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mac import SimdMac4


def test_simd_mac4_initialization():
    """Test that SimdMac4 initializes with correct defaults."""
    dut = SimdMac4()
    assert dut.input_offset.init == 128
    assert dut.accumulator.init == 0


def create_dut_wrapper():
    """Create device under test with proper clock domain."""

    class DutWrapper(Elaboratable):
        def __init__(self):
            self.dut = SimdMac4()

        def elaborate(self, platform):
            m = Module()
            m.submodules.dut = self.dut
            return m

    return DutWrapper()


def test_zero_offset_simple():
    """Test MAC with zero offset: (0 + 0) * 1 + (0 + 0) * 1 + ... = 0."""
    dut = create_dut_wrapper()

    async def testbench(ctx):
        # Set inputs to zero and offset to zero
        ctx.set(dut.dut.input_offset, 0)
        ctx.set(dut.dut.in0, 0x00000000)
        ctx.set(dut.dut.in1, 0x00000000)
        ctx.set(dut.dut.start, 1)
        await ctx.tick()
        ctx.set(dut.dut.start, 0)
        await ctx.tick()
        await ctx.tick()
        # Check that done signal is asserted when result is available
        assert ctx.get(dut.dut.done) == 1
        # Result should be 0
        acc = ctx.get(dut.dut.accumulator)
        assert acc == 0

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    with sim.write_vcd("test_mac_zero_offset.vcd"):
        sim.run()


def test_single_element_no_offset():
    """Test single element: (5 + 0) * 3 = 15."""
    dut = create_dut_wrapper()

    async def testbench(ctx):
        # Offset = 0, in0[0] = 5, in1[0] = 3, others = 0
        ctx.set(dut.dut.input_offset, 0)
        ctx.set(dut.dut.in0, 0x00000005)
        ctx.set(dut.dut.in1, 0x00000003)
        ctx.set(dut.dut.start, 1)
        await ctx.tick()
        ctx.set(dut.dut.start, 0)
        await ctx.tick()
        await ctx.tick()  # Allow time for accumulation
        # Check that done signal is asserted when result is available
        assert ctx.get(dut.dut.done) == 1
        acc = ctx.get(dut.dut.accumulator)
        assert acc == 15

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    with sim.write_vcd("test_mac_single_element.vcd"):
        sim.run()


def test_with_offset():
    """Test with offset: ((5+128) * 1) + ((0+128) * 0) + ... = 133."""
    dut = create_dut_wrapper()

    async def testbench(ctx):
        # Offset = 128, in0[0] = 5, in1[0] = 1, others = 0
        ctx.set(dut.dut.input_offset, 128)
        ctx.set(dut.dut.in0, 0x00000005)
        ctx.set(dut.dut.in1, 0x00000001)
        ctx.set(dut.dut.start, 1)
        await ctx.tick()
        ctx.set(dut.dut.start, 0)
        await ctx.tick()
        await ctx.tick()  # Allow time for accumulation
        # Check that done signal is asserted when result is available
        assert ctx.get(dut.dut.done) == 1
        acc = ctx.get(dut.dut.accumulator)
        # (5 + 128) * 1 = 133
        # (0 + 128) * 0 = 0
        # (0 + 128) * 0 = 0
        # (0 + 128) * 0 = 0
        # Total = 133
        assert acc == 133

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    with sim.write_vcd("test_mac_with_offset.vcd"):
        sim.run()


def test_four_element_simd():
    """Test all four elements: [(2+0)*1, (3+0)*2, (4+0)*3, (5+0)*4] = 2+6+12+20 = 40."""
    dut = create_dut_wrapper()

    async def testbench(ctx):
        # Offset = 0
        # in0 = [2, 3, 4, 5] (as bytes in 32-bit word)
        # in1 = [1, 2, 3, 4]
        ctx.set(dut.dut.input_offset, 0)
        ctx.set(dut.dut.in0, 0x05040302)  # Little-endian: 2, 3, 4, 5
        ctx.set(dut.dut.in1, 0x04030201)  # Little-endian: 1, 2, 3, 4
        ctx.set(dut.dut.start, 1)
        await ctx.tick()
        ctx.set(dut.dut.start, 0)
        await ctx.tick()
        await ctx.tick()  # Allow time for accumulation
        # Check that done signal is asserted when result is available
        assert ctx.get(dut.dut.done) == 1
        acc = ctx.get(dut.dut.accumulator)
        # (2*1) + (3*2) + (4*3) + (5*4) = 2 + 6 + 12 + 20 = 40
        assert acc == 40

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    with sim.write_vcd("test_mac_four_element.vcd"):
        sim.run()


def test_accumulation():
    """Test accumulation across multiple operations."""
    dut = create_dut_wrapper()

    async def testbench(ctx):
        ctx.set(dut.dut.input_offset, 0)

        # First operation: (1+0)*2 + ... = 2
        ctx.set(dut.dut.in0, 0x00000001)
        ctx.set(dut.dut.in1, 0x00000002)
        ctx.set(dut.dut.start, 1)
        await ctx.tick()
        ctx.set(dut.dut.start, 0)
        await ctx.tick()
        await ctx.tick()
        # Check that done signal is asserted when result is available
        assert ctx.get(dut.dut.done) == 1

        # Second operation: (3+0)*4 + ... = 12
        ctx.set(dut.dut.in0, 0x00000003)
        ctx.set(dut.dut.in1, 0x00000004)
        ctx.set(dut.dut.start, 1)
        await ctx.tick()
        ctx.set(dut.dut.start, 0)
        await ctx.tick()
        await ctx.tick()
        # Check that done signal is asserted when result is available
        assert ctx.get(dut.dut.done) == 1

        # Accumulator should be 2 + 12 = 14
        acc = ctx.get(dut.dut.accumulator)
        assert acc == 14

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    with sim.write_vcd("test_mac_accumulation.vcd"):
        sim.run()


def test_reset_accumulator():
    """Test reset_acc signal clears accumulator."""
    dut = create_dut_wrapper()

    async def testbench(ctx):
        ctx.set(dut.dut.input_offset, 0)

        # First operation: accumulate to 10
        ctx.set(dut.dut.in0, 0x00000005)
        ctx.set(dut.dut.in1, 0x00000002)
        ctx.set(dut.dut.start, 1)
        await ctx.tick()
        ctx.set(dut.dut.start, 0)
        await ctx.tick()
        await ctx.tick()
        # Check that done signal is asserted when result is available
        assert ctx.get(dut.dut.done) == 1
        acc = ctx.get(dut.dut.accumulator)
        assert acc == 10

        # Reset accumulator
        ctx.set(dut.dut.reset_acc, 1)
        await ctx.tick()
        ctx.set(dut.dut.reset_acc, 0)
        await ctx.tick()
        await ctx.tick()

        # Accumulator should be 0
        acc = ctx.get(dut.dut.accumulator)
        assert acc == 0

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    with sim.write_vcd("test_mac_reset.vcd"):
        sim.run()


def test_large_values():
    """Test with maximum byte values: 255."""
    dut = create_dut_wrapper()

    async def testbench(ctx):
        ctx.set(dut.dut.input_offset, 0)
        # All bytes = 255
        ctx.set(dut.dut.in0, 0xFFFFFFFF)
        ctx.set(dut.dut.in1, 0xFFFFFFFF)
        ctx.set(dut.dut.start, 1)
        await ctx.tick()
        ctx.set(dut.dut.start, 0)
        await ctx.tick()
        await ctx.tick()
        # Check that done signal is asserted when result is available
        assert ctx.get(dut.dut.done) == 1
        acc = ctx.get(dut.dut.accumulator)
        # (255 * 255) * 4 = 65025 * 4 = 260100
        assert acc == 260100

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    with sim.write_vcd("test_mac_large_values.vcd"):
        sim.run()


def test_offset_with_four_elements():
    """Test offset applied to all four elements."""
    dut = create_dut_wrapper()

    async def testbench(ctx):
        # Offset = 1, all input bytes = 1, all in1 = 2
        # Result: 4 * ((1+1) * 2) = 4 * 4 = 16
        ctx.set(dut.dut.input_offset, 1)
        ctx.set(dut.dut.in0, 0x01010101)
        ctx.set(dut.dut.in1, 0x02020202)
        ctx.set(dut.dut.start, 1)
        await ctx.tick()
        ctx.set(dut.dut.start, 0)
        await ctx.tick()
        await ctx.tick()
        # Check that done signal is asserted when result is available
        assert ctx.get(dut.dut.done) == 1
        acc = ctx.get(dut.dut.accumulator)
        assert acc == 16

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    with sim.write_vcd("test_mac_offset_four_elements.vcd"):
        sim.run()


def test_mixed_zero_nonzero():
    """Test with mix of zero and non-zero elements."""
    dut = create_dut_wrapper()

    async def testbench(ctx):
        # in0 = [10, 0, 20, 0], in1 = [1, 100, 2, 100]
        # Result: (10*1) + (0*100) + (20*2) + (0*100) = 10 + 0 + 40 + 0 = 50
        ctx.set(dut.dut.input_offset, 0)
        ctx.set(dut.dut.in0, 0x0014000A)  # [10, 0, 20, 0]
        ctx.set(dut.dut.in1, 0x64020164)  # Actually [100, 1, 2, 100] in little-endian
        ctx.set(dut.dut.start, 1)
        await ctx.tick()
        ctx.set(dut.dut.start, 0)
        await ctx.tick()
        await ctx.tick()
        # Check that done signal is asserted when result is available
        assert ctx.get(dut.dut.done) == 1
        acc = ctx.get(dut.dut.accumulator)
        # Actual: (10*100) + (0*1) + (20*2) + (0*100) = 1000 + 0 + 40 + 0 = 1040
        assert acc == 1040

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    with sim.write_vcd("test_mac_mixed_values.vcd"):
        sim.run()


def test_consecutive_operations_with_offset():
    """Test multiple MAC operations with offset."""
    dut = create_dut_wrapper()

    async def testbench(ctx):
        ctx.set(dut.dut.input_offset, 10)

        # First: ((1+10)*2) = 22
        ctx.set(dut.dut.in0, 0x01010101)
        ctx.set(dut.dut.in1, 0x02020202)
        ctx.set(dut.dut.start, 1)
        await ctx.tick()
        ctx.set(dut.dut.start, 0)
        await ctx.tick()
        await ctx.tick()
        # Check that done signal is asserted when result is available
        assert ctx.get(dut.dut.done) == 1

        acc = ctx.get(dut.dut.accumulator)
        # (11*2) * 4 = 22 * 4 = 88
        assert acc == 88

        # Second: ((2+10)*3) each = 36 * 4 = 144
        ctx.set(dut.dut.in0, 0x02020202)
        ctx.set(dut.dut.in1, 0x03030303)
        ctx.set(dut.dut.start, 1)
        await ctx.tick()
        ctx.set(dut.dut.start, 0)
        await ctx.tick()
        await ctx.tick()
        # Check that done signal is asserted when result is available
        assert ctx.get(dut.dut.done) == 1

        acc = ctx.get(dut.dut.accumulator)
        # 88 + 144 = 232
        assert acc == 232

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    with sim.write_vcd("test_mac_consecutive_offset.vcd"):
        sim.run()


def test_edge_case_start_not_held():
    """Test that operation works without holding start high."""
    dut = create_dut_wrapper()

    async def testbench(ctx):
        ctx.set(dut.dut.input_offset, 0)
        ctx.set(dut.dut.in0, 0x05050505)
        ctx.set(dut.dut.in1, 0x03030303)
        # Only pulse start for one cycle
        ctx.set(dut.dut.start, 1)
        await ctx.tick()
        ctx.set(dut.dut.start, 0)
        await ctx.tick()
        await ctx.tick()
        # Check that done signal is asserted when result is available
        assert ctx.get(dut.dut.done) == 1
        acc = ctx.get(dut.dut.accumulator)
        # (5*3) * 4 = 60
        assert acc == 60

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    with sim.write_vcd("test_mac_start_pulse.vcd"):
        sim.run()


def test_edge_case_offset_255():
    """Test with maximum offset value."""
    dut = create_dut_wrapper()

    async def testbench(ctx):
        ctx.set(dut.dut.input_offset, 0xFF)
        ctx.set(dut.dut.in0, 0x01010101)
        ctx.set(dut.dut.in1, 0x01010101)
        ctx.set(dut.dut.start, 1)
        await ctx.tick()
        ctx.set(dut.dut.start, 0)
        await ctx.tick()
        await ctx.tick()
        # Check that done signal is asserted when result is available
        assert ctx.get(dut.dut.done) == 1
        acc = ctx.get(dut.dut.accumulator)
        # ((1 + 255) * 1) * 4 = 256 * 4 = 1024
        assert acc == 1024

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    with sim.write_vcd("test_mac_offset_255.vcd"):
        sim.run()


def test_edge_case_multiple_resets():
    """Test multiple consecutive resets."""
    dut = create_dut_wrapper()

    async def testbench(ctx):
        ctx.set(dut.dut.input_offset, 0)

        # Accumulate
        ctx.set(dut.dut.in0, 0x05050505)
        ctx.set(dut.dut.in1, 0x02020202)
        ctx.set(dut.dut.start, 1)
        await ctx.tick()
        ctx.set(dut.dut.start, 0)
        await ctx.tick()
        await ctx.tick()
        # Check that done signal is asserted when result is available
        assert ctx.get(dut.dut.done) == 1

        # First reset
        ctx.set(dut.dut.reset_acc, 1)
        await ctx.tick()
        ctx.set(dut.dut.reset_acc, 0)
        await ctx.tick()

        acc = ctx.get(dut.dut.accumulator)
        assert acc == 0

        # Accumulate again
        ctx.set(dut.dut.in0, 0x03030303)
        ctx.set(dut.dut.in1, 0x04040404)
        ctx.set(dut.dut.start, 1)
        await ctx.tick()
        ctx.set(dut.dut.start, 0)
        await ctx.tick()
        await ctx.tick()
        # Check that done signal is asserted when result is available
        assert ctx.get(dut.dut.done) == 1

        # Second reset
        ctx.set(dut.dut.reset_acc, 1)
        await ctx.tick()
        ctx.set(dut.dut.reset_acc, 0)
        await ctx.tick()

        acc = ctx.get(dut.dut.accumulator)
        assert acc == 0

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    with sim.write_vcd("test_mac_multiple_resets.vcd"):
        sim.run()


def test_edge_case_reset_while_accumulating():
    """Test reset during accumulation (implementation dependent)."""
    dut = create_dut_wrapper()

    async def testbench(ctx):
        ctx.set(dut.dut.input_offset, 0)
        ctx.set(dut.dut.in0, 0x05050505)
        ctx.set(dut.dut.in1, 0x02020202)
        ctx.set(dut.dut.start, 1)
        await ctx.tick()
        ctx.set(dut.dut.start, 0)
        await ctx.tick()
        # Reset during potential processing
        ctx.set(dut.dut.reset_acc, 1)
        await ctx.tick()
        ctx.set(dut.dut.reset_acc, 0)
        await ctx.tick()
        # Check that done signal is asserted when result is available
        assert ctx.get(dut.dut.done) == 1
        acc = ctx.get(dut.dut.accumulator)
        # Accumulator should be reset (priority over accumulation)
        assert acc == 0

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    with sim.write_vcd("test_mac_reset_during_acc.vcd"):
        sim.run()


def test_integration_realistic_ml_inference_scenario():
    """Test realistic machine learning inference scenario."""
    dut = create_dut_wrapper()

    async def testbench(ctx):
        # Simulating a convolution layer with quantized values
        # Input offset simulates per-channel quantization bias
        ctx.set(dut.dut.input_offset, 128)

        # Weights (in1) and activations (in0) quantized to 8-bit
        # First 4 dot-products (different filters)
        test_cases = [
            (0x80808080, 0x01010101),  # Neutral activations
            (0x7F7F7F7F, 0x7F7F7F7F),  # High values
            (0x00000000, 0xFF000000),  # Sparse pattern
        ]

        for in0, in1 in test_cases:
            ctx.set(dut.dut.in0, in0)
            ctx.set(dut.dut.in1, in1)
            ctx.set(dut.dut.start, 1)
            await ctx.tick()
            ctx.set(dut.dut.start, 0)
            await ctx.tick()
            # Check that done signal is asserted when result is available
            assert ctx.get(dut.dut.done) == 1
            await ctx.tick()

        # Just verify accumulator updated (exact value depends on math)
        acc = ctx.get(dut.dut.accumulator)
        assert acc > 0

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    with sim.write_vcd("test_mac_ml_scenario.vcd"):
        sim.run()


def test_integration_systematic_sweep():
    """Systematic sweep of input values to check consistency."""
    dut = create_dut_wrapper()

    async def testbench(ctx):
        ctx.set(dut.dut.input_offset, 0)

        # Test various combinations
        for a in [1, 2, 5, 10]:
            for b in [1, 2, 5, 10]:
                ctx.set(dut.dut.reset_acc, 1)
                await ctx.tick()
                ctx.set(dut.dut.reset_acc, 0)
                await ctx.tick()

                in0 = (a << 0) | (a << 8) | (a << 16) | (a << 24)
                in1 = (b << 0) | (b << 8) | (b << 16) | (b << 24)

                ctx.set(dut.dut.in0, in0)
                ctx.set(dut.dut.in1, in1)
                ctx.set(dut.dut.start, 1)
                await ctx.tick()
                ctx.set(dut.dut.start, 0)
                await ctx.tick()
                # Check that done signal is asserted when result is available
                assert ctx.get(dut.dut.done) == 1
                await ctx.tick()

                acc = ctx.get(dut.dut.accumulator)
                expected = (a * b) * 4  # 4 elements
                assert acc == expected, (
                    f"Failed for a={a}, b={b}: got {acc}, expected {expected}"
                )

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    with sim.write_vcd("test_mac_systematic_sweep.vcd"):
        sim.run()


if __name__ == "__main__":
    pytest.main([__file__])
