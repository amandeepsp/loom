"""Tests for SimdMac4 instruction.

SimdMac4: for each byte lane i in [0..3]:
    output = base + Σ(in0[i] + 128) * in1[i]

where base = accumulator (funct7[0]=0) or 0 (funct7[0]=1, reset).
"""

import sys
from pathlib import Path

import pytest
from amaranth import Elaboratable, Module
from amaranth.sim import Simulator

sys.path.insert(0, str(Path(__file__).parent))
from mac import SimdMac4


class DutWrapper(Elaboratable):
    def __init__(self):
        self.dut = SimdMac4()

    def elaborate(self, platform):
        m = Module()
        m.submodules.dut = self.dut
        return m


def run_sim(testbench, vcd_name="test.vcd"):
    wrapper = DutWrapper()
    sim = Simulator(wrapper)
    sim.add_clock(1e-6)

    async def _tb(ctx):
        await testbench(ctx, wrapper.dut)

    sim.add_testbench(_tb)
    with sim.write_vcd(vcd_name):
        sim.run()


def mac4_expected(in0: int, in1: int, acc: int = 0) -> int:
    """Compute expected MAC4 result: acc + Σ(in0[i]+128)*in1[i]."""
    total = acc
    for i in range(4):
        a = (in0 >> (i * 8)) & 0xFF
        b = (in1 >> (i * 8)) & 0xFF
        total += (a + 128) * b
    return total


def test_zeros():
    """All zeros: Σ(0+128)*0 = 0."""

    async def tb(ctx, dut):
        ctx.set(dut.funct7, 1)  # reset
        ctx.set(dut.in0, 0)
        ctx.set(dut.in1, 0)
        ctx.set(dut.start, 1)
        await ctx.tick()
        ctx.set(dut.start, 0)
        await ctx.tick()
        assert ctx.get(dut.accumulator) == 0

    run_sim(tb, "test_zeros.vcd")


def test_offset_only():
    """in0=0, in1=1s: Σ(0+128)*1 = 512."""

    async def tb(ctx, dut):
        ctx.set(dut.funct7, 1)
        ctx.set(dut.in0, 0x00000000)
        ctx.set(dut.in1, 0x01010101)
        ctx.set(dut.start, 1)
        await ctx.tick()
        ctx.set(dut.start, 0)
        await ctx.tick()
        assert ctx.get(dut.accumulator) == mac4_expected(0, 0x01010101)

    run_sim(tb, "test_offset_only.vcd")


def test_ones():
    """in0=1s, in1=1s: Σ(1+128)*1 = 516."""

    async def tb(ctx, dut):
        ctx.set(dut.funct7, 1)
        ctx.set(dut.in0, 0x01010101)
        ctx.set(dut.in1, 0x01010101)
        ctx.set(dut.start, 1)
        await ctx.tick()
        ctx.set(dut.start, 0)
        await ctx.tick()
        assert ctx.get(dut.accumulator) == mac4_expected(0x01010101, 0x01010101)

    run_sim(tb, "test_ones.vcd")


def test_mixed():
    """Mixed byte values: (1+128)*4 + (2+128)*3 + (3+128)*2 + (4+128)*1 = 1300."""

    async def tb(ctx, dut):
        ctx.set(dut.funct7, 1)
        ctx.set(dut.in0, 0x04030201)
        ctx.set(dut.in1, 0x01020304)
        ctx.set(dut.start, 1)
        await ctx.tick()
        ctx.set(dut.start, 0)
        await ctx.tick()
        assert ctx.get(dut.accumulator) == mac4_expected(0x04030201, 0x01020304)

    run_sim(tb, "test_mixed.vcd")


def test_max_values():
    """All 0xFF: Σ(255+128)*255 = 4 * 383 * 255 = 390660."""

    async def tb(ctx, dut):
        ctx.set(dut.funct7, 1)
        ctx.set(dut.in0, 0xFFFFFFFF)
        ctx.set(dut.in1, 0xFFFFFFFF)
        ctx.set(dut.start, 1)
        await ctx.tick()
        ctx.set(dut.start, 0)
        await ctx.tick()
        assert ctx.get(dut.accumulator) == mac4_expected(0xFFFFFFFF, 0xFFFFFFFF)

    run_sim(tb, "test_max_values.vcd")


def test_accumulation():
    """Two operations accumulate: first + second."""

    async def tb(ctx, dut):
        # First: reset + compute
        ctx.set(dut.funct7, 1)
        ctx.set(dut.in0, 0x01010101)
        ctx.set(dut.in1, 0x01010101)
        ctx.set(dut.start, 1)
        await ctx.tick()
        ctx.set(dut.start, 0)
        await ctx.tick()
        first = mac4_expected(0x01010101, 0x01010101)
        assert ctx.get(dut.accumulator) == first

        # Second: accumulate (funct7=0)
        ctx.set(dut.funct7, 0)
        ctx.set(dut.in0, 0x01010101)
        ctx.set(dut.in1, 0x01010101)
        ctx.set(dut.start, 1)
        await ctx.tick()
        ctx.set(dut.start, 0)
        await ctx.tick()
        assert ctx.get(dut.accumulator) == first * 2

    run_sim(tb, "test_accumulation.vcd")


def test_reset_clears_accumulation():
    """funct7[0]=1 resets accumulator before computing."""

    async def tb(ctx, dut):
        # Accumulate something
        ctx.set(dut.funct7, 1)
        ctx.set(dut.in0, 0x05050505)
        ctx.set(dut.in1, 0x02020202)
        ctx.set(dut.start, 1)
        await ctx.tick()
        ctx.set(dut.start, 0)
        await ctx.tick()
        assert ctx.get(dut.accumulator) > 0

        # Accumulate more (funct7=0)
        ctx.set(dut.funct7, 0)
        ctx.set(dut.in0, 0x01010101)
        ctx.set(dut.in1, 0x01010101)
        ctx.set(dut.start, 1)
        await ctx.tick()
        ctx.set(dut.start, 0)
        await ctx.tick()
        before_reset = ctx.get(dut.accumulator)

        # Reset + compute (funct7=1) — accumulator should NOT include previous
        ctx.set(dut.funct7, 1)
        ctx.set(dut.in0, 0x00000000)
        ctx.set(dut.in1, 0x01010101)
        ctx.set(dut.start, 1)
        await ctx.tick()
        ctx.set(dut.start, 0)
        await ctx.tick()
        fresh = mac4_expected(0x00000000, 0x01010101)
        assert ctx.get(dut.accumulator) == fresh
        assert ctx.get(dut.accumulator) < before_reset

    run_sim(tb, "test_reset_clears.vcd")


def test_systematic_sweep():
    """Sweep various input bytes and verify against reference."""

    async def tb(ctx, dut):
        for a_byte in [0, 1, 5, 127, 255]:
            for b_byte in [0, 1, 5, 127, 255]:
                in0 = a_byte | (a_byte << 8) | (a_byte << 16) | (a_byte << 24)
                in1 = b_byte | (b_byte << 8) | (b_byte << 16) | (b_byte << 24)
                expected = mac4_expected(in0, in1)

                ctx.set(dut.funct7, 1)  # reset
                ctx.set(dut.in0, in0)
                ctx.set(dut.in1, in1)
                ctx.set(dut.start, 1)
                await ctx.tick()
                ctx.set(dut.start, 0)
                await ctx.tick()
                actual = ctx.get(dut.accumulator)
                assert actual == expected, (
                    f"a={a_byte} b={b_byte}: got {actual}, expected {expected}"
                )

    run_sim(tb, "test_sweep.vcd")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
