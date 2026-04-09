import struct

import pytest
from amaranth.sim import Simulator

from hardware.epilogue.quant import SRDHM, RoundingDividebyPOT

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


def to_signed32(val):
    if val >= (1 << 31):
        val -= 1 << 32
    return val


def ref_srdhm(a: int, b: int) -> int:
    if a == INT32_MIN and b == INT32_MIN:
        return INT32_MAX
    ab = a * b
    nudge = 1 << 30
    result = (ab + nudge) >> 31
    return max(INT32_MIN, min(INT32_MAX, result))


def ref_rdbpot(x: int, exponent: int) -> int:
    if exponent == 0:
        return x
    mask = (1 << exponent) - 1
    remainder = x & mask
    sign_bit = (x >> 31) & 1
    threshold = (mask >> 1) + sign_bit
    rounding = 1 if remainder > threshold else 0
    return (x >> exponent) + rounding


class TestSRDHM:
    def test_saturation(self):
        """INT32_MIN * INT32_MIN saturates to INT32_MAX."""

        async def testbench(ctx):
            ctx.set(dut.a, INT32_MIN)
            ctx.set(dut.b, INT32_MIN)
            ctx.set(dut.start, 1)
            await ctx.tick()
            assert ctx.get(dut.done) == 1
            result = to_signed32(ctx.get(dut.out))
            assert result == INT32_MAX

        dut = SRDHM()
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_srdhm_sat.vcd"):
            sim.run()

    @pytest.mark.parametrize(
        "a, b",
        [
            (1000000, 1000000),
            (-500000, 300000),
            (INT32_MAX, 1),
            (1, INT32_MAX),
            (INT32_MIN + 1, INT32_MIN + 1),
            (0, 12345),
            (123456789, -987654321),
        ],
    )
    def test_reference(self, a, b):
        """SRDHM matches Python reference for various inputs."""
        expected = ref_srdhm(a, b)

        async def testbench(ctx):
            ctx.set(dut.a, a)
            ctx.set(dut.b, b)
            ctx.set(dut.start, 1)
            await ctx.tick()
            assert ctx.get(dut.done) == 1
            result = to_signed32(ctx.get(dut.out))
            assert result == expected, f"srdhm({a}, {b}): expected {expected}, got {result}"

        dut = SRDHM()
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_srdhm.vcd"):
            sim.run()

    def test_zero(self):
        """SRDHM with zero input gives zero."""

        async def testbench(ctx):
            ctx.set(dut.a, 0)
            ctx.set(dut.b, 12345)
            ctx.set(dut.start, 1)
            await ctx.tick()
            assert ctx.get(dut.done) == 1
            assert to_signed32(ctx.get(dut.out)) == 0

        dut = SRDHM()
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_srdhm_zero.vcd"):
            sim.run()


class TestRDBPOT:
    @pytest.mark.parametrize(
        "x, exponent, expected",
        [
            (100, 2, 25),       # 100 / 4 = 25
            (101, 2, 25),       # 101 / 4 = 25.25 → 25
            (102, 2, 26),       # 102 / 4 = 25.5  → 26
            (103, 2, 26),       # 103 / 4 = 25.75 → 26
            (-100, 2, -25),
            (-101, 2, -25),
            (-102, 2, -26),
            (-103, 2, -26),
            (256, 8, 1),
            (255, 8, 1),
            (127, 8, 0),
            (0, 5, 0),
            (1, 1, 1),
            (-1, 1, -1),
        ],
    )
    def test_reference(self, x, exponent, expected):
        """Python reference produces correct rounding."""
        result = ref_rdbpot(x, exponent)
        assert result == expected, f"rdbpot({x}, {exponent}): expected {expected}, got {result}"
