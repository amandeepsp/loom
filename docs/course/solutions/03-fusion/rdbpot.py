"""Rdbpot — RoundingDivideByPowerOfTwo as a CFU instruction.

Instruction encoding:
    funct7 = 2  (instruction select — slot 2 in the CFU dispatch table)
    in0    = a  (INT32 value to shift, typically output of SRDHM)
    in1    = shift amount (lower 5 bits used, valid range 0-31)
    output = arithmetic right shift of a by shift, with rounding toward zero

This is the second step of TFLite's requantization pipeline, after SRDHM.
It's much cheaper than SRDHM — no DSP blocks needed, just a mux tree and
some comparators.

Resource cost on Gowin GW2AR-18C:
    - 0 DSP blocks
    - ~80-120 LUTs (barrel shifter, mask generation, comparator)
    - 0 flip-flops (purely combinational, single-cycle)

Connection to TFLite's quantization:
    After the MAC loop produces an INT32 accumulator and SRDHM scales it
    by the per-channel multiplier, RDBPOT applies the per-channel power-of-two
    shift to bring the value into a range suitable for INT8 output.
    The shift values are typically 2-12 for real models.
"""

from amaranth import Module, Signal, signed

from cfu import Instruction


class Rdbpot(Instruction):
    """RoundingDivideByPowerOfTwo — single-cycle combinational.

    Why rounding toward zero (not toward -inf)?
    ────────────────────────────────────────────
    Plain arithmetic right shift (>>) rounds toward negative infinity:
        -7 >> 2 = -2  (in most hardware/languages)
    But -7 / 4 = -1.75, and rounding toward zero gives -1.

    For neural network quantization, consistent rounding matters. Systematic
    rounding toward -inf introduces a negative bias in activations that
    accumulates across layers. Round-toward-zero is symmetric and matches
    TFLite's gemmlowp reference implementation.

    Actually, the gemmlowp rounding mode is "round half up" (toward +inf),
    not strictly round-toward-zero. For the half case (remainder == divisor/2):
    - Positive: rounds up (same as toward +inf)
    - Negative: rounds up (toward zero)
    The implementation below matches gemmlowp exactly.

    Why the threshold differs for negative numbers:
    ────────────────────────────────────────────────
    For positive a: threshold = (mask >> 1), i.e., half the divisor
        remainder > half means we're past the midpoint → round up
    For negative a: threshold = (mask >> 1) + 1
        The +1 raises the bar for rounding up. Since negative >> gives a
        too-negative result, rounding up moves toward zero — but we want
        to round up only when clearly past the midpoint, not at exactly half.
        This gives the "round half up" behavior for negatives.
    """

    def elaborate(self, platform):
        m = Module()

        a = Signal(signed(32))
        m.d.comb += a.eq(self.in0.as_signed())

        # Only use lower 5 bits of shift amount (0-31 range).
        # Real models use shift values of 2-12; anything above 31 is nonsensical.
        shift = Signal(5)
        m.d.comb += shift.eq(self.in1[:5])

        # mask = (1 << shift) - 1
        # This covers the bits that will be shifted out.
        # We compute 1 << shift using a 32-bit signal (max shift = 31).
        one_shifted = Signal(32)
        m.d.comb += one_shifted.eq(1 << shift)

        mask = Signal(32)
        m.d.comb += mask.eq(one_shifted - 1)

        # remainder = a & mask (the bits that get shifted out)
        remainder = Signal(32)
        m.d.comb += remainder.eq(a & mask)

        # threshold = (mask >> 1) + (1 if a < 0 else 0)
        # The sign-dependent offset implements the asymmetric rounding.
        is_negative = Signal()
        m.d.comb += is_negative.eq(a[-1])  # MSB = sign bit

        threshold = Signal(32)
        m.d.comb += threshold.eq((mask >> 1) + is_negative)

        # Base result: arithmetic right shift.
        # Amaranth's >> on a signed signal produces arithmetic shift.
        base = Signal(signed(32))
        m.d.comb += base.eq(a >> shift)

        # Round up if remainder exceeds threshold.
        do_round = Signal()
        m.d.comb += do_round.eq(remainder > threshold)

        # shift == 0 special case: no division, pass through unchanged.
        # When shift=0: mask=0, remainder=0, threshold=0, 0>0 is false → base = a.
        # So the general formula already handles shift=0 correctly!
        # No special case needed — the math just works.

        m.d.comb += self.output.eq(base + do_round)

        # Single-cycle: always done immediately.
        m.d.comb += self.done.eq(1)

        return m
