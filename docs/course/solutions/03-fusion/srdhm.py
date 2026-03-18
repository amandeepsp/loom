"""Srdhm — SaturatingRoundingDoubleHighMul as a CFU instruction.

Instruction encoding:
    funct7 = 1  (instruction select — slot 1 in the CFU dispatch table)
    in0    = a  (INT32 accumulator value)
    in1    = b  (INT32 fixed-point multiplier, Q0.31)
    output = round((int64(a) * int64(b)) / 2^31), saturated to INT32 range

This is the expensive step of TFLite's requantization pipeline. On a RV32IM
core without 64-bit multiply, this takes 10-15 cycles in software (multiple
32x32 multiplies with manual carry propagation). In hardware: single-cycle
combinational.

Resource cost on Gowin GW2AR-18C:
    - 2-4 DSP blocks for the 32x32 multiply (Gowin pDSPs are 18x18, so
      Amaranth/Yosys will decompose the 32x32 into multiple DSP primitives)
    - ~50 LUTs for saturation check + nudge addition
    - 0 flip-flops (purely combinational, single-cycle)
"""

from amaranth import Module, Signal, signed

from cfu import Instruction


class Srdhm(Instruction):
    """SaturatingRoundingDoubleHighMul — single-cycle combinational.

    Why sign extension to 64 bits?
    ─────────────────────────────
    We need the full 64-bit product of two 32-bit signed values.
    32 x 32 = up to 63 bits (plus sign bit = 64 bits). If we stayed in
    32 bits, we'd lose the upper half — which is exactly the half we want
    (the "high mul" in the name).

    Why nudge = 1 << 30?
    ────────────────────
    The nudge implements rounding. We're computing:
        result = (a * b) >> 31
    Without the nudge, >> truncates toward negative infinity.
    Adding 1<<30 before the shift is equivalent to adding 0.5 before
    truncation — this gives round-half-up behavior, matching TFLite.

    In Q0.31 fixed-point terms: 1<<30 = 0.5, so we're rounding to the
    nearest integer in the output scale.

    Why saturation?
    ───────────────
    INT32_MIN * INT32_MIN = (-2^31) * (-2^31) = 2^62.
    The "double" part (>> 31 instead of >> 32) doubles this to 2^63.
    But int64 max is 2^63 - 1, so this overflows. It's the ONLY input
    pair that causes overflow. We saturate to INT32_MAX.

    How this maps to Gowin DSPs:
    ────────────────────────────
    Gowin's pDSP blocks are 18x18 signed multipliers (MULT18X18).
    A 32x32 multiply decomposes into 4 partial products:
        a = a_hi(16) : a_lo(16)
        b = b_hi(16) : b_lo(16)
        product = a_lo*b_lo + (a_lo*b_hi + a_hi*b_lo)<<16 + (a_hi*b_hi)<<32
    Yosys/Amaranth handles this decomposition automatically. The synthesis
    tool maps each 16x16 sub-product to one DSP block, using 2-4 DSPs total
    depending on how the tool optimizes the partial product accumulation.
    """

    def elaborate(self, platform):
        m = Module()

        INT32_MIN = -(1 << 31)
        INT32_MAX = (1 << 31) - 1
        NUDGE = 1 << 30

        # Sign-extend inputs to 64 bits for full-precision multiply.
        # Signal(signed(64)) tells Amaranth these are signed — critical
        # for correct sign extension in the multiply.
        a_ext = Signal(signed(64))
        b_ext = Signal(signed(64))
        m.d.comb += [
            a_ext.eq(self.in0.as_signed()),
            b_ext.eq(self.in1.as_signed()),
        ]

        # Full 64-bit product. Amaranth will infer DSP blocks for this.
        product = Signal(signed(64))
        m.d.comb += product.eq(a_ext * b_ext)

        # Add the rounding nudge, then extract upper 32 bits via >> 31.
        # The +NUDGE adds 0.5 in Q31, implementing round-half-up.
        nudged = Signal(signed(64))
        m.d.comb += nudged.eq(product + NUDGE)

        # >> 31 extracts bits [62:31] of the product — the "double high"
        # part (one bit more than a standard high-mul which would >> 32).
        result = Signal(signed(64))
        m.d.comb += result.eq(nudged >> 31)

        # Saturation check: only needed when BOTH inputs are INT32_MIN.
        # This is rare in practice but required for correctness.
        both_min = Signal()
        m.d.comb += both_min.eq(
            (self.in0 == INT32_MIN) & (self.in1 == INT32_MIN)
        )

        with m.If(both_min):
            m.d.comb += self.output.eq(INT32_MAX)
        with m.Else():
            # Truncate 64-bit result to 32 bits for output.
            # The value is guaranteed to fit in int32 (except the saturating case).
            m.d.comb += self.output.eq(result[:32])

        # Single-cycle: always done immediately.
        m.d.comb += self.done.eq(1)

        return m
