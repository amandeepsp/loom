from amaranth import Elaboratable, Module, Mux, Signal, signed

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


class SRDHM(Elaboratable):
    """1-stage pipelined SRDHM.

    Registers the 32x32 multiply (the timing bottleneck at 48 MHz).
    Nudge, extract, and saturation are combinational from the registered product.

    Latency: 1 cycle (start -> done).
    Throughput: 1 result/cycle.
    """

    def __init__(self) -> None:
        self.a = Signal(signed(32))
        self.b = Signal(signed(32))
        self.start = Signal()
        self.out = Signal(signed(32))
        self.done = Signal()

    def elaborate(self, platform):
        m = Module()

        pos_nudge = 1 << 30

        # Combinational: multiply + saturation detect
        ab = Signal(signed(64))
        saturate = Signal()
        m.d.comb += [
            ab.eq(self.a * self.b),
            saturate.eq((self.a == INT32_MIN) & (self.b == INT32_MIN)),
        ]

        # Register stage: latch multiply result (the long pole for timing)
        reg_ab = Signal(signed(64))
        reg_saturate = Signal()

        m.d.sync += self.done.eq(0)
        with m.If(self.start):
            m.d.sync += [
                reg_ab.eq(ab),
                reg_saturate.eq(saturate),
                self.done.eq(1),
            ]

        # Combinational tail: rounded signed division by 2^31.
        # Matches gemmlowp SaturatingRoundingDoublingHighMul:
        #   (ab + (1 - 2^30)) / 2^31, truncating toward zero, when ab < 0.
        # For negative values, trunc(x / d) == floor((x + d - 1) / d).
        # Combining d - 1 with gemmlowp's negative nudge leaves +2^30, the
        # same nudge as the positive case. This avoids a second wide adder and
        # remainder reduction in hardware while preserving scalar C++ results.
        m.d.comb += self.out.eq(
            Mux(reg_saturate, INT32_MAX, (reg_ab + pos_nudge)[31:])
        )

        return m


class RoundingDividebyPOT(Elaboratable):
    """
    This divides by a power of two, rounding to the nearest whole number.
    """

    def __init__(self):
        self.x = Signal(signed(32))
        self.exponent = Signal(6)
        self.result = Signal(signed(32))

    def elaborate(self, platform):
        m = Module()
        mask = (1 << self.exponent) - 1
        remainder = self.x & mask
        threshold = (mask >> 1) + self.x[31]
        rounding = Mux(remainder > threshold, 1, 0)
        m.d.comb += self.result.eq((self.x >> self.exponent) + rounding)
        return m
