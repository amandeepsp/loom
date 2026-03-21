"""SimdMac4 — 4-lane SIMD multiply-accumulate with input offset.

For each byte lane i in [0..3]:
    output += (in0[i] + INPUT_OFFSET) * in1[i]

Run simulation:
    uv run mac.py
"""

from amaranth import Module, Signal, signed
from cfu import Instruction


class SimdMac4(Instruction):
    def __init__(self):
        super().__init__()
        self.input_offset = Signal(signed(32))
        self.accumulator = Signal(signed(32))
        self.reset_acc = Signal()

    def elaborate(self, platform):
        m = super().elaborate(platform)
        in_vals = [Signal(signed(8), name=f"in_val_{i}") for i in range(4)]
        filter_vals = [Signal(signed(8), name=f"filter_val_{i}") for i in range(4)]
        mults = [Signal(signed(18), name=f"mult_{i}") for i in range(4)]
        for i in range(4):
            m.d.comb += [
                in_vals[i].eq(self.in0.word_select(i, 8).as_signed()),
                filter_vals[i].eq(self.in1.word_select(i, 8).as_signed()),
                mults[i].eq((in_vals[i] + self.input_offset) * filter_vals[i]),
            ]

        with m.If(self.reset_acc):
            m.d.sync += self.accumulator.eq(0)
        with m.Elif(self.start):
            m.d.sync += self.accumulator.eq(self.accumulator + sum(mults))
        self.signal_done(m)
        return m
