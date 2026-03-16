"""SimdMac4 — 4-lane SIMD multiply-accumulate with input offset.

For each byte lane i in [0..3]:
    output += (in0[i] + INPUT_OFFSET) * in1[i]

Control via funct7:
    funct7[0] = 0: accumulate (output = accumulator + dot product)
    funct7[0] = 1: reset + compute (output = 0 + dot product)
"""

from amaranth import Module, Signal

from cfu import Instruction


class SimdMac4(Instruction):
    INPUT_OFFSET = 128

    def __init__(self):
        super().__init__()
        self.accumulator = Signal(32)

    def elaborate(self, platform):
        m = Module()

        # funct7[0] selects base: 0 = accumulate, 1 = reset
        base = Signal(32)
        with m.If(self.funct7[0]):
            m.d.comb += base.eq(0)
        with m.Else():
            m.d.comb += base.eq(self.accumulator)

        products = []
        for i in range(4):
            a = Signal(8, name=f"a{i}")
            b = Signal(8, name=f"b{i}")
            p = Signal(32, name=f"p{i}")
            m.d.comb += [
                a.eq(self.in0.word_select(i, 8)),
                b.eq(self.in1.word_select(i, 8)),
                p.eq((a + self.INPUT_OFFSET) * b),
            ]
            products.append(p)

        m.d.comb += [
            self.output.eq(base + sum(products)),
            self.done.eq(1),
        ]

        with m.If(self.start):
            m.d.sync += self.accumulator.eq(self.output)

        return m
