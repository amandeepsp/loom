from amaranth import Module, Signal
from amaranth.build import Platform
from cfu import Instruction


class SimdMac4(Instruction):
    def __init__(self) -> None:
        super().__init__()
        self.input_offset = Signal(32, init=128)
        self.accumulator = Signal(32)
        self.reset_acc = Signal()

    def elaborate(self, platform: Platform):
        m = Module()

        in0s = [Signal(8, name=f"in0_{i}") for i in range(4)]
        in1s = [Signal(8, name=f"in1_{i}") for i in range(4)]

        accs = [Signal(32, name=f"accs_{i}") for i in range(4)]

        for i in range(4):
            m.d.comb += [
                in0s[i].eq(self.in0.word_select(i, 8)),
                in1s[i].eq(self.in1.word_select(i, 8)),
                accs[i].eq((in0s[i] + self.input_offset) * in1s[i]),
            ]

        m.d.sync += self.done.eq(1)
        with m.If(self.start):
            m.d.sync += self.accumulator.eq(self.accumulator + sum(accs))
            m.d.sync += self.done.eq(0)
        with m.Elif(self.reset_acc):
            m.d.sync += self.accumulator.eq(0)

        return m
