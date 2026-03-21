from amaranth import Module, Signal, signed
from amaranth.build import Platform
from cfu import Instruction


class WriteRegs(Instruction):
    """
    Write CFU registers.

    """

    def __init__(self) -> None:
        super().__init__()
        self.input_offset = Signal(signed(32))
        self.reset_acc = Signal()

    def elaborate(self, platform: Platform):
        m = super().elaborate(platform)
        with m.If(self.start):
            with m.Switch(self.in0[:8]):
                with m.Case(0):
                    m.d.sync += self.input_offset.eq(self.in1)
                with m.Case(1):
                    m.d.comb += self.reset_acc.eq(1)
            m.d.comb += self.done.eq(1)
        return m


class ReadRegs(Instruction):
    """
    Read CFU registers
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_offset = Signal(signed(32))
        self.accumulator = Signal(signed(32))

    def elaborate(self, platform: Platform):
        m = super().elaborate(platform)
        with m.If(self.start):
            with m.Switch(self.in0[:4]):
                with m.Case(0):
                    m.d.comb += self.output.eq(self.input_offset)
                with m.Case(1):
                    m.d.comb += self.output.eq(self.accumulator)
            m.d.comb += self.done.eq(1)
        return m
