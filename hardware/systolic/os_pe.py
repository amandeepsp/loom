from amaranth import Module, Signal, signed
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out


class OutputStationaryPE(wiring.Component):
    """
    Output-stationary PE with registered input passthrough.

    Accumulator stays in-place: acc += act_in * w_in each cycle.
    Both act_out and w_out are registered (1-cycle delay) for systolic flow.

    Modes (priority order):
      psum_load=1: acc ← 0         (reset for new tile)
      else:        acc += prod      (compute)

    The product is registered (prod ← act * w) before accumulating, so the
    DSP multiply and the 32-bit add are separate pipeline stages. This adds
    one cycle of latency (the sequencer flushes one extra cycle); prod is 0
    whenever the array inputs are idle, so nothing leaks across tiles.

    psum_out = acc (combinational read, directly exposed for R×C readback).
    """

    def __init__(self, in_width=8, acc_width=32):
        self.in_width = in_width
        self.acc_width = acc_width

        super().__init__(
            {
                "act_in": In(signed(in_width)),
                "act_out": Out(signed(in_width)),
                "w_in": In(signed(in_width)),
                "w_out": Out(signed(in_width)),
                "psum_load": In(1),
                "psum_out": Out(signed(acc_width)),
            }
        )

    def elaborate(self, _platform):
        m = Module()
        acc = Signal(signed(self.acc_width))
        prod = Signal(signed(2 * self.in_width))

        # Registered passthrough for systolic flow
        m.d.sync += self.act_out.eq(self.act_in)
        m.d.sync += self.w_out.eq(self.w_in)

        m.d.sync += prod.eq(self.act_in * self.w_in)
        with m.If(self.psum_load):
            m.d.sync += acc.eq(0)
        with m.Else():
            m.d.sync += acc.eq(acc + prod)

        m.d.comb += self.psum_out.eq(acc)

        return m
