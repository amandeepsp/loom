from amaranth import Module, Signal, signed

from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out


class WeightStationaryPE(wiring.Component):
    """
    Weight-stationary PE with registered psum flow (TPU-style).

    No local accumulator. Each cycle:
      psum_out ← psum_in + act_in * w_reg  (registered pipeline stage)
      act_out  ← act_in                     (registered, flows right)

    Partial sums flow top-to-bottom through the column. After the pipeline
    fills (num_rows cycles), one result per cycle exits the bottom row.
    External accumulators handle K-tiling across passes.
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
                "w_load": In(1),
                "psum_in": In(signed(acc_width)),
                "psum_out": Out(signed(acc_width)),
            }
        )

    def elaborate(self, platform):
        m = Module()
        w_reg = Signal(signed(self.in_width))

        with m.If(self.w_load):
            m.d.sync += w_reg.eq(self.w_in)

        m.d.sync += self.psum_out.eq(self.psum_in + self.act_in * w_reg)
        m.d.sync += self.act_out.eq(self.act_in)
        m.d.comb += self.w_out.eq(w_reg)

        return m
