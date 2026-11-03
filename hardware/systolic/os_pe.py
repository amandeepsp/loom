from amaranth import ClockSignal, Instance, Module, Signal, signed
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

    def __init__(self, in_width=8, acc_width=32, gowin_dsp=False):
        self.in_width = in_width
        self.acc_width = acc_width
        # Emit the multiply as a Gowin MULT9X9 with its internal output
        # register (OUT_REG) instead of a fabric register: same 1-cycle
        # product latency, but no DSP->DFF packing constraint for the placer.
        # Simulation uses the behavioural register (identical semantics).
        self.gowin_dsp = gowin_dsp

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

        if self.gowin_dsp:
            assert self.in_width <= 9, "MULT9X9 takes 9-bit operands"
            a9 = Signal(signed(9))
            b9 = Signal(signed(9))
            dout = Signal(18)
            m.d.comb += [a9.eq(self.act_in), b9.eq(self.w_in), prod.eq(dout[:2 * self.in_width])]
            m.submodules.mult = Instance(
                "MULT9X9",
                p_AREG=0, p_BREG=0, p_OUT_REG=1, p_PIPE_REG=0,
                p_ASIGN_REG=0, p_BSIGN_REG=0, p_SOA_REG=0,
                p_MULT_RESET_MODE="SYNC",
                i_A=a9, i_B=b9, i_ASIGN=1, i_BSIGN=1, i_ASEL=0, i_BSEL=0,
                i_SIA=0, i_SIB=0, i_CE=1, i_CLK=ClockSignal(), i_RESET=0,
                o_DOUT=dout,
            )
        else:
            m.d.sync += prod.eq(self.act_in * self.w_in)
        with m.If(self.psum_load):
            m.d.sync += acc.eq(0)
        with m.Else():
            m.d.sync += acc.eq(acc + prod)

        m.d.comb += self.psum_out.eq(acc)

        return m
