from amaranth import Module, Signal, signed
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out

from hardware.systolic.ws_pe import WeightStationaryPE
from hardware.systolic.skew import SkewBuffer


class WeightStationaryPEArray(wiring.Component):
    """
    Weight-stationary systolic array with registered psum flow (TPU-style).

    Built-in activation skew buffer handles input alignment.
    Partial sums flow top-to-bottom through registered pipeline stages.
    After the pipeline fills (num_rows cycles), one result per cycle exits
    the bottom, staggered by column (col c at cycle num_rows - 1 + c).

    No drain or acc_load signals — the pipeline is always flowing.
    External accumulators handle K-tiling across passes.

    Phases:
      1. Weight load (w_load=1): stream weights top-to-bottom, num_rows cycles
      2. Feed activations (all rows simultaneously). Skew buffer staggers them.
         Results stream out of bottom row.
    """

    def __init__(self, num_rows: int, num_cols: int, in_width=8, acc_width=32):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.in_width = in_width
        self.acc_width = acc_width

        ports = {}
        for r in range(num_rows):
            ports[f"act_in_{r}"] = In(signed(in_width))
        for c in range(num_cols):
            ports[f"w_in_{c}"] = In(signed(in_width))
            ports[f"psum_out_{c}"] = Out(signed(acc_width))
        ports["w_load"] = In(1)

        super().__init__(ports)

    def elaborate(self, _platform):
        m = Module()

        rows = self.num_rows
        cols = self.num_cols

        # Activation skew buffer: row r delayed by r cycles
        m.submodules["act_skew"] = act_skew = SkewBuffer(
            rows, self.in_width)

        for r in range(rows):
            m.d.comb += getattr(act_skew, f"in_{r}").eq(
                getattr(self, f"act_in_{r}"))

        pes = [[WeightStationaryPE(self.in_width, self.acc_width)
                for _ in range(cols)] for _ in range(rows)]

        for r in range(rows):
            for c in range(cols):
                m.submodules[f"pe_{r}_{c}"] = pes[r][c]

        # Activations: skewed input feeds left column, flows right (registered)
        for r in range(rows):
            m.d.comb += pes[r][0].act_in.eq(getattr(act_skew, f"out_{r}"))
            for c in range(1, cols):
                m.d.comb += pes[r][c].act_in.eq(pes[r][c - 1].act_out)

        # Weights: shift-register top to bottom
        for c in range(cols):
            m.d.comb += pes[0][c].w_in.eq(getattr(self, f"w_in_{c}"))
            for r in range(1, rows):
                m.d.comb += pes[r][c].w_in.eq(pes[r - 1][c].w_out)

        # Psum: always flows registered, top row starts from 0
        for c in range(cols):
            m.d.comb += pes[0][c].psum_in.eq(0)
            for r in range(1, rows):
                m.d.comb += pes[r][c].psum_in.eq(pes[r - 1][c].psum_out)
            m.d.comb += getattr(self, f"psum_out_{c}").eq(
                pes[rows - 1][c].psum_out)

        # Broadcast w_load
        for r in range(rows):
            for c in range(cols):
                m.d.comb += pes[r][c].w_load.eq(self.w_load)

        return m
