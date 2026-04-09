from amaranth import Module, Signal, signed
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out

from hardware.systolic.os_pe import OutputStationaryPE
from hardware.systolic.skew import SkewBuffer


class OutputStationaryPEArray(wiring.Component):
    """
    Output-stationary systolic array with built-in skew buffers.

    Activations flow left-to-right, weights flow top-to-bottom, both registered.
    Input skew buffers align data so PE(r,c) receives a[k] and w[k] at cycle k+r+c.
    Each PE accumulates: psum += act * w over K cycles.

    After K + num_rows + num_cols - 2 cycles, all PEs hold their output.

    Output: R×C direct ports. Each PE's accumulator is exposed as
    psum_out_{r}_{c}, readable combinationally after computation completes.

    Ports:
      act_in_{r}       - activation inputs (pre-skew), one per row
      w_in_{c}         - weight inputs (pre-skew), one per column
      psum_out_{r}_{c} - direct accumulator output per PE
      psum_load       - reset all accumulators
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
        for r in range(num_rows):
            for c in range(num_cols):
                ports[f"psum_out_{r}_{c}"] = Out(signed(acc_width))
        ports["psum_load"] = In(1)

        super().__init__(ports)

    def elaborate(self, _platform):
        m = Module()

        rows = self.num_rows
        cols = self.num_cols

        # Skew buffers: activation by row, weights by column
        m.submodules["act_skew"] = act_skew = SkewBuffer(
            rows, self.in_width)
        m.submodules["w_skew"] = w_skew = SkewBuffer(
            cols, self.in_width)

        for r in range(rows):
            m.d.comb += getattr(act_skew, f"in_{r}").eq(
                getattr(self, f"act_in_{r}"))
        for c in range(cols):
            m.d.comb += getattr(w_skew, f"in_{c}").eq(
                getattr(self, f"w_in_{c}"))

        # PEs
        pes = [[OutputStationaryPE(self.in_width, self.acc_width)
                for _ in range(cols)] for _ in range(rows)]

        for r in range(rows):
            for c in range(cols):
                m.submodules[f"pe_{r}_{c}"] = pes[r][c]

        # Activations: skewed input feeds left column, flows right (registered)
        for r in range(rows):
            m.d.comb += pes[r][0].act_in.eq(getattr(act_skew, f"out_{r}"))
            for c in range(1, cols):
                m.d.comb += pes[r][c].act_in.eq(pes[r][c - 1].act_out)

        # Weights: skewed input feeds top row, flows down (registered)
        for c in range(cols):
            m.d.comb += pes[0][c].w_in.eq(getattr(w_skew, f"out_{c}"))
            for r in range(1, rows):
                m.d.comb += pes[r][c].w_in.eq(pes[r - 1][c].w_out)

        # Direct R×C accumulator outputs
        for r in range(rows):
            for c in range(cols):
                m.d.comb += getattr(self, f"psum_out_{r}_{c}").eq(
                    pes[r][c].psum_out)

        # Broadcast control
        for r in range(rows):
            for c in range(cols):
                m.d.comb += pes[r][c].psum_load.eq(self.psum_load)

        return m
