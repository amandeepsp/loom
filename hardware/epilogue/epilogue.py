"""Epilogue — 1-stage requantization pipeline with result storage.

Accepts INT32 accumulator values one-per-cycle from the sequencer,
produces INT8 outputs stored in internal result registers.

Pipeline: +bias (comb) -> SRDHM (1 reg stage) -> RDBPOT (comb)
          -> +offset (comb) -> clamp (comb) -> INT8

Latency: 1 cycle (SRDHM register). Throughput: 1 result/cycle after fill.
"""

from amaranth import Array, Module, Signal, signed, unsigned
from amaranth.lib.data import StructLayout
from amaranth.lib.enum import IntEnum
from amaranth.lib.memory import Memory
from amaranth.lib.wiring import Component, In, Out
from amaranth.utils import ceil_log2

from hardware.epilogue.quant import SRDHM, RoundingDividebyPOT


class Epilogue(Component):
    """Per-channel params (bias, multiplier, shift) must be presented on the
    input ports each cycle, synchronized with the data. The external wiring
    (top.py) is responsible for indexing into a param table using epi_index
    from the sequencer.

    Per-layer params (output_offset, activation_min, activation_max) are
    constant across all results in a tile.

    first_in / last_in mark the boundaries of a result stream.
    Data is valid on every cycle from first_in through last_in (inclusive).
    """

    def __init__(self, num_results=16, acc_width=32, out_width=8):
        self._num_results = num_results
        self._acc_width = acc_width
        self._out_width = out_width

        ports = {
            # Input stream (from sequencer)
            "data_in": In(signed(acc_width)),
            "first_in": In(1),
            "last_in": In(1),
            # Per-channel params
            "bias": In(signed(18)),
            "multiplier": In(signed(32)),
            "shift": In(unsigned(5)),
            # Per-layer params (constant across tile)
            "output_offset": In(signed(16)),
            "activation_min": In(signed(out_width)),
            "activation_max": In(signed(out_width)),
            # Done signal back to sequencer
            "done": Out(1),
            # Result readback (active any time, outside pipeline)
            "out_addr": In(ceil_log2(num_results)),
            "out_data": Out(signed(out_width)),
        }

        super().__init__(ports)

    def elaborate(self, _platform):
        m = Module()

        num = self._num_results

        # --- Pipeline stages ---
        m.submodules.srdhm = srdhm = SRDHM()
        m.submodules.rdbpot = rdbpot = RoundingDividebyPOT()

        # Data is valid whenever first or last is asserted, or we're between them
        active = Signal()
        valid = Signal()
        m.d.comb += valid.eq(self.first_in | self.last_in | active)
        with m.If(self.first_in):
            m.d.sync += active.eq(1)
        with m.If(self.last_in):
            m.d.sync += active.eq(0)

        # Stage 0 (combinational): bias add -> SRDHM input
        m.d.comb += [
            srdhm.a.eq(self.data_in + self.bias),
            srdhm.b.eq(self.multiplier),
            srdhm.start.eq(valid),
        ]

        # Delay shift by 1 cycle to align with SRDHM output
        reg_shift = Signal(unsigned(5))
        m.d.sync += reg_shift.eq(self.shift)

        # Stage 1 (combinational from registered SRDHM output): RDBPOT
        m.d.comb += [
            rdbpot.x.eq(srdhm.out),
            rdbpot.exponent.eq(reg_shift),
        ]

        # Combinational tail: offset + clamp
        with_offset = Signal(signed(32))
        clamped = Signal(signed(self._out_width))
        m.d.comb += with_offset.eq(rdbpot.result + self.output_offset)
        with m.If(with_offset > self.activation_max):
            m.d.comb += clamped.eq(self.activation_max)
        with m.Elif(with_offset < self.activation_min):
            m.d.comb += clamped.eq(self.activation_min)
        with m.Else():
            m.d.comb += clamped.eq(with_offset)

        # --- Valid tracking (1-cycle delay matching SRDHM) ---
        valid_out = Signal()
        last_out = Signal()
        m.d.sync += [
            valid_out.eq(valid),
            last_out.eq(self.last_in),
        ]

        # --- Result storage ---
        results = Array(
            Signal(signed(self._out_width), name=f"epi_r_{i}") for i in range(num)
        )
        write_idx = Signal(range(num))

        m.d.sync += self.done.eq(0)
        with m.If(valid_out):
            m.d.sync += [
                results[write_idx].eq(clamped),
                write_idx.eq(write_idx + 1),
            ]
            with m.If(last_out):
                m.d.sync += [
                    self.done.eq(1),
                    write_idx.eq(0),
                ]

        # Readback - always active
        m.d.comb += self.out_data.eq(results[self.out_addr])

        return m


PerChannelParamLayout = StructLayout(
    {
        "bias": signed(18),
        "multiplier": signed(32),
        "shift": unsigned(5),
    }
)


class PerChannelWriteSelect(IntEnum):
    BIAS = 0x0
    MULT = 0x1
    SHIFT = 0x2


class PerChannelStore(Component):
    """Mem for per-channel epilogue params.

    Write: CPU selects field via wr_sel, writes one entry at a time.
    Read:  Sequencer drives rd_addr; data appears combinationally (same cycle).
    """

    def __init__(self, depth=16):
        self.depth = depth
        ports = {
            # Write port (from CFU instruction)
            "wr_addr": In(range(depth)),
            "wr_data": In(signed(32)),
            "wr_sel": In(PerChannelWriteSelect),
            "wr_en": In(1),
            # Read port
            "rd_addr": In(range(depth)),
            "bias": Out(signed(18)),
            "multiplier": Out(signed(32)),
            "shift": Out(unsigned(5)),
        }
        super().__init__(ports)

    def elaborate(self, _platform):
        m = Module()

        bias_mem = Memory(shape=signed(18), depth=self.depth, init=[])
        mult_mem = Memory(shape=signed(32), depth=self.depth, init=[])
        shift_mem = Memory(shape=unsigned(5), depth=self.depth, init=[])
        m.submodules.bias_mem = bias_mem
        m.submodules.mult_mem = mult_mem
        m.submodules.shift_mem = shift_mem

        # Write ports — gate enable by wr_sel
        mems = {
            PerChannelWriteSelect.BIAS: bias_mem,
            PerChannelWriteSelect.MULT: mult_mem,
            PerChannelWriteSelect.SHIFT: shift_mem,
        }
        for sel, mem in mems.items():
            wr = mem.write_port()
            m.d.comb += [
                wr.addr.eq(self.wr_addr),
                wr.data.eq(self.wr_data),
                wr.en.eq(self.wr_en & (self.wr_sel == sel)),
            ]

        # Read ports — combinational (same-cycle), aligned with sequencer epi_data
        rd_bias = bias_mem.read_port(domain="comb")
        rd_mult = mult_mem.read_port(domain="comb")
        rd_shift = shift_mem.read_port(domain="comb")

        m.d.comb += [
            rd_bias.addr.eq(self.rd_addr),
            rd_mult.addr.eq(self.rd_addr),
            rd_shift.addr.eq(self.rd_addr),
        ]

        m.d.comb += [
            self.bias.eq(rd_bias.data),
            self.multiplier.eq(rd_mult.data),
            self.shift.eq(rd_shift.data),
        ]

        return m
