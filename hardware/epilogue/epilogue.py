"""Epilogue — 1-stage requantization pipeline with result storage.

Accepts INT32 accumulator values one-per-cycle from the sequencer,
produces INT8 outputs stored in internal result registers.

Pipeline: +bias (comb) -> SRDHM (1 reg stage) -> RDBPOT (comb)
          -> +offset (comb) -> clamp (comb) -> INT8 (reg) -> optional LUT

Latency: 2 cycles (SRDHM register, then clamp/LUT register).
Throughput: 1 result/cycle after fill.

The optional 256-entry LUT maps each INT8 result through the next layer's
input requantization table (stored as 64 little-endian words, indexed by the
result's two's-complement byte), so fused stores need no CPU table lookup.
"""

from amaranth import Array, Cat, Const, Module, Mux, Signal, signed, unsigned
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

    def __init__(self, num_results=16, acc_width=32, out_width=8, cols=None):
        self._num_results = num_results
        self._cols = cols
        self._acc_width = acc_width
        self._out_width = out_width

        ports = {
            # Input stream (from sequencer)
            "data_in": In(signed(acc_width)),
            "first_in": In(1),
            "last_in": In(1),
            # Per-channel params
            "bias": In(signed(32)),
            "multiplier": In(signed(32)),
            "shift": In(unsigned(6)),
            # Per-layer params (constant across tile)
            "output_offset": In(signed(16)),
            "activation_min": In(signed(out_width)),
            "activation_max": In(signed(out_width)),
            # Done signal back to sequencer
            "done": Out(1),
            # Result readback (active any time, outside pipeline)
            "out_addr": In(ceil_log2(num_results)),
            "out_data": Out(signed(out_width)),
            # Column readback: rows r0..r0+3 of column c (index r0*cols + c).
            "out_data4col": Out(32),
            # Write-back readback: all rows of column wb_col.
            "wb_col": In(3),
            "out_col8": Out(64),
            # Result LUT (64 x 32-bit words) and its enable
            "lut_wr_addr": In(6),
            "lut_wr_data": In(32),
            "lut_wr_en": In(1),
            "lut_en": In(1),
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
        reg_shift = Signal(unsigned(6))
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

        # --- Stage 2: register the clamped result; LUT read in parallel ---
        clamped_r = Signal(signed(self._out_width))
        byte_sel = Signal(2)
        valid_out2 = Signal()
        last_out2 = Signal()
        m.d.sync += [
            clamped_r.eq(clamped),
            byte_sel.eq(clamped[0:2]),
            valid_out2.eq(valid_out),
            last_out2.eq(last_out),
        ]
        lut_mem = Memory(shape=32, depth=64, init=[])
        m.submodules.lut_mem = lut_mem
        lut_wr = lut_mem.write_port()
        m.d.comb += [
            lut_wr.addr.eq(self.lut_wr_addr),
            lut_wr.data.eq(self.lut_wr_data),
            lut_wr.en.eq(self.lut_wr_en),
        ]
        lut_rd = lut_mem.read_port(domain="sync")
        m.d.comb += lut_rd.addr.eq(clamped[2:8])
        final = Signal(signed(self._out_width))
        with m.If(self.lut_en):
            m.d.comb += final.eq(lut_rd.data.word_select(byte_sel, 8))
        with m.Else():
            m.d.comb += final.eq(clamped_r)

        # --- Result storage ---
        results = Array(
            Signal(signed(self._out_width), name=f"epi_r_{i}") for i in range(num)
        )
        write_idx = Signal(range(num))

        m.d.sync += self.done.eq(0)
        with m.If(valid_out2):
            m.d.sync += [
                results[write_idx].eq(final),
                write_idx.eq(write_idx + 1),
            ]
            with m.If(last_out2):
                m.d.sync += [
                    self.done.eq(1),
                    write_idx.eq(0),
                ]

        # Readback - always active
        m.d.comb += self.out_data.eq(results[self.out_addr])
        # Column readback: all rows of column wb_col (result write-back), and
        # READ_RESULT's 4-row column reads as one half of it (the CFU drives
        # wb_col from the read index when no write-back runs).
        cols = self._cols
        if cols and cols & (cols - 1) == 0 and num // cols == 8:
            lc = ceil_log2(cols)
            m.d.comb += self.out_col8.eq(Cat(*(
                results[Cat(self.wb_col[:lc], Const(r, 3))] for r in range(8))))
            # out_addr = r0 * cols + c with r0 in {0, 4}: bit lc + 2 picks the half.
            m.d.comb += self.out_data4col.eq(
                Mux(self.out_addr[lc + 2], self.out_col8[32:64], self.out_col8[0:32]))

        return m


PerChannelParamLayout = StructLayout(
    {
        "bias": signed(32),
        "multiplier": signed(32),
        "shift": unsigned(6),
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
            "bias": Out(signed(32)),
            "multiplier": Out(signed(32)),
            "shift": Out(unsigned(6)),
        }
        super().__init__(ports)

    def elaborate(self, _platform):
        m = Module()

        bias_mem = Memory(shape=signed(32), depth=self.depth, init=[])
        mult_mem = Memory(shape=signed(32), depth=self.depth, init=[])
        shift_mem = Memory(shape=unsigned(6), depth=self.depth, init=[])
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
