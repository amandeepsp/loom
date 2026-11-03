"""LineGen: hardware 3x3 depthwise im2col into the activation fill bank.

The activation DMA (idle during fused depthwise layers) copies one group of
8 channel planes (CHW, `plane` bytes each, 8 * plane <= 8 KiB) into a plane
buffer. For each M tile of 8 output pixels on one output row, GEN produces
the 72 activation lines the systolic array consumes, channel-major
(line = c * 9 + ky * 3 + kx), byte r = input pixel
(iy0 + ky, ix0 + kx + r * stride) of channel c, or the zero point when that
is padding. Stride 1 makes one line per cycle; stride 2 takes two cycles
(two contiguous 8-byte reads, even bytes kept). Two-row mode serves 4-wide
outputs (a tile is two output rows of 4 pixels): bytes 0..3 come from input
row iy0 + ky, bytes 4..7 from the row `stride` further down, also two reads.

The plane buffer is split into even / odd 64-bit word banks so the two
consecutive words holding any 8 contiguous bytes are read in one cycle;
a 128 -> 64-bit byte funnel then aligns them.
"""

from amaranth import Array, Cat, Const, Module, Mux, Signal, signed, unsigned
from amaranth.lib.memory import Memory
from amaranth.lib.wiring import Component, In, Out


class LineGen(Component):
    def __init__(self, words: int = 1024, lines_addr_width: int = 9):
        assert words & (words - 1) == 0
        self.words = words
        self.waddr_bits = (words - 1).bit_length()
        super().__init__({
            # Plane-buffer fill (from the activation DMA bridge).
            "dma_addr": In(self.waddr_bits),
            "dma_data": In(64),
            "dma_en": In(1),
            # Geometry (per layer).
            "iw": In(10),
            "ih": In(10),
            "plane": In(14),
            "stride2": In(1),
            "tworow": In(1),
            "zp": In(8),
            # GEN command: tile origin (input row iy0, its byte offset row0
            # within a plane = iy0 * iw, input column ix0 of output byte 0, kx 0).
            "start": In(1),
            "iy0": In(signed(11)),
            "row0": In(signed(16)),
            "ix0": In(signed(12)),
            "busy": Out(1),
            # Activation fill-bank write port.
            "wr_addr": Out(lines_addr_width),
            "wr_data": Out(64),
            "wr_en": Out(1),
        })

    def elaborate(self, platform):
        m = Module()
        half_words = self.words // 2
        idx_bits = (half_words - 1).bit_length()

        even = Memory(shape=64, depth=half_words, init=[])
        odd = Memory(shape=64, depth=half_words, init=[])
        m.submodules.even = even
        m.submodules.odd = odd
        for bank, parity in ((even, 0), (odd, 1)):
            wp = bank.write_port()
            m.d.comb += [
                wp.addr.eq(self.dma_addr[1:]),
                wp.data.eq(self.dma_data),
                wp.en.eq(self.dma_en & (self.dma_addr[0] == parity)),
            ]
        rd_even = even.read_port(domain="sync")
        rd_odd = odd.read_port(domain="sync")

        # ---- Stage 0: walk (c, ky, kx[, half]) and issue the bank reads ----
        cl = Signal(range(8))
        ky = Signal(range(3))
        kx = Signal(range(3))
        half = Signal()
        line = Signal(range(72))
        plane_base = Signal(signed(17))
        rowoff = Signal(signed(17))
        active = Signal()

        two_reads = Signal()
        m.d.comb += two_reads.eq(self.stride2 | self.tworow)
        # Second read: next 8 bytes (stride 2) or `stride` rows down (two-row).
        row_step = Signal(signed(12))
        m.d.comb += row_step.eq(Mux(self.stride2, self.iw << 1, self.iw))
        half_off = Signal(signed(13))
        m.d.comb += half_off.eq(Mux(half, Mux(self.tworow, row_step, 8), 0))
        byte_addr = Signal(signed(18))
        m.d.comb += byte_addr.eq(plane_base + rowoff + self.ix0 + kx + half_off)
        word = Signal(signed(16))
        m.d.comb += word.eq(byte_addr >> 3)
        w_hi = Signal(idx_bits)
        m.d.comb += w_hi.eq(word[1:1 + idx_bits])
        with m.If(word[0]):
            m.d.comb += [rd_odd.addr.eq(w_hi), rd_even.addr.eq(w_hi + 1)]
        with m.Else():
            m.d.comb += [rd_even.addr.eq(w_hi), rd_odd.addr.eq(w_hi)]

        iy = Signal(signed(12))
        m.d.comb += iy.eq(self.iy0 + ky
                          + Mux(half & self.tworow, Mux(self.stride2, 2, 1), 0))
        iy_ok = Signal()
        m.d.comb += iy_ok.eq((iy >= 0) & (iy < self.ih))

        # Registered for stage 1 (aligned with the sync read data).
        s1_valid = Signal()
        s1_last_half = Signal()
        s1_half = Signal()
        s1_odd_first = Signal()
        s1_shift = Signal(3)
        s1_iy_ok = Signal()
        s1_ix = Signal(signed(13))
        s1_line = Signal(range(72))
        m.d.sync += [
            s1_valid.eq(active),
            s1_last_half.eq(~two_reads | half),
            s1_half.eq(half),
            s1_odd_first.eq(word[0]),
            s1_shift.eq(byte_addr[0:3]),
            s1_iy_ok.eq(iy_ok),
            s1_ix.eq(self.ix0 + kx),
            s1_line.eq(line),
        ]

        with m.If(self.start & ~active):
            m.d.sync += [
                active.eq(1), cl.eq(0), ky.eq(0), kx.eq(0), half.eq(0), line.eq(0),
                plane_base.eq(0), rowoff.eq(self.row0),
            ]
        with m.Elif(active):
            with m.If(two_reads & ~half):
                m.d.sync += half.eq(1)
            with m.Else():
                m.d.sync += [half.eq(0), line.eq(line + 1)]
                with m.If(kx == 2):
                    m.d.sync += kx.eq(0)
                    with m.If(ky == 2):
                        m.d.sync += [ky.eq(0), rowoff.eq(self.row0),
                                     plane_base.eq(plane_base + self.plane)]
                        with m.If(cl == 7):
                            m.d.sync += active.eq(0)
                        with m.Else():
                            m.d.sync += cl.eq(cl + 1)
                    with m.Else():
                        m.d.sync += [ky.eq(ky + 1), rowoff.eq(rowoff + self.iw)]
                with m.Else():
                    m.d.sync += kx.eq(kx + 1)

        # ---- Stage 1: funnel, select, pad, write ----
        lo = Signal(64)
        hi = Signal(64)
        m.d.comb += [
            lo.eq(Mux(s1_odd_first, rd_odd.data, rd_even.data)),
            hi.eq(Mux(s1_odd_first, rd_even.data, rd_odd.data)),
        ]
        both = Signal(128)
        m.d.comb += both.eq(Cat(lo, hi))
        src = Array(both.word_select(s, 8) for s in range(16))
        got = [Signal(8, name=f"got{i}") for i in range(8)]
        for i in range(8):
            m.d.comb += got[i].eq(src[s1_shift + i])

        out = [Signal(8, name=f"out{r}") for r in range(8)]
        held = Signal(32)  # two-read modes: output bytes 0..3 from the first read
        for r in range(8):
            # Input column of output byte r (two-row: both halves use r % 4).
            rr = r % 4
            ix_r = Signal(signed(14), name=f"ix{r}")
            m.d.comb += ix_r.eq(s1_ix + Mux(self.tworow, Mux(self.stride2, 2 * rr, rr),
                                            Mux(self.stride2, 2 * r, r)))
            ok = Signal(name=f"ok{r}")
            m.d.comb += ok.eq(s1_iy_ok & (ix_r >= 0) & (ix_r < self.iw))
            if r < 4:
                # Stride 1: byte r. Stride 2 / two-row s2: byte 2r. Two-row s1: byte r.
                raw = Mux(self.stride2, got[2 * r], got[r])
            else:
                # Single read (s1): byte r. Second read: byte (r-4)*stride.
                raw = Mux(self.tworow, Mux(self.stride2, got[2 * (r - 4)], got[r - 4]),
                          Mux(self.stride2, got[2 * (r - 4)], got[r]))
            m.d.comb += out[r].eq(Mux(ok, raw, self.zp))
        with m.If(s1_valid & two_reads & ~s1_half):
            m.d.sync += held.eq(Cat(*out[0:4]))

        m.d.comb += [
            self.wr_en.eq(s1_valid & s1_last_half),
            self.wr_addr.eq(s1_line),
            self.wr_data.eq(Mux(two_reads, Cat(held, *out[4:8]), Cat(*out))),
        ]
        # busy also covers the stage-1 write still in flight.
        m.d.comb += self.busy.eq(active | self.start | s1_valid)
        return m
