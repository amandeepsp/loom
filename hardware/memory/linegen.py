"""LineGen: hardware 3x3 depthwise im2col into the activation fill bank.

The activation DMA (idle during fused depthwise layers) copies one group of
8 channel planes (CHW, `plane` bytes each, 8 * plane <= 8 KiB) into a plane
buffer. For each M tile of 8 output pixels on one output row, GEN produces
the 72 activation lines the systolic array consumes, channel-major
(line = c * 9 + ky * 3 + kx), byte r = input pixel
(iy0 + ky, ix0 + kx + r * stride) of channel c, or the zero point when that
is padding. Each line takes R = 1, 2 or 4 reads (cycles): one 8-byte read at
stride 1; two contiguous reads, even bytes kept, at stride 2; and for 4- or
2-wide outputs (a tile spans 2 or 4 output rows) one read per output row, W =
8 / R bytes each. Dense mode writes tap-major lines (tap * nch + c) for an
ordinary 3x3 conv over nch <= 8 input planes.

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
            # Reads per line (1 << reads_log2); rowwise: reads step input rows.
            "reads_log2": In(2),
            "rowwise": In(1),
            # Dense conv: tap-major lines (tap * nch + c), nch channels (0 = 8).
            "dense": In(1),
            "nch": In(4),
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

        # Reads per line R = 1 << reads_log2, each giving W = 8 / R bytes:
        #   rowwise = 0 (OW >= 8): read q is 8 bytes further along the row;
        #   rowwise = 1 (OW = 4 / 2): read q is `stride` input rows lower.
        # Chunk byte j of a read = funnel byte j * stride.
        nch = Signal(4)
        m.d.comb += nch.eq(Mux(self.nch == 0, 8, self.nch))
        last_q = Signal(2)
        m.d.comb += last_q.eq((1 << self.reads_log2) - 1)

        # ---- Stage 0: walk (c, ky, kx, q) and issue the bank reads ----
        cl = Signal(range(8))
        ky = Signal(range(3))
        kx = Signal(range(3))
        q = Signal(2)
        line = Signal(range(72))
        cl_line = Signal(range(72))  # dense: line of (cl, tap 0) = cl
        plane_base = Signal(signed(17))
        rowoff = Signal(signed(17))
        q_off = Signal(signed(14))   # byte offset of read q
        q_row = Signal(range(8))     # input-row offset of read q
        active = Signal()

        s_val = Signal(range(3))
        m.d.comb += s_val.eq(Mux(self.stride2, 2, 1))
        q_step = Signal(signed(13))
        m.d.comb += q_step.eq(Mux(self.rowwise, Mux(self.stride2, self.iw << 1, self.iw), 8))

        byte_addr = Signal(signed(18))
        m.d.comb += byte_addr.eq(plane_base + rowoff + self.ix0 + kx + q_off)
        word = Signal(signed(16))
        m.d.comb += word.eq(byte_addr >> 3)
        w_hi = Signal(idx_bits)
        m.d.comb += w_hi.eq(word[1:1 + idx_bits])
        with m.If(word[0]):
            m.d.comb += [rd_odd.addr.eq(w_hi), rd_even.addr.eq(w_hi + 1)]
        with m.Else():
            m.d.comb += [rd_even.addr.eq(w_hi), rd_odd.addr.eq(w_hi)]

        iy = Signal(signed(12))
        m.d.comb += iy.eq(self.iy0 + ky + q_row)
        iy_ok = Signal()
        m.d.comb += iy_ok.eq((iy >= 0) & (iy < self.ih))

        # Registered for stage 1 (aligned with the sync read data).
        s1_valid = Signal()
        s1_last = Signal()
        s1_odd_first = Signal()
        s1_shift = Signal(3)
        s1_iy_ok = Signal()
        s1_ix = Signal(signed(14))
        s1_line = Signal(range(72))
        m.d.sync += [
            s1_valid.eq(active),
            s1_last.eq(q == last_q),
            s1_odd_first.eq(word[0]),
            s1_shift.eq(byte_addr[0:3]),
            s1_iy_ok.eq(iy_ok),
            s1_ix.eq(self.ix0 + kx + Mux(self.rowwise, 0, q_off)),
            s1_line.eq(line),
        ]

        with m.If(self.start & ~active):
            m.d.sync += [
                active.eq(1), cl.eq(0), ky.eq(0), kx.eq(0), q.eq(0), line.eq(0),
                cl_line.eq(0), plane_base.eq(0), rowoff.eq(self.row0), q_off.eq(0), q_row.eq(0),
            ]
        with m.Elif(active):
            with m.If(q != last_q):
                m.d.sync += [q.eq(q + 1), q_off.eq(q_off + q_step),
                             q_row.eq(q_row + Mux(self.rowwise, s_val, 0))]
            with m.Else():
                m.d.sync += [q.eq(0), q_off.eq(0), q_row.eq(0),
                             line.eq(line + Mux(self.dense, nch, 1))]
                with m.If(kx == 2):
                    m.d.sync += kx.eq(0)
                    with m.If(ky == 2):
                        m.d.sync += [ky.eq(0), rowoff.eq(self.row0),
                                     plane_base.eq(plane_base + self.plane)]
                        with m.If(cl == nch - 1):
                            m.d.sync += active.eq(0)
                        with m.Else():
                            m.d.sync += cl.eq(cl + 1)
                            with m.If(self.dense):
                                m.d.sync += [line.eq(cl_line + 1), cl_line.eq(cl_line + 1)]
                    with m.Else():
                        m.d.sync += [ky.eq(ky + 1), rowoff.eq(rowoff + self.iw)]
                with m.Else():
                    m.d.sync += kx.eq(kx + 1)

        # ---- Stage 1: funnel, chunk, pad, assemble, write ----
        lo = Signal(64)
        hi = Signal(64)
        m.d.comb += [
            lo.eq(Mux(s1_odd_first, rd_odd.data, rd_even.data)),
            hi.eq(Mux(s1_odd_first, rd_even.data, rd_odd.data)),
        ]
        both = Signal(128)
        m.d.comb += both.eq(Cat(lo, hi))
        src = Array(both.word_select(i, 8) for i in range(16))
        got = [Signal(8, name=f"got{i}") for i in range(8)]
        for i in range(8):
            m.d.comb += got[i].eq(src[s1_shift + i])

        chunk = [Signal(8, name=f"chunk{j}") for j in range(8)]
        for j in range(8):
            ix_j = Signal(signed(15), name=f"ix{j}")
            m.d.comb += ix_j.eq(s1_ix + Mux(self.stride2, 2 * j, j))
            ok = Signal(name=f"ok{j}")
            m.d.comb += ok.eq(s1_iy_ok & (ix_j >= 0) & (ix_j < self.iw))
            raw = Mux(self.stride2, got[2 * j], got[j]) if j < 4 else got[j]
            m.d.comb += chunk[j].eq(Mux(ok, raw, self.zp))
        chunk_v = Signal(64)
        m.d.comb += chunk_v.eq(Cat(*chunk))

        # Shift each read's W bytes in from the top; after R reads the line
        # holds read 0's bytes at the bottom.
        acc = Signal(64)
        merged = Signal(64)
        with m.Switch(self.reads_log2):
            with m.Case(0):
                m.d.comb += merged.eq(chunk_v)
            with m.Case(1):
                m.d.comb += merged.eq(Cat(acc[32:64], chunk_v[0:32]))
            with m.Default():
                m.d.comb += merged.eq(Cat(acc[16:64], chunk_v[0:16]))
        with m.If(s1_valid):
            m.d.sync += acc.eq(merged)

        m.d.comb += [
            self.wr_en.eq(s1_valid & s1_last),
            self.wr_addr.eq(s1_line),
            self.wr_data.eq(merged),
            # busy also covers the stage-1 write still in flight.
            self.busy.eq(active | self.start | s1_valid),
        ]
        return m
