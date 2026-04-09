from amaranth import Module, Signal
from amaranth.lib import data, wiring
from amaranth.lib.wiring import In, Out
from amaranth.lib.memory import Memory


class DmaWriteSignature(wiring.Signature):
    """Write-only port for DMA → scratchpad fill.

    Used by DoubleScratchpad as its grouped write port (``wr``).
    """

    def __init__(self, addr_width, data_width=32):
        super().__init__({
            "addr": In(addr_width),
            "data": In(data_width),
            "en":   In(1),
        })


class DoubleScratchpad(wiring.Component):
    """
    Double-buffered scratchpad backed by two SRAM banks.

    Write port writes to the fill bank. Read port reads from the compute bank.
    Assert swap for one cycle to toggle which bank is fill vs compute.

    Each bank: depth × line_shape (maps to B-SRAM blocks).
    Read latency: 1 cycle (synchronous SRAM).

    bank_sel=0: write→A, read→B
    bank_sel=1: write→B, read→A
    """

    def __init__(self, depth=512, line_shape=32):
        self.depth = depth
        self.line_shape = line_shape
        self.addr_bits = (depth - 1).bit_length()

        super().__init__(
            {
                "wr": Out(DmaWriteSignature(addr_width=self.addr_bits, data_width=line_shape)),
                "rd_addr": In(self.addr_bits),
                "rd_data": Out(line_shape),
                "swap": In(1),
            }
        )

    def elaborate(self, _platform):
        m = Module()

        bank_sel = Signal()

        with m.If(self.swap):
            m.d.sync += bank_sel.eq(~bank_sel)

        m.submodules["mem_a"] = mem_a = Memory(
            shape=self.line_shape, depth=self.depth, init=[])
        m.submodules["mem_b"] = mem_b = Memory(
            shape=self.line_shape, depth=self.depth, init=[])

        wr_a = mem_a.write_port()
        wr_b = mem_b.write_port()
        rd_a = mem_a.read_port()
        rd_b = mem_b.read_port()

        # Write: gate enable by bank_sel — only fill bank accepts writes
        m.d.comb += [
            wr_a.addr.eq(self.wr.addr),
            wr_a.data.eq(self.wr.data),
            wr_a.en.eq(self.wr.en & ~bank_sel),
            wr_b.addr.eq(self.wr.addr),
            wr_b.data.eq(self.wr.data),
            wr_b.en.eq(self.wr.en & bank_sel),
        ]

        # Read: both banks read same address, mux output by bank_sel
        m.d.comb += [
            rd_a.addr.eq(self.rd_addr),
            rd_a.en.eq(1),
            rd_b.addr.eq(self.rd_addr),
            rd_b.en.eq(1),
        ]

        with m.If(bank_sel):
            m.d.comb += self.rd_data.eq(rd_a.data)
        with m.Else():
            m.d.comb += self.rd_data.eq(rd_b.data)

        return m
