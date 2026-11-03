"""SoC glue for the CFU result write-back stream (shared by sim and board).

The CFU emits one beat per output column: a 64-bit word (8 rows) and its
byte address. A LiteDRAM DMA writer on its own 64-bit crossbar port stores
it (the crossbar converts to the DRAM width). `wb_dma_pending` counts beats
accepted from the CFU but not yet handed to DRAM; firmware waits for 0
before the next step reads the data with a DMA or the CPU.
"""

from migen import If, Signal

from litedram.frontend.dma import LiteDRAMDMAWriter
from litex.soc.interconnect.csr import CSRStatus


def add_cfu_result_writer(soc, *, data_width=64, fifo_depth=16):
    port = soc.sdram.crossbar.get_port(mode="write", data_width=data_width)
    writer = LiteDRAMDMAWriter(port, fifo_depth=fifo_depth, with_csr=False)
    soc.submodules.wb_dma = writer

    wb_valid = Signal(name="cfu_wb_valid")
    wb_ready = Signal(name="cfu_wb_ready")
    wb_addr = Signal(32, name="cfu_wb_addr")
    wb_data = Signal(data_width, name="cfu_wb_data")
    shift = (data_width // 8 - 1).bit_length()
    aw = len(writer.sink.address)
    soc.comb += [
        writer.sink.valid.eq(wb_valid),
        writer.sink.address.eq(wb_addr[shift:shift + aw]),
        writer.sink.data.eq(wb_data),
        wb_ready.eq(writer.sink.ready),
    ]

    pending = Signal(8, name="cfu_wb_pending")
    accepted = Signal()
    written = Signal()
    soc.comb += [
        accepted.eq(writer.sink.valid & writer.sink.ready),
        written.eq(port.wdata.valid & port.wdata.ready),
    ]
    soc.sync += If(accepted & ~written, pending.eq(pending + 1)).Elif(
        written & ~accepted, pending.eq(pending - 1))
    writer._pending = CSRStatus(8, name="pending",
        description="Result write-back beats accepted but not yet written to DRAM.")
    soc.comb += writer._pending.status.eq(pending)

    return {
        "o_wb_valid": wb_valid,
        "i_wb_ready": wb_ready,
        "o_wb_addr": wb_addr,
        "o_wb_data": wb_data,
    }
