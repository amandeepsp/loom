"""CFU top-level: OS systolic array + sequencer + epilogue, DMA-fed scratchpads.

Datapath:
  DMA → DoubleScratchpad (fill bank)
  Sequencer reads compute bank → feeds OS PE Array → drains psums to Epilogue
  Epilogue requantizes INT32→INT8, stores results for CPU readback
"""

from amaranth import ClockSignal, Module, ResetSignal, Signal, signed, unsigned
from amaranth.back.verilog import convert
from amaranth.lib import wiring

from cfu import Cfu
from scratchpad import DmaWriteSignature, DoubleScratchpad
from os_pe_array import OutputStationaryPEArray
from sequencer import Sequencer
from epilogue import Epilogue, PerChannelStore
from decoder.instructions import (
    ComputeStartInstruction,
    ComputeWaitInstruction,
    EpiParamInstruction,
    ConfigInstruction,
    ReadResultInstruction,
)


class Top(Cfu):
    ROWS = 4
    COLS = 4
    STORE_DEPTH = 512

    def __init__(self):
        super().__init__()
        addr_bits = (self.STORE_DEPTH - 1).bit_length()

        # DMA write ports — exposed on Verilog boundary for SoC adapter
        sig = DmaWriteSignature(addr_width=addr_bits)
        self.dma_act = sig.create(path=("dma_act",))
        self.dma_wgt = sig.create(path=("dma_wgt",))

        self.ports += [
            self.dma_act.addr, self.dma_act.data, self.dma_act.en,
            self.dma_wgt.addr, self.dma_wgt.data, self.dma_wgt.en,
        ]

    def elab_instructions(self, m):
        # See hardware/decoder/README for encoding.
        m.submodules.i_compute_start = self.i_compute_start = ComputeStartInstruction()
        m.submodules.i_compute_wait = self.i_compute_wait = ComputeWaitInstruction()
        m.submodules.i_epi_param = self.i_epi_param = EpiParamInstruction()
        m.submodules.i_config = self.i_config = ConfigInstruction()
        m.submodules.i_read_result = self.i_read_result = ReadResultInstruction()
        return {
            0: self.i_compute_start,
            1: self.i_compute_wait,
            2: self.i_epi_param,
            3: self.i_config,
            4: self.i_read_result,
        }

    def elaborate(self, platform):
        m = super().elaborate(platform)

        rows, cols = self.ROWS, self.COLS

        # === Submodules ===================================================

        self.act_scratch = act_sp = DoubleScratchpad(depth=self.STORE_DEPTH, line_shape=32)
        self.wgt_scratch = wgt_sp = DoubleScratchpad(depth=self.STORE_DEPTH, line_shape=32)
        m.submodules.act_scratch = act_sp
        m.submodules.wgt_scratch = wgt_sp

        self.array = array = OutputStationaryPEArray(rows, cols)
        m.submodules.array = array

        self.seq = seq = Sequencer(rows=rows, cols=cols, scratchpad_depth=self.STORE_DEPTH)
        m.submodules.seq = seq

        self.epi = epi = Epilogue(num_results=rows * cols)
        m.submodules.epi = epi

        self.params = params = PerChannelStore(depth=rows * cols)
        m.submodules.params = params

        # Aliases for global config (owned by ConfigInstruction)
        i_cfg = self.i_config
        output_offset = i_cfg.output_offset
        activation_min = i_cfg.activation_min
        activation_max = i_cfg.activation_max

        # === DMA → Scratchpad fill bank ===================================

        wiring.connect(m, wiring.flipped(self.dma_act), act_sp.wr)
        wiring.connect(m, wiring.flipped(self.dma_wgt), wgt_sp.wr)

        # === Scratchpad ↔ Sequencer =======================================

        m.d.comb += [
            act_sp.rd_addr.eq(seq.act_rd_addr),
            wgt_sp.rd_addr.eq(seq.wgt_rd_addr),
            act_sp.swap.eq(seq.act_swap),
            wgt_sp.swap.eq(seq.wgt_swap),
            # Scratchpad 32-bit → sequencer struct (4×int8 packed)
            seq.act_rd_data.as_value().eq(act_sp.rd_data),
            seq.wgt_rd_data.as_value().eq(wgt_sp.rd_data),
        ]

        # === Sequencer → Array ============================================

        for r in range(rows):
            m.d.comb += getattr(array, f"act_in_{r}").eq(
                getattr(seq.arr_act_in, f"r{r}"))
        for c in range(cols):
            m.d.comb += getattr(array, f"w_in_{c}").eq(
                getattr(seq.arr_w_in, f"c{c}"))
        m.d.comb += array.psum_load.eq(seq.arr_psum_load)

        # === Array → Sequencer (psum readback) ============================

        for r in range(rows):
            for c in range(cols):
                m.d.comb += getattr(seq, f"arr_psum_out_{r}_{c}").eq(
                    getattr(array, f"psum_out_{r}_{c}"))

        # === Sequencer → Epilogue =========================================

        m.d.comb += [
            epi.data_in.eq(seq.epi_data),
            epi.first_in.eq(seq.epi_first),
            epi.last_in.eq(seq.epi_last),
            seq.epi_done.eq(epi.done),
        ]

        # === Param store → Epilogue (comb read, same cycle as epi_data) ===

        m.d.comb += [
            params.rd_addr.eq(seq.epi_index),
            epi.bias.eq(params.bias),
            epi.multiplier.eq(params.multiplier),
            epi.shift.eq(params.shift),
        ]

        # === Global regs → Epilogue ======================================

        m.d.comb += [
            epi.output_offset.eq(output_offset),
            epi.activation_min.eq(activation_min),
            epi.activation_max.eq(activation_max),
        ]

        # === Instruction → Datapath wiring ================================

        # COMPUTE_START → Sequencer
        i_cs = self.i_compute_start
        m.d.comb += [
            seq.first.eq(i_cs.seq_first),
            seq.last.eq(i_cs.seq_last),
            seq.k_count.eq(i_cs.seq_k_count),
            seq.start.eq(i_cs.seq_start),
        ]

        # COMPUTE_WAIT ← Sequencer (latched to prevent race with done pulse)
        seq_done_latch = Signal()
        with m.If(seq.done):
            m.d.sync += seq_done_latch.eq(1)
        with m.If(i_cs.seq_start):
            m.d.sync += seq_done_latch.eq(0)
        m.d.comb += self.i_compute_wait.seq_done.eq(seq.done | seq_done_latch)

        # EPI_PARAM → PerChannelStore
        i_ep = self.i_epi_param
        m.d.comb += [
            params.wr_addr.eq(i_ep.wr_addr),
            params.wr_data.eq(i_ep.wr_data),
            params.wr_sel.eq(i_ep.wr_sel),
            params.wr_en.eq(i_ep.wr_en),
        ]

        # READ_RESULT ↔ Epilogue
        i_rr = self.i_read_result
        m.d.comb += [
            epi.out_addr.eq(i_rr.out_addr),
            i_rr.out_data.eq(epi.out_data),
        ]

        return m


if __name__ == "__main__":
    import re

    top = Top()
    ports = top.ports + [ClockSignal("sync"), ResetSignal("sync")]
    v = convert(top, name="Cfu", ports=ports, strip_internal_attrs=True)
    v = re.sub(r"^.*dump_module.*\n", "", v, flags=re.MULTILINE)
    with open("top.v", "w") as f:
        f.write(v)
    print("Wrote top.v")
