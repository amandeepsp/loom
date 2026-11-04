"""CFU top-level: OS systolic array + sequencer + epilogue, DMA-fed scratchpads.

Datapath:
  DMA → DoubleScratchpad (fill bank)
  Sequencer reads compute bank → feeds OS PE Array → drains psums to Epilogue
  Epilogue requantizes INT32→INT8, stores results for CPU readback
"""

import argparse
import re
from dataclasses import dataclass

from amaranth import ClockSignal, Module, Mux, ResetSignal, Signal, signed, unsigned
from amaranth.back.verilog import convert
from amaranth.lib import wiring

from hardware.cfu import Cfu
from hardware.memory.linegen import LineGen
from hardware.memory.scratchpad import DmaWriteSignature, DoubleScratchpad
from hardware.systolic.os_pe_array import OutputStationaryPEArray
from hardware.control.os_sequencer import OSSequencer
from hardware.epilogue.epilogue import Epilogue, PerChannelStore
from hardware.decoder.instructions import (
    ComputeStartInstruction,
    ComputeWaitInstruction,
    EpiParamInstruction,
    ConfigInstruction,
    ReadResultInstruction,
    PerfCounterInstruction,
    WriteActInstruction,
    LineGenInstruction,
)


@dataclass(frozen=True)
class TopConfig:
    rows: int = 8
    cols: int = 8
    store_depth: int = 512
    in_width: int = 8
    acc_width: int = 32
    # Hardware depthwise im2col (LineGen) with a plane buffer of this many
    # 64-bit words (0 disables it).
    linegen_words: int = 1024

    @property
    def act_line_width(self) -> int:
        return self.rows * self.in_width

    @property
    def wgt_line_width(self) -> int:
        return self.cols * self.in_width

    @property
    def num_results(self) -> int:
        return self.rows * self.cols


class Top(Cfu):
    def __init__(self, config: TopConfig | None = None):
        super().__init__()
        self.config = config or TopConfig()
        addr_bits = (self.config.store_depth - 1).bit_length()

        # DMA write ports — exposed on Verilog boundary for SoC adapter
        self.rows = self.config.rows
        self.cols = self.config.cols
        # The activation DMA also fills LineGen's (larger) plane buffer.
        act_addr_bits = max(addr_bits, (self.config.linegen_words - 1).bit_length())
        sig_act = DmaWriteSignature(addr_width=act_addr_bits, data_width=self.config.act_line_width)
        sig_wgt = DmaWriteSignature(addr_width=addr_bits, data_width=self.config.wgt_line_width)
        self.dma_act = sig_act.create(path=("dma_act",))
        self.dma_wgt = sig_wgt.create(path=("dma_wgt",))
        self.dma_act_busy = Signal(name="dma_act_busy")
        self.dma_wgt_busy = Signal(name="dma_wgt_busy")

        self.ports += [
            self.dma_act.addr, self.dma_act.data, self.dma_act.en,
            self.dma_wgt.addr, self.dma_wgt.data, self.dma_wgt.en,
            self.dma_act_busy, self.dma_wgt_busy,
        ]
        # Result write-back stream (to a LiteDRAM DMA writer in the SoC):
        # one beat per output column, 8 rows = 64 bits, at a byte address.
        self.wb_valid = Signal(name="wb_valid")
        self.wb_ready = Signal(name="wb_ready")
        self.wb_addr = Signal(32, name="wb_addr")
        self.wb_data = Signal(64, name="wb_data")
        self.ports += [self.wb_valid, self.wb_ready, self.wb_addr, self.wb_data]
        self.seq_state_debug = Signal(8, name="seq_state_debug")
        self.seq_busy_debug = Signal(name="seq_busy_debug")
        self.error_warn_debug = Signal(name="error_warn_debug")
        self.ports += [
            self.seq_state_debug,
            self.seq_busy_debug,
            self.error_warn_debug,
        ]

    def elab_instructions(self, m):
        # See hardware/decoder/README for encoding.
        m.submodules.i_compute_start = self.i_compute_start = ComputeStartInstruction()
        m.submodules.i_compute_wait = self.i_compute_wait = ComputeWaitInstruction()
        m.submodules.i_epi_param = self.i_epi_param = EpiParamInstruction()
        m.submodules.i_config = self.i_config = ConfigInstruction()
        m.submodules.i_read_result = self.i_read_result = ReadResultInstruction()
        m.submodules.i_perf_counter = self.i_perf_counter = PerfCounterInstruction()
        m.submodules.i_write_act = self.i_write_act = WriteActInstruction()
        m.submodules.i_linegen = self.i_linegen = LineGenInstruction()
        return {
            0: self.i_compute_start,
            1: self.i_compute_wait,
            2: self.i_epi_param,
            3: self.i_config,
            4: self.i_read_result,
            5: self.i_perf_counter,
            6: self.i_write_act,
            7: self.i_linegen,
        }

    def elaborate(self, platform):
        m = super().elaborate(platform)

        rows, cols = self.config.rows, self.config.cols
        assert cols & (cols - 1) == 0, "per-column param indexing needs power-of-two cols"

        # === Submodules ===================================================

        self.act_scratch = act_sp = DoubleScratchpad(
            depth=self.config.store_depth, line_shape=self.config.act_line_width
        )
        self.wgt_scratch = wgt_sp = DoubleScratchpad(
            depth=self.config.store_depth, line_shape=self.config.wgt_line_width
        )
        m.submodules.act_scratch = act_sp
        m.submodules.wgt_scratch = wgt_sp

        self.array = array = OutputStationaryPEArray(
            rows, cols, self.config.in_width, self.config.acc_width
        )
        self.seq = seq = OSSequencer(
            rows=rows,
            cols=cols,
            in_width=self.config.in_width,
            acc_width=self.config.acc_width,
            scratchpad_depth=self.config.store_depth,
        )
        m.submodules.array = array
        m.submodules.seq = seq

        self.epi = epi = Epilogue(
            num_results=self.config.num_results,
            acc_width=self.config.acc_width,
            out_width=self.config.in_width,
            cols=cols,
        )
        m.submodules.epi = epi

        self.params = params = PerChannelStore(
            depth=self.config.num_results,
        )
        m.submodules.params = params

        # Aliases for global config (owned by ConfigInstruction)
        i_cfg = self.i_config
        output_offset = i_cfg.output_offset
        activation_min = i_cfg.activation_min
        activation_max = i_cfg.activation_max

        # === DMA → Scratchpad fill bank ===================================

        # Activation fill bank: LineGen lines, CPU lines via WRITE_ACT
        # (funct3=6), or the DMA. In plane-buffer mode the DMA fills LineGen.
        i_wa = self.i_write_act
        i_lg = self.i_linegen
        lg = None
        if self.config.linegen_words:
            m.submodules.linegen = lg = self.linegen = LineGen(
                words=self.config.linegen_words,
                lines_addr_width=len(act_sp.wr.addr))
            m.d.comb += [
                lg.dma_addr.eq(self.dma_act.addr),
                lg.dma_data.eq(self.dma_act.data),
                lg.dma_en.eq(self.dma_act.en & i_lg.planebuf_mode),
                lg.iw.eq(i_lg.iw), lg.ih.eq(i_lg.ih), lg.plane.eq(i_lg.plane),
                lg.stride2.eq(i_lg.stride2), lg.rowwise.eq(i_lg.rowwise),
                lg.reads_log2.eq(i_lg.reads_log2), lg.dense.eq(i_lg.dense), lg.nch.eq(i_lg.nch),
                lg.zp.eq(i_lg.zp),
                lg.start.eq(i_lg.gen_start), lg.iy0.eq(i_lg.iy0),
                lg.row0.eq(i_lg.row0), lg.ix0.eq(i_lg.ix0),
                i_lg.gen_busy.eq(lg.busy),
            ]
        with m.If(lg.wr_en if lg is not None else 0):
            m.d.comb += [
                act_sp.wr.addr.eq(lg.wr_addr),
                act_sp.wr.data.eq(lg.wr_data),
                act_sp.wr.en.eq(1),
            ]
        with m.Elif(i_wa.wr_en):
            m.d.comb += [
                act_sp.wr.addr.eq(i_wa.wr_addr),
                act_sp.wr.data.eq(i_wa.wr_data),
                act_sp.wr.en.eq(1),
            ]
        with m.Else():
            m.d.comb += [
                act_sp.wr.addr.eq(self.dma_act.addr),
                act_sp.wr.data.eq(self.dma_act.data),
                act_sp.wr.en.eq(self.dma_act.en & ~i_lg.planebuf_mode),
            ]
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
            # Per-channel params depend only on the column: read slot c of
            # result (r, c), so software uploads row 0 only (`col_params`).
            # Software that writes every row still works.
            params.rd_addr.eq(seq.epi_index & (cols - 1)),  # cols: power of two
            epi.bias.eq(params.bias),
            epi.multiplier.eq(params.multiplier),
            epi.shift.eq(params.shift),
        ]

        # === Global regs → Epilogue ======================================

        m.d.comb += [
            epi.output_offset.eq(output_offset),
            epi.activation_min.eq(activation_min),
            epi.activation_max.eq(activation_max),
            epi.lut_wr_addr.eq(i_cfg.lut_wr_addr),
            epi.lut_wr_data.eq(i_cfg.lut_wr_data),
            epi.lut_wr_en.eq(i_cfg.lut_wr_en),
            epi.lut_en.eq(i_cfg.lut_en),
        ]

        # === Instruction → Datapath wiring ================================

        # COMPUTE_START → Sequencer
        i_cs = self.i_compute_start
        m.d.comb += [
            seq.first.eq(i_cs.seq_first),
            seq.last.eq(i_cs.seq_last),
            seq.keep_wgt.eq(i_cs.seq_keep_wgt),
            seq.keep_act.eq(i_cs.seq_keep_act),
            seq.k_count.eq(i_cs.seq_k_count),
            seq.start.eq(i_cs.seq_start),
        ]

        # COMPUTE_WAIT ← Sequencer (latched to prevent race with done pulse)
        # Completion seen by COMPUTE_WAIT. A new COMPUTE_START clears it in the
        # op's own cycle, and done is masked until the (registered) start
        # pulse reaches the sequencer, so a WAIT issued right after START can
        # never return on the previous tile's done.
        seq_done_latch = Signal()
        start_pending = i_cs.start | i_cs.seq_start
        with m.If(i_cs.start):
            m.d.sync += seq_done_latch.eq(0)
        with m.Elif(seq.done & ~i_cs.seq_start):
            m.d.sync += seq_done_latch.eq(1)
        # Result write-back: armed by a COMPUTE_START of a last K chunk while
        # enabled (base latched then), runs after the epilogue, and holds
        # COMPUTE_WAIT until every column beat is accepted.
        i_cfg = self.i_config
        wb_armed = Signal()
        wb_active = Signal()
        wb_base = Signal(32)
        wb_col = Signal(4)
        wb_addr = Signal(32)
        wb_ncols = Signal(4)
        m.d.comb += wb_ncols.eq(Mux(i_cfg.wb_ncols == 0, 8, i_cfg.wb_ncols))
        with m.If(i_cs.start):
            m.d.sync += [
                wb_armed.eq(i_cfg.wb_enable & i_cs.in0[1]),
                wb_base.eq(i_cfg.wb_base_next),
            ]
        with m.Elif(wb_armed & seq.done & ~i_cs.seq_start):
            m.d.sync += [wb_armed.eq(0), wb_active.eq(1), wb_col.eq(0), wb_addr.eq(wb_base)]
        with m.If(wb_active & self.wb_ready):
            m.d.sync += [wb_col.eq(wb_col + 1), wb_addr.eq(wb_addr + i_cfg.wb_cstride)]
            with m.If(wb_col == wb_ncols - 1):
                m.d.sync += wb_active.eq(0)
        m.d.comb += [
            self.wb_valid.eq(wb_active),
            self.wb_addr.eq(wb_addr),
            self.epi.wb_col.eq(wb_col[:3]),
            self.epi.wb_active.eq(wb_active),
            self.wb_data.eq(self.epi.out_col8),
        ]
        m.d.comb += self.i_compute_wait.seq_done.eq(
            ~start_pending & (seq.done | seq_done_latch) & ~wb_armed & ~wb_active)

        m.d.comb += [
            self.seq_state_debug.eq(seq.state_debug),
            self.seq_busy_debug.eq(seq.busy_debug),
            self.error_warn_debug.eq(0),
        ]

        # === Performance counters =======================================

        # Selectors are part of the loom CUSTOM0 ISA; keep this order in
        # sync with firmware/cfu.zig and shared/protocol.{zig,py}.
        perf = self.i_perf_counter
        # 26 bits covers a single 49M-cycle MobileNet inference while saving
        # enough registers/LUTs to keep the design placeable on this FPGA.
        perf_counters = [Signal(26, name=name) for name in (
            "perf_seq_busy_cycles",
            "perf_dma_active_cycles",
            "perf_overlap_cycles",
        )]
        (
            seq_busy_cycles,
            dma_active_cycles,
            overlap_cycles,
        ) = perf_counters

        dma_active = self.dma_act_busy | self.dma_wgt_busy
        with m.If(perf.reset_counters):
            m.d.sync += [counter.eq(0) for counter in perf_counters]
        with m.Else():
            with m.If(seq.busy_debug):
                m.d.sync += seq_busy_cycles.eq(seq_busy_cycles + 1)
            with m.If(dma_active):
                m.d.sync += dma_active_cycles.eq(dma_active_cycles + 1)
            with m.If(dma_active & seq.busy_debug):
                m.d.sync += overlap_cycles.eq(overlap_cycles + 1)

        for port, counter in zip(
            (
                perf.seq_busy_cycles,
                perf.dma_active_cycles,
                perf.overlap_cycles,
            ),
            perf_counters,
        ):
            m.d.comb += port.eq(counter)

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
            i_rr.out_data4col.eq(epi.out_data4col),
        ]

        return m


def build_config_from_args(args: argparse.Namespace) -> TopConfig:
    return TopConfig(
        rows=args.cfu_rows,
        cols=args.cfu_cols,
        store_depth=args.cfu_store_depth,
        in_width=args.cfu_in_width,
        acc_width=args.cfu_acc_width,
    )


def write_verilog(config: TopConfig, output: str) -> None:
    top = Top(config)
    ports = top.ports + [ClockSignal("sync"), ResetSignal("sync")]
    v = convert(top, name="Cfu", ports=ports, strip_internal_attrs=True)
    v = re.sub(r"^.*dump_module.*\n", "", v, flags=re.MULTILINE)
    with open(output, "w") as f:
        f.write(v)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CFU top-level Verilog.")
    parser.add_argument("--cfu-rows", default=8, type=int, help="CFU array row count.")
    parser.add_argument("--cfu-cols", default=8, type=int, help="CFU array column count.")
    parser.add_argument("--cfu-store-depth", default=512, type=int, help="CFU scratchpad depth.")
    parser.add_argument("--cfu-in-width", default=8, type=int, help="CFU activation and weight element width in bits.")
    parser.add_argument("--cfu-acc-width", default=32, type=int, help="CFU accumulator width in bits.")
    parser.add_argument("--output", default="top.v", help="Output Verilog path.")
    args = parser.parse_args()

    write_verilog(build_config_from_args(args), args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
