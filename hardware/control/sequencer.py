"""
Compute Sequencer — drives the OS systolic array for one tile.

FSM: IDLE → [PRIME if first] → FEED → FLUSH → [EPILOGUE if last] → DONE

The sequencer does NOT own the requantization pipeline. During EPILOGUE
it walks the PE accumulators and emits them one-per-cycle via epi_valid/epi_data.
An external Epilogue module consumes them.
"""

from amaranth import Array, Module, Signal, signed
from amaranth.lib import data, wiring
from amaranth.lib.wiring import In, Out
from amaranth.utils import ceil_log2


class Sequencer(wiring.Component):
    """Output-stationary compute sequencer.

    Coordinates scratchpad reads, array control signals, and epilogue drain
    for one COMPUTE tile (R×C output block over K inner-dimension steps).

    first/last control:
      first=1: run PRIME (reset PE accumulators). Set on first K-tile.
      last=1:  run EPILOGUE (drain accumulators to external pipeline).
               Set on final K-tile.
    PE accumulators persist across tiles when first=0, enabling K-tiling
    without external result registers.
    """

    def __init__(self, in_width=8, acc_width=32, rows=4, cols=4, scratchpad_depth=512):
        ports = {}

        assert rows >= 2 and cols >= 2, "Array must be at least 2x2"

        self.in_width = in_width
        self.acc_width = acc_width
        self.rows = rows
        self.cols = cols
        self.scratchpad_depth = scratchpad_depth

        scratchpad_addr_width = ceil_log2(scratchpad_depth)

        self.act_layout = data.StructLayout(
            {f"r{r}": signed(in_width) for r in range(rows)}
        )
        self.wgt_layout = data.StructLayout(
            {f"c{c}": signed(in_width) for c in range(cols)}
        )

        # Config
        ports["k_count"] = In(scratchpad_addr_width + 1)

        # Control
        ports["start"] = In(1)
        ports["done"] = Out(1)
        ports["first"] = In(1)
        ports["last"] = In(1)
        ports["state_debug"] = Out(8)
        ports["busy_debug"] = Out(1)

        # Scratchpad interface
        ports["act_rd_addr"] = Out(scratchpad_addr_width)
        ports["wgt_rd_addr"] = Out(scratchpad_addr_width)
        ports["act_swap"] = Out(1)
        ports["wgt_swap"] = Out(1)
        ports["act_rd_data"] = In(self.act_layout)
        ports["wgt_rd_data"] = In(self.wgt_layout)

        # Array control
        ports["arr_psum_load"] = Out(1)
        ports["arr_act_in"] = Out(self.act_layout)
        ports["arr_w_in"] = Out(self.wgt_layout)

        for r in range(rows):
            for c in range(cols):
                ports[f"arr_psum_out_{r}_{c}"] = In(signed(acc_width))

        # Epilogue handshake — sequencer emits psums, external module consumes
        ports["epi_data"] = Out(signed(acc_width))
        ports["epi_index"] = Out(range(rows * cols))
        ports["epi_first"] = Out(1)
        ports["epi_last"] = Out(1)
        ports["epi_done"] = In(1)

        super().__init__(ports)

    def elaborate(self, _platform):
        m = Module()

        cycle = Signal(range(self.scratchpad_depth + self.rows + self.cols))
        num_results = self.rows * self.cols
        flush_cycles = self.rows + self.cols - 2

        # Mux PE psum outputs into a flat array for indexed access
        psum_array = Array(
            getattr(self, f"arr_psum_out_{r}_{c}")
            for r in range(self.rows)
            for c in range(self.cols)
        )

        # Epilogue drain counter
        epi_counter = Signal(range(num_results))

        with m.FSM(name="fsm") as fsm:
            with m.State("IDLE"):
                m.d.comb += self.done.eq(0)
                with m.If(self.start):
                    m.d.sync += cycle.eq(0)
                    with m.If(self.first):
                        m.next = "PRIME"
                    with m.Else():
                        m.next = "FEED"

            with m.State("PRIME"):
                m.d.comb += self.arr_psum_load.eq(1)
                m.d.comb += self.act_rd_addr.eq(0)
                m.d.comb += self.wgt_rd_addr.eq(0)
                m.next = "FEED"

            with m.State("FEED"):
                m.d.comb += self.arr_act_in.eq(self.act_rd_data)
                m.d.comb += self.arr_w_in.eq(self.wgt_rd_data)

                m.d.comb += self.act_rd_addr.eq(cycle + 1)
                m.d.comb += self.wgt_rd_addr.eq(cycle + 1)

                with m.If(cycle == self.k_count - 1):
                    m.d.sync += cycle.eq(0)
                    m.next = "FLUSH"
                with m.Else():
                    m.d.sync += cycle.eq(cycle + 1)

            with m.State("FLUSH"):
                with m.If(cycle == flush_cycles - 1):
                    m.d.sync += cycle.eq(0)
                    with m.If(self.last):
                        m.d.sync += epi_counter.eq(0)
                        m.next = "EPILOGUE"
                    with m.Else():
                        m.next = "DONE"
                with m.Else():
                    m.d.sync += cycle.eq(cycle + 1)

            with m.State("EPILOGUE"):
                m.d.comb += [
                    self.epi_index.eq(epi_counter),
                    self.epi_first.eq(epi_counter == 0),
                    self.epi_last.eq(epi_counter == num_results - 1),
                    self.epi_data.eq(psum_array[epi_counter]),
                ]

                with m.If(epi_counter == num_results - 1):
                    m.next = "EPILOGUE_WAIT"
                with m.Else():
                    m.d.sync += epi_counter.eq(epi_counter + 1)

            with m.State("EPILOGUE_WAIT"):
                with m.If(self.epi_done):
                    m.next = "DONE"

            with m.State("DONE"):
                m.d.comb += self.done.eq(1)
                m.d.comb += self.act_swap.eq(1)
                m.d.comb += self.wgt_swap.eq(1)

                with m.If(~self.start):
                    m.next = "IDLE"
                with m.Else():
                    m.next = "WAIT_START_DEASSERT"

            with m.State("WAIT_START_DEASSERT"):
                m.d.comb += self.done.eq(1)
                with m.If(~self.start):
                    m.next = "IDLE"

        m.d.comb += [
            self.state_debug.eq(
                fsm.ongoing("IDLE") * 0
                | fsm.ongoing("PRIME") * 1
                | fsm.ongoing("FEED") * 2
                | fsm.ongoing("FLUSH") * 3
                | fsm.ongoing("EPILOGUE") * 4
                | fsm.ongoing("EPILOGUE_WAIT") * 5
                | fsm.ongoing("DONE") * 6
                | fsm.ongoing("WAIT_START_DEASSERT") * 7
            ),
            self.busy_debug.eq(~fsm.ongoing("IDLE")),
        ]

        return m
