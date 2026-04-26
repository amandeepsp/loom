"""
Compute Sequencer — drives the WS systolic array for one tile.

Weight-stationary: weights load once, bottom-row-first, then R activation rows
are fed back-to-back. Results from the bottom row are de-skewed, captured
row-wide into one tile buffer, and drained sequentially to the epilogue.

This intentionally implements the simple ADR-006 shape:
  - one result buffer, not ping-pong buffers
  - optional 2-wide drain to cut epilogue drain time in half
  - no compute/drain overlap

Only supports single K-tile workloads where K <= rows.
"""

from amaranth import Array, Module, Signal, signed
from amaranth.lib import data, wiring
from amaranth.lib.memory import Memory
from amaranth.lib.wiring import In, Out
from amaranth.utils import ceil_log2

from hardware.systolic.skew import SkewBuffer


class WSSequencer(wiring.Component):
    """Weight-stationary compute sequencer.

    The result buffer is a deterministic tile store rather than a streaming
    FIFO: STREAM_ROWS writes rows 0..R-1, then DRAIN reads rows 0..R-1 and
    emits one or two adjacent columns per cycle.
    """

    def __init__(
        self,
        in_width=8,
        acc_width=32,
        rows=4,
        cols=4,
        scratchpad_depth=512,
        read_latency=0,
        wide=1,
    ):
        assert rows >= 2 and cols >= 2, "Array must be at least 2x2"
        assert wide in (1, 2), "wide must be 1 or 2"
        if wide == 2:
            assert cols % 2 == 0, "cols must be even when wide=2"

        self.in_width = in_width
        self.acc_width = acc_width
        self.rows = rows
        self.cols = cols
        self.scratchpad_depth = scratchpad_depth
        self.read_latency = read_latency
        self.wide = wide

        scratchpad_addr_width = ceil_log2(scratchpad_depth)
        num_results = rows * cols

        self.act_layout = data.StructLayout(
            {f"r{r}": signed(in_width) for r in range(rows)}
        )
        self.wgt_layout = data.StructLayout(
            {f"c{c}": signed(in_width) for c in range(cols)}
        )

        ports = {
            # Config
            "k_count": In(scratchpad_addr_width + 1),
            # Control
            "start": In(1),
            "done": Out(1),
            "first": In(1),
            "last": In(1),
            "state_debug": Out(8),
            "busy_debug": Out(1),
            "fifo_wr_en_debug": Out(1),
            "fifo_wr_row_debug": Out(range(rows + 1)),
            "capture_countdown_debug": Out(range(rows + cols + 1)),
            "wgt_cycle_debug": Out(range(rows)),
            "cycle_debug": Out(range(scratchpad_depth + 2 * rows + cols)),
            # Scratchpad interface
            "act_rd_addr": Out(scratchpad_addr_width),
            "wgt_rd_addr": Out(scratchpad_addr_width),
            "act_swap": Out(1),
            "wgt_swap": Out(1),
            "act_rd_data": In(self.act_layout),
            "wgt_rd_data": In(self.wgt_layout),
            # Array control
            "arr_w_load": Out(1),
            "arr_act_in": Out(self.act_layout),
            "arr_w_in": Out(self.wgt_layout),
            # Epilogue handshake
            "epi_data": Out(signed(acc_width)),
            "epi_index": Out(range(num_results)),
            "epi_first": Out(1),
            "epi_last": Out(1),
            "epi_done": In(1),
        }

        for c in range(cols):
            ports[f"arr_psum_out_{c}"] = In(signed(acc_width))

        if wide == 2:
            ports["epi_data_1"] = Out(signed(acc_width))

        super().__init__(ports)

    def elaborate(self, _platform):
        m = Module()

        rows = self.rows
        cols = self.cols
        num_results = rows * cols
        wide = self.wide
        has_latency = self.read_latency == 1

        # --- De-skew bottom-row psum outputs into a full result row ---
        m.submodules.psum_deskew = psum_deskew = SkewBuffer(
            cols, self.acc_width, reverse=True
        )
        for c in range(cols):
            m.d.comb += getattr(psum_deskew, f"in_{c}").eq(
                getattr(self, f"arr_psum_out_{c}")
            )
        deskewed = [getattr(psum_deskew, f"out_{c}") for c in range(cols)]

        # --- One row-addressed result buffer ---
        row_layout = data.StructLayout(
            {f"c{c}": signed(self.acc_width) for c in range(cols)}
        )
        m.submodules.result_mem = result_mem = Memory(
            shape=row_layout, depth=rows, init=[]
        )
        wr_port = result_mem.write_port()
        rd_port = result_mem.read_port(domain="comb")

        fifo_wr_row = Signal(range(rows + 1))
        fifo_wr_en = Signal()
        m.d.comb += [
            wr_port.addr.eq(fifo_wr_row),
            wr_port.en.eq(fifo_wr_en),
        ]
        for c in range(cols):
            m.d.comb += getattr(wr_port.data, f"c{c}").eq(deskewed[c])

        # --- Counters and latched controls ---
        cycle = Signal(range(self.scratchpad_depth + 2 * rows + cols))
        wgt_cycle = Signal(range(rows))
        capture_countdown = Signal(range(rows + cols + 1))
        drain_row = Signal(range(rows))
        drain_col = Signal(range(cols))
        drain_idx = Signal(range(num_results))
        latched_first = Signal()
        latched_last = Signal()

        rd_cols = Array(getattr(rd_port.data, f"c{c}") for c in range(cols))
        m.d.comb += rd_port.addr.eq(drain_row)

        with m.FSM(name="ws_seq_fsm") as fsm:
            with m.State("IDLE"):
                with m.If(self.start):
                    m.d.sync += [
                        cycle.eq(0),
                        fifo_wr_row.eq(0),
                        latched_first.eq(self.first),
                        latched_last.eq(self.last),
                    ]
                    with m.If(self.first):
                        m.d.sync += wgt_cycle.eq(0)
                        if has_latency:
                            m.next = "SWAP"
                        else:
                            m.next = "LOAD_WEIGHTS"
                    with m.Else():
                        m.d.sync += capture_countdown.eq(rows + cols - 1)
                        m.next = "STREAM_ROWS"

            with m.State("SWAP"):
                # Synchronous scratchpad reads need one cycle after bank swap
                # before LOAD_WEIGHTS consumes wgt_rd_data.
                m.d.comb += [
                    self.act_swap.eq(latched_first),
                    self.wgt_swap.eq(latched_first),
                    self.wgt_rd_addr.eq(rows - 1),
                ]
                m.next = "LOAD_WEIGHTS"

            with m.State("LOAD_WEIGHTS"):
                m.d.comb += [
                    self.arr_w_load.eq(1),
                    self.arr_w_in.eq(self.wgt_rd_data),
                ]
                if not has_latency:
                    m.d.comb += [
                        self.act_swap.eq(latched_first & (wgt_cycle == 0)),
                        self.wgt_swap.eq(latched_first & (wgt_cycle == 0)),
                    ]

                if has_latency:
                    with m.If(wgt_cycle == rows - 1):
                        m.d.comb += [
                            self.wgt_rd_addr.eq(0),
                            self.act_rd_addr.eq(0),
                        ]
                    with m.Else():
                        m.d.comb += self.wgt_rd_addr.eq(rows - 2 - wgt_cycle)
                else:
                    m.d.comb += self.wgt_rd_addr.eq(rows - 1 - wgt_cycle)

                with m.If(wgt_cycle == rows - 1):
                    m.d.sync += [
                        cycle.eq(0),
                        capture_countdown.eq(rows + cols - 1),
                    ]
                    m.next = "STREAM_ROWS"
                with m.Else():
                    m.d.sync += wgt_cycle.eq(wgt_cycle + 1)

            with m.State("STREAM_ROWS"):
                if has_latency:
                    with m.If(cycle < rows):
                        m.d.comb += [
                            self.act_rd_addr.eq(cycle + 1),
                            self.arr_act_in.eq(self.act_rd_data),
                        ]
                else:
                    with m.If(cycle < rows):
                        m.d.comb += [
                            self.act_rd_addr.eq(cycle),
                            self.arr_act_in.eq(self.act_rd_data),
                        ]

                with m.If(capture_countdown > 0):
                    m.d.sync += capture_countdown.eq(capture_countdown - 1)
                with m.Elif(fifo_wr_row < rows):
                    m.d.comb += fifo_wr_en.eq(1)
                    m.d.sync += fifo_wr_row.eq(fifo_wr_row + 1)

                with m.If(cycle == 2 * rows + cols - 2):
                    m.d.sync += [
                        cycle.eq(0),
                        drain_row.eq(0),
                        drain_col.eq(0),
                        drain_idx.eq(0),
                    ]
                    with m.If(latched_last):
                        m.next = "DRAIN"
                    with m.Else():
                        m.next = "DONE"
                with m.Else():
                    m.d.sync += cycle.eq(cycle + 1)

            with m.State("DRAIN"):
                m.d.comb += [
                    self.epi_first.eq(drain_idx == 0),
                    self.epi_last.eq(drain_idx == num_results - wide),
                    self.epi_index.eq(drain_idx),
                    self.epi_data.eq(rd_cols[drain_col]),
                ]
                if wide == 2:
                    m.d.comb += self.epi_data_1.eq(rd_cols[drain_col + 1])

                with m.If(drain_idx == num_results - wide):
                    m.next = "DRAIN_WAIT"
                with m.Else():
                    m.d.sync += drain_idx.eq(drain_idx + wide)
                    with m.If(drain_col + wide >= cols):
                        m.d.sync += [
                            drain_col.eq(0),
                            drain_row.eq(drain_row + 1),
                        ]
                    with m.Else():
                        m.d.sync += drain_col.eq(drain_col + wide)

            with m.State("DRAIN_WAIT"):
                with m.If(self.epi_done):
                    m.next = "DONE"

            with m.State("DONE"):
                m.d.comb += [
                    self.done.eq(1),
                    self.act_swap.eq(1),
                    self.wgt_swap.eq(1),
                ]
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
                | fsm.ongoing("SWAP") * 1
                | fsm.ongoing("LOAD_WEIGHTS") * 2
                | fsm.ongoing("STREAM_ROWS") * 3
                | fsm.ongoing("DRAIN") * 4
                | fsm.ongoing("DRAIN_WAIT") * 5
                | fsm.ongoing("DONE") * 6
                | fsm.ongoing("WAIT_START_DEASSERT") * 7
            ),
            self.busy_debug.eq(~fsm.ongoing("IDLE")),
            self.fifo_wr_en_debug.eq(fifo_wr_en),
            self.fifo_wr_row_debug.eq(fifo_wr_row),
            self.capture_countdown_debug.eq(capture_countdown),
            self.wgt_cycle_debug.eq(wgt_cycle),
            self.cycle_debug.eq(cycle),
        ]

        return m
