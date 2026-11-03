from amaranth import Cat, Mux, Signal, signed
from amaranth.lib.wiring import In, Out
from hardware.cfu import Instruction


class ComputeStartInstruction(Instruction):
    """funct3=0: Non-blocking — latch config, pulse seq.start, return immediately."""

    seq_first: Out(1)
    seq_last: Out(1)
    seq_keep_wgt: Out(1)
    seq_keep_act: Out(1)
    seq_k_count: Out(10)
    seq_start: Out(1)

    def elaborate(self, platform):
        m = super().elaborate(platform)

        with m.If(self.start):
            m.d.sync += [
                self.seq_first.eq(self.in0[0]),
                self.seq_last.eq(self.in0[1]),
                # Keep the weight bank (no swap): consecutive M tiles of one
                # N tile reuse the weights already in the compute bank.
                self.seq_keep_wgt.eq(self.in0[2]),
                # Keep the activation bank: consecutive N tiles of one M tile
                # reuse the activations already in the compute bank.
                self.seq_keep_act.eq(self.in0[3]),
                self.seq_k_count.eq(self.in1),
            ]

        # Pulse seq.start one cycle after latch
        m.d.sync += self.seq_start.eq(self.start)

        m.d.comb += self.output.eq(1)
        self.signal_done(m)
        return m


class ComputeWaitInstruction(Instruction):
    """funct3=1: Blocking — holds done=0 until sequencer completes."""

    seq_done: In(1)

    def elaborate(self, platform):
        m = super().elaborate(platform)

        m.d.comb += self.done.eq(self.seq_done)
        m.d.comb += self.output.eq(1)
        return m


class EpiParamInstruction(Instruction):
    """funct3=2: Single-cycle — write per-channel quantization parameter."""

    wr_addr: Out(32)
    wr_data: Out(signed(32))
    wr_sel: Out(7)
    wr_en: Out(1)

    def elaborate(self, platform):
        m = super().elaborate(platform)

        m.d.comb += [
            self.wr_addr.eq(self.in0),
            self.wr_data.eq(self.in1),
            self.wr_sel.eq(self.funct7),
            self.wr_en.eq(self.start),
        ]

        self.signal_done(m)
        return m


class ConfigInstruction(Instruction):
    """funct3=3: Single-cycle — write global config register."""

    output_offset: Out(signed(16))
    activation_min: Out(signed(8))
    activation_max: Out(signed(8))
    # funct7=3: result-LUT word in1 at index in0 (0..63); funct7=4: LUT enable = in1[0]
    lut_wr_addr: Out(6)
    lut_wr_data: Out(32)
    lut_wr_en: Out(1)
    lut_en: Out(1)

    def elaborate(self, platform):
        m = super().elaborate(platform)

        m.d.comb += [
            self.lut_wr_addr.eq(self.in0[:6]),
            self.lut_wr_data.eq(self.in1),
            self.lut_wr_en.eq(self.start & (self.funct7 == 3)),
        ]
        with m.If(self.start & (self.funct7 == 4)):
            m.d.sync += self.lut_en.eq(self.in1[0])

        with m.If(self.start):
            with m.Switch(self.funct7):
                with m.Case(0):
                    m.d.sync += self.output_offset.eq(self.in1[:16])
                with m.Case(1):
                    m.d.sync += self.activation_min.eq(self.in1[:8])
                with m.Case(2):
                    m.d.sync += self.activation_max.eq(self.in1[:8])

        self.signal_done(m)
        return m


class ReadResultInstruction(Instruction):
    """funct3=4: Single-cycle — read results from the epilogue store.

    funct7=0: one INT8 result (sign-extended) at index in0.
    funct7=1: four INT8 results at indices in0..in0+3 (in0 a multiple of 4),
              packed little-endian into one 32-bit word.
    funct7=2: four INT8 results down a column: rows r0..r0+3 of column c,
              in0 = r0 * cols + c (r0 a multiple of 4), packed little-endian.
    """

    out_addr: Out(32)
    out_data: In(signed(8))
    out_data4: In(32)
    out_data4col: In(32)

    def elaborate(self, platform):
        m = super().elaborate(platform)

        m.d.comb += self.out_addr.eq(self.in0)
        with m.If(self.funct7 == 1):
            m.d.comb += self.output.eq(self.out_data4)
        with m.Elif(self.funct7 == 2):
            m.d.comb += self.output.eq(self.out_data4col)
        with m.Else():
            m.d.comb += self.output.eq(self.out_data)

        self.signal_done(m)
        return m


class WriteActInstruction(Instruction):
    """funct3=6: CPU write into the activation scratchpad fill bank.

    funct7=0: write line {in1, in0} (in0 = rows 0..3, the lower DMA bytes)
              at the write pointer, then increment the pointer.
    funct7=1: set the write pointer to in0.

    Lets the CPU stream im2col words straight into the CFU instead of through
    a packed DRAM buffer and the activation DMA. Software never mixes it with
    an activation DMA in flight.
    """

    wr_addr: Out(16)
    wr_data: Out(64)
    wr_en: Out(1)

    def elaborate(self, platform):
        m = super().elaborate(platform)

        ptr = Signal(16)
        m.d.comb += [
            self.wr_addr.eq(ptr),
            self.wr_data.eq(Cat(self.in0, self.in1)),
            self.wr_en.eq(self.start & (self.funct7 == 0)),
        ]
        with m.If(self.start):
            with m.If(self.funct7 == 1):
                m.d.sync += ptr.eq(self.in0)
            with m.Else():
                m.d.sync += ptr.eq(ptr + 1)

        self.signal_done(m)
        return m


class LineGenInstruction(Instruction):
    """funct3=7: hardware depthwise im2col (LineGen) control.

    funct7=0: geometry: in0 = iw[0:10] | ih[10:20] | stride2[20] | tworow[21];
              in1 = plane bytes [0:14].
    funct7=1: in0[0] = plane-buffer mode (activation DMA fills LineGen's
              plane buffer instead of the activation scratchpad);
              in1[0:8] = input zero point.
    funct7=2: GEN one tile: in0 = row0[0:16] (byte offset of row iy0 within a
              plane) | iy0[16:27]; in1 = ix0[0:12]. Non-blocking; the 72
              lines land in the activation fill bank.
    funct7=3: wait until GEN is done (blocking).
    """

    iw: Out(10)
    ih: Out(10)
    plane: Out(14)
    stride2: Out(1)
    tworow: Out(1)
    zp: Out(8)
    planebuf_mode: Out(1)
    gen_start: Out(1)
    iy0: Out(signed(11))
    row0: Out(signed(16))
    ix0: Out(signed(12))
    gen_busy: In(1)

    def elaborate(self, platform):
        m = super().elaborate(platform)

        # GEN operands are latched and the start pulse issued a cycle later,
        # so LineGen never sees them change mid-tile.
        m.d.sync += self.gen_start.eq(self.start & (self.funct7 == 2))
        with m.If(self.start):
            with m.Switch(self.funct7):
                with m.Case(0):
                    m.d.sync += [
                        self.iw.eq(self.in0[0:10]),
                        self.ih.eq(self.in0[10:20]),
                        self.stride2.eq(self.in0[20]),
                        self.tworow.eq(self.in0[21]),
                        self.plane.eq(self.in1[0:14]),
                    ]
                with m.Case(1):
                    m.d.sync += [
                        self.planebuf_mode.eq(self.in0[0]),
                        self.zp.eq(self.in1[0:8]),
                    ]
                with m.Case(2):
                    m.d.sync += [
                        self.row0.eq(self.in0[0:16]),
                        self.iy0.eq(self.in0[16:27]),
                        self.ix0.eq(self.in1[0:12]),
                    ]

        m.d.comb += self.output.eq(0)
        # The command payload (funct7) is only valid in the start cycle: the
        # CPU may change it while a multi-cycle op waits, so remember that
        # this op is a WAIT instead of re-reading funct7.
        waiting = Signal()
        with m.If(self.start):
            m.d.sync += waiting.eq(self.funct7 == 3)
        is_wait = Signal()
        m.d.comb += is_wait.eq(Mux(self.start, self.funct7 == 3, waiting))
        with m.If(is_wait):
            # Busy covers the registered start pulse too (gen_start feeds
            # LineGen.busy), so a WAIT right after GEN still blocks.
            m.d.comb += self.done.eq(~self.gen_busy & ~self.gen_start)
        with m.Else():
            self.signal_done(m)
        return m


class PerfCounterInstruction(Instruction):
    """funct3=5: reset or read one hardware performance counter.

    funct7=0 resets every counter. Selectors 1..3 return the counters wired
    by :class:`hardware.top.Top`. Keeping the counters in the top level lets
    them observe the DMA ports and sequencer at the same time.
    """

    seq_busy_cycles: In(32)
    dma_active_cycles: In(32)
    overlap_cycles: In(32)
    reset_counters: Out(1)

    def elaborate(self, platform):
        m = super().elaborate(platform)

        m.d.comb += [
            self.output.eq(0),
            self.reset_counters.eq(self.start & (self.funct7 == 0)),
        ]
        with m.Switch(self.funct7):
            for selector, counter in enumerate(
                (
                    self.seq_busy_cycles,
                    self.dma_active_cycles,
                    self.overlap_cycles,
                ),
                start=1,
            ):
                with m.Case(selector):
                    m.d.comb += self.output.eq(counter)

        self.signal_done(m)
        return m
