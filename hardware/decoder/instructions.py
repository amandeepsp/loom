from amaranth import signed
from amaranth.lib.wiring import In, Out
from hardware.cfu import Instruction


class ComputeStartInstruction(Instruction):
    """funct3=0: Non-blocking — latch config, pulse seq.start, return immediately."""

    seq_first: Out(1)
    seq_last: Out(1)
    seq_k_count: Out(10)
    seq_start: Out(1)

    def elaborate(self, platform):
        m = super().elaborate(platform)

        with m.If(self.start):
            m.d.sync += [
                self.seq_first.eq(self.in0[0]),
                self.seq_last.eq(self.in0[1]),
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

    def elaborate(self, platform):
        m = super().elaborate(platform)

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
    """funct3=4: Single-cycle — read INT8 result from epilogue store."""

    out_addr: Out(32)
    out_data: In(signed(8))

    def elaborate(self, platform):
        m = super().elaborate(platform)

        m.d.comb += self.out_addr.eq(self.in0)
        m.d.comb += self.output.eq(self.out_data)

        self.signal_done(m)
        return m
