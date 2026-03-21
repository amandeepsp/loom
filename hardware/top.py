"""CFU top-level: single MAC4 instruction at funct3=0."""

from amaranth import ClockSignal, ResetSignal
from amaranth.back.verilog import convert
from cfu import Cfu
from mac import SimdMac4
from rw import ReadRegs, WriteRegs


class Top(Cfu):
    def elab_instructions(self, m):
        m.submodules["mac4"] = mac4 = SimdMac4()
        m.submodules["read"] = read = ReadRegs()
        m.submodules["write"] = write = WriteRegs()

        m.d.comb += [
            read.input_offset.eq(write.input_offset),
            read.accumulator.eq(mac4.accumulator),
            mac4.reset_acc.eq(write.reset_acc),
            mac4.input_offset.eq(write.input_offset),
        ]
        return {0: mac4, 1: read, 2: write}


if __name__ == "__main__":
    import re

    top = Top()
    ports = top.ports + [ClockSignal("sync"), ResetSignal("sync")]
    v = convert(top, name="Cfu", ports=ports, strip_internal_attrs=True)
    # Yosys emits dump_module debug regs that break GTKWave's VCD parser.
    v = re.sub(r"^.*dump_module.*\n", "", v, flags=re.MULTILINE)
    with open("top.v", "w") as f:
        f.write(v)
    print("Wrote /top.v")
