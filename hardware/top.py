"""CFU top-level: single MAC4 instruction at funct3=0."""

from amaranth.back.verilog import convert

from cfu import Cfu
from mac import SimdMac4


class Top(Cfu):
    def elab_instructions(self, m):
        m.submodules["mac4"] = mac4 = SimdMac4()
        return {0: mac4}


if __name__ == "__main__":
    top = Top()
    v = convert(top, name="Cfu", ports=top.ports)
    with open("../top.v", "w") as f:
        f.write(v)
    print("Wrote ../top.v")
