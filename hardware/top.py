from amaranth.back.verilog import convert
from cfu import Cfu
from mac import SimdMac4


class Top(Cfu):
    def elab_instructions(self, m):
        mac_inst = SimdMac4()
        m.submodules += mac_inst
        return {0: mac_inst}


top = Top()
v = convert(top, name="Cfu", ports=top.ports)
with open("top.v", "w") as f:
    f.write(v)
