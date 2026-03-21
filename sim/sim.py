#!/usr/bin/env python3
"""Build LiteX SoC metadata for Renode simulation.

Produces build/sim/csr.json and build/sim/software/bios/crt0.o
without synthesizing gateware. Run with: uv run python sim/sim.py
"""

import os

from litex.build.sim import SimPlatform
from litex.build.generic_platform import Pins, Subsignal
from litex.soc.integration.soc_core import SoCCore
from litex.soc.integration.builder import Builder

# Minimal I/O — only serial pins so LiteX creates a standard LiteX_UART.
_io = [
    ("serial", 0,
        Subsignal("tx", Pins(1)),
        Subsignal("rx", Pins(1)),
    ),
]


class SimSoC(SoCCore):
    # Override SRAM origin to match firmware/linker.ld (0x10000000).
    mem_map = {**SoCCore.mem_map, "sram": 0x10000000}

    def __init__(self):
        platform = SimPlatform("SIM", _io)
        super().__init__(
            platform,
            clk_freq=48_000_000,
            cpu_type="vexriscv",
            cpu_variant="full+cfu",
            cpu_cfu=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "top.v"),
            integrated_rom_size=0x00020000,     # 128 KB
            integrated_sram_size=0x00002000,    #   8 KB
            integrated_main_ram_size=0x00800000, #   8 MB
            with_uart=True,
            with_timer=True,
        )


def main():
    soc = SimSoC()
    builder = Builder(
        soc,
        output_dir="build/sim",
        compile_gateware=False,
        compile_software=True,
    )
    # SimPlatform's Verilator toolchain errors on gateware finalization
    # (no clock domain) even with compile_gateware=False.  The artifacts
    # we need — csr.json and crt0.o — are already written by this point.
    try:
        builder.build()
    except NotImplementedError:
        pass

    print(f"  csr.json → {builder.output_dir}/csr.json")
    print(f"  crt0.o   → {builder.output_dir}/software/bios/crt0.o")


if __name__ == "__main__":
    main()
