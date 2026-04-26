#!/usr/bin/env python3
"""LiteX Verilator simulation with CFU + DMA engines for e2e testing.

Usage:
    # 1. Generate CFU Verilog
    just verilog

    # 2. Generate SoC (csr.json) without compiling gateware
    uv run python -m soc.sim --no-compile-gateware

    # 3. Build firmware against sim CSRs
    zig build firmware -Dbuild-dir=build/sim

    # 4. Run simulation with firmware
    uv run python -m soc.sim --sdram-init=zig-out/bin/firmware.bin
"""

import argparse
import math
import os
import sys

from migen import *

from litex.build.generic_platform import *
from litex.build.sim import SimPlatform
from litex.build.sim.config import SimConfig
from litex.soc.integration.soc_core import *
from litex.soc.integration.soc import SoCRegion
from litex.soc.integration.builder import *
from litex.soc.integration.common import get_mem_data, get_boot_address
from litex.soc.interconnect.csr import AutoCSR
from litex.soc.cores.cpu import CPUS

from litedram import modules as litedram_modules
from litedram.modules import parse_spd_hexdump
from litedram.phy.model import sdram_module_nphases, get_sdram_phy_settings, SDRAMPHYModel
from litedram.frontend.dma import LiteDRAMDMAReader

# IOs ----------------------------------------------------------------------------------------------

_io = [
    ("sys_clk", 0, Pins(1)),
    ("sys_rst", 0, Pins(1)),
    ("serial", 0,
        Subsignal("source_valid", Pins(1)),
        Subsignal("source_ready", Pins(1)),
        Subsignal("source_data",  Pins(8)),
        Subsignal("sink_valid",   Pins(1)),
        Subsignal("sink_ready",   Pins(1)),
        Subsignal("sink_data",    Pins(8)),
    ),
]

# Platform -----------------------------------------------------------------------------------------

class Platform(SimPlatform):
    def __init__(self):
        SimPlatform.__init__(self, "SIM", _io)

# CFU DMA adapter ----------------------------------------------------------------------------------

class CfuDmaAdapter(Module):
    """Bridges a LiteDRAMDMAReader stream to CFU scratchpad write ports.

    The DMA reader produces `data_width`-bit words from DRAM.  This adapter
    converts the stream into address/data/enable signals for the CFU's
    scratchpad DMA write port.
    """

    def __init__(self, dma_reader, scratchpad_depth):
        addr_bits = (scratchpad_depth - 1).bit_length()

        self.addr = Signal(addr_bits)
        self.data = Signal(len(dma_reader.source.data))
        self.en   = Signal()

        addr_counter = Signal(addr_bits)

        self.comb += [
            self.data.eq(dma_reader.source.data),
            self.en.eq(dma_reader.source.valid),
            self.addr.eq(addr_counter),
            dma_reader.source.ready.eq(1),
        ]

        self.sync += [
            If(dma_reader.source.valid & dma_reader.source.ready,
                If(dma_reader.source.last,
                    addr_counter.eq(0),
                ).Else(
                    addr_counter.eq(addr_counter + 1),
                )
            ),
            If(~dma_reader.enable,
                addr_counter.eq(0),
            ),
        ]

# Simulation SoC -----------------------------------------------------------------------------------

class AccelSimSoC(SoCCore):
    def __init__(self,
        cfu_rows        = 8,
        cfu_cols        = 8,
        cfu_store_depth = 512,
        cfu_in_width    = 8,
        sdram_module    = "MT48LC16M16",
        sdram_data_width = 32,
        sdram_verbosity  = 0,
        sdram_init       = [],
        sim_debug        = False,
        trace_reset_on   = False,
        **kwargs):

        platform     = Platform()
        sys_clk_freq = int(1e6)

        # CRG --------------------------------------------------------------------------------------
        self.crg = CRG(platform.request("sys_clk"))

        # SoCCore ----------------------------------------------------------------------------------
        SoCCore.__init__(self, platform, clk_freq=sys_clk_freq,
            ident="Accel CFU Simulation",
            **kwargs)

        # SDRAM ------------------------------------------------------------------------------------
        if not self.integrated_main_ram_size:
            sdram_clk_freq   = int(100e6)
            sdram_module_cls = getattr(litedram_modules, sdram_module)
            sdram_rate       = "1:{}".format(sdram_module_nphases[sdram_module_cls.memtype])
            sdram_module_obj = sdram_module_cls(sdram_clk_freq, sdram_rate)
            self.sdrphy = SDRAMPHYModel(
                module     = sdram_module_obj,
                data_width = sdram_data_width,
                clk_freq   = sdram_clk_freq,
                verbosity  = sdram_verbosity,
                init       = sdram_init,
            )
            self.add_sdram("sdram",
                phy                     = self.sdrphy,
                module                  = sdram_module_obj,
                l2_cache_size           = kwargs.get("l2_size", 8192),
                l2_cache_min_data_width = kwargs.get("min_l2_data_width", 128),
                l2_cache_reverse        = False,
            )
            if sdram_init:
                self.add_constant("SDRAM_TEST_DISABLE")
            else:
                self.add_constant("MEMTEST_DATA_SIZE", 8 * 1024)
                self.add_constant("MEMTEST_ADDR_SIZE", 8 * 1024)

            # CFU DMA engines ------------------------------------------------------------------
            line_width = cfu_rows * cfu_in_width  # 64 for 8×8

            act_port = self.sdram.crossbar.get_port(mode="read", data_width=line_width)
            act_dma = LiteDRAMDMAReader(act_port, fifo_depth=16, with_csr=True)
            act_adapter = CfuDmaAdapter(act_dma, cfu_store_depth)
            # Use flat names so CSRs become act_dma_{base,length,...}
            self.submodules.act_dma = act_dma
            self.submodules += act_adapter

            wgt_port = self.sdram.crossbar.get_port(mode="read", data_width=line_width)
            wgt_dma = LiteDRAMDMAReader(wgt_port, fifo_depth=16, with_csr=True)
            wgt_adapter = CfuDmaAdapter(wgt_dma, cfu_store_depth)
            self.submodules.wgt_dma = wgt_dma
            self.submodules += wgt_adapter

            # Wire DMA adapters → CFU Verilog ports --------------------------------------------
            if hasattr(self.cpu, "cfu_params"):
                self.cpu.cfu_params.update({
                    "i_dma_act__addr": act_adapter.addr,
                    "i_dma_act__data": act_adapter.data,
                    "i_dma_act__en":   act_adapter.en,
                    "i_dma_wgt__addr": wgt_adapter.addr,
                    "i_dma_wgt__data": wgt_adapter.data,
                    "i_dma_wgt__en":   wgt_adapter.en,
                })

        # Debug ------------------------------------------------------------------------------------
        if sim_debug:
            platform.add_debug(self, reset=1 if trace_reset_on else 0)
        else:
            self.comb += platform.trace.eq(1)

# Build --------------------------------------------------------------------------------------------

def main():
    from litex.build.parser import LiteXArgumentParser
    parser = LiteXArgumentParser(description="Accel CFU Simulation")
    parser.set_platform(SimPlatform)

    # CFU parameters.
    parser.add_argument("--cfu-rows",         default=8,    type=int,   help="CFU array row count.")
    parser.add_argument("--cfu-cols",         default=8,    type=int,   help="CFU array column count.")
    parser.add_argument("--cfu-store-depth",  default=512,  type=int,   help="CFU scratchpad depth.")
    parser.add_argument("--cfu-in-width",     default=8,    type=int,   help="CFU input element width in bits.")

    # SDRAM.
    parser.add_argument("--sdram-module",     default="MT48LC16M16",    help="Select SDRAM chip.")
    parser.add_argument("--sdram-data-width", default=32,   type=int,   help="Set SDRAM chip data width.")
    parser.add_argument("--sdram-init",       default=None,             help="SDRAM init file (.bin or .json).")
    parser.add_argument("--sdram-verbosity",  default=0,    type=int,   help="Set SDRAM verbosity.")

    # Debug.
    parser.add_argument("--sim-debug",        action="store_true",      help="Add simulation debugging modules.")
    parser.add_argument("--non-interactive",  action="store_true",      help="Run without user input.")

    args = parser.parse_args()

    soc_kwargs = soc_core_argdict(args)

    # Force VexRiscv with CFU.
    soc_kwargs["cpu_type"]    = "vexriscv"
    soc_kwargs["cpu_variant"] = "full+cfu"
    soc_kwargs["cpu_cfu"]     = os.path.join(os.path.dirname(__file__), "..", "top.v")

    # UART → sim console.
    if soc_kwargs.get("uart_name") == "serial":
        soc_kwargs["uart_name"] = "sim"

    sys_clk_freq = int(1e6)
    sim_config   = SimConfig()
    sim_config.add_clocker("sys_clk", freq_hz=sys_clk_freq)
    # Always use serial2console; tools/sim_run.py bridges stdio<->TCP for tests.
    # (LiteX serial2tcp is broken on macOS — bytes never flow to the client.)
    sim_config.add_module("serial2console", "serial")

    # Prepare SDRAM init data.
    sdram_init = []
    ram_boot_address = None
    if args.sdram_init is not None:
        conf_soc = AccelSimSoC(
            cfu_rows        = args.cfu_rows,
            cfu_cols        = args.cfu_cols,
            cfu_store_depth = args.cfu_store_depth,
            cfu_in_width    = args.cfu_in_width,
            **soc_kwargs,
        )
        sdram_init = get_mem_data(args.sdram_init,
            data_width = conf_soc.bus.data_width,
            endianness = conf_soc.cpu.endianness,
            offset     = conf_soc.mem_map["main_ram"],
        )
        ram_boot_address = get_boot_address(args.sdram_init)

    soc = AccelSimSoC(
        cfu_rows         = args.cfu_rows,
        cfu_cols         = args.cfu_cols,
        cfu_store_depth  = args.cfu_store_depth,
        cfu_in_width     = args.cfu_in_width,
        sdram_module     = args.sdram_module,
        sdram_data_width = args.sdram_data_width,
        sdram_verbosity  = args.sdram_verbosity,
        sdram_init       = sdram_init,
        sim_debug        = args.sim_debug,
        trace_reset_on   = int(float(getattr(args, "trace_start", 0))) > 0,
        **soc_kwargs,
    )

    if ram_boot_address is not None:
        if ram_boot_address == 0:
            ram_boot_address = soc.mem_map["main_ram"]
        soc.add_constant("ROM_BOOT_ADDRESS", ram_boot_address)

    builder = Builder(soc, **parser.builder_argdict)
    builder.build(
        sim_config  = sim_config,
        interactive = not args.non_interactive,
        **parser.toolchain_argdict,
    )

if __name__ == "__main__":
    main()
