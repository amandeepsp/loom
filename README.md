# accel — RISC-V Custom Function Unit Accelerator

Hardware ML accelerator built around a VexRiscv soft-core with a Custom Function Unit (CFU) running on a LiteX SoC.

The current stack is:
- host Python sends requests over UART
- bare-metal Zig firmware decodes them
- the firmware issues a `CUSTOM_0` instruction
- CFU hardware executes a 4-lane SIMD MAC (`mac4`)

```
Host PC ──UART──► VexRiscv firmware ──CUSTOM_0 insn──► CFU hardware
                       ▲                                    │
                       └────────────────result──────────────┘
```

## Layout

| Directory    | Language       | Purpose                                       |
|-------------|----------------|-----------------------------------------------|
| `hardware/` | Python/Amaranth | CFU RTL — instruction definitions, SIMD MAC4  |
| `firmware/` | Zig            | Bare-metal firmware: UART link protocol, CFU driver |
| `host/`     | Python          | Host-side client that talks to the simulator    |
| `top.v`     | Verilog (generated) | CFU Verilog output consumed by LiteX       |

## Quick Start (Sim)

```bash
# 1. Generate the Verilog CFU
cd hardware && python top.py && cd ..

# 2. Make sure the LiteX sim build artifacts exist
#    Needed by firmware/build.zig:
#    - build/sim/csr.json
#    - build/sim/software/bios/crt0.o

# 3. Build the firmware
cd firmware && zig build && cd ..   # defaults to -Dbuild-dir=../build/sim

# 4. Run end-to-end test through the LiteX simulator
uv run python host/client.py --test -v

# 5. Run the NumPy example through the simulator
uv run python host/numpy_sim.py
```

If `build/sim/software/bios/crt0.o` is missing, rebuild it explicitly:

```bash
riscv64-elf-gcc -c -march=rv32i2p0_m -mabi=ilp32 -D__vexriscv__ \
  "$(python -c "import litex.soc.cores.cpu.vexriscv, os; print(os.path.join(os.path.dirname(litex.soc.cores.cpu.vexriscv.__file__), 'crt0.S'))")" \
  -o build/sim/software/bios/crt0.o
```

The firmware build reads `csr.json` and `crt0.o` directly from the LiteX build
directory, so CSR addresses stay in sync with the SoC.

## Build And Test On Tang Nano 20K

Prerequisites: add `oss-cad-suite` to `PATH`.

```bash
# 1. Build the LiteX SoC for the board
uv run python -m litex_boards.targets.sipeed_tang_nano_20k \
    --build --toolchain apicula \
    --cpu-type vexriscv --cpu-variant full+cfu --cpu-cfu top.v

# 2. Flash the bitstream
uv run python -m litex_boards.targets.sipeed_tang_nano_20k --flash

# 3. Build firmware targeting the hardware SoC
cd firmware && zig build -Dbuild-dir=../build/sipeed_tang_nano_20k && cd ..

# 4. Run tests on the real FPGA (resets board, uploads firmware, runs tests)
uv run python host/client.py --test --serial /dev/ttyUSB1 -v
```

## Notes

- `host/client.py` supports both `SimLink` and `SerialLink`.
- The response header includes `cycles_lo`, which is reserved for lightweight timing/telemetry.

## Docs

- `docs/cfu.md`
- `docs/firmware.md`
- `docs/tutorial/`
