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
| `driver/`   | Zig            | Host-side native driver (serial)               |
| `shared/`   | Zig            | Wire protocol + CFU inline asm (firmware & driver) |
| `sim/`      | C++/Renode     | Renode simulation: platform def, Verilated CFU |
| `top.v`     | Verilog (generated) | CFU Verilog output consumed by LiteX       |

## Quick Start (Renode Simulation)

No FPGA needed. The full SoC runs in Renode with the actual CFU Verilog
co-simulated via Verilator.

Prerequisites: `renode`, `verilator`, `cmake`, `zig`, `uv`, a RISC-V GCC toolchain.

```bash
# 1. Generate the Verilog CFU
uv run python hardware/top.py

# 2. Build LiteX SoC metadata (csr.json + crt0.o)
uv run python sim/sim.py

# 3. Build the Verilated CFU shared library
mkdir -p sim/cfu/build && cd sim/cfu/build
cmake .. && make -j$(nproc)
cd ../../..

# 4. Build firmware
zig build firmware

# 5. Boot in Renode
renode --disable-xwt --console -e "include @sim/accel.resc; start"
# → UART prints "[link] ready"
```

To run the driver E2E test against the simulation (in a second terminal):

```bash
# Bridge Renode's TCP UART (port 3456) to a PTY
socat pty,raw,echo=0,link=/tmp/renode-uart tcp:localhost:3456 &

# Run the driver — ping + mac4 [1,2,3,4]·[5,6,7,8] = 70
zig build run -- /tmp/renode-uart
```

The firmware build reads `csr.json` and `crt0.o` from the LiteX build
directory (`build/sim/` by default), so CSR addresses stay in sync with the SoC.

## Build And Test On Tang Nano 20K

Prerequisites: `oss-cad-suite` on `PATH`.

```bash
# 1. Build the LiteX SoC for the board
uv run python -m litex_boards.targets.sipeed_tang_nano_20k \
    --build --toolchain apicula \
    --cpu-type vexriscv --cpu-variant full+cfu --cpu-cfu top.v

# 2. Flash the bitstream
uv run python -m litex_boards.targets.sipeed_tang_nano_20k --flash

# 3. (Optional) Reset the board
openFPGALoader --board tangnano20k --reset

# 4. Build firmware targeting the hardware SoC
zig build firmware -Dbuild-dir=build/sipeed_tang_nano_20k

# 5. Upload firmware via serial boot
uv run litex_term /dev/ttyUSB1 --kernel zig-out/bin/firmware.bin
# → wait for "serialboot" prompt, then firmware loads and prints "[link] ready"

# 6. Run driver against the real FPGA
zig build run -- /dev/ttyUSB1
```

## Notes

- The response header includes `cycles_lo`, reserved for lightweight timing/telemetry.
- The driver accepts a port path as its first argument (defaults to `/dev/ttyUSB1`).
- For simulation, the Verilated CFU runs the actual RTL cycle-accurately — any
  hardware bug will reproduce without the FPGA.

## Docs

- `docs/cfu.md`
- `docs/firmware.md`
- `docs/tutorial/`
