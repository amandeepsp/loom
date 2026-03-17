# accel — RISC-V Custom Function Unit Accelerator

Hardware ML accelerator built around a VexRiscv soft-core with a Custom Function Unit (CFU) running on a LiteX SoC.

## What this project does (5-minute overview)

A host PC sends computation requests over UART to a RISC-V CPU (VexRiscv) running inside a simulator (or on an FPGA). The CPU has a hardware accelerator — a **Custom Function Unit** — wired directly into the instruction pipeline. When the firmware receives a request, it issues a single custom RISC-V instruction (`CUSTOM_0`) that executes the operation in hardware (one cycle), then sends the result back.

The current accelerator implements **SIMD MAC4**: a 4-lane multiply-accumulate over packed byte vectors with a configurable input offset — the core kernel in quantised neural-network inference.

```
Host PC ──UART──► VexRiscv firmware ──CUSTOM_0 insn──► CFU hardware
                       ▲                                    │
                       └────────────────result──────────────┘
```

### Project layout

| Directory    | Language       | Purpose                                       |
|-------------|----------------|-----------------------------------------------|
| `hardware/` | Python/Amaranth | CFU RTL — instruction definitions, SIMD MAC4  |
| `firmware/` | Zig            | Bare-metal firmware: UART link protocol, CFU driver |
| `host/`     | Python          | Host-side client that talks to the simulator    |
| `top.v`     | Verilog (generated) | CFU Verilog output consumed by LiteX       |

### Quick start (simulator)

```bash
# 1. Generate the Verilog CFU
cd hardware && python top.py && cd ..

# 2. Build the firmware (reads csr.json + crt0.o from the LiteX build dir)
cd firmware && zig build && cd ..   # defaults to -Dbuild-dir=../build/sim

# 3. Run end-to-end test through the LiteX simulator
uv run python host/client.py --test -v
```

### Build & test on Tang Nano 20K

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

The firmware build reads `csr.json` and `crt0.o` directly from the LiteX build
directory, so CSR addresses are always in sync with the SoC — no manual codegen
step needed.


## Architecture (50-minute deep dive)

### The SoC

LiteX generates a VexRiscv-based SoC with the `full+cfu` variant. This variant includes the **CfuPlugin**, which adds a custom instruction bus to the CPU pipeline. The SoC provides:

- **128 KB main RAM** at `0x40000000` — holds firmware code + read-only data
- **8 KB SRAM** at `0x10000000` — `.data` and `.bss` sections
- **CSR peripherals** at `0xf0000000` — UART, timer, control registers
- **CFU bus** — directly connected to the VexRiscv pipeline

### Custom Function Unit (CFU)

See [`docs/cfu.md`](docs/cfu.md) for a detailed guide.

The CFU sits inside the CPU pipeline and executes custom R-type instructions in the `CUSTOM_0` opcode space (`0x0B`). It uses a **valid/ready handshake** protocol:

1. CPU asserts `cmd_valid` with function ID + two 32-bit operands
2. CFU computes the result (single or multi-cycle)
3. CFU asserts `rsp_valid` with the 32-bit result
4. CPU reads the result into the destination register

The current implementation (`hardware/top.py`) wires a single instruction at `funct3=0`: **SimdMac4**.

### SimdMac4 — 4-lane SIMD multiply-accumulate

Defined in `hardware/mac.py`. For each byte lane `i` in `[0..3]`:

```
output = base + Σ (in0[i] + INPUT_OFFSET) * in1[i]
```

Where:
- `INPUT_OFFSET = 128` (matches quantised NN zero-point conventions)
- `funct7[0] = 0` → accumulate: `base = accumulator`
- `funct7[0] = 1` → reset: `base = 0`

This is a single-cycle instruction — the CFU responds in the same cycle the command arrives.

### Firmware

See [`docs/firmware.md`](docs/firmware.md) for a detailed guide.

The firmware is a bare-metal Zig program targeting `riscv32-freestanding-ilp32`. It implements:

1. **MMIO / CSR access** (`mmio.zig`) — volatile reads/writes at fixed addresses
2. **UART driver** (`uart.zig`) — byte-level I/O over LiteX's UART CSRs
3. **Link protocol** (`link.zig`) — framed binary request/response over UART
4. **CFU driver** (`cfu.zig`) — inline assembly for `CUSTOM_0` instructions
5. **Command dispatch** (`dispatch.zig`) — routes opcodes to handlers

### Link protocol

A simple framed binary protocol over UART:

**Request** (8 bytes header + payload):
```
[magic=0xCF] [opcode] [payload_len:u16] [seq_id:u16] [reserved:u16] [payload...]
```

**Response** (8 bytes header + payload):
```
[magic=0xFC] [status] [payload_len:u16] [seq_id:u16] [cycles_lo:u16] [payload...]
```

### Host client

`host/client.py` supports two transports:
- **SimLink** (default): spawns `litex_sim` and talks over stdin/stdout pipes
- **SerialLink** (`--serial`): uploads firmware via the LiteX SFL boot protocol and talks over a serial port to a real FPGA

### Build pipeline

1. **Amaranth → Verilog**: `python hardware/top.py` generates `top.v`
2. **LiteX SoC build**: generates the SoC, producing `csr.json` and `crt0.o` in the build directory
3. **Zig cross-compile**: `zig build -Dbuild-dir=<litex-build>` reads `csr.json` for CSR addresses and links `crt0.o`, producing `firmware.bin`
4. **Test (sim)**: `host/client.py --test` spawns the LiteX simulator
5. **Test (FPGA)**: `host/client.py --test --serial /dev/ttyUSB1` resets the board, uploads firmware via SFL, and runs tests
