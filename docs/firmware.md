# Firmware Guide

Bare-metal Zig firmware for a VexRiscv soft-core with CFU, running on a LiteX SoC.

## 5-Minute Understanding

The firmware is a loop: wait for a UART command, execute a hardware instruction via the CFU, send back the result. It runs freestanding on a RISC-V CPU with no OS, no allocator, and no standard library runtime.

```
main():
    enable CFU
    init UART
    loop:
        header = recv_header()   ← 8-byte binary frame from host
        dispatch(header)         ← decode opcode, read payload, call CFU, respond
```

### Source files

| File          | Role                                                    |
|---------------|---------------------------------------------------------|
| `main.zig`    | Entry point — enables CFU, inits UART, runs dispatch loop |
| `uart.zig`    | UART driver — read/write bytes via LiteX MMIO CSRs      |
| `link.zig`    | Link protocol — framed request/response over UART        |
| `dispatch.zig`| Command dispatch — routes opcodes to handlers            |
| `cfu.zig`     | CFU driver — inline asm to issue `CUSTOM_0` instructions |
| `mmio.zig`    | MMIO/CSR primitives — volatile register + CSR access     |
| `csr.zig`     | CSR address constants generated from LiteX               |
| `linker.ld`   | Linker script — memory map for the LiteX SoC             |

### Building

```bash
cd firmware
zig build -Dcrt0=crt0.o
# Output: zig-out/bin/firmware.bin
```

The `crt0.o` file is pre-built from LiteX's `crt0.S` startup code:

```bash
riscv64-elf-gcc -c -march=rv32im_zicsr -mabi=ilp32 -nostdlib \
  $(python -c "import litex.soc.cores.cpu.vexriscv; import os; \
    print(os.path.join(os.path.dirname(litex.soc.cores.cpu.vexriscv.__file__), 'crt0.S'))") \
  -o crt0.o
```

## 50-Minute Understanding

### Memory map

Defined in `linker.ld`, matching the LiteX SoC configuration:

| Region     | Base         | Size    | Contents                         |
|-----------|-------------|---------|----------------------------------|
| `rom`      | `0x00000000` | 128 KB  | (unused — boot ROM)             |
| `sram`     | `0x10000000` | 8 KB    | `.data` + `.bss` + stack         |
| `main_ram` | `0x40000000` | 8 MB    | `.text` + `.rodata` (firmware loaded here) |
| `csr`      | `0xf0000000` | 64 KB   | Memory-mapped peripheral CSRs    |

The `.data` section is loaded at `main_ram` (in the firmware binary) but copied to `sram` at boot by `crt0.S`. The stack grows downward from the top of SRAM (`0x10002000`).

### Boot sequence

1. LiteX loads `firmware.bin` into `main_ram` at `0x40000000`
2. CPU jumps to `_start` in `crt0.S` (from `crt0.o`)
3. `crt0.S` copies `.data` from ROM to SRAM, zeroes `.bss`, sets up the stack, then calls `main()`
4. `main()` enables the CFU via CSR `0xBC0`, initialises UART, and enters the dispatch loop

### MMIO layer (`mmio.zig`)

Two comptime-parameterised register types:

- **`Reg(addr)`** — volatile `u32` read/write at a fixed memory address. Used for LiteX peripheral CSRs (UART, timer, etc.).
- **`Csr(addr)`** — RISC-V CSR access via `csrr`/`csrw` inline assembly. Used for CPU-internal registers (e.g., the CfuPlugin enable at `0xBC0`).

Both are `inline fn` so they compile to a single load/store instruction with no function call overhead.

### UART driver (`uart.zig`)

Talks to the LiteX UART peripheral via five MMIO registers:

| Register      | Address        | Purpose                     |
|--------------|----------------|-----------------------------|
| `uart_rxtx`   | `0xf0001800`  | TX/RX data register          |
| `uart_txfull`  | `0xf0001804`  | TX FIFO full flag            |
| `uart_rxempty` | `0xf0001808`  | RX FIFO empty flag           |
| `uart_ev_pending` | `0xf0001810` | Event pending (ack writes) |
| `uart_ev_enable`  | `0xf0001814` | Event enable (TX/RX bits)  |

`write_byte()` spins until `txfull == 0`, then writes to `rxtx`. `read_byte_blocking()` spins until `rxempty == 0`, reads from `rxtx`, and acknowledges the RX event.

### Link protocol (`link.zig`)

Binary framing over UART for host↔firmware communication.

**Request header** (8 bytes, packed little-endian):
```
u8   magic        = 0xCF
u8   op           (OpType enum — e.g., 0x01 = mac4)
u16  payload_len
u16  seq_id
u16  _reserved
```

**Response header** (8 bytes, packed little-endian):
```
u8   magic        = 0xFC
u8   status       (0x00 = OK, else error code)
u16  payload_len
u16  seq_id       (echoed from request)
u16  cycles_lo    (performance counter, low 16 bits)
```

`recv_header()` synchronises by scanning for the magic byte `0xCF`, then reads the remaining 7 bytes of the header. This handles any stale data or debug output mixed in.

### CFU driver (`cfu.zig`)

Emits RISC-V R-type custom instructions using inline assembly:

```
.insn r CUSTOM_0, funct3, funct7, rd, rs1, rs2
```

This maps to opcode `0x0B` with `funct3` and `funct7` as the function selectors. The assembler encodes the full 32-bit instruction at compile time; at runtime it's a single instruction.

Two public functions:
- **`mac4(a, b)`** — accumulate: `funct7=0`, adds to the running accumulator
- **`mac4_first(a, b)`** — reset + compute: `funct7=1`, clears accumulator first

### Command dispatch (`dispatch.zig`)

Routes incoming opcodes to handler functions. Currently supports one opcode:

- **`mac4` (0x01)**: reads 8 bytes of payload (two `i32` values), calls `cfu.mac4_first(a, b)`, sends back the 4-byte result.

Unknown opcodes get an error response with status `0x01`. Short payloads get status `0x02`.

### Build system (`build.zig`)

The Zig build script:
1. Cross-compiles for `riscv32-freestanding-ilp32` with the `M` extension (integer multiply/divide) and without `F`, `D`, `C`, `A` extensions
2. Links against `crt0.o` (pre-built from LiteX's startup code) and `linker.ld`
3. Produces an ELF, then runs `riscv64-elf-objcopy -O binary` to create a flat binary for LiteX's `--ram-init`

The objcopy step uses `b.findProgram` to locate the toolchain binary, with fallback names for different distro conventions.
