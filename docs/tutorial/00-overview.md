# Part 0 — Overview & Architecture

> **Series:** **[00-overview](00-overview.md)** → [01-mac](01-mac.md) → [02-vertical-slice](02-vertical-slice.md) → [03-autonomous](03-autonomous.md) → [04-tinygrad](04-tinygrad.md) → [05-scaling](05-scaling.md)
> **Deep Dive:** [Prior Art & Architecture Decisions](appendix-prior-art.md)

---

## 0.1  The Architecture Shift

This project uses **selective lowering** — the same execution model as a
GPU. The host machine runs the ML model. Only specific operations are
offloaded to the FPGA accelerator.

```
  ┌────────────────────────────────────────────────────────────────┐
  │                     Host (x86/ARM Linux)                       │
  │                                                                │
  │  TinyGrad                                                      │
  │  ┌──────────┐     ┌────────────┐     ┌───────────────────┐     │
  │  │  Model   │────►│  Schedule  │────►│  Selective Lower  │     │
  │  │  Graph   │     │  (fuse ops)│     │                   │     │
  │  └──────────┘     └────────────┘     │  Can HW do this?  │     │
  │                                      │  YES → serialize  │     │
  │                                      │         & send    │     │
  │                                      │  NO  → run on CPU │     │
  │                                      └────────┬──────────┘     │
  │                                               │                │
  │                     UART serial link           │                │
  └───────────────────────────────────────────────┼────────────────┘
                                                  │
                                    ┌─────────────▼──────────────┐
                                    │  FPGA (Tang Nano 20K)      │
                                    │                            │
                                    │  VexRiscv firmware (Zig)   │
                                    │  ┌──────────────────────┐  │
                                    │  │ Wait for command     │  │
                                    │  │ Decode OpType        │  │
                                    │  │ Execute via CFU      │  │
                                    │  │ Send result back     │  │
                                    │  └──────────────────────┘  │
                                    │                            │
                                    │  CFU hardware (Amaranth)   │
                                    └────────────────────────────┘
```

**🤔 Before reading further, answer these:**

1. *What does "selective lowering" mean in a GPU context?* When you write
   `torch.matmul(a, b)` in Python, what runs on the CPU and what runs on
   the GPU? Your system is the same idea at a smaller scale.

2. *Why not compile the entire model to run on the RISC-V?* Open
   `firmware/linker.ld`. How much SRAM does the firmware have? How much
   RAM does a MobileNet-v2 0.25 need for weights alone?

3. *When does offloading help? When does it hurt?* Sending data over UART
   costs time. If the operation takes fewer cycles than the transfer, the
   host CPU would be faster. This is the fundamental tension.

---

## 0.2  What You're Building

End-to-end, the system has five layers:

```
  Layer 5:  TinyGrad model graph (Python)
               │
  Layer 4:  Selective lowering decision (Python)
               │  "Is this op worth offloading?"
               │
  Layer 3:  Link protocol — packet framing (Zig / Python)
               │  [MAGIC][OpType][len][payload...]
               │
  Layer 2:  UART serial transport
               │  115200 baud, ~11.5 KB/s
               │
  Layer 1:  Firmware command loop (Zig, bare-metal RISC-V)
               │  receive → decode → execute → respond
               │
  Layer 0:  CFU hardware (Amaranth HDL → Verilog → FPGA)
               4-lane INT8 SIMD MAC, custom RISC-V instruction
```

Each tutorial part builds one or two layers. By the end you can run a
TinyGrad model and watch specific operations execute on your FPGA.

---

## 0.3  The Hardware You Have

**Tang Nano 20K** — Gowin GW2AR-18C FPGA:

| Resource | Total | Used by SoC | Free |
|---|---|---|---|
| LUT4 | 20,736 | ~8,000–10,000 | ~10,000–12,000 |
| Flip-flops | 15,552 | ~4,000–6,000 | ~10,000 |
| DSP (9×9 multipliers) | 48 | 0 | **48** |
| BSRAM (18 Kbit each) | 46 | ~28–30 | **~16–18** |
| On-package PSRAM | 8 MiB | Not connected | 8 MiB (stretch goal) |

**Memory map** (from `firmware/linker.ld`):

```
  0x00000000  ┌──────────────┐
              │     ROM      │  128 KiB (bootloader)
  0x00020000  ├──────────────┤
              │              │
  0x10000000  ├──────────────┤
              │     SRAM     │  8 KiB  (stack + .data + .bss)
  0x10002000  ├──────────────┤
              │              │
  0x40000000  ├──────────────┤
              │   main_ram   │  8 MiB  (firmware code + rodata)
  0x40800000  ├──────────────┤
              │              │
  0xf0000000  ├──────────────┤
              │     CSR      │  64 KiB (UART, Timer, LEDs, ...)
  0xf0010000  └──────────────┘
```

**🤔 Exercise:** Open `firmware/linker.ld`. Where does `.text` (code) go?
Where does `.bss` (zero-initialized data) go? Where is the stack? *Why
does this matter for your firmware design?* (Hint: SRAM is 8 KiB. Every
buffer you declare eats into that.)

---

## 0.4  The Software Stack

| Layer | Language | What it does |
|---|---|---|
| Hardware blocks | Amaranth HDL (Python) | MAC unit, CFU bus logic → generates Verilog |
| SoC integration | LiteX (Python) | Wires VexRiscv + UART + peripherals, generates CSR headers |
| Firmware | Zig | Bare-metal on RISC-V. Cross-compiled, freestanding. Uses `.insn` inline asm for custom instructions |
| Host-side driver | Python | Speaks the UART link protocol, integrates with TinyGrad |

**Key files right now:**

```
  firmware/
  ├── build.zig          Target: riscv32, freestanding, ReleaseSmall
  ├── linker.ld          Memory layout
  └── src/
      ├── main.zig       Entry point (currently empty loop)
      ├── cfu.zig        .insn wrapper: mac4(acc, a, b)
      └── link.zig       Protocol: magic bytes, OpType enum

  hardware/
  ├── cfu.py             CFU bus FSM, 8-instruction dispatch via funct3
  └── mac.py             SimdMac4: 4-lane INT8 SIMD MAC
```

---

## 0.5  The Design Spectrum

Google's [CFU-Playground](https://github.com/google/CFU-Playground) built
four tiers of ML accelerator. Studying them reveals where the performance
actually comes from:

```
  COMPLEXITY ──────────────────────────────────────────────────────►

  Tier 1              Tier 2              Tier 3              Tier 4
  Scalar MAC          SIMD MAC +          Autonomous          Systolic Array +
  (proj_accel_1)      HW requant          inner loop          LRAM + PostProcess
                      (avg_pdti8)         (mnv2_first)        (hps_accel)

  ~1 MAC/cyc          ~4 MACs/cyc         ~4 MACs/cyc         ~32 MACs/cyc
  CPU drives ALL      CPU outer loop      CPU: load+start     CPU: config+poll

  ~100 lines          ~200 lines          ~800 lines          ~2000+ lines
```

**The key insight:** The jump from Tier 2 to Tier 3 — taking the CPU off
the hot path — is where most of the architectural benefit lives. The
systolic array (Tier 4) is a throughput multiplier on top of an already-
efficient design. *Don't skip Tier 3 to jump to Tier 4.*

Our tutorial maps to these tiers:

| Tutorial Part | CFU-Playground Tier | What you learn |
|---|---|---|
| Part 1: MAC | Tier 1–2 | Custom instruction encoding, CFU bus, SIMD |
| Part 2: Vertical Slice | — | End-to-end data path, protocol design |
| Part 3: Autonomous | Tier 3 | BSRAM stores, HW requant, CPU off hot path |
| Part 4: TinyGrad | — | Selective lowering, UOp mapping |
| Part 5: Scaling | Tier 4 | Systolic array, PSRAM, clock speed |

See [Prior Art & Architecture Decisions](appendix-prior-art.md) for the
full analysis, including challenged assumptions and revised designs.

---

## 0.6  Series Roadmap

| Part | Build | Learn | Effort |
|---|---|---|---|
| **1. MAC** ✅ | SIMD MAC hardware + Zig `.insn` wrapper | Custom RISC-V instructions, CFU protocol | Done |
| **2. Vertical Slice** | UART echo → packet framing → MAC over serial → host test | End-to-end data path, bare-metal Zig, protocol design | Medium |
| **3. Autonomous** | HW requant, BSRAM filter stores, output FIFO, sequencer | CPU off hot path, on-chip memory architecture | High |
| **4. TinyGrad** | CUSTOM_FUNCTION hook, lowering decision logic | Framework integration, UOp mapping, when to offload | Medium |
| **5. Scaling** | Systolic array, higher clock, PSRAM (stretch) | Spatial parallelism, memory bandwidth wall | High |

**Work in order.** Each part builds on the previous one. Part 2 is the
critical foundation — without a working vertical slice, nothing else
matters.

---

## 0.7  Prerequisites

- **Hardware:** Tang Nano 20K, USB cable, `/dev/ttyUSB1` accessible
- **Zig:** 0.15.2+ (for firmware cross-compilation)
- **Python:** 3.10+ with UV (for Amaranth, LiteX, host-side tooling)
- **FPGA toolchain:** OSS CAD Suite (Yosys, nextpnr-gowin, openFPGALoader)
- **Serial terminal:** picocom (firmware upload via `--kernel`)
- **Working SoC:** LiteX build with VexRiscv + CFU, bitstream loaded
- **Working MAC:** `hardware/mac.py` passes simulation tests

**🤔 Sanity check:** Before starting Part 2, verify:
1. You can build the firmware: `cd firmware && zig build`
2. You can upload: `zig build load` (or picocom manually)
3. You can see the LiteX BIOS prompt over UART
4. `build/sipeed_tang_nano_20k/software/include/generated/csr.h` exists

---

**Next:** [Part 1 — The MAC: Your First Custom Instruction](01-mac.md)
