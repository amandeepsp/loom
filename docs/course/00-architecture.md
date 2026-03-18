# Unit 0: Architecture — How GPUs Actually Work

> **Course:** **[00-architecture](00-architecture.md)** > [01-compute](01-compute.md)

---

## 0.1 What Does It Mean to "Run a Model on a GPU"?

When you write `model.cuda()` in PyTorch, you are splitting your program across two separate computers with separate memories connected by a bus. The CPU (host) orchestrates. The GPU (device) computes. Every tensor operation becomes a message: "here's the data, here's the kernel, run it, give me the result."

This is the **host/device split**, and it is the foundational abstraction of every ML accelerator ever built — from NVIDIA's A100 to Google's TPU to Apple's Neural Engine.

The pattern is always the same:

```
  Host (general-purpose)          Bus              Device (specialized)
  ┌─────────────────────┐    ┌──────────┐    ┌──────────────────────┐
  │  Python / framework  │    │          │    │  Massively parallel   │
  │  Model graph         │───►│  PCIe    │───►│  compute units       │
  │  Scheduling          │    │  NVLink  │    │  Local VRAM          │
  │  Memory management   │◄───│  UART    │◄───│  Fixed-function HW   │
  └─────────────────────┘    └──────────┘    └──────────────────────┘
```

The bus technology changes. The device architecture changes. The split does not.

> **MLSys Connection:** In CUDA, the host enqueues *kernel launches* into *command queues* (streams). The GPU pulls commands from these queues and executes them asynchronously. The host doesn't stall unless it explicitly synchronizes. This is how GPU utilization stays high — the command queue decouples the host from the device.

---

## 0.2 Your Architecture: The Same Pattern at FPGA Scale

Your system follows exactly this pattern:

| GPU System | Your System |
|---|---|
| CPU (x86/ARM) | CPU (x86/ARM) — same host |
| PCIe bus (16 GB/s) | UART serial (11.5 KB/s) |
| GPU | Tang Nano 20K FPGA |
| CUDA cores | CFU (Custom Function Unit) |
| GPU VRAM (80 GB on A100) | BSRAM (5.2 KiB) + SRAM (8 KiB) |
| Kernel launch | Custom RISC-V instruction |
| CUDA driver | Python host client (`host/client.py`) |
| GPU command processor | Zig firmware on VexRiscv soft CPU |
| PyTorch / TinyGrad | TinyGrad with custom lowering |

```
  ┌────────────────────────────────────────────────────────────────┐
  │                     Host (x86/ARM Linux)                       │
  │                                                                │
  │  TinyGrad                                                      │
  │  ┌──────────┐     ┌────────────┐     ┌───────────────────┐     │
  │  │  Model   │────>│  Schedule  │────>│  Selective Lower  │     │
  │  │  Graph   │     │  (fuse ops)│     │                   │     │
  │  └──────────┘     └────────────┘     │  Can HW do this?  │     │
  │                                      │  YES -> serialize  │     │
  │                                      │         & send    │     │
  │                                      │  NO  -> run on CPU │     │
  │                                      └────────┬──────────┘     │
  │                                               │                │
  │                     UART serial link           │                │
  └───────────────────────────────────────────────┼────────────────┘
                                                  │
                                    ┌─────────────v──────────────┐
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

The Python host client is your "GPU driver" — it builds commands, sends them to the device, and reads back results. The VexRiscv firmware is your "GPU command processor" — it sits on the device, decodes incoming commands, dispatches them to hardware, and pushes results back onto the bus. The CFU is your "shader core." The UART is your "PCIe."

> **MLSys Connection:** On a real GPU, the **driver** runs on the host CPU (NVIDIA's kernel-mode driver, Mesa's Nouveau, etc.) and submits command buffers over PCIe. The **command processor** is a fixed-function unit *on the GPU die* that reads those command buffers, decodes them, and dispatches work to the SMs. Your Python host code is the driver. Your Zig firmware is the command processor. Don't confuse the two — the driver never touches the hardware directly; it talks to the device through the bus.

---

## 0.3 The Hardware You Have

**Tang Nano 20K** — Gowin GW2AR-18C FPGA:

| Resource | Total | Used by SoC | Free for you |
|---|---|---|---|
| LUT4 | 20,736 | ~8,000-10,000 | ~10,000-12,000 |
| Flip-flops | 15,552 | ~4,000-6,000 | ~10,000 |
| DSP (9x9 multipliers) | 48 | 0 | **48** |
| BSRAM (18 Kbit each) | 46 | ~28-30 | **~16-18** |
| On-package PSRAM | 8 MiB | Not connected | 8 MiB (stretch goal) |

**VexRiscv soft CPU:** RV32IM, ~27 MHz, UART at 115200 baud.

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

---

## 0.4 Exercise: Map Your Hardware to GPU Concepts

Open your Tang Nano 20K resource table above. Open `firmware/linker.ld`. Answer these questions:

**Question 1: What is your "VRAM"?**

On a GPU, VRAM (HBM/GDDR) is where model weights and activations live. What's the equivalent in your system? How much do you have?

<details><summary>Hint 1</summary>

Your device has two kinds of memory: SRAM (fast, 8 KiB, used for stack and variables) and BSRAM (on-FPGA block RAM, ~16-18 free blocks of 18 Kbit each). Which one is analogous to VRAM?

</details>

<details><summary>Hint 2</summary>

BSRAM is the closest analog — it's local to the compute hardware and the CFU can access it directly. But 16 blocks x 18 Kbit = ~36 KiB total free. An A100 has 80 GB. You have about 2 million times less memory.

</details>

<details><summary>Solution</summary>See solutions/00-unit/vram-analysis.md</details>

**Question 2: What is your "PCIe"?**

PCIe 4.0 x16 delivers ~32 GB/s to the GPU. What's your equivalent? How does the bandwidth compare?

<details><summary>Hint 1</summary>

Your host-to-device link is UART at 115200 baud. How many bytes per second is that?

</details>

<details><summary>Hint 2</summary>

115200 baud with 8N1 encoding = ~11,520 bytes/sec = ~11.5 KB/s. PCIe 4.0 x16 is ~32 GB/s. The ratio is roughly 1:2,800,000. This makes your "offload or compute locally?" decision very different from a GPU's.

</details>

<details><summary>Solution</summary>See solutions/00-unit/pcie-analysis.md</details>

**Question 3: Where does the analogy break?**

Think about at least two ways your system differs fundamentally from a real GPU, beyond just being smaller.

<details><summary>Hint 1</summary>

Consider: Does a GPU have a general-purpose CPU on the device side? Does your FPGA share memory with the host? Can a GPU run thousands of threads simultaneously?

</details>

<details><summary>Hint 2</summary>

Key differences: (1) Your FPGA has a full CPU (VexRiscv) on the device — GPUs have fixed-function command processors, not general-purpose CPUs. This gives you flexibility but costs area. (2) Your link is synchronous and blocking — the host sends a command and waits. Real GPUs use asynchronous command queues with thousands of in-flight operations. (3) GPUs achieve throughput through massive parallelism (thousands of threads). Your CFU does one operation at a time.

</details>

<details><summary>Solution</summary>See solutions/00-unit/analogy-breaks.md</details>

---

## 0.5 The Memory Hierarchy: Four Tiers, Just Like a GPU

GPUs have a memory hierarchy: registers (fastest, tiniest) → shared memory → L2 cache → HBM/GDDR (slowest, biggest). Your system has the same structure:

| Tier | GPU | Your System | Size | Bandwidth |
|---|---|---|---|---|
| **Registers** | Register file (per-thread) | CFU internal signals (accumulator) | ~32 bits | 1 cycle |
| **Shared memory** | `__shared__` (per-SM, 48-228 KiB) | BSRAM (~36 KiB free, ~16 blocks) | ~5 KiB usable | 1 cycle |
| **Global memory** | HBM/GDDR (up to 80 GB) | SPI flash (W25Q64, 8 MB) | 8 MB | ~10-40 MB/s (QSPI) |
| **Host memory** | CPU system RAM (via PCIe) | Host RAM (via UART) | Unbounded | ~11.5 KB/s |

The SPI flash on-board is the interesting middle tier. It's too slow for cycle-by-cycle access but large enough to hold all MobileNet v2 0.25 weights (~200 KB). The strategy: **pre-load weights from SPI flash into BSRAM before each layer, then compute from BSRAM at full speed.** This is exactly what GPU kernels do — load tiles from global memory into shared memory, synchronize, then compute from shared memory.

```
  Host RAM         SPI Flash         BSRAM          CFU
  (weights,        (all model        (active         (compute)
   activations)     weights)          tile)

  ────────────►  ────────────►  ────────────►  ────────►
  UART 11.5KB/s  QSPI ~20MB/s  Wire speed      1 cycle
  one-time load  per-layer DMA  per-cycle read
```

The practical implication: model weights can live in SPI flash permanently. At inference time, the firmware (or a DMA controller) copies the current layer's weight tile from flash into BSRAM filter stores. This eliminates the UART bottleneck for weight transfer — only activations and results cross the UART.

> **🔗 MLSys Connection:** This is exactly the GPU memory hierarchy strategy. CUDA kernels tile their work to fit in shared memory, load from global memory (HBM) in bulk, compute from shared memory at full bandwidth. The ratio of your BSRAM-to-SPI-flash bandwidth (~1000:1) mirrors the ratio of shared-memory-to-HBM bandwidth on a real GPU (~10-30:1). The principle is the same: make the fast memory work harder by reusing data.

---

## 0.6 Does the CPU Earn Its Keep?

Your system has a VexRiscv soft CPU on the FPGA (~8-10K LUTs — nearly half your free budget). A legitimate question: **should we cut the CPU and connect the UART directly to hardware?**

| | VexRiscv (current) | Pure FPGA fabric (FFHW) |
|---|---|---|
| **LUTs** | ~8-10K | ~1-2K for UART+FSM |
| **Dev velocity** | Firmware recompile: seconds | Resynthesis: minutes |
| **Debugging** | `uart.write("debug\n")` | Waveform viewer only |
| **Protocol changes** | Edit Zig, reflash | Edit HDL, resynthesize |
| **SPI flash driver** | Write a Zig driver | Custom SPI controller FSM |
| **Inference latency** | Instruction overhead per byte | Deterministic, pipelined |
| **Freed LUTs** | 0 | ~8K → bigger systolic array |

As the sequencer becomes autonomous (Unit 4+), the CPU's job during inference shrinks to: "write config registers → assert START → wait DONE → drain FIFO." That's a 5-state FSM — you don't need a 32-bit RISC-V CPU for that.

**Why we keep VexRiscv:** Development velocity. Firmware changes in seconds; resynthesis takes minutes. The SPI flash driver is trivial in Zig, painful in pure HDL. And `uart.write()` for debugging is invaluable. Google's CFU-Playground makes the same choice for the same reasons.

**When to consider FFHW:** If you finish the course and want to push performance, removing VexRiscv frees ~8K LUTs — enough for a larger systolic array or double-buffered memory controller. This is a "graduate-level" exercise: replace the soft CPU with a minimal UART-to-register-file FSM.

> **🔗 MLSys Connection:** Real GPUs have this same split. The GPU command processor is a small microcontroller (not a full CPU) that decodes command buffers and configures the SMs. Google's TPU v1 similarly has a minimal host interface — the systolic array does the real work. The question "CPU vs fixed-function control" is a fundamental accelerator design tradeoff: flexibility vs area efficiency.

---

## 0.7 The Design Spectrum: Five Tiers of Accelerator

Google's [CFU-Playground](https://github.com/google/CFU-Playground) built four tiers of ML accelerator on FPGAs. Each tier maps to a concept from real GPU architecture:

```
  COMPLEXITY ──────────────────────────────────────────────────────>

  Tier 1              Tier 2              Tier 3              Tier 4
  Scalar MAC          SIMD MAC +          Autonomous          Systolic Array +
  (proj_accel_1)      HW requant          inner loop          LRAM + PostProcess
                      (avg_pdti8)         (mnv2_first)        (hps_accel)

  ~1 MAC/cyc          ~4 MACs/cyc         ~4 MACs/cyc         ~32 MACs/cyc
  CPU drives ALL      CPU outer loop      CPU: load+start     CPU: config+poll
```

Now, in GPU terms:

| Tier | FPGA Design | GPU Analog |
|---|---|---|
| **Tier 1** — Scalar MAC | CPU issues one multiply per instruction | Like calling a single CUDA core with one thread. Absurd overhead per operation. |
| **Tier 2** — SIMD MAC | 4 multiplies per instruction, CPU still drives the inner loop | Like a warp executing a vectorized instruction (`dp4a`). Better throughput, but the CPU is the bottleneck. |
| **Tier 3** — Autonomous inner loop | Hardware runs the full dot product loop. CPU just loads weights and starts execution. | Like a GPU kernel — the host launches it and waits. The device manages its own memory access pattern. |
| **Tier 4** — Systolic array | Spatial array of MACs with local memory and post-processing pipeline. | Tensor Core / TPU systolic array. Matrix multiply is a single operation, not a loop. |

> **MLSys Connection:** The jump from Tier 2 to Tier 3 is the most important architectural transition. It corresponds to moving from "the CPU calls a math library function in a loop" to "the CPU launches a kernel and the device runs autonomously." This is where GPU programming actually begins — and where most of the speedup comes from, even before you add more compute units.

Our tutorial maps to these tiers:

| Tutorial Part | Tier | What you learn |
|---|---|---|
| Part 1: MAC | Tier 1-2 | Custom instruction encoding, CFU bus, SIMD |
| Part 2: Vertical Slice | -- | End-to-end data path, protocol design |
| Part 3: Autonomous | Tier 3 | BSRAM stores, HW requant, CPU off hot path |
| Part 4: TinyGrad | -- | Selective lowering, UOp mapping |
| Part 5: Scaling | Tier 4 | Systolic array, PSRAM, clock speed |

---

## 0.8 Selective Lowering: The GPU Execution Model

When PyTorch executes `y = model(x)`, it doesn't ship the entire Python program to the GPU. It walks the computation graph, finds operations the GPU can accelerate (matmul, conv2d, attention), and *lowers* only those to GPU kernels. Everything else stays on the CPU.

Your system does the same thing:

```
  TinyGrad computation graph:

    input --> Conv2D --> ReLU --> Conv2D --> Softmax --> output

  Selective lowering decision:

    input --> [OFFLOAD Conv2D] --> ReLU (CPU) --> [OFFLOAD Conv2D] --> Softmax (CPU) --> output
```

The "lowering decision" is: **can the hardware do this operation faster than the host, accounting for transfer overhead?**

On a GPU, the answer is almost always "yes" for large tensor operations because PCIe bandwidth is high and GPU throughput is enormous. On your system, the calculus is different. UART is slow. Your CFU does 4 MACs per cycle at 27 MHz. The breakeven point — where offloading beats the host CPU — depends on the operation size.

> **MLSys Connection:** This is exactly what `torch.compile()` does internally. The compiler traces the graph, identifies subgraphs that can run on the device, and generates kernel code for those subgraphs. Everything else falls back to eager CPU execution. TinyGrad's scheduler makes the same decision — and in Unit 4, you'll write the hook that tells it about your hardware.

---

## 0.9 The Full Stack: GPU vs. Your System

Side by side, layer for layer:

| Layer | GPU System | Your System |
|---|---|---|
| **Framework** | PyTorch / TinyGrad | TinyGrad |
| **Compiler/Scheduler** | torch.compile / TinyGrad scheduler | TinyGrad scheduler + custom lowering |
| **Driver** | CUDA driver / OpenCL runtime | Python host client (`host/client.py`) |
| **Transport** | PCIe DMA, command queues | UART serial link, packet framing |
| **Device firmware** | GPU microcontroller firmware | Zig firmware on VexRiscv |
| **Compute hardware** | CUDA cores / Tensor Cores | CFU (Amaranth HDL -> Verilog -> FPGA) |
| **Device memory** | HBM2e / GDDR6X | BSRAM (5.2 KiB free) + SRAM (8 KiB) |

Every layer exists in both systems. Yours is smaller, simpler, and fully open — you can read every line from the Python framework down to the gate-level hardware description.

---

## 0.10 Checkpoint

Before moving to Unit 1, you should be able to:

- [ ] Explain the host/device split and why it exists (specialization vs. generality)
- [ ] Name the GPU analog for each component in your system (host Python=driver, UART=PCIe, VexRiscv firmware=command processor, CFU=shader core, BSRAM=shared memory)
- [ ] Articulate why offloading a single MAC over UART is like launching a GPU kernel that does one multiply — technically correct but catastrophically inefficient
- [ ] Explain selective lowering: the host runs the model, only specific operations go to the device
- [ ] Identify at least two ways your system differs from a real GPU beyond scale
- [ ] Read the Tang Nano 20K resource table and know your constraints: ~12K free LUTs, 48 DSP blocks, ~16 free BSRAM blocks, 8 KiB SRAM

---

## Suggested Readings

- **tinygrad source:** `tinygrad/runtime/ops_gpu.py` — how tinygrad talks to a GPU. Notice the same host/device pattern.
- **tinygrad source:** `tinygrad/engine/schedule.py` — the scheduler that decides what to lower and what to fuse.
- **Blog:** Fabien Sanglard, ["How GPUs Work"](https://fabiensanglard.net/gpu/index.html) — visual walkthrough of the GPU pipeline from command queue to pixel output.
- **Paper:** Jouppi et al., ["In-Datacenter Performance Analysis of a Tensor Processing Unit"](https://arxiv.org/abs/1704.04760) (2017) — Google's TPU paper. Section 2 describes the host/device split for TPUs. Compare their systolic array to what you'll build in Unit 5.
- **Blog:** Fabian Giesen, ["A Trip Through the Graphics Pipeline"](https://fgiesen.wordpress.com/2011/07/09/a-trip-through-the-graphics-pipeline-2011-index/) — deep dive into GPU pipeline stages. The command processor section maps directly to your VexRiscv firmware.
- **Reference:** RISC-V ISA Manual, Section 2.2 (Base Instruction Formats) — the R-type encoding you'll use for custom instructions in Unit 1.

---

**Next:** [Unit 1: Custom Compute — ALUs and Intrinsics](01-compute.md)
