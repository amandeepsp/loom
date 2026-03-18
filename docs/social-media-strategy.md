# ACCEL Project — Social Media Strategy

A piecemeal posting plan for X (Twitter) and LinkedIn as you hit each
project milestone. Each post is a self-contained "micro-story" that works
even if someone hasn't seen the previous ones.

---

## Principles

1. **Show, don't tell.** Every post should have a visual — terminal output,
   waveform screenshot, block diagram, or board photo.
2. **One post per milestone.** Don't batch. Frequent small posts build more
   engagement than rare long threads.
3. **Teach one thing.** Each post should leave the reader knowing something
   they didn't before (a trick, a number, a concept).
4. **Tag the ecosystem.** Mention relevant projects/people: `@taborsky_cz`
   (LiteX), `@enjoy_digital`, `@__tinygrad__`, `@SpinalHDL`, `@Sipeed`.
5. **X is punchy, LinkedIn is narrative.** Same milestone, different tone.

---

## Post Schedule by Milestone

### Post 1 — Project Kickoff

**When:** Now (or whenever you start posting)

**X (Twitter):**
> Building a tiny ML accelerator from scratch on a $20 FPGA board 🧵
>
> Tang Nano 20K + LiteX + VexRiscv RISC-V CPU
> Goal: INT8 systolic array for neural net inference
>
> Starting with what I have: a working SoC with UART and hello-world firmware.
> Next: wire up an INT8 MAC unit.
>
> [photo of Tang Nano 20K board on desk]

**LinkedIn:**
> **Building an ML Accelerator from Scratch on a $20 FPGA**
>
> I'm documenting my journey building a custom INT8 neural network
> accelerator using open-source tools: Amaranth HDL for RTL, LiteX for the
> SoC, VexRiscv for the RISC-V CPU, all on a Sipeed Tang Nano 20K.
>
> The board costs $20 and has a Gowin GW2AR-18C with 20K LUTs, 48 DSP
> multipliers, and on-package PSRAM. It's small — but that's the point.
> Constraints force good design.
>
> Current status: working SoC with UART, serial boot, and firmware reload
> without reprogramming the FPGA. Next step: integrating an INT8 MAC
> custom function unit.
>
> [photo of board + terminal showing BIOS prompt]

**📸 Visual:** Photo of Tang Nano 20K with a serial terminal open showing
the LiteX BIOS banner. Clean desk, good lighting, board in focus.

---

### Post 2 — CSR MAC Integration

**When:** After Milestone 1 (MAC wired into SoC, firmware prints results)

**X:**
> Got my first custom hardware peripheral working in LiteX! 🎉
>
> 4-lane INT8 SIMD MAC unit, memory-mapped via CSR.
> Firmware writes operands → reads accumulator → prints result.
>
> One line in the SoC to add it:
>   self.mac = SIMDMacCSR()
>
> LiteX auto-generates the C headers. Zero bus plumbing.
>
> [screenshot of terminal output showing MAC result + csr.h snippet]

**LinkedIn:**
> **Adding a Custom Accelerator to a RISC-V SoC in One Line of Python**
>
> LiteX's AutoCSR system is remarkably elegant. To add a memory-mapped
> INT8 MAC unit to my VexRiscv SoC, I wrote the Migen module, then added
> `self.mac = SIMDMacCSR()` to the SoC constructor. LiteX automatically:
> - Assigned Wishbone bus addresses
> - Generated C header macros for every register
> - Integrated it into the CSR memory map
>
> The result: firmware can write packed INT8 operands and read back a 32-bit
> accumulator, all through auto-generated accessor functions.
>
> Key learning: start with CSRs, not custom instructions. The bus overhead
> (2-4 cycles) is negligible compared to development speed.
>
> [screenshot of csr.h showing MAC registers + terminal output]

**📸 Visual:** Side-by-side of (1) the generated `csr.h` MAC section and
(2) terminal output showing software vs hardware MAC results with cycle
counts.

---

### Post 3 — Software vs Hardware Benchmark

**When:** After Milestone 2 (benchmark numbers)

**X:**
> First benchmark: software INT8 MAC vs hardware CFU on VexRiscv @ 27 MHz
>
> Vector length 64:
>   Software: XXX cycles
>   Hardware (CSR): XXX cycles
>   Speedup: X.Xx
>
> The bus overhead is real — but the CFU wins at longer vectors.
> At length 256: X.Xx faster.
>
> Raw numbers are the best teacher.
>
> [screenshot of terminal with cycle counts table]

**LinkedIn:**
> **Measuring the Real Cost of Memory-Mapped Accelerators**
>
> Everyone says "just add a hardware accelerator." But what does it actually
> cost to talk to it?
>
> I benchmarked a 4-lane INT8 MAC unit on my RISC-V SoC. Each MAC
> invocation through CSR registers requires ~3 bus cycles of overhead per
> register write. For short vectors, the software loop is actually competitive.
>
> The crossover point is around length XX — below that, the bus overhead
> dominates. Above it, the CFU's 4-wide parallelism pays off.
>
> This is exactly why systolic arrays exist: you amortize the data movement
> cost across many compute operations.
>
> [chart/table of cycle counts at different vector lengths]

**📸 Visual:** A simple table or bar chart comparing software vs CFU cycle
counts at vector lengths 16, 32, 64, 128, 256. Can be a terminal printout
or a quick matplotlib chart.

---

### Post 4 — First Convolution Layer

**When:** After running a single INT8 conv layer end-to-end

**X:**
> First INT8 convolution layer running on my custom accelerator! 🔥
>
> 1×1 conv, 8→8 channels, 4×4 spatial
> Quantized on host (PyTorch) → exported weights → cross-compiled → serial boot
>
> Output matches the Python reference to the bit.
> Correctness before speed. Always.
>
> [terminal showing "PASS: output matches reference"]

**LinkedIn:**
> **From Quantized Model to Silicon (Well, FPGA) — End-to-End**
>
> Today I ran my first complete neural network layer on custom hardware:
>
> 1. Quantized a convolution layer to INT8 using PyTorch
> 2. Exported weights as flat C arrays
> 3. Cross-compiled firmware with RISC-V GCC
> 4. Serial-booted onto the Tang Nano 20K
> 5. Output matches the host-side Python reference exactly
>
> The layer is tiny (1×1 conv, 8→8 channels), but the *path* is complete.
> Every subsequent layer uses the same infrastructure.
>
> The hardest part wasn't the hardware — it was getting the requantization
> math right (scale × multiplier >> shift + zero_point, clamped to [-128, 127]).
>
> [diagram of the quantize → export → compile → boot pipeline]

**📸 Visual:** Flow diagram (can be a simple Mermaid render or hand-drawn):
`PyTorch → Quantize → Export .c → GCC riscv32 → serial boot → FPGA → PASS`.

---

### Post 5 — Systolic Array in Simulation

**When:** After 4×4 systolic array passes simulation tests

**X:**
> 4×4 weight-stationary systolic array passing simulation tests ✅
>
> 16 INT8 MACs running in parallel
> Activation skew, pipeline fill/drain all working
> Result matches numpy.matmul to the bit
>
> Uses 16 of 48 DSP blocks on the GW2AR-18C (33%)
>
> Next: wire it into the SoC and measure real throughput.
>
> [VCD waveform screenshot showing the systolic pipeline]

**LinkedIn:**
> **Designing a Systolic Array: What the Papers Don't Tell You**
>
> I built a 4×4 weight-stationary systolic array in Migen (Python → hardware).
> The concept is elegant: weights stay in place, activations flow through,
> partial sums cascade down. But the implementation has sharp edges:
>
> 1. **Activation skew:** row i must start receiving data i cycles after
>    row 0. Off by one? Every result is wrong.
> 2. **Pipeline drain:** after the last activation enters, you need
>    R+C-1 extra cycles for results to flush out. Miss this and you
>    lose the last outputs.
> 3. **Weight loading:** 16 CSR writes to load 16 weights. That's 48
>    cycles of bus overhead — amortized over many activations, it's fine.
>    For depthwise convolutions? Not fine.
>
> The simulation testbench (comparing against numpy.matmul) caught all
> three bugs before I touched hardware. Simulation-first development works.
>
> [waveform screenshot with annotations pointing out skew + drain]

**📸 Visual:** GTKWave / Surfer screenshot of the VCD waveform. Annotate
with arrows showing (1) weight load phase, (2) activation streaming with
skew, (3) valid outputs appearing at the bottom edge.

---

### Post 6 — Systolic Array on Hardware

**When:** After the array works on the actual FPGA

**X:**
> 4×4 INT8 systolic array running on real hardware! 🎯
>
> Tang Nano 20K @ 27 MHz
> 16 MACs/cycle = 432 MMAC/s peak
>
> Synthesis report:
>   LUTs: XX% used
>   DSPs: 16/48 (33%)
>   BSRAM: XX/46
>
> Firmware drives it through CSRs. Matches simulation exactly.
>
> [photo of board + terminal with benchmark results]

**LinkedIn:**
> **From Simulation to Silicon: My Systolic Array on a $20 FPGA**
>
> The 4×4 INT8 systolic array is now running on real hardware. Some numbers:
>
> - **16 DSP multipliers** used out of 48 available (33%)
> - **Peak throughput:** 16 MACs/cycle × 27 MHz = 432 MMAC/s
> - **Actual throughput:** XX MMAC/s (bus overhead eats ~XX%)
>
> The gap between peak and actual is the entire motivation for DMA engines
> and custom instructions — the compute fabric is fast, but feeding it data
> through memory-mapped registers is the bottleneck.
>
> Total project cost: $20 board + open-source tools. No Vivado, no Quartus,
> no vendor lock-in.
>
> [photo of board + synthesis resource utilization summary]

**📸 Visual:** Two-part: (1) photo of the board running, (2) cropped
synthesis report showing resource utilization. Bonus: annotate the FPGA
chip on the board with "16 MACs running in here."

---

### Post 7 — TinyGrad Integration (if you reach it)

**When:** After any TinyGrad backend milestone

**X:**
> Wrote a TinyGrad custom backend for my FPGA accelerator
>
> ACCEL=1 python model.py
>
> TinyGrad compiles the model → emits C with CFU intrinsics →
> cross-compiles for RISC-V → serial boots to Tang Nano 20K
>
> Not fast. But it works end-to-end. That's the point.
>
> [terminal showing ACCEL=1 python output]

**📸 Visual:** Terminal showing the TinyGrad device initialization message
and inference output.

---

## Recurring Content Ideas

These can be posted between milestones to maintain cadence:

| Type | Example | Visual |
|---|---|---|
| **TIL (Today I Learned)** | "TIL: LiteX's `AutoCSR` generates C headers automatically. No manual register map maintenance." | Screenshot of generated `csr.h` |
| **Bug story** | "Spent 3 hours debugging why my systolic array output was shifted by one. The activation skew was off by a cycle. VCD waveforms saved me." | Annotated waveform showing the bug |
| **Number of the day** | "The GW2AR-18C has 48 DSP blocks. Each can do one 18×18 multiply per cycle. That's 1.3 GOPS at 27 MHz. My laptop's CPU can do ~100 GOPS. 77× gap — but at 0.5W vs 65W." | Simple comparison graphic |
| **Tool tip** | "Serial boot workflow: keep the bitstream loaded, rebuild firmware, press S1 to reset to BIOS, run serialboot. No FPGA reprogram needed." | Terminal recording (asciinema) |
| **Reading rec** | "If you're designing a systolic array, read the Eyeriss paper first. Their dataflow taxonomy (weight/output/row-stationary) is the clearest framework I've found." | Paper screenshot or diagram |
| **Code snippet** | "Custom RISC-V instructions in 3 lines of inline asm: `.insn r 0x0B, 0, 0, %0, %1, %2`" | Syntax-highlighted code block |
| **Board photo** | "Current state of the desk. Tang Nano 20K, USB serial, and way too many terminal windows." | Aesthetic desk/lab photo |

---

## Platform-Specific Tips

### X (Twitter)
- **Thread format** for multi-image posts (4 images max per tweet)
- **Alt text on images** — the FPGA/embedded community appreciates accessibility
- **Hashtags:** `#FPGA` `#RISCV` `#tinygrad` `#OpenSourceHardware` `#MachineLearning` `#TangNano`
- **Best times:** Weekday mornings (US Pacific / European afternoon)
- **Engagement:** Reply to other FPGA/RISC-V builders. The community is small and reciprocal.

### LinkedIn
- **Longer form** — 3-5 short paragraphs with line breaks
- **No hashtag spam** — 3-5 relevant ones at the end: `#FPGA` `#RISCV` `#EdgeAI` `#OpenHardware`
- **Personal angle** — "I learned", "I was surprised by", "the hardest part was"
- **Tag companies/projects** where relevant: Sipeed, LiteX, Gowin Semiconductor
- **Document format** posts (with images) get more reach than link posts
- **Best times:** Tuesday–Thursday, morning

---

## Photo/Screenshot Checklist

Keep these ready as you hit each milestone:

- [ ] Clean photo of Tang Nano 20K board (good lighting, minimal clutter)
- [ ] Terminal screenshot: LiteX BIOS banner
- [ ] Terminal screenshot: MAC benchmark results (sw vs hw cycle counts)
- [ ] `csr.h` snippet showing auto-generated MAC registers
- [ ] VCD waveform: single MAC operation
- [ ] VCD waveform: systolic array pipeline (annotated with skew/drain)
- [ ] Synthesis report: resource utilization summary (LUTs, DSPs, BSRAM)
- [ ] Terminal screenshot: convolution layer PASS output
- [ ] Flow diagram: quantize → export → compile → boot → verify pipeline
- [ ] Terminal screenshot: TinyGrad ACCEL=1 output (if reached)
- [ ] "Number comparison" graphic: FPGA throughput vs CPU vs GPU
- [ ] Photo of the full desk/setup (aspirational lab aesthetic)

---

## Posting Cadence

| Week | Post | Platform |
|---|---|---|
| 1 | Kickoff: project intro + board photo | X + LinkedIn |
| 2 | CSR MAC working | X |
| 3 | Benchmark numbers | X + LinkedIn |
| 4 | TIL or bug story (filler) | X |
| 5 | First conv layer | X + LinkedIn |
| 6 | Tool tip or reading rec (filler) | X |
| 7–8 | Systolic array in sim | X + LinkedIn |
| 9 | Systolic on hardware | X + LinkedIn |
| 10+ | TinyGrad / advanced topics | X + LinkedIn |

Adjust based on your actual pace. The key rule: **post when you have
something visual to show, not on a fixed schedule.**
