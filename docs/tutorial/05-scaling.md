# Part 5 — Scaling Up

> **Series:** [00-overview](00-overview.md) → [01-mac](01-mac.md) → [02-vertical-slice](02-vertical-slice.md) → [03-autonomous](03-autonomous.md) → [04-tinygrad](04-tinygrad.md) → **[05-scaling](05-scaling.md)**
> **Deep Dive:** [Prior Art & Architecture Decisions](appendix-prior-art.md)

Parts 1–4 gave you a working selective-lowering accelerator with a single
4-wide SIMD MAC. This part covers what comes next: spatial parallelism
(systolic array), higher clock speeds, and external memory.

**Only read this after Parts 1–4 are solid.** The autonomous inner loop
from Part 3 captures most of the architectural benefit. Everything here
is a throughput multiplier on top of that foundation.

---

## 5.1  When to Scale

**🤔 Exercise:** Before scaling, measure what you have. From Part 3:

```
  Single 4-wide MAC, autonomous BSRAM, 27 MHz:
    4 MACs/cycle × 27 MHz = 108 MOPS peak

  Is that enough for your target model?

  MobileNet-v2 0.25, 96×96 input:
    ~5.6 million MACs total
    At 108 MOPS: ~52 ms per inference

  Is 52 ms acceptable? For real-time video (30 fps): need < 33 ms. Close.
  For classification of still images: plenty fast.
```

*If your Part 3 design is fast enough, you don't need a systolic array.*
A systolic array adds ~800–2000 lines of HDL complexity. Make sure the
payoff justifies the effort.

---

## 5.2  The Systolic Array

### A Single PE

Each processing element has: weight register, multiplier, adder,
activation passthrough.

```
                  ┌─────────────────────────────────┐
   weight_load ──►│   ┌───────┐                     │
                  │   │weight │  (loaded once)       │
                  │   │  reg  │                      │
                  │   └───┬───┘                      │
                  │       │                          │
   act_in ──────► │───┬───┼──────────────────────────│──► act_out
                  │   │   ▼                          │
                  │   │ ┌─────┐                      │
                  │   └►│  ×  │  INT8 × INT8         │
                  │     └──┬──┘                      │
                  │        ▼                         │
   psum_in ─────► │──► ┌─────┐                       │
                  │    │  +  │  16 + 32 → 32         │
                  │    └──┬──┘                       │
                  │       ▼                          │
                  │   psum_out                       │
                  └──────────────────────────────────┘
```

**Weight-stationary:** weights are loaded once per tile, activations
stream left-to-right, partial sums flow top-to-bottom.

### The 4×4 Array

```
          ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐
  act ──► │PE0,0│─►│PE0,1│─►│PE0,2│─►│PE0,3│──►
          └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘
  act ──► │PE1,0│─►│PE1,1│─►│PE1,2│─►│PE1,3│
          └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘
  act ──► │PE2,0│─►│PE2,1│─►│PE2,2│─►│PE2,3│
          └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘
  act ──► │PE3,0│─►│PE3,1│─►│PE3,2│─►│PE3,3│
          └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘
             ▼        ▼        ▼        ▼
         (partial sums — 4 output columns)
```

- **16 PEs × 1 DSP each = 16 DSPs** (33% of budget)
- Plus 4 DSPs for requantization = 20 DSPs (42%)
- Remaining: 28 DSPs free

### Pipeline Timing

**🤔 Exercise:** For `C[4×4] = A[4×K] × B[K×4]`, the activations must
be **skewed** — row `i` starts `i` cycles after row 0:

```
  Cycle:    0     1     2     3     4     5     6
  Row 0:   a0,0  a0,1  a0,2  a0,3   -     -     -
  Row 1:    -    a1,0  a1,1  a1,2  a1,3   -     -
  Row 2:    -     -    a2,0  a2,1  a2,2  a2,3   -
  Row 3:    -     -     -    a3,0  a3,1  a3,2  a3,3
```

Total cycles from first input to last output:
**K + (R - 1) + (C - 1) = K + 6** for a 4×4 array.

*Draw this for K=8.* Mark when each PE receives matching activation and
partial sum. Verify they align.

### Build Incrementally

1. **Single PE** — test: `psum_out = psum_in + act × weight`
2. **1D row** (4 PEs) — test: dot product of two 4-element vectors
3. **2D array** (4×4) — test: `C = A × B` vs `numpy.matmul`
4. **Integration** — wire to CFU bus, drive from firmware

**Do NOT skip to step 4.** Each step catches different bugs. Off-by-one
timing in the skew is the most common systolic array bug.

---

## 5.3  Data Tiling

For layers larger than 4×4, tile the matmul:

```
  C[M×N] = A[M×K] × B[K×N]

  Tile B into ⌈N/4⌉ column-strips (weight tiles)
  Tile A into ⌈M/4⌉ row-strips (activation tiles)

  Total tile operations: ⌈M/4⌉ × ⌈N/4⌉
  Each tile: K + 6 cycles of array compute + weight reload
```

**🤔 Exercise:** MobileNet-v2 0.25's first pointwise conv:
- 1×1 conv, 16 input channels, 8 output channels, 48×48 spatial
- In matmul terms: M = 48×48 = 2304, K = 16, N = 8
- Tiles: ⌈2304/4⌉ × ⌈8/4⌉ = 576 × 2 = 1152 tile operations
- Each tile: 16 + 6 = 22 cycles of compute
- Total: 1152 × 22 = 25,344 cycles ≈ 0.94 ms at 27 MHz

*How does this compare to Part 3's single-MAC version?* Part 3 at 4
MACs/cycle: 2304 × 8 × 16 / 4 = 73,728 cycles ≈ 2.73 ms. The 4×4 array
is ~2.9× faster for this layer. Not 4× because of tiling overhead.

---

## 5.4  Higher Clock Frequency

The quickest win. Your SoC runs at 27 MHz (raw crystal). The PLL can go
higher.

```
  27 MHz  →  432 MOPS peak  (baseline)
  54 MHz  →  864 MOPS peak  (2× — usually closes timing)
  81 MHz  → 1296 MOPS peak  (3× — may need pipeline registers)
 108 MHz  → 1728 MOPS peak  (4× — likely needs work)
```

**🤔 Exercise:** Change `sys_clk_freq` in the SoC configuration. Rebuild.
Check the synthesis report:
- Does timing close?
- What's the critical path? Is it in the array or in VexRiscv?
- At what frequency does it fail?

If the critical path is in the systolic array, add pipeline registers to
PE outputs. This adds 1 cycle of latency per PE but shortens the
combinational chain.

---

## 5.5  External Memory (PSRAM)

The memory wall. Your on-chip BSRAM (18 KiB of activations) limits the
layer sizes you can process without spatial tiling.

The Tang Nano 20K has 8 MiB of on-package PSRAM (not yet connected):

| Memory | Capacity | Bandwidth | Latency |
|---|---|---|---|
| BSRAM | ~18 KiB free | 32b × 27 MHz = 108 MB/s | 1 cycle |
| PSRAM | 8 MiB | ~40–100 MB/s (burst) | ~100ns initial |

With PSRAM, the entire MobileNet-v2 model (~100 KiB weights) fits. But:

```
  Can you feed the array from PSRAM?

  4×4 array needs: 16 bytes/cycle × 27 MHz = 432 MB/s
  PSRAM provides:  ~100 MB/s (burst)

  432 MB/s needed vs 100 MB/s available → MEMORY BOUND (4× gap)
```

**🤔 Exercise:** What helps?
- **Double-buffering:** Load next tile into BSRAM from PSRAM while
  computing current tile. Overlaps transfer and compute.
- **Weight compression:** 4-bit weights → 2× less bandwidth needed.
- **Tiling strategy:** Maximize data reuse within BSRAM before going
  back to PSRAM.
- **DMA engine:** Dedicated hardware for PSRAM→BSRAM transfers,
  freeing the CPU.

PSRAM integration requires the Gowin PSRAM controller IP or
`litehyperbus`. This is a separate project.

---

## 5.6  Depthwise Convolutions

MobileNet-v2 uses depthwise separable convolutions. The depthwise part
doesn't map to matmul — each output channel depends on only one input
channel (no reduction across channels).

**Options:**
1. Fall back to single SIMD MAC (Part 1) — simple, correct, moderate speed
2. Reshape as batched element-wise multiply — creative but complex
3. Add a dedicated depthwise mode to the array — more hardware

**🤔 Exercise:** What fraction of MobileNet-v2's total MACs are in
depthwise vs pointwise layers? If depthwise is <20% of compute, fallback
to Part 1's MAC is fine. Don't over-optimize the minority case.

---

## 5.7  Optimization Roadmap

Ordered by effort-to-payoff:

| Optimization | Effort | Speedup | When |
|---|---|---|---|
| Higher clock (54 MHz) | Low | 2× | Now |
| Fused requant (Part 3) | Low | 1.1–1.3× | Part 3 |
| Double-buffered weights | Low | 1.2–1.5× | Now |
| 4×4 systolic array | Medium | 2–4× | This part |
| Higher clock (81+ MHz) | Medium | 3–4× | After array |
| PSRAM integration | High | Enables large models | Stretch |
| DMA engine | Medium | 2–5× (with PSRAM) | Stretch |
| Weight compression | Medium | 1.5–2× (bandwidth) | Stretch |

**🤔 Final exercise:** Given your actual measurements from Parts 1–4,
*rank these by expected real-world impact for your specific system.* The
table above is generic — your bottleneck profile may change the ordering.

---

## 5.8  Checkpoint

- [ ] I can explain weight-stationary vs output-stationary tradeoffs
- [ ] I know the DSP/BSRAM budget and what the largest array is
- [ ] I can draw the systolic timing diagram (with skew) from memory
- [ ] I have a tiling strategy for layers larger than 4×4
- [ ] I understand the memory bandwidth bottleneck
- [ ] I have a prioritized optimization roadmap based on real measurements
- [ ] I've decided whether the systolic array is worth building for my use case

---

**Previous:** [Part 4 — Selective Lowering from TinyGrad](04-tinygrad.md)
**Deep Dive:** [Prior Art & Architecture Decisions](appendix-prior-art.md)

## References

| Topic | Source |
|---|---|
| Google TPU v1 (systolic arrays) | [arxiv.org/abs/1704.04760](https://arxiv.org/abs/1704.04760) |
| Eyeriss (dataflow taxonomy) | [eyeriss.mit.edu](https://eyeriss.mit.edu/) |
| CFU-Playground (all 4 tiers) | [github.com/google/CFU-Playground](https://github.com/google/CFU-Playground) |
| TinyGrad UOps | [docs.tinygrad.org/developer/uop](https://docs.tinygrad.org/developer/uop/) |
| Gowin GW2AR-18C datasheet | Gowin Semiconductor |
| Gowin pDSP user guide | UG289E |
| RISC-V `.insn` directive | [sourceware.org](https://sourceware.org/binutils/docs/as/RISC_002dV_002dFormats.html) |
