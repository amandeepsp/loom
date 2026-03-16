# Appendix — Prior Art Deep Dive & Architecture Decisions

> **Parent:** [05-scaling](05-scaling.md)
>
> This document records everything learned from studying existing ML
> accelerator designs and challenges every architectural assumption made
> in the tutorial series. The goal: build the most efficient accelerator
> possible on the GW2AR-18C, not just a "good enough" learning exercise.

---

## A.1  Google CFU-Playground: What They Actually Built

The [CFU-Playground](https://github.com/google/CFU-Playground) is the
project this accelerator is most directly inspired by. Studying its most
sophisticated example (`hps_accel/gen2`) reveals that **our initial
architecture assumptions were wrong in several important ways**.

### A.1.1  The `hps_accel` Gen2 Architecture

Google's most advanced CFU-Playground project uses a **4×2 systolic array**
(not 4×4, not 8×8), where each cell is a 4-wide SIMD MACC block:

```
  hps_accel Gen2 Systolic Array:

  4 rows × 2 columns × 4 MACs/block = 32 MACs/cycle

  ┌──────────────────────────────────────────────────────┐
  │                                                      │
  │  FilterStore[0]              FilterStore[1]           │
  │  (512×32b BSRAM)            (512×32b BSRAM)          │
  │       │                          │                   │
  │       ▼ (4×INT8)                ▼ (4×INT8)          │
  │  ┌─────────┐  act_pass   ┌─────────┐               │
  │  │MACC[0,0]│────────────►│MACC[0,1]│               │
  │  │ 4 MACs  │             │ 4 MACs  │               │
  │  └────┬────┘  filt_pass  └────┬────┘               │
  │       │ ▼                     │ ▼                   │
  │  ┌─────────┐             ┌─────────┐               │
  │  │MACC[1,0]│────────────►│MACC[1,1]│               │
  │  └────┬────┘             └────┬────┘               │
  │       │ ▼                     │ ▼                   │
  │  ┌─────────┐             ┌─────────┐               │
  │  │MACC[2,0]│────────────►│MACC[2,1]│               │
  │  └────┬────┘             └────┬────┘               │
  │       │ ▼                     │ ▼                   │
  │  ┌─────────┐             ┌─────────┐               │
  │  │MACC[3,0]│────────────►│MACC[3,1]│               │
  │  └─────────┘             └─────────┘               │
  │       │                       │                     │
  │       ▼ (8 accumulators)     ▼                     │
  │  ┌────────────────────────────────┐                 │
  │  │  PostProcess Pipeline (7 cyc)  │                 │
  │  │  SRDHM → RDBPOT → Clamp → INT8│                 │
  │  └───────────────┬────────────────┘                 │
  │                  ▼                                  │
  │  ┌────────────────────────────────┐                 │
  │  │  Output FIFO (1024 × 32b)     │                 │
  │  │  packs 4 × INT8 → 32b word   │                 │
  │  └────────────────────────────────┘                 │
  └──────────────────────────────────────────────────────┘
```

### A.1.2  Key Design Decisions That Differ From Ours

| Decision | Our Plan | Google hps_accel | Why It Matters |
|---|---|---|---|
| **Array shape** | 4×4 scalar | 4×2 × 4-wide SIMD | Same throughput (16 vs 32 MACs), but 4×2 needs only 2 filter stores vs 4 |
| **Dataflow** | Weight-stationary | Output-stationary (accumulator stays in PE) | Weights AND activations both flow; accumulator is pinned |
| **Weight storage** | CPU writes via CSR/insn | Pre-loaded into BSRAM filter stores, free-running cyclic read | Zero CPU involvement during compute |
| **Activation source** | CPU writes via CSR/insn | 4-bank LRAM, autonomous fetch | 16 bytes/cycle, no CPU involvement |
| **Data on hot path** | Everything crosses CPU bus | NOTHING crosses CPU bus during compute | CPU only does setup + read output |
| **Post-processing** | Software requantization | Full 7-stage HW pipeline: SRDHM → RDBPOT → clamp → INT8 | Zero CPU cycles for requantization |
| **Output delivery** | CPU reads individual CSRs | 1024-deep FIFO, CPU drains at leisure | Decouples compute from output read |

### A.1.3  The Data Movement Architecture

**This is the single most important lesson from hps_accel.**

The CPU is NOT in the data path during inference. The architecture is:

```
  ┌─────────────────────────────────────────────────────────┐
  │  BEFORE inference (slow path — CPU involved):           │
  │                                                         │
  │  CPU ──SET──► FilterStore BSRAM  (write weights)        │
  │  CPU ──SET──► ParamMem BSRAM    (write requant params)  │
  │  CPU ──SET──► Config registers  (mode, dimensions)      │
  │  CPU ──SET──► LRAM              (write activations)     │
  │  CPU ──SET──► START pulse                               │
  └─────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────┐
  │  DURING inference (hot path — CPU NOT involved):        │
  │                                                         │
  │  LRAM banks ──► RamMux ──► InputFetcher ──► Array       │
  │       (16 bytes/cycle, autonomous)           │          │
  │                                              │          │
  │  FilterStore ──► Array (cyclic read)         │          │
  │       (4 bytes/cycle/col, autonomous)        │          │
  │                                              ▼          │
  │                                         PostProcess     │
  │  ParamMem ──► PostProcess                    │          │
  │       (per-channel, autonomous)              ▼          │
  │                                         Output FIFO     │
  └─────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────┐
  │  AFTER inference (slow path — CPU involved):            │
  │                                                         │
  │  CPU ◄──GET── Output FIFO  (read packed INT8 results)   │
  └─────────────────────────────────────────────────────────┘
```

**The CPU touches data only at setup and teardown.** During the actual
matrix multiply, all data movement is autonomous hardware reading from
on-chip SRAM. This is fundamentally different from our plan where the CPU
is the data pump.

### A.1.4  The 4-Bank LRAM Trick

Activations are stored in 4 interleaved LRAM banks. A `RamMux` rotates
which bank each address port maps to on each cycle:

```
  Bank-conflict-free 4-bank interleaved access:

  Cycle:    0         1         2         3         4    ...
  Port 0: Bank0     Bank1     Bank2     Bank3     Bank0
  Port 1:   -       Bank0     Bank1     Bank2     Bank3
  Port 2:   -         -       Bank0     Bank1     Bank2
  Port 3:   -         -         -       Bank0     Bank1

  By cycle 3, all 4 ports are active on different banks.
  Sustained throughput: 4 × 32b = 128 bits/cycle = 16 bytes/cycle.

  The phase rotation ensures zero bank conflicts even though
  all 4 ports are accessing sequential addresses.
```

**We don't have LRAM on Gowin.** But we have 16–18 free BSRAMs (18Kbit
each). We could use 4 BSRAMs as activation banks (4 × 2KiB = 8KiB), plus
2 BSRAMs as filter stores (2 × 2KiB = 4KiB). Total: 6 BSRAMs for data,
leaving ~10 for other uses.

### A.1.5  The `first`/`last` Accumulator Control

Instead of tiling the K dimension and managing partial sums externally,
hps_accel uses `first` and `last` signals that propagate through the
systolic array:

```
  first/last control (eliminates K-dimension tiling overhead):

  For a conv layer with input_depth=64 (K=64):
  The filter store has 64/4 = 16 entries.

  Cycle:  0     1     2    ...   15    16    17   ...
  first:  1     0     0    ...    0     1     0   ...
  last:   0     0     0    ...    1     0     0   ...

  On first=1: accumulator resets to 0
  On last=1:  accumulator value is latched to output register
  Between:    accumulator keeps accumulating

  The array processes the ENTIRE K dimension in one pass.
  No tiling, no partial sum management, no CPU intervention.

  For K=64 with 4 MACs/block: 16 cycles of accumulation
  then output is ready. Next output channel starts immediately.
```

This is much cleaner than our proposed approach of tiling K and reading
partial sums after each tile.

---

## A.2  Google CFU-Playground: CfuBase Design Patterns

### A.2.1  The Reference `Cfu` Class

Google's `Cfu` class (in `python/amaranth_cfu/cfu.py`) uses a **3-state
FSM** — more robust than our simple one-entry response buffer:

```
  Google Cfu FSM:

  ┌──────────┐   cmd_valid & done     ┌──────────────┐
  │          │   & rsp_ready          │              │
  │ WAIT_CMD │◄──────────────────────│ WAIT_TRANSFER│
  │          │                        │ (CPU not     │
  │ cmd_ready│   cmd_valid & done    │  ready yet)  │
  │ = 1      │   & !rsp_ready        │ rsp_valid=1  │
  │          │───────────────────────►│              │
  │          │                        └──────┬───────┘
  │          │   cmd_valid & !done           │ rsp_ready
  │          │──────────┐                    │
  └──────────┘          │            ┌───────┘
                        ▼            │
               ┌──────────────┐      │
               │              │      │
               │WAIT_INSTRUCT.│──────┘
               │ (multi-cycle │  done & !rsp_ready
               │  instruction)│
               │ cmd_ready=0  │──────┐
               │              │      │ done & rsp_ready
               └──────────────┘      │
                        ▲            │
                        │            ▼
                        │     ┌──────────┐
                        └─────│ WAIT_CMD │
                              └──────────┘
```

Key differences from our implementation:
1. **`stored_output` register** latches the result when the instruction
   finishes before the CPU is ready — prevents result loss
2. **`stored_function_id`** remembers which instruction is executing
   after `cmd_valid` deasserts
3. **Fallback instructions** for all 8 funct3 slots — unregistered opcodes
   return `in0` immediately instead of hanging
4. **LRAM ports** — 4 direct SRAM read ports bypassing the main bus

### A.2.2  Instruction Dispatch via `funct3`

Google's `Cfu` dispatches on `funct3` (bits 0–2 of `cmd_function_id`),
using up to **8 independent instruction modules**. Each instruction gets
`funct7` (bits 3–9) as a sub-opcode. Our `SimdMac` dispatches on `funct7`
only, which limits us to a single instruction class.

```
  Google dispatch:                    Our dispatch:

  cmd_function_id[9:0]               cmd_function_id[9:0]
  ┌───────────┬────────┐             ┌───────────┬────────┐
  │  funct7   │ funct3 │             │  funct7   │ funct3 │
  │  (7 bits) │(3 bits)│             │  (7 bits) │(3 bits)│
  └─────┬─────┴───┬────┘             └─────┬─────┴────────┘
        │         │                         │     (unused)
        │    ┌────┴────┐                    │
        │    │ 8-way   │                    │
        │    │ dispatch │               ┌───┴────┐
        │    └────┬────┘               │ funct7 │
        │    ┌────┴─────┐              │ == 0?  │
        │    │Instruction│              └───┬────┘
        │    │  module   │              yes │ no
        │    │  receives │                  │
        ▼    │  funct7   │              MAC  RESET
             └───────────┘

  Google: 8 instructions × 128 sub-ops = 1024 functions
  Ours:   1 instruction  × 128 sub-ops = 128 functions
```

**Implication for systolic array:** We should use `funct3` to dispatch
between LOAD_WEIGHT, STREAM_ACT, READ_OUTPUT, and RESET — exactly as
planned in Part 3, Section 3.9. This matches Google's pattern.

---

## A.3  The Requantization Pipeline: Hardware Details

Both `hps_accel` and `mnv2_first` implement the full TFLite INT32→INT8
requantization in hardware. This is **not optional** for a high-efficiency
design — software requantization dominates the cycle count once data
movement is solved.

### A.3.1  The Math

```
  TFLite quantized convolution output transform:

  acc += bias                          // per-channel INT32 bias
  acc = SRDHM(acc, multiplier)         // SaturatingRoundingDoubleHighMul
                                       //   acc × multiplier → high 32 bits
                                       //   effectively: acc × M where M ∈ [0.5, 1.0)
  acc = RDBPOT(acc, shift)             // RoundingDivideByPowerOfTwo
                                       //   acc >> shift, with correct rounding
  acc += output_offset                 // per-layer INT9
  acc = clamp(acc, min, max)           // saturate to [-128, 127]
  output = (int8_t)acc
```

### A.3.2  Hardware Pipeline

```
  PostProcess Pipeline (hps_accel: 7 cycles, mnv2_first: 4 cycles):

  INT32 accumulator
      │
      ▼ ─── + bias (per-channel, signed(18)) ─── combinational
      │
      ▼ ─── SRDHM: 3 pipeline stages ────────────────────────
      │     Cycle 0: register abs(a), b
      │     Cycle 1: multiply 32×32 → 63-bit product
      │     Cycle 2: nudge, extract high 32 bits, restore sign
      │
      │     ┌──────────────────────────────────────────────┐
      │     │ This is the expensive part:                  │
      │     │ One 32×32 multiplier = 2–4 DSP blocks       │
      │     │ On Gowin: 4 × MULT9X9 or 2 × MULT18X18     │
      │     │ This is UNAVOIDABLE for correct requant.     │
      │     └──────────────────────────────────────────────┘
      │
      ▼ ─── RDBPOT: 1–3 pipeline stages ─────────────────────
      │     Variable right-shift by per-channel amount (2–12)
      │     Implemented as mux tree over shift values
      │     Cost: ~50–80 LUTs (no DSPs)
      │
      ▼ ─── + output_offset (per-layer) ──── combinational
      │
      ▼ ─── clamp to [min, max] ────────── 1 stage
      │     Two comparators + mux
      │
      ▼
    INT8 output

  Per-channel parameters stored in BSRAM:
  ┌────────────────────────────────────────────────┐
  │  ParamMem: up to 512 channels                  │
  │  Per entry: bias(18b) + multiplier(32b) +      │
  │             shift(4b) = 54 bits                │
  │  Storage: ~512 × 54b ≈ 27 Kbit ≈ 2 BSRAMs    │
  └────────────────────────────────────────────────┘

  Throughput: 1 INT8 output / cycle (fully pipelined)
```

### A.3.3  Resource Cost for Requantization

| Resource | Cost | Notes |
|---|---|---|
| DSP blocks | 2–4 (for 32×32 multiply) | Depends on Gowin MULT primitive packing |
| LUTs | ~150–200 | Adders, mux trees, comparators |
| Flip-flops | ~130–230 | Pipeline registers |
| BSRAM | 2 | Per-channel parameter store (54b × 512) |

**Impact on our DSP budget:** If the 32×32 multiply uses 4 DSPs, that
leaves 48 - 16 (array) - 4 (requant) = 28 DSPs free. Still comfortable.

---

## A.4  Challenging Our Architecture Assumptions

### A.4.1  Assumption: "Weight-Stationary Is Best for Inference"

**Challenge:** Google's hps_accel is effectively **output-stationary** —
the accumulator stays in the PE while both weights and activations flow.

```
  Weight-Stationary (our plan):       Output-Stationary (hps_accel):

  Pinned: weights                     Pinned: accumulator (partial sum)
  Flows:  activations (→),            Flows:  activations (→),
          partial sums (↓)                    weights (↓)
  Reload: weights per tile            Reload: nothing (acc resets via first/last)

  WS advantage:                       OS advantage:
  • Weights loaded once, reused       • No partial sum management
    across many activations           • K dimension handled in one pass
  • Simple control                    • No output tiling needed
                                      • first/last signals eliminate
  WS disadvantage:                      explicit reset/read cycles
  • Must tile K dimension if
    K > array width                   OS disadvantage:
  • CPU must read partial sums        • Weights must stream continuously
    and accumulate across tiles       • Needs filter store BSRAM
  • Output tiling adds overhead       • More complex control
```

**Verdict:** For our use case (small array, Conv2D layers with moderate
channel depth), output-stationary with `first`/`last` control is superior.
It eliminates the K-dimension tiling overhead that makes our CSR-based
approach so wasteful.

**However:** Weight-stationary is simpler to implement and debug. For a
learning project, it's a valid Phase 1 choice. The key insight is that
**the dataflow choice matters less than the data delivery mechanism** —
whether you use WS or OS, if the CPU is the data pump, you lose.

### A.4.2  Assumption: "4×4 Is the Right Array Shape"

**Challenge:** Google chose 4×2 with 4-wide SIMD inside each block.

```
  Why 4×2 × 4-SIMD instead of 4×4 × 1-scalar:

  4×4 scalar array:                   4×2 × 4-SIMD array:
  • 16 PEs                            • 8 MACC blocks
  • 16 multipliers                    • 32 multipliers (8 × 4)
  • 16 accumulators                   • 8 accumulators
  • 4 filter stores needed            • 2 filter stores needed
  • 4 activation inputs               • 4 activation inputs (4 rows)
  • 16 MACs/cycle                     • 32 MACs/cycle

  Filter store bandwidth:
  4×4: needs 4 × 32b/cycle = 128b    4×2: needs 2 × 32b/cycle = 64b
       (4 BSRAMs for filters)              (2 BSRAMs for filters)

  The bottleneck is filter store bandwidth.
  More columns = more filter stores = more BSRAM.
  Fewer columns + wider SIMD = same throughput, half the BSRAM.
```

**For Gowin GW2AR-18C:**
- 48 DSPs available
- A 4×2 × 4-SIMD array uses 32 DSPs (67% — aggressive but feasible)
- A 4×4 scalar array uses 16 DSPs (33% — conservative)
- **But:** Gowin pDSP blocks are 9×9 or 18×18 multipliers, NOT
  Lattice's `MULTADDSUB9X9WIDE` which does 4 multiplies in one block.
  We need 1 DSP per INT8×INT8 multiply. So 4-wide SIMD costs 4 DSPs
  per MACC block, same as 4 separate PEs.

**Verdict:** On Gowin, 4×4 and 4×2×4-SIMD use the same number of DSPs
(16). The advantage of 4×2 is fewer filter stores (2 vs 4 BSRAMs). The
advantage of 4×4 is simpler control and more output channels per tile.
**4×4 is fine for our FPGA.** But consider keeping it 4×2 if BSRAM is
tight.

### A.4.3  Assumption: "CSR → Custom Instructions → DMA Is the Right Path"

**Challenge:** Google skipped CSR entirely. They went straight to custom
instructions with on-chip SRAM buffering. There is no CSR-mapped version
of hps_accel.

```
  Our planned progression:           Google's actual progression:

  Phase 1: CSR-mapped array           Phase 1: Simple SIMD MAC via CFU
           (CPU writes operands)               (example_cfu, avg_pdti8)
           ↓                                   ↓
  Phase 2: CfuPlugin + .insn          Phase 2: SIMD MAC + filter BSRAM
           (CPU still data pump,               (mnv2_first — weights in
            but 3× faster)                      BSRAM, acts via CFU insn)
           ↓                                   ↓
  Phase 3: DMA from PSRAM             Phase 3: Full systolic array +
           (autonomous fetch)                   LRAM activation banks +
                                                PostProcess pipeline
                                                (hps_accel/gen2)
```

**Key insight:** Google's Phase 2 (`mnv2_first`) already had weights in
BSRAM. They never had a phase where the CPU writes weights on the hot
path. The BSRAM filter store is not a "stretch optimization" — it's a
**prerequisite for any non-trivial speedup**.

**Revised recommendation:**
- Phase 1: CSR-mapped array (for correctness only, no performance claims)
- Phase 1.5: Add BSRAM filter stores (weights pre-loaded, free-running)
- Phase 2: CfuPlugin + .insn (CPU streams activations only)
- Phase 3: BSRAM activation banks (fully autonomous compute)

### A.4.4  Assumption: "We Need External Memory Before DMA Matters"

**Challenge:** hps_accel uses **on-chip** LRAM (256 KiB on Lattice NX),
not external memory. DMA from on-chip SRAM is what matters, not DMA from
external memory.

We don't have LRAM, but we have **16–18 free BSRAMs × 18 Kbit = ~36 KiB**.
That's enough for:
- 4 × 2KiB activation banks = 8 KiB
- 2 × 2KiB filter stores = 4 KiB
- 2 × ~1.5KiB parameter stores = 3 KiB
- Remaining: ~21 KiB for other uses

```
  BSRAM budget (16 free BSRAMs):

  ┌────────────────────────────────────────────┐
  │ Usage              BSRAMs    Capacity       │
  ├────────────────────────────────────────────┤
  │ Activation bank 0    1      2,048 bytes    │
  │ Activation bank 1    1      2,048 bytes    │
  │ Activation bank 2    1      2,048 bytes    │
  │ Activation bank 3    1      2,048 bytes    │
  │ Filter store 0       1      2,048 bytes    │
  │ Filter store 1       1      2,048 bytes    │
  │ Requant params       2      ~3,456 bytes   │
  │ Output FIFO          1      2,048 bytes    │
  ├────────────────────────────────────────────┤
  │ TOTAL                9      ~18 KiB        │
  │ Remaining            7      ~16 KiB free   │
  └────────────────────────────────────────────┘

  This gives us 8 KiB of activation buffering.
  MobileNet-v2 0.25 largest activation tensor: ~72 KiB (96×96×8)
  So we still need spatial tiling — but 8 KiB is enough for
  several rows at a time.
```

**Verdict:** We can build a hps_accel-like autonomous data path using
only on-chip BSRAM. No external memory needed for the data feeding
mechanism. External memory (PSRAM) is only needed to store the full
model's weights and activations — a separate concern.

### A.4.5  Assumption: "INT8 Is the Right Precision"

**Not challenged.** INT8 is correct for this project:
- MobileNet-v2 is designed for INT8 quantization
- Gowin DSPs are 9×9 or 18×18 — perfect for INT8 (or INT9 with offset)
- Sub-8-bit (INT4, binary) would need a different model architecture
- Going wider (INT16) would halve throughput for negligible accuracy gain

### A.4.6  Assumption: "Requantization Can Wait"

**Challenge:** Google implemented fused requantization from their second
project (`mnv2_first`), not as a late optimization. Without it, the CPU
spends significant cycles on the INT32→INT8 conversion between layers.

```
  Cost of software requantization per output element:

  // This is what the CPU does without HW fusion:
  int32_t scaled = (int64_t)acc * multiplier;  // 4-8 cycles (no HW mul64)
  scaled >>= 31;                                // 1 cycle
  scaled += nudge;                              // 1 cycle
  scaled >>= shift;                             // 1 cycle
  scaled += offset;                             // 1 cycle
  if (scaled < -128) scaled = -128;             // 1-2 cycles
  if (scaled > 127) scaled = 127;               // 1-2 cycles

  Total: ~10-15 cycles per output element on RV32IM (no mul64)
  For 4 outputs per tile: ~40-60 cycles
  For 16 outputs per tile: ~160-240 cycles

  With HW fusion: 0 CPU cycles (pipelined, 1 output/cycle after fill)
```

**Verdict:** Fused requantization should be Phase 1.5, not Phase 3. It's
moderate effort and eliminates a significant CPU bottleneck.

---

## A.5  Gowin pDSP Block Capabilities

### A.5.1  What We Know

The GW2AR-18C has 48 pDSP blocks. Each can be configured as:
- **9×9 signed multiplier** (MULT9X9) — 1 INT8×INT8 multiply per DSP
- **18×18 signed multiplier** (MULT18X18) — 1 INT16×INT16 or wider
- **Pre-adder + multiplier** configurations
- **Multiply-accumulate** modes (with built-in accumulator)

### A.5.2  Can We Pack Two INT8 Multiplies Per DSP?

**The key question for doubling our throughput.**

On Xilinx DSP48, you can pack two INT8×INT8 multiplies into one 18×18 by
exploiting non-overlapping bit ranges:

```
  Dual INT8 packing trick (Xilinx DSP48):

  A = a_hi << 9 | a_lo         (pack two 8-bit values, separated by gap)
  B = b_val                     (one 8-bit value)
  P = A × B = (a_hi × b_val) << 9 | (a_lo × b_val)

  Extract: prod_lo = P[15:0], prod_hi = P[24:9]

  But this only works for A×B where B is shared.
  For independent multiplies (a0×b0 and a1×b1), you need:

  A = a0 | (a1 << 11)          B = b0 | (b1 << 11)
  P = a0×b0 + (a0×b1 + a1×b0)<<11 + (a1×b1)<<22
                ^^^^^^^^^^^^^^^^^^^^
                cross terms contaminate!

  So simple packing does NOT give independent multiplies.
```

**For Gowin pDSP:** The 9×9 mode gives one INT8×INT8 per DSP. The 18×18
mode gives one multiply with wider operands, not two independent
multiplies. **No dual-packing shortcut on Gowin.**

However, the Gowin pDSP in 9×9 mode may be chained with adjacent DSPs
for accumulation. Check the Gowin DSP user guide (UG289E) for the exact
`MULTALU` and `MULTADDALU` primitive configurations.

### A.5.3  DSP Budget Allocation

```
  Optimal DSP allocation for GW2AR-18C (48 DSPs):

  Option A: 4×4 array + requant
  ├── Systolic array: 16 DSPs (16 × MULT9X9)
  ├── Requantization: 4 DSPs  (1 × 32×32 multiply = 4 × MULT9X9)
  ├── Remaining:     28 DSPs
  └── Utilization:   42%

  Option B: 4×2 × 4-SIMD array + requant
  ├── Systolic array: 32 DSPs (8 blocks × 4 × MULT9X9)
  ├── Requantization: 4 DSPs
  ├── Remaining:     12 DSPs
  └── Utilization:   75%

  Option C: 8×4 array + requant (maximum)
  ├── Systolic array: 32 DSPs
  ├── Requantization: 4 DSPs
  ├── Remaining:     12 DSPs
  └── Utilization:   75%

  Option D: 4×4 array + 2× requant (dual output pipeline)
  ├── Systolic array: 16 DSPs
  ├── Requantization: 8 DSPs  (2 pipelines for 2× throughput)
  ├── Remaining:     24 DSPs
  └── Utilization:   50%
```

---

## A.6  Revised Architecture Proposal

Based on the prior art study, here is what a "tour de force" accelerator
on the GW2AR-18C should look like:

```
  ┌──────────────────────────────────────────────────────────┐
  │                    Revised Architecture                   │
  │                                                          │
  │  ┌──────────┐    CfuPlugin        ┌──────────────────┐  │
  │  │ VexRiscv  │◄──── .insn ────────►│  CFU Controller  │  │
  │  │ full+cfu  │    (setup/read)     │  (3-state FSM)   │  │
  │  └──────────┘                      └────────┬─────────┘  │
  │                                             │            │
  │  ┌──────────────────────────────────────────┼──────────┐ │
  │  │              Autonomous Datapath         │          │ │
  │  │                                          │          │ │
  │  │  ┌──────────────┐   ┌──────────────┐    │          │ │
  │  │  │ Act BSRAM ×4 │   │Filter BSRAM×2│    │          │ │
  │  │  │ (8 KiB total)│   │(4 KiB total) │    │          │ │
  │  │  └──────┬───────┘   └──────┬───────┘    │          │ │
  │  │         │                   │            │          │ │
  │  │    ┌────┴─────┐       ┌────┴─────┐      │          │ │
  │  │    │  RamMux   │       │  Cyclic  │      │          │ │
  │  │    │  4-phase  │       │  Reader  │      │          │ │
  │  │    └────┬─────┘       └────┬─────┘      │          │ │
  │  │         │                   │            │          │ │
  │  │         ▼                   ▼            │          │ │
  │  │  ┌────────────────────────────────┐      │          │ │
  │  │  │     Systolic Array 4×4         │      │          │ │
  │  │  │     (16 DSPs, 16 MACs/cycle)   │      │          │ │
  │  │  │     first/last accumulator     │      │          │ │
  │  │  │     control (no K tiling)      │      │          │ │
  │  │  └────────────┬──────────────────┘      │          │ │
  │  │               │                          │          │ │
  │  │               ▼                          │          │ │
  │  │  ┌────────────────────────────────┐      │          │ │
  │  │  │  PostProcess Pipeline (4-7 cyc)│      │          │ │
  │  │  │  + ParamMem BSRAM (2 blocks)   │      │          │ │
  │  │  │  SRDHM → RDBPOT → Clamp → INT8│      │          │ │
  │  │  │  (4 DSPs for 32×32 multiply)  │      │          │ │
  │  │  └────────────┬──────────────────┘      │          │ │
  │  │               │                          │          │ │
  │  │               ▼                          │          │ │
  │  │  ┌────────────────────────────────┐      │          │ │
  │  │  │  Output FIFO (1 BSRAM)        │◄─────┘          │ │
  │  │  │  CPU drains via GET insn       │                 │ │
  │  │  └────────────────────────────────┘                 │ │
  │  └─────────────────────────────────────────────────────┘ │
  │                                                          │
  │  Resource budget:                                        │
  │  DSPs:  16 (array) + 4 (requant) = 20 / 48 (42%)       │
  │  BSRAM: 4 (act) + 2 (filt) + 2 (params) + 1 (fifo)    │
  │         = 9 / 16 free (56%)                              │
  │  LUTs:  ~3000-4000 (array + control + postprocess)       │
  │         out of ~10,000 free                              │
  └──────────────────────────────────────────────────────────┘
```

### Key Differences From Original Plan

| Aspect | Original Plan | Revised Plan |
|---|---|---|
| Data delivery | CPU is data pump (CSR or .insn) | Autonomous BSRAM fetch |
| Weight storage | CPU writes per tile | BSRAM filter stores, pre-loaded |
| Activation storage | CPU writes per column | BSRAM activation banks, pre-loaded |
| K tiling | CPU manages tiles | first/last signals, single-pass |
| Requantization | Software (CPU) | Hardware pipeline (4 DSPs) |
| Output delivery | CPU reads CSRs | Output FIFO, CPU drains |
| CPU role during compute | 100% busy pumping data | Idle (or prefetching next layer) |

---

## A.7  The Full CFU-Playground Design Spectrum

Google's CFU-Playground contains four distinct complexity tiers. Each tier
teaches different lessons about the complexity/performance trade-off:

```
  The CFU Design Spectrum:

  COMPLEXITY / AREA ──────────────────────────────────────────────►

  proj_accel_1      avg_pdti8         mnv2_first         hps_accel
  ─────────────     ─────────────     ─────────────     ─────────────
  Scalar MAC        4-wide SIMD       4-wide SIMD +     4×2 Systolic
  + bounds check    + HW requant      autonomous loop   + LRAM + PP

  ~1 MAC/cycle      ~4 MACs/cycle     ~4 MACs/cycle     ~32 MACs/cycle
  CPU drives ALL    CPU drives outer  CPU: load+start   CPU: config+poll
                    loop              HW: inner loop    HW: everything

  ~100 lines        ~200 lines        ~800 lines        ~2000+ lines
  ─────────────     ─────────────     ─────────────     ─────────────
```

### A.7.1  Tier 1: Scalar MAC (`proj_accel_1`)

The simplest useful CFU. One multiply-accumulate per instruction:

```
  acc += filter_val × (input_val + offset)
```

Interesting non-obvious instruction: `DoubleCompare` — checks
`(0 ≤ x < W) && (0 ≤ y < H)` in one cycle, eliminating a branch-heavy
bounds check from the convolution padding logic. **Lesson: look for
control-flow hotspots, not just arithmetic ones.**

### A.7.2  Tier 2: 4-Wide SIMD + Quantization (`avg_pdti8`)

This is essentially what our `SimdMac` does, plus two critical additions:

1. **Hardware SRDHM** — `SaturatingRoundingDoubleHighMul(a, multiplier)`
   in 2 cycles instead of ~40 CPU cycles
2. **Hardware RDBPOT** — `RoundingDivideByPowerOfTwo(x, shift)` in 1
   cycle instead of ~15 CPU cycles

```c
  // avg_pdti8 firmware hot path:
  for (int i = 0; i < input_depth; i += 4) {
      uint32_t in4  = *(uint32_t*)&input_data[i];
      uint32_t flt4 = *(uint32_t*)&filter_data[i];
      cfu_op2(in4, flt4);                          // 4 MACs, 1 cycle
  }
  int32_t acc = cfu_op1(0, 0);                     // read accumulator
  acc = cfu_op7(0, acc, output_multiplier[ch]);     // SRDHM, 2 cycles
  acc = cfu_op6(0, acc, -output_shift[ch]);         // RDBPOT, 1 cycle
  acc += output_offset;                             // CPU
  acc = clamp(acc, -128, 127);                      // CPU
```

**Key lesson:** Moving SRDHM + RDBPOT to hardware is a **high-ROI
optimization independent of MAC width**. We should do this even before
building the systolic array. The accumulator value is already in the CFU —
requantizing it there costs almost nothing in hardware but saves ~55 CPU
cycles per output element.

### A.7.3  Tier 3: Autonomous Inner Loop (`mnv2_first`)

This is the crucial intermediate step we were missing. `mnv2_first` adds:

1. **Filter BSRAM stores** — weights pre-loaded into 4 EBRAM banks
   (8 KiB total), cyclic read during compute
2. **Input double-buffering** — CPU writes next pixel's input into
   buffer B while hardware reads from buffer A
3. **Autonomous sequencer** — hardware iterates over all input channels
   × output channels without CPU involvement
4. **Hardware post-processing** — full SRDHM → RDBPOT → clamp pipeline
5. **Output FIFO** — CPU drains packed INT8 results at leisure

```
  mnv2_first data flow:

  ┌──────────┐
  │   CPU    │──► Write weights to filter BSRAM (once per layer)
  │          │──► Write input pixels (double-buffered, per pixel)
  │          │──► START
  │          │
  │          │◄── Read output FIFO (packed 4×INT8 per word)
  └──────────┘

  ┌──────────────────────────────────────────────────┐
  │  HARDWARE (runs autonomously after START):       │
  │                                                  │
  │  FilterStore   InputStore     Madd4Pipeline      │
  │  (4×EBRAM)     (double-buf)  (4 MACs/cycle)     │
  │     │              │              │              │
  │     └──────────────┴──────►  Accumulator         │
  │                                   │              │
  │                              PostProcess         │
  │                              (SRDHM+RDBPOT)      │
  │                                   │              │
  │                              Output FIFO         │
  │                              (512 × 32b)         │
  └──────────────────────────────────────────────────┘

  CPU involvement during compute: ZERO for inner loop
  CPU only writes next pixel input (overlapped with compute)
```

**Throughput:** Still 4 MACs/cycle (single pipeline), but the CPU is free
to do other work. The autonomous sequencer handles the entire
`input_depth × output_channels` iteration.

**Key lesson:** You don't need a systolic array to get autonomous
operation. A single 4-wide MACC with an autonomous sequencer + BSRAM
stores achieves the same CPU-freedom as hps_accel, just at lower
throughput. **This is our ideal Phase 2 target.**

### A.7.4  Tier 4: Systolic Array (`hps_accel`)

Covered in detail in sections A.1–A.3. The jump from Tier 3 to Tier 4
adds spatial parallelism (4×2 array = 8× the throughput) but at 2.5×
the code complexity and requiring LRAM banks for activation bandwidth.

### A.7.5  Implications for Our Design

```
  Revised phased approach based on CFU-Playground spectrum:

  Our Phase 1  →  avg_pdti8 pattern
  ─────────────────────────────────────────────────
  • 4-wide SIMD MAC (already have this: SimdMac)
  • ADD: Hardware SRDHM + RDBPOT instructions
  • ADD: Input offset as configurable register
  • CPU drives all loops
  • Effort: Low (2 new instructions, ~100 lines)
  • Payoff: ~55 CPU cycles saved per output element

  Our Phase 2  →  mnv2_first pattern
  ─────────────────────────────────────────────────
  • Same 4-wide MACC, but with:
  • ADD: Filter BSRAM stores (pre-loaded weights)
  • ADD: Input double-buffer
  • ADD: Autonomous sequencer (HW inner loop)
  • ADD: Full PostProcess pipeline
  • ADD: Output FIFO
  • CPU: loads weights once, writes inputs, drains FIFO
  • Effort: Medium (~800 lines gateware)
  • Payoff: CPU freed from inner loop entirely

  Our Phase 3  →  hps_accel pattern (if needed)
  ─────────────────────────────────────────────────
  • 4×4 (or 4×2) systolic array
  • BSRAM activation banks (replaces LRAM)
  • Autonomous activation fetch
  • CPU: configure + poll output FIFO
  • Effort: High (~2000 lines gateware)
  • Payoff: 4–8× throughput over Phase 2

  KEY INSIGHT: Phase 2 (mnv2_first pattern) captures most of
  the architectural benefit. Phase 3 (systolic array) is a
  throughput multiplier on top of an already-efficient design.
  Don't skip Phase 2 to jump to Phase 3.
```

---

## A.8  Open Questions for Further Research

1. **Gowin pDSP chaining:** Can adjacent pDSP blocks be chained for
   built-in accumulation (like Xilinx DSP48 cascading)? This would
   eliminate the LUT-based adder tree in the PE.

2. **BSRAM dual-port:** Gowin BSRAM supports true dual-port (read +
   write simultaneously). Can we use this for ping-pong buffering
   within a single BSRAM — write next tile while reading current?

3. **Clock speed ceiling:** What is the maximum clock frequency for a
   4×4 systolic array on GW2AR-18C? The PLL goes to 625 MHz but the
   fabric likely limits to 100–150 MHz. Pipeline registers in each PE
   would help timing closure.

4. **Depthwise convolution:** hps_accel has a Mode0/Mode1 split. For
   depthwise conv, falling back to the single SIMD MAC CFU (Part 1)
   may be simpler than adding a depthwise mode to the systolic array.

5. **Activation bank sizing:** With 4 × 2KiB banks (8 KiB total), how
   many spatial rows of MobileNet can we buffer? For the largest layer
   (96×96×8 = 72 KiB), we'd need to tile to ~10 rows at a time.

---

## A.9  References

- [CFU-Playground source](https://github.com/google/CFU-Playground) — all hps_accel/mnv2_first code
- [Google TPU v1 paper](https://arxiv.org/abs/1704.04760) — systolic array at datacenter scale
- [Eyeriss](https://eyeriss.mit.edu/) — row-stationary dataflow, energy analysis
- [Eyeriss v2 (2019)](https://arxiv.org/abs/1807.07928) — flexible dataflow, sparse acceleration
- Gowin GW2AR-18C datasheet — DSP, BSRAM, LUT specifications
- Gowin UG289E — pDSP block user guide (MULT9X9, MULT18X18, MULTALU)
- [LiteX CFU integration](https://github.com/enjoy-digital/litex/blob/master/litex/soc/cores/cpu/vexriscv/core.py) — add_cfu() method
