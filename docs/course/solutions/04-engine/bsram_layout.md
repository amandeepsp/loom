# BSRAM Allocation Map

The Gowin GW2AR-18C has 48 BSRAM blocks, each 18 Kbit (2,048 bytes usable
in 8-bit-wide mode). After the SoC (VexRiscv + peripherals) claims some,
approximately 16 remain free for the accelerator.

This layout is based on the hps_accel architecture adapted for Gowin's BSRAM
instead of Lattice's LRAM.

---

## Allocation Table

| BSRAM Index | Purpose | Size | Access Pattern |
|---|---|---|---|
| 0 | Activation bank 0 | 2 KiB | Phase-rotated read, port 0 |
| 1 | Activation bank 1 | 2 KiB | Phase-rotated read, port 1 |
| 2 | Activation bank 2 | 2 KiB | Phase-rotated read, port 2 |
| 3 | Activation bank 3 | 2 KiB | Phase-rotated read, port 3 |
| 4 | Filter store 0 | 2 KiB | Cyclic read: wraps at K/4 boundary |
| 5 | Filter store 1 | 2 KiB | Cyclic read: same as store 0, for column 2 |
| 6 | Param store (multipliers) | 2 KiB | Sequential read, per output channel |
| 7 | Param store (shifts+bias) | 2 KiB | Sequential read, per output channel |
| 8 | Output FIFO | 2 KiB | Write by sequencer, drain by CPU |
| 9-15 | Free | 14 KiB | Available for future use |

**Total used: 9 BSRAMs (18 KiB) — 56% of free budget.**

---

## Activation Banks (BSRAM 0-3)

**Total capacity:** 4 x 2 KiB = 8 KiB

Activations are interleaved across 4 banks with phase-rotated addressing
to ensure conflict-free parallel reads. A `RamMux` module rotates which
bank each read port maps to on each cycle:

```
  Cycle:    0         1         2         3         4    ...
  Port 0: Bank 0    Bank 1    Bank 2    Bank 3    Bank 0
  Port 1:   -       Bank 0    Bank 1    Bank 2    Bank 3
  Port 2:   -         -       Bank 0    Bank 1    Bank 2
  Port 3:   -         -         -       Bank 0    Bank 1

  By cycle 3, all 4 ports are active on different banks.
  Sustained throughput: 4 x 8 bits = 32 bits/cycle = 4 bytes/cycle
  (or 4 x 32 bits if using word-wide access = 16 bytes/cycle)
```

**Sizing check for MobileNet-v2 0.25:**
- Largest activation tensor: 96 x 96 x 8 = 72 KiB (first depthwise layer)
- 8 KiB fits ~10 rows: 10 x 96 x 8 = 7,680 bytes
- Requires spatial tiling: process 10 rows at a time, reload between tiles
- Smallest layers (e.g., 6 x 6 x 32 = 1,152 bytes) fit entirely

**Write pattern:** CPU writes activations sequentially during IDLE phase.
Interleaving is handled by address bits: `bank = addr[1:0]`, `bank_addr = addr >> 2`.

---

## Filter Stores (BSRAM 4-5)

**Total capacity:** 2 x 2 KiB = 4 KiB

Weights are pre-loaded by the CPU before compute starts. During RUNNING,
the sequencer reads them in a cyclic pattern:

```
  For N output channels, K input channels, 4 bytes per SIMD word:
  Filter store layout: [channel_0_word_0, channel_0_word_1, ..., channel_N-1_word_K/4-1]

  Read address = (output_channel * (K/4) + k_counter) % store_depth

  The cyclic reader wraps around when processing multiple spatial positions
  with the same output channel — same weights reused for every spatial position.
```

**Why 2 stores?** Double-buffering. While the sequencer reads from store 0,
the CPU can write the next layer's weights into store 1. On layer switch,
the roles swap. This hides the weight loading latency.

**Sizing check:** MobileNet-v2 0.25 largest pointwise layer: 32 output channels
x 16 input channels = 512 bytes. Fits in 1 store with room to spare. Even
the largest layer (96 x 32 = 3,072 bytes) fits in 2 stores.

---

## Parameter Stores (BSRAM 6-7)

**Total capacity:** 2 x 2 KiB = 4 KiB (~3,456 bytes used)

Per-channel requantization parameters, read during the post-processing
pipeline after each output channel's MAC loop completes:

```
  BSRAM 6 — Multipliers (32 bits each):
  [mult_ch0, mult_ch1, ..., mult_chN-1]
  Max channels: 2048 / 4 = 512 channels per store

  BSRAM 7 — Shifts + Bias:
  [shift_ch0(8b) + bias_ch0(24b), shift_ch1 + bias_ch1, ...]
  Packed: 4 bits shift + 18 bits bias = 22 bits, padded to 32 bits per entry
```

The sequencer reads one entry per output channel, which happens once every
K/4 MAC cycles — bandwidth is not a concern here.

---

## Output FIFO (BSRAM 8)

**Capacity:** 2 KiB = 512 x 32-bit words = 2048 INT8 outputs (packed 4 per word)

Written by the sequencer after requantization, drained by the CPU after DONE.

```
  Write side (hardware):
    On each completed output: pack 4 INT8 values into one 32-bit word
    Write pointer increments by 1 word per 4 outputs
    If FIFO full: stall the sequencer (backpressure)

  Read side (CPU):
    CPU reads via CFU instruction: output = cfu_op(READ_FIFO, 0)
    Each read returns 4 packed INT8 values
    Read pointer increments automatically

  Depth check:
    Largest layer output: 48 x 48 x 32 = 73,728 bytes = 36 x FIFO capacity
    → CPU must drain periodically, or tile the output dimension
    Smallest layer: 6 x 6 x 8 = 288 bytes — fits entirely in FIFO
```

---

## Comparison to CUDA Shared Memory

| Concept | CUDA Shared Memory | Your BSRAM Layout |
|---|---|---|
| Declaration | `__shared__ float smem[SIZE]` | Fixed at synthesis time |
| Allocation | Runtime, per-kernel | Static, per-design |
| Capacity | 48-164 KiB per SM | 18 KiB total (9 BSRAMs) |
| Banking | 32 banks, 4 bytes each | 4 banks (activation), 2 banks (filter) |
| Conflict resolution | Bank conflicts stall | Phase rotation eliminates conflicts |
| Programmer control | Explicit indexing | Hardware sequencer auto-indexes |
| Flexibility | Any data layout | Fixed purpose per BSRAM |

The key parallel: in CUDA, you declare `__shared__` arrays and manually manage
what data lives there to maximize data reuse and minimize global memory traffic.
You're doing exactly the same thing — except your "shared memory" allocation
is decided at hardware design time, not at kernel launch time.

This is the fundamental tradeoff of fixed-function vs programmable:
- CUDA: flexible allocation, but programmer must get it right every kernel
- Your design: fixed allocation, but the hardware guarantees optimal access
  patterns for the one operation it supports

In a GPU, bad shared memory usage (bank conflicts, insufficient tiling) leads
to performance cliffs. In your design, the access patterns are baked into
the sequencer FSM — correct by construction, but only for conv/matmul.
