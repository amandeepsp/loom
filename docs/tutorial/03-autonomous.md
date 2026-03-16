# Part 3 — Autonomous Compute: Taking the CPU Off the Hot Path

> **Series:** [00-overview](00-overview.md) → [01-mac](01-mac.md) → [02-vertical-slice](02-vertical-slice.md) → **[03-autonomous](03-autonomous.md)** → [04-tinygrad](04-tinygrad.md) → [05-scaling](05-scaling.md)
> **Deep Dive:** [Prior Art & Architecture Decisions](appendix-prior-art.md)

Part 2 proved the full stack works. It also revealed the bottleneck: the
CPU (or UART) sits on the data path during every computation. This part
fixes that by moving data into on-chip SRAM and letting the hardware
compute autonomously.

This is where the real performance comes from — not bigger arrays, not
higher clocks, but **getting the CPU out of the way**.

---

## 3.1  The Problem You Measured

From Part 2, you have timing numbers. Even ignoring UART, the CPU-driven
inner loop has poor hardware utilization:

```
  CPU as data pump (CSR writes):
  ├── Write operands:  ~3 cycles per CSR write × 2 writes = 6 cycles
  ├── Read result:     ~3 cycles
  ├── HW compute:      1 cycle
  └── Utilization:     1 / 10 ≈ 10%

  CPU as data pump (.insn custom instructions):
  ├── Load operands:   ~1 cycle per instruction
  ├── Read result:     ~1 cycle
  ├── HW compute:      1 cycle
  └── Utilization:     1 / 3 ≈ 33%

  Autonomous BSRAM fetch:
  ├── HW reads from SRAM:  1 cycle (pipelined)
  ├── HW compute:          1 cycle (pipelined)
  └── Utilization:         ~67–90%
```

The pattern is clear: **every cycle the CPU spends feeding data is a cycle
the compute hardware sits idle.**

---

## 3.2  The CFU-Playground Design Spectrum

Google's CFU-Playground has four tiers of accelerator. Each teaches a
different lesson:

```
  Tier 1: Scalar MAC               Tier 2: SIMD + HW Requant
  (proj_accel_1)                    (avg_pdti8)
  ─────────────────                 ─────────────────────────
  • 1 MAC/cycle                     • 4 MACs/cycle
  • CPU drives ALL                  • CPU drives outer loop
  • ~100 lines of HDL               • ~200 lines
                                    • KEY: adds SRDHM + RDBPOT
                                      in hardware (saves ~55
                                      CPU cycles per output)

  Tier 3: Autonomous Loop           Tier 4: Systolic Array
  (mnv2_first)                      (hps_accel)
  ─────────────────────             ─────────────────────────
  • Still 4 MACs/cycle              • 32 MACs/cycle
  • But CPU is FREE                 • LRAM activation banks
  • Filter BSRAM stores             • Full PostProcess pipeline
  • Autonomous sequencer            • CPU: config + poll
  • HW inner loop                   • ~2000+ lines
  • Output FIFO
  • ~800 lines

  ┌──────────────────────────────────────────────────────┐
  │  The jump from Tier 2 to Tier 3 is where most of     │
  │  the architectural benefit lives.                     │
  │                                                       │
  │  Tier 4 (systolic array) is a throughput multiplier   │
  │  on top of an already-efficient design.               │
  │  Don't skip Tier 3 to jump to Tier 4.                │
  └──────────────────────────────────────────────────────┘
```

**Our tutorial mapping:**
- Part 1 = Tier 1–2 (MAC + inline asm)
- **Part 3 = Tier 3** (autonomous loop — this part)
- Part 5 = Tier 4 (systolic array)

---

## 3.3  Phase A: Hardware Requantization

Start here. It's the highest ROI optimization and it's independent of
the data delivery mechanism.

### The Requantization Math

After a convolution's MAC loop, the INT32 accumulator must be converted
back to INT8 for the next layer:

```
  acc += bias                         // per-channel, INT32
  acc = SRDHM(acc, multiplier)        // SaturatingRoundingDoubleHighMul
  acc = RDBPOT(acc, shift)            // RoundingDivideByPowerOfTwo
  acc += output_offset                // per-layer
  acc = clamp(acc, -128, 127)         // saturate to INT8
```

**SRDHM** (Saturating Rounding Double High Mul): multiply two 32-bit
values, take the upper 32 bits of the 64-bit product, with correct
rounding and saturation. This is the expensive part.

**RDBPOT** (Rounding Divide By Power Of Two): right-shift with correct
rounding. Cheaper — it's a mux tree over shift amounts.

### Why This Matters

```
  Software requantization per output element on RV32IM:

  int64 product = (int64)acc × multiplier;     // 4–8 cycles (no HW mul64)
  int32 high = product >> 31;                  // 1 cycle
  high += nudge;                                // 1 cycle
  result = high >> shift;                       // 1 cycle
  result += offset;                             // 1 cycle
  result = clamp(result, -128, 127);            // 2 cycles
                                                ─────────
                                                ~10–15 cycles per element

  Hardware requantization: 0 CPU cycles (pipelined, 1 output/cycle)
```

For a layer producing 2304 output elements (48×48), that's ~30,000 CPU
cycles saved.

### Implementation

**🤔 Exercise:** Implement SRDHM in software first (Python or Zig).
Understand exactly what "saturating rounding double high mul" means:

1. Compute `(int64)a × (int64)b` → 64-bit product
2. Round: add `1 << 30` (the rounding nudge)
3. Extract upper 32 bits: `product >> 31`
4. Handle saturation: if both inputs are `INT32_MIN`, the result saturates

*Why is this expensive on RV32IM?* The ISA has no 64-bit multiply. You
need multiple 32×32 multiplies and manual carry propagation.

**🤔 Exercise:** Add SRDHM and RDBPOT as CFU instructions:
- SRDHM → funct3 = 1 in `hardware/cfu.py`, new inline fn in `cfu.zig`
- RDBPOT → funct3 = 2
- Update `OpType` in `link.zig`

Resource cost: 2–4 DSPs for the 32×32 multiply, ~150 LUTs. Still leaves
28+ DSPs free.

---

## 3.4  Phase B: BSRAM Filter Stores

Pre-load weights into on-chip BSRAM. The hardware reads them
autonomously during compute — no CPU involvement.

### The Idea

```
  Before (CPU as data pump):           After (BSRAM filter store):

  CPU ──► CSR write ──► MAC            BSRAM ──► MAC
  CPU ──► CSR write ──► MAC            BSRAM ──► MAC
  CPU ──► CSR write ──► MAC            BSRAM ──► MAC
  ...                                  ...
  (CPU busy every cycle)               (CPU free — hardware reads BSRAM)
```

The CPU loads weights ONCE into BSRAM before starting compute. The
hardware reads them in a cyclic pattern using an address counter that
wraps around.

### The first/last Pattern

Google's hps_accel uses `first` and `last` signals instead of explicit
K-dimension tiling:

```
  For input_depth=64 (K=64), with 4 MACs per block:
  The filter store has 64/4 = 16 entries.

  Cycle:  0     1     2    ...   15    16    17   ...
  first:  1     0     0    ...    0     1     0   ...
  last:   0     0     0    ...    1     0     0   ...

  On first=1: accumulator resets to 0
  On last=1:  accumulator value is latched to output register
  Between:    accumulator keeps accumulating
```

The hardware processes the ENTIRE K dimension in one pass. No tiling, no
partial-sum management, no CPU intervention.

**🤔 Exercise:** How many BSRAMs do you need for filter stores? Each
BSRAM is 18 Kbit = 2 KiB = 2048 bytes. For MobileNet-v2 0.25:
- Largest pointwise layer: 32 output channels × 16 input channels = 512
  bytes. Fits in 1 BSRAM.
- With 2 BSRAMs (4 KiB) you can double-buffer: load the next layer's
  weights while computing the current layer.

---

## 3.5  Phase C: Output FIFO

Instead of the CPU reading individual results via CSR, the hardware
packs INT8 results into a FIFO. The CPU drains it at leisure.

```
  Without FIFO:                    With FIFO:

  HW: compute ─► CPU: read         HW: compute ─► FIFO ─► CPU: drain
  HW: wait...    CPU: process      HW: compute ─► FIFO     (later)
  HW: compute ─► CPU: read         HW: compute ─► FIFO
  HW: wait...    CPU: process      HW: compute ─► FIFO

  Array stalls every output.       Array runs continuously.
  CPU and HW take turns.           CPU and HW decouple.
```

**🤔 Exercise:** How deep should the FIFO be? One BSRAM (2 KiB = 512
32-bit words). If the hardware produces one INT8 output per cycle and
packs 4 per word, that's 2048 outputs before it fills. Is the CPU fast
enough to drain periodically? (At 1 read per cycle, the CPU drains 4
outputs per cycle — faster than production. So any depth works as long
as the CPU doesn't stall for too long.)

---

## 3.6  Phase D: The Autonomous Sequencer

The final piece: a hardware FSM that iterates over all
`input_depth × output_channels` without CPU involvement.

```
  Before (Part 2):

  firmware main loop:
    for each output_channel:
        reset accumulator
        for each input_depth chunk:
            read input from UART
            read weight from UART
            call cfu.mac4()
        send result over UART

  After (Part 3):

  firmware main loop:
    receive weights → load into BSRAM
    receive activations → load into BSRAM
    write config: input_depth, output_channels
    write START
    wait for DONE
    drain output FIFO → send over UART
```

The inner loop moves from firmware to hardware. The firmware only does
setup and teardown.

**🤔 Exercise:** Draw the FSM states for the autonomous sequencer:

```
  IDLE ─── START ───► LOAD_WEIGHT
                          │
                     COMPUTE_ROW ◄─── (for each output channel)
                          │
                     COMPUTE_ELEMENT ◄── (for each K chunk)
                          │
                     OUTPUT_ELEMENT ──► PostProcess → FIFO
                          │
                     (next output channel)
                          │
                     DONE ───► IDLE
```

*What configuration does the CPU need to provide?* At minimum:
- `input_depth` (K dimension)
- `output_channels` (N dimension)
- Filter store base address (implicit — start of BSRAM)
- Activation base address (implicit)
- Requantization parameters (loaded into param BSRAM)

---

## 3.7  What Changes in the Firmware

The firmware's job simplifies dramatically:

```
  Part 2 firmware:                   Part 3 firmware:
  ─────────────────                  ─────────────────
  receive data byte by byte          receive data in bulk
  loop calling cfu.mac4()            load BSRAM (DMA-like burst write)
  track accumulator in software      configure sequencer registers
  send result                        write START
                                     wait for DONE
                                     drain output FIFO
                                     send results
```

The protocol changes too. Instead of streaming operands with the request,
the host sends weight and activation blobs that get loaded into BSRAM.
Then a short "execute" command triggers the sequencer. Then the host
reads back the output FIFO.

**🤔 Exercise:** Redesign the `link.zig` OpType enum for Part 3:

```
  OpType:
    load_weights = 0x01    // bulk-load weight data into filter BSRAM
    load_acts    = 0x02    // bulk-load activation data into act BSRAM
    load_params  = 0x03    // load requant params into param BSRAM
    execute      = 0x04    // start autonomous compute
    read_output  = 0x05    // drain output FIFO
```

*How does this change the host-side Python?*

---

## 3.8  BSRAM Budget

Where does everything fit?

```
  BSRAM allocation (16–18 free BSRAMs):

  ┌────────────────────────────────────────────────┐
  │ Usage                BSRAMs    Capacity         │
  ├────────────────────────────────────────────────┤
  │ Activation bank 0      1      2,048 bytes      │
  │ Activation bank 1      1      2,048 bytes      │
  │ Activation bank 2      1      2,048 bytes      │
  │ Activation bank 3      1      2,048 bytes      │
  │ Filter store 0         1      2,048 bytes      │
  │ Filter store 1         1      2,048 bytes      │
  │ Requant params         2      ~3,456 bytes     │
  │ Output FIFO            1      2,048 bytes      │
  ├────────────────────────────────────────────────┤
  │ TOTAL                  9      ~18 KiB          │
  │ Remaining            7–9      ~14–18 KiB free  │
  └────────────────────────────────────────────────┘
```

Activation banks: 4 × 2 KiB = 8 KiB total. MobileNet-v2 0.25's largest
activation tensor is ~72 KiB (96×96×8). You need spatial tiling — process
a few rows at a time. 8 KiB is enough for ~10 rows.

**🤔 Exercise:** For a 1×1 conv layer with 8 input channels and 4×4
spatial size: how many activation bytes? 4 × 4 × 8 = 128 bytes. *Does
that fit in one activation bank?* Yes, easily. Start with layers that fit
entirely, then add tiling for larger layers.

---

## 3.9  Putting It All Together

The autonomous compute architecture:

```
  ┌─────────────────────────────────────────────────────────┐
  │  BEFORE compute (slow path — CPU loads data):           │
  │                                                         │
  │  CPU ──UART──► filter BSRAM    (write weights, once)    │
  │  CPU ──UART──► param BSRAM     (write requant params)   │
  │  CPU ──UART──► act BSRAM       (write activations)      │
  │  CPU ──CSR──► config regs      (depth, channels)        │
  │  CPU ──CSR──► START                                     │
  └─────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────┐
  │  DURING compute (hot path — CPU NOT involved):          │
  │                                                         │
  │  filter BSRAM ──► MAC (cyclic read, autonomous)         │
  │  act BSRAM ───► MAC (sequential read, autonomous)       │
  │  param BSRAM ──► PostProcess (per-channel, autonomous)  │
  │                      │                                  │
  │                      ▼                                  │
  │                  output FIFO                            │
  └─────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────┐
  │  AFTER compute (slow path — CPU reads results):         │
  │                                                         │
  │  CPU ◄── output FIFO  (drain packed INT8 results)       │
  │  CPU ──UART──► host   (send results back)               │
  └─────────────────────────────────────────────────────────┘
```

**🤔 Exercise:** Compare the cycle count for a 1×1 conv (8 in, 8 out,
4×4 spatial) in Part 2 vs Part 3:

Part 2: 16 spatial positions × 8 output channels × 2 MAC invocations ×
~10 cycles (CSR overhead) = ~2560 cycles, plus UART time.

Part 3: Load weights (~64 bytes into BSRAM via CSR writes = ~192 cycles).
Load activations (~128 bytes = ~384 cycles). START. Compute: 16 × 8 × 2
= 256 MAC cycles (pipelined). Drain FIFO: ~32 cycles. Total: ~860 cycles
of CPU time, and the compute runs concurrently.

*Where did the savings come from?* The CPU isn't waiting for each MAC to
complete. It does setup, triggers hardware, and reads results.

---

## 3.10  Checkpoint

- [ ] I understand the 4-tier design spectrum and where each part fits
- [ ] I can implement SRDHM in software as a reference
- [ ] I have a BSRAM allocation plan
- [ ] I understand the first/last accumulator control pattern
- [ ] I can sketch the autonomous sequencer FSM
- [ ] I can explain why CPU-off-hot-path matters more than array size
- [ ] I've identified which operations need new funct3 slots

---

**Previous:** [Part 2 — Vertical Slice](02-vertical-slice.md)
**Next:** [Part 4 — Selective Lowering from TinyGrad](04-tinygrad.md)
