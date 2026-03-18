# Sequencer FSM Design

The autonomous sequencer replaces the CPU's inner loop. Instead of the CPU
feeding operands one at a time, it writes configuration registers, asserts
START, and waits for DONE. The hardware iterates over the entire computation
without further CPU involvement.

This is the architectural equivalent of a GPU's instruction fetch + warp
scheduler — but hardcoded for a single operation (conv/matmul). TPUs do the
same thing: the control logic is a fixed dataflow, not a programmable
instruction stream.

---

## State Diagram

```
              ┌────────────────────────────────────────────────┐
              │                                                │
              ▼                                                │
         ┌──────────┐                                          │
         │          │   START signal                           │
         │   IDLE   │──────────────────┐                       │
         │          │                  │                       │
         │ config   │                  ▼                       │
         │ regs     │           ┌──────────┐                   │
         │ writable │           │          │                   │
         └──────────┘           │ LOADING  │ (optional)        │
                                │          │                   │
                                │ DMA wts/ │                   │
                                │ acts to  │                   │
                                │ BSRAM    │                   │
                                └────┬─────┘                   │
                                     │ load complete           │
                                     ▼                         │
                                ┌──────────┐                   │
                          ┌────►│          │                   │
                          │     │ RUNNING  │                   │
                          │     │          │                   │
                          │     │ triple   │                   │
                          │     │ nested   │                   │
                          │     │ loop     │                   │
                          │     └────┬─────┘                   │
                          │          │ all iterations done     │
                          │          ▼                         │
                          │     ┌──────────┐                   │
                          │     │          │                   │
                          │     │  DONE    │───────────────────┘
                          │     │          │  CPU acknowledges
                          │     │ assert   │
                          │     │ done sig │
                          │     └──────────┘
                          │          │
                          └──────────┘
                           (loop back for
                            next spatial tile
                            if tiling)
```

---

## State Descriptions

### IDLE
- Waiting for the CPU to assert START.
- All configuration registers are writable (N, K, spatial_size, etc.).
- Output FIFO is empty (or has been drained from previous run).
- The CPU uses this time to load weights/activations into BSRAM if needed.

### LOADING (optional)
- DMA weights and/or activations into BSRAM.
- Only needed if weights weren't pre-loaded during the previous IDLE phase.
- In the simplest design, the CPU loads BSRAM before asserting START, so
  this state can be skipped entirely.

### RUNNING
- The sequencer executes a triple-nested loop:

```
  for s in range(spatial_size):           # spatial positions
      for n in range(N):                  # output channels
          accumulator = 0
          for k in range(K // 4):         # input channels, 4 at a time (SIMD)
              act = activation_bsram[s * (K//4) + k]      # 4 x INT8
              wt  = filter_bsram[n * (K//4) + k]          # 4 x INT8
              accumulator += mac4(act, wt)                 # 4 MACs
          # After K loop completes for this (s, n):
          result = srdhm(accumulator, multiplier[n])
          result = rdbpot(result, shift[n])
          result = clamp(result + output_offset, -128, 127)
          output_fifo.push(result)
```

- **first/last signals** (from hps_accel pattern): instead of explicit loop
  counters visible to the MAC, the sequencer emits `first=1` on the first
  K iteration (resets accumulator) and `last=1` on the final K iteration
  (triggers requantization + FIFO write). This simplifies the MAC datapath.

- Filter store reads use a cyclic address counter that wraps at K/4 and
  advances the base by K/4 for each new output channel.

- Activation store reads use a sequential counter that resets at the start
  of each output channel (same activations reused for all N channels at
  a given spatial position).

### DONE
- Assert the `done` signal so the CPU knows computation is complete.
- The output FIFO now contains `spatial_size * N` INT8 results, packed
  4 per 32-bit word.
- CPU drains the FIFO at its own pace, then can write new config for the
  next layer.

---

## Configuration Registers

Written by the CPU before asserting START:

| Register | Bits | Description |
|---|---|---|
| `N` | 16 | Number of output channels |
| `K` | 16 | Number of input channels (must be multiple of 4) |
| `spatial_size` | 16 | Number of spatial positions (height * width for 1x1 conv) |
| `output_offset` | 9 | Per-layer output zero point (signed) |
| `quant_mult_addr` | 12 | Base address of per-channel multipliers in param BSRAM |
| `quant_shift_addr` | 12 | Base address of per-channel shifts in param BSRAM |

The CPU writes these via CFU custom instructions (funct7 selects the register,
in0 carries the value). This takes ~6 instruction cycles — negligible compared
to the compute time.

---

## Timing Analysis

For a layer with S spatial positions, N output channels, K input channels:

```
  MAC cycles:         S * N * (K / 4)
  Requant cycles:     S * N * ~3         (SRDHM + RDBPOT + clamp, pipelined)
  Total compute:      S * N * (K/4 + 3)

  Example: 1x1 conv, 48x48 spatial, 16 in channels, 8 out channels
    S = 2304, N = 8, K = 16
    MAC:    2304 * 8 * 4 = 73,728 cycles
    Requant: 2304 * 8 * 3 = 55,296 cycles (overlaps with MAC if pipelined)
    Total:  ~73,728 cycles (requant hidden behind MAC pipeline)
    At 27 MHz: ~2.73 ms
```

If the requantization pipeline is fully pipelined (throughput = 1 output/cycle
after fill), it overlaps with the next spatial position's MAC iterations,
so the total time is dominated by MAC cycles alone.

---

## Comparison to GPU SM

| Concept | GPU Streaming Multiprocessor | Your Sequencer |
|---|---|---|
| Instruction fetch | Fetches from instruction cache | Hardcoded FSM — no instruction memory |
| Warp scheduler | Selects next warp to execute | Fixed triple-nested loop order |
| Register file | General purpose, 32K x 32b | Purpose-built accumulators + params |
| Shared memory | Software-managed SRAM | BSRAM with fixed allocation |
| Compute units | CUDA cores (flexible ALU) | SIMD MAC + requant pipeline |
| Flexibility | Runs any kernel | Runs exactly one operation (conv/matmul) |

The tradeoff is clear: your sequencer is far simpler (tens of states vs
thousands of transistors for fetch/decode/schedule), but it can only do one
thing. This is exactly the TPU tradeoff — Google's TPU v1 has no instruction
fetch either, just a sequencer that drives a systolic array.

For the workloads we care about (INT8 convolutions in MobileNet), this fixed
function is all we need. The CPU handles everything else.
