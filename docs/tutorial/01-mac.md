# Part 1 — The MAC: Your First Custom Instruction

> **Series:** [00-overview](00-overview.md) → **[01-mac](01-mac.md)** → [02-vertical-slice](02-vertical-slice.md) → [03-autonomous](03-autonomous.md) → [04-tinygrad](04-tinygrad.md) → [05-scaling](05-scaling.md)

This part is **already built**. The hardware exists in `hardware/mac.py`
and the firmware wrapper exists in `firmware/src/cfu.zig`. This document
is a condensed recap of what was built and why, with exercises to deepen
your understanding before moving on.

---

## 1.1  What the MAC Does

Quantized neural networks represent activations and weights as INT8 values.
The inner loop of every convolution is a multiply-accumulate:

```
  acc += (input[i] + offset) × weight[i]
```

The `SimdMac4` does **four** of these per clock cycle, packed into 32-bit
registers:

```
              byte 3      byte 2      byte 1      byte 0
         ┌───────────┬───────────┬───────────┬───────────┐
  in0  = │ input[3]  │ input[2]  │ input[1]  │ input[0]  │   (rs1)
         └───────────┴───────────┴───────────┴───────────┘
         ┌───────────┬───────────┬───────────┬───────────┐
  in1  = │weight[3]  │weight[2]  │weight[1]  │weight[0]  │   (rs2)
         └───────────┴───────────┴───────────┴───────────┘

  output = Σ (in0[i] + offset) × in1[i]    for i = 0..3

  Accumulator += output
```

**🤔 Why the offset?** INT8 quantization maps floating-point values to
[-128, 127]. The formula is `real_value = scale × (int8_value - zero_point)`.
TFLite uses `zero_point = -128`, so the firmware adds 128 before
multiplying. This converts from signed INT8 to the unsigned offset domain:

```
  int8 value:   -128  -127  ...   0   ...  126   127
  after +128:      0     1  ... 128   ...  254   255
```

**🤔 Exercise:** The offset is hardcoded to 128 in `hardware/mac.py`. Real
models have *per-layer* zero-points that vary. If a layer's zero-point is
-135 instead of -128, what happens? *How would you make the offset
configurable?* (Look at `self.input_offset = Signal(32, reset=128)` — the
signal is there, but who sets it?)

---

## 1.2  The R-Type Custom Instruction

The MAC is invoked via a custom RISC-V instruction using the `CUSTOM_0`
opcode space (0x0B). The encoding is a standard R-type:

```
  31        25 24    20 19    15 14  12 11     7 6       0
  ┌──────────┬────────┬────────┬──────┬────────┬─────────┐
  │  funct7  │  rs2   │  rs1   │funct3│   rd   │ opcode  │
  │  7 bits  │ 5 bits │ 5 bits │3 bits│ 5 bits │ 7 bits  │
  └──────────┴────────┴────────┴──────┴────────┴─────────┘
  │ 0000000  │weights │inputs  │ 000  │ result │ 0001011 │
  └──────────┴────────┴────────┴──────┴────────┴─────────┘
                                                 CUSTOM_0
```

In Zig (`firmware/src/cfu.zig`), this is encoded via inline assembly:

```
  .insn r CUSTOM_0, %[f3], %[f7], %[rd], %[rs1], %[rs2]
```

The CFU bus in `hardware/cfu.py` dispatches on **funct3** (3 bits → 8
instruction slots). Each instruction module receives **funct7** (7 bits)
as a sub-opcode. Currently only slot 0 is used.

**🤔 Exercise:** Decode the MAC instruction by hand. If `funct7=0x00`,
`funct3=0x0`, `rs1=a1`, `rs2=a2`, `rd=a0`, what are all 32 bits? Write
them out in binary. Verify it matches `0x0B` in bits [6:0].

**🤔 Exercise:** The Cfu class has 8 funct3 slots but only slot 0 is used.
*What would you put in the other 7?* Think about the operations that come
after a MAC: requantization (SRDHM, RDBPOT), accumulator reset, offset
configuration. Each could be a separate instruction. We'll use this in
Part 3.

---

## 1.3  The CFU Bus Protocol

The CPU and CFU communicate via a valid/ready handshake:

```
  CPU → CFU:  cmd_valid, cmd_function_id[9:0], cmd_inputs_0, cmd_inputs_1
  CFU → CPU:  cmd_ready, rsp_valid, rsp_outputs_0
  CPU → CFU:  rsp_ready
```

A transfer occurs when `valid & ready` are both high on the same clock
edge. The CFU uses a 3-state FSM:

```
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
  └──────────┘          ▼            ┌───────┘
                ┌──────────────┐      │
                │WAIT_INSTRUCT.│──────┘
                │ (multi-cycle │  done
                │  instruction)│
                └──────────────┘
```

**🤔 Exercise:** Why does the FSM have a `WAIT_TRANSFER` state? What would
happen if the CFU produced a result but the CPU wasn't ready to read it?
(Look at the `stored_output` register in `hardware/cfu.py`.)

**🤔 Exercise:** The `cfu.zig` function `mac4(acc, a, b)` returns
`acc + cfu_call(...)`. The accumulation happens in *software*. But the
hardware `SimdMac4` *also* has an internal accumulator. *When would you use
the software accumulator? When the hardware one?* (Hint: the hardware
accumulator persists across calls — useful for long reductions. The
software one gives you more control.)

---

## 1.4  Checkpoint

Before moving to Part 2, verify:

- [ ] I understand why the MAC adds 128 (INT8 zero-point convention)
- [ ] I can decode the `.insn r CUSTOM_0, ...` encoding by hand
- [ ] I understand the 3-state CFU FSM and why back-pressure matters
- [ ] I know where the 8 instruction slots are and what funct3 vs funct7 do
- [ ] The MAC hardware passes simulation tests (`hardware/test_mac.py`)
- [ ] I've read `firmware/src/cfu.zig` and understand the inline assembly

---

**Previous:** [Part 0 — Overview](00-overview.md)
**Next:** [Part 2 — Vertical Slice: Host to Hardware and Back](02-vertical-slice.md)
