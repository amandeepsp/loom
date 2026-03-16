# Custom Function Unit (CFU) Guide

The CFU is a hardware accelerator wired directly into the VexRiscv CPU pipeline via the CfuPlugin. It executes custom RISC-V instructions in the `CUSTOM_0` opcode space.

## 5-Minute Understanding

The CFU receives two 32-bit register values and a function ID from the CPU, computes a result, and writes it back to a destination register — all as a single RISC-V instruction. The firmware calls it with inline assembly:

```zig
// funct3=0, funct7=1: reset accumulator + compute
result = .insn r CUSTOM_0, 0, 1, rd, rs1, rs2
```

The current CFU implements one instruction: **SimdMac4** — a 4-lane byte-wise multiply-accumulate used in quantised neural network inference.

### Instruction encoding

R-type instruction in the `CUSTOM_0` slot (opcode `0x0B`):

```
31        25 24    20 19    15 14  12 11     7 6       0
┌──────────┬────────┬────────┬──────┬────────┬─────────┐
│  funct7  │  rs2   │  rs1   │funct3│   rd   │ opcode  │
│  7 bits  │ 5 bits │ 5 bits │3 bits│ 5 bits │ 7 bits  │
└──────────┴────────┴────────┴──────┴────────┴─────────┘
                                                0001011
                                                CUSTOM_0
```

- **funct3** (3 bits): selects one of 8 instruction slots
- **funct7** (7 bits): per-instruction control flags
- **rs1, rs2**: 32-bit input operands
- **rd**: 32-bit result

### Hardware handshake

The CPU and CFU communicate via a valid/ready protocol:

```
CPU                          CFU
 │  cmd_valid=1               │
 │  function_id, in0, in1 ───►│
 │                             │ compute...
 │◄─── rsp_valid=1, output ───│
 │  rsp_ready=1                │
```

## 50-Minute Understanding

### Architecture (`hardware/cfu.py`)

The `Cfu` class is the top-level Amaranth `Elaboratable`. It manages:

1. **Instruction dispatch**: routes `funct3` (3 bits from the function ID) to one of 8 instruction slots
2. **Handshake FSM**: coordinates the valid/ready protocol between CPU and instruction logic
3. **Instruction lifecycle**: wires `start`, `done`, inputs, and outputs for each instruction

#### Signal interface

| Signal                         | Dir | Width | Purpose                        |
|-------------------------------|-----|-------|--------------------------------|
| `cmd_valid`                    | in  | 1     | CPU has a command ready        |
| `cmd_ready`                    | out | 1     | CFU can accept a command       |
| `cmd_payload_function_id`      | in  | 10    | funct3 (bits 2:0) + funct7 (bits 9:3) |
| `cmd_payload_inputs_0`         | in  | 32    | rs1 value                      |
| `cmd_payload_inputs_1`         | in  | 32    | rs2 value                      |
| `rsp_valid`                    | out | 1     | CFU has a result ready         |
| `rsp_ready`                    | in  | 1     | CPU can accept the result      |
| `rsp_payload_outputs_0`        | out | 32    | rd value (result)              |
| `reset`                        | in  | 1     | Reset signal                   |

#### FSM states

```
┌──────────┐   cmd_valid & done      ┌──────────────┐
│          │   & rsp_ready           │              │
│ WAIT_CMD │◄───────────────────────│ WAIT_TRANSFER│
│          │                         │ (CPU not     │
│ cmd_ready│   cmd_valid & done      │  ready yet)  │
│ = 1      │   & !rsp_ready         │ rsp_valid=1  │
│          │────────────────────────►│              │
│          │                         └──────┬───────┘
│          │   cmd_valid & !done            │ rsp_ready
│          │──────────┐                     │
└──────────┘          │             ┌───────┘
                       ▼             │
              ┌──────────────┐       │
              │              │       │
              │WAIT_INSTRUCT.│───────┘
              │ (multi-cycle │  done & rsp_ready
              │  instruction)│
              │ cmd_ready=0  │──────┐
              │              │      │ done & !rsp_ready
              └──────────────┘      │
                       ▲            │
                       │            ▼
                       │     ┌──────────────┐
                       └─────│ WAIT_TRANSFER│
                             └──────────────┘
```

- **WAIT_CMD**: CFU is idle, `cmd_ready=1`. On `cmd_valid`, dispatches `funct3` to the matching instruction, asserts `start`.
- **WAIT_INSTRUCTION**: multi-cycle instruction is executing. Waits for `done`.
- **WAIT_TRANSFER**: result is ready (`rsp_valid=1`) but CPU hasn't asserted `rsp_ready` yet. Buffers the output in `stored_output`.

Single-cycle instructions (like SimdMac4) take the fast path: WAIT_CMD → back to WAIT_CMD in one cycle when `rsp_ready` is already high.

#### Adding a new instruction

1. Create a class that extends `Instruction` (`hardware/cfu.py`):

```python
class MyInstruction(Instruction):
    def elaborate(self, platform):
        m = Module()
        # Use self.in0, self.in1, self.funct7 as inputs
        m.d.comb += self.output.eq(...)   # set result
        m.d.comb += self.done.eq(1)       # single-cycle: always done
        return m
```

2. Register it in your `Cfu` subclass at a free `funct3` slot (0–7):

```python
class Top(Cfu):
    def elab_instructions(self, m):
        m.submodules["mac4"] = mac4 = SimdMac4()
        m.submodules["my_new"] = my_new = MyInstruction()
        return {0: mac4, 1: my_new}
```

3. Add a firmware wrapper in `cfu.zig`:

```zig
pub inline fn my_new(a: i32, b: i32) i32 {
    return cfu_op(1, 0, a, b);  // funct3=1
}
```

4. Add an opcode to `link.zig`'s `OpType` and a handler in `dispatch.zig`.

### SimdMac4 (`hardware/mac.py`)

4-lane SIMD multiply-accumulate with input offset:

```
For each byte lane i in [0..3]:
    output = base + Σ (in0[i] + 128) * in1[i]
```

- **`funct7[0] = 0`** (accumulate): `base = accumulator` — adds to running sum
- **`funct7[0] = 1`** (reset): `base = 0` — starts fresh

The `INPUT_OFFSET = 128` matches the zero-point convention for unsigned 8-bit quantised weights in TFLite models. This lets the firmware process 4 multiply-accumulates per instruction instead of one.

The accumulator is updated on the `start` pulse (`m.d.sync`), so it persists across instructions until explicitly reset.

### Testing

Hardware tests use Amaranth's simulator:

```bash
cd hardware
uv run pytest test_mac.py -v
```

End-to-end tests run through the full LiteX simulator:

```bash
uv run python host/client.py --test -v
```

### Verilog generation

```bash
cd hardware
python top.py
# Writes ../top.v
```

This runs Amaranth's `convert()` to produce synthesisable Verilog that LiteX imports as the CFU module. The generated Verilog is committed to the repo for convenience.
