# AGENTS.md

## Project Overview

**accel** — RISC-V Custom Function Unit (CFU) accelerator for ML on FPGA (Tang Nano 20K).
Cross-layer stack: Amaranth (RTL), Zig (firmware + driver), Python (SoC integration).

### Directory Map

- `hardware/` — Amaranth Python RTL: CFU datapath, systolic array, DMA scratchpads, sequencer, epilogue
- `firmware/` — Zig: bare-metal RISC-V (VexRISCV), UART protocol, KIR interpreter, CFU/DMA drivers
- `shared/` — Zig + Python: wire protocol (`protocol.zig`), IR definitions (`ir.zig`, `ir.py`)
- `host/` — Zig: native driver + C API (`libaccel.so`) for serial communication with board
- `tvm/` — Python: TVM Relax patterns, codegen, quantization utils, runtime bridge
- `models/` — Python: int8 MNIST training, static quantization, ONNX export
- `tools/` — Python: GEMM test harness (`test_gemm.py`), TVM sim test (`tvm_sim_test.py`), IR bytecode builder (`ir.py`), LiteX flash utils
- `docs/` — Architecture Decision Records
- `top.v` — **Generated** Verilog (from `hardware/top.py`)

## Essential Commands

### Hardware RTL (Python/Amaranth)

```sh
# Regenerate CFU Verilog from Amaranth (always before hw-build)
just verilog

# Full flow: Verilog → LiteX SoC → bitstream → flash, can take a very long time
just hw-all

# Just build/flash without firmware upload
just hw-build
just hw-flash

# Reset FPGA board
just hw-reset
```

**Required tools on PATH:** `oss-cad-suite/bin` (Yosys, nextpnr, openFPGALoader).
Set `oss_cad_bin` variable in Justfile if not default.

### Firmware (Zig → RISC-V binary)

```sh
# Build firmware (requires prior hw-build to generate build/sipeed_tang_nano_20k/csr.json)
just hw-firmware

# Upload to running board (blocks waiting for response)
just hw-upload

# Upload + auto-release serial port when transfer completes
just hw-upload-once

# End-to-end GEMM test against board with firmware running
just hw-gemm
```

**Build requirement:** `hw-build` must run first (generates CSR register addresses in `csr.json`).
Firmware build via `zig build firmware -Dbuild-dir=build/sipeed_tang_nano_20k` will fail until SoC is built.

### Tests (Python/pytest via uv)

```sh
# Run all hardware tests (Amaranth simulation)
uv run pytest hardware

# Run single test file
uv run pytest hardware/systolic/test_*.py -v

# Sim regression (Verilator + firmware, no board)
just sim-gemm                      # 8×8×8 default
just sim-gemm m=4 k=16 n=4         # custom sizes

# Sim TVM path (MNIST model on Verilator)
just sim-tvm
```

Tests live in: `hardware/systolic/`, `hardware/memory/`, `hardware/epilogue/`.
pytest configured in `pyproject.toml` to include these paths.

## Build Order (Critical)

When making changes:

1. **If RTL changes:** `just verilog` → `just hw-build` (generates `top.v` and CSR addresses)
2. **If firmware changes:** `zig build firmware -Dbuild-dir=build/sipeed_tang_nano_20k` (depends on CSR from step 1)
3. **If hardware tests fail:** `uv run pytest hardware -v` (simulation, no board needed)
4. **If e2e fails:** `just hw-reset && just hw-upload-once && just hw-gemm`

Missing any earlier step causes cryptic build failures (missing csr.json, undefined CSR addresses).

## Configuration Parameters (Justfile Defaults)

These change CFU array size and memory depth; passed to Verilog gen, LiteX, firmware:

```
cfu_rows = "8"           # Systolic rows (default 8, max limited by FPGA)
cfu_cols = "8"           # Systolic columns
cfu_store_depth = "512"  # Scratchpad depth in 32-bit words
cfu_in_width = "8"       # Input data width (8 = int8)
cfu_acc_width = "32"     # Accumulator width
port = "/dev/ttyUSB1"    # Serial port for board
```

When tuning hardware, update Justfile variables — they propagate to both Verilog generation and firmware build.

## IR & Protocol

- `shared/ir.zig`: Instruction format (load-weight, load-act, tile-mma, store, epilogue params, done).
  Program header: magic 0x4B495200 ("KIR\0"), version, tensor count, instruction count.
- `shared/ir.py`: Python IR definitions and `ProgramBuilder` — single source of truth for bytecode format.
- `shared/protocol.zig`: UART framing (CRC, control sequences).
- `host/driver.zig`: Host-side serial communication + CFU instruction dispatch.
- `host/c_api.zig`: C FFI surface (`libaccel.so`) — `accel_open`, `accel_exec`, `accel_write_mem`, etc.
- `firmware/interpreter.zig`: Firmware IR interpreter (executes instructions, drives DMA, sequencer).
- `tools/ir.py`: Legacy IR bytecode builder (used by `tools/test_gemm.py`).

Changing IR requires sync across `shared/ir.zig`, `shared/ir.py`, `firmware/interpreter.zig`, `host/driver.zig`.

## TVM Integration

- `tvm/patterns.py`: Relax DPL patterns for quantized matmul composites (QDQ format from ONNX).
- `tvm/codegen.py`: Lowers partitioned regions to `call_dps_packed`, extracts composite constants.
- `tvm/runtime.py`: `AccelRuntime` class — bridges TVM packed functions to `libaccel.so`, memory layout, KIR generation.
- `tvm/quant_utils.py`: Derives per-channel epilogue params (bias, multiplier, shift) from scale/zero-point.
- `tvm/relax.py`: `lower_pipeline()` — tiling → partitioning → codegen → lambda lift.

The TVM integration is out-of-tree; it imports from `../tvm` (local Apache TVM build).

## Hardware Quirks

### Datapath Flow

```
Serial Input → UART Rx (firmware) → IR Interpreter
  ↓
DMA (fill) → Scratchpad (double-buffered) → Sequencer reads compute bank
  ↓
Systolic Array (output-stationary GEMM) → psum pipeline
  ↓
Epilogue (int32→int8 requantization) → Results readback
```

### Code Generation

- `top.v` is **generated** from `hardware/top.py` by `just verilog`. Never edit `top.v` directly.
- Regenerate before every LiteX build if RTL changes.

### Simulation vs. Hardware

- Simulation tests (pytest) run Amaranth simulation (no board, 100x slower, good for debug).
- E2E tests require real board + running firmware.
- Snapshot workflows: `test_top.py` generates reference traces; compare against live runs.

### CSR Addresses

Firmware reads CSR register addresses from `build/sipeed_tang_nano_20k/csr.json` (generated by LiteX).
Zig build script parses JSON → options → `csr_options` module. If CSR.json missing/stale, firmware build fails.

## Python Environment

- **Dependency manager:** `uv` (faster than pip, respects lock file).
- **Dependencies:** Amaranth, LiteX, LiteX-boards, pytest, pyserial.
- **External dependencies resolved locally:** See `pyproject.toml` `[tool.uv.sources]` — points to adjacent `../litex/` paths.
  If those are missing, clone/symlink them or pin published versions.

Commands always via `uv run`:
```sh
uv run pytest hardware           # not pytest directly
uv run python -m hardware.top    # not python directly
uv run litex_term /dev/ttyUSB1  # not litex_term directly
```

## Zig Build System

- **Firmware target:** riscv32, bare-metal, no float, no atomics.
- **Host driver target:** native (x86_64 Linux).
- **Linker script:** `firmware/linker.ld` (VexRISCV memory map).
- **Build options:** `-Dcfu-rows=N`, `-Dcfu-cols=N`, `-Dcfu-store-depth=D` (must match Verilog + Justfile).
- **Output:** `zig-out/bin/firmware.bin` (objcopy'd ELF → binary for serial boot).

## Common Pitfalls

1. **"csr.json not found"** → Run `just hw-build` before `zig build firmware`.
2. **Stale `top.v`** → Always run `just verilog` after RTL changes.
3. **Parameter mismatch** → If you change `cfu_rows`, update Justfile variables.
4. **Serial port collision** → Only one process can access `/dev/ttyUSB1`; close `litex_term` before `hw-gemm`.
5. **E2E test hangs** → Board may be in bad state; run `just hw-reset && just hw-upload-once` to recover.

## Debugging Firmware

### Debug Output (`link.sendDebug`)

Firmware has `link.sendDebug(value: u32)` in `firmware/link.zig` that prints `DEBG<4 bytes>` to UART before error responses. Useful for debugging early failures.

### Debug Buffer in Errors

`firmware/interpreter.zig` populates a debug buffer passed to `execute()`. On error, this buffer is included in the response payload. Format:
- `[0] = payload_len`
- `[1] = remaining bytes`
- `[2] = num_instructions or context-specific`
- `[3] = instruction index or error context`
- `[4+] = opcode or additional data`

Parse debug payload in driver output when seeing errors.

### Common Firmware Errors

- **`BadMagic`** → IR program header mismatch (wrong magic or version)
- **`BadPayloadLen`** → Byte counting error in `readInstruction` (was using wrong subtraction)
- **`BadAddress`** → Tensor descriptor validation failed (dtype, bounds, offset out of range, etc.)
- **`IllegalInstruction`** → Unknown opcode in IR program

### Common Fixes Applied

1. **`tile_store` field order**: Python emitted `n_offset` before `m_offset`, but firmware struct expects `m_offset` first. Fixed in `tools/ir.py`.
2. **Input `k_offset`**: For `[K, tile]` layout, k_offset should be `k_base * tile` (byte offset within M-tile), not `k_base * 4`.
3. **Input tensor stride**: Should be `K * tile` bytes (size of one M-tile), matching the `[K, tile]` layout.
4. **Byte counting**: `readInstruction` was subtracting `@sizeOf(T)` instead of `@sizeOf(T) - 1`.
5. **Software N tiling**: Added N-tiling loop when N > 8, with proper `n_offset` and `n_count` for all operations.

### Verify IR Struct Alignment

Python `tile_store(m_offset, n_offset)` emits in the same order as Zig `TileStore{m_offset, n_offset}` expects. Always verify byte order matches between `tools/ir.py` and `shared/ir.zig`.

## Debugging Hardware

### LED Status Indicators

Board RGB LED shows firmware state:
- **Blue (slow pulse)** → IDLE, waiting for commands
- **Green** → Compute in progress
- **Red** → Error state

### Simulation Tests

Before testing on hardware, verify RTL with simulation:
```sh
uv run pytest hardware -v
```

### GEMM Test Variants

`accel-gemm` supports two test variants:
- **non-pipelined** → Load → DMA wait → Compute → Store (no overlap)
- **pipelined** → Full K-tiling with DMA/compute overlap

Test parameters:
- `--m`, `--k`, `--n` for matrix dimensions
- Supports software M and N tiling for matrices larger than 8x8

Use `--no-verify` to just run and check cycles, or `--verify-tolerance 1` for ±1 tolerance.

### Driver Debugging

Driver outputs to stderr. Use `-v` flag for verbose output showing all driver commands.

## Pre-commit / CI

No pre-commit hooks or CI workflows currently defined. Manual checks:
- `uv run pytest hardware` must pass (Amaranth simulation).
- `just hw-all` must complete end-to-end on Tang Nano 20K.
- `accel-gemm /dev/ttyUSB1 all --verify-tolerance 1` should pass.
- `just sim-gemm` should pass (GEMM regression on Verilator).
- Don't commit `top.v` if it's stale; regenerate before committing.
