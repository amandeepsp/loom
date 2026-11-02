# AGENTS.md

## Project Overview

**loom** — RISC-V Custom Function Unit (CFU) accelerator for ML on FPGA (Tang Nano 20K).
Cross-layer stack: Amaranth (RTL), Zig (firmware), Python/xDSL (compiler, host transport, SoC integration).

### Directory Map

- `hardware/` — Amaranth Python RTL: CFU datapath, systolic array, DMA scratchpads, sequencer, epilogue
- `firmware/` — Zig: bare-metal RISC-V (VexRiscv), UART protocol, `.lmb` model executor, KIR interpreter, CFU/DMA drivers
- `shared/` — Zig + Python: wire protocol (`protocol.zig`, `protocol.py` incl. Serial/TCP transports), IR definitions (`ir.zig`, `ir.py`)
- `compiler/` — Python: ONNX importer, xDSL/IRDL dialects, rewrite passes, affine lowering, RV32I + CUSTOM0 backend
- `models/` — Python: int8 MNIST training, static quantization, ONNX export
- `tests/` — Python: E2E GEMM test harness (`test_gemm.py`), Verilator sim tests (`test_sim.py`)
- `tools/` — Python: LiteX flash utils, IR debug helpers
- `docs/` — Architecture Decision Records
- `top.v` — **Generated** Verilog (from `hardware/top.py`)

## Essential Commands

### Build and board

Run/test recipes take `target=hw|sim` (default `hw`). Set `LOOM_PORT` and
`OSS_CAD_BIN` in the environment or a `.env` file if not the defaults.

```sh
just verilog          # regenerate top.v from Amaranth
just soc              # verilog → LiteX SoC → bitstream (slow; writes csr.json)
just flash            # write bitstream to SPI flash
just reset            # reload FPGA from flash
just firmware         # build hw firmware (needs csr.json from `just soc`)
just up               # build firmware, reset, serial-boot it, ping
just all              # soc → flash → up
just gemm             # E2E GEMM on the board (needs `just up`)
just gemm-sweep       # small GEMM + K-tiled GEMM
```

**Required tools:** `oss-cad-suite/bin` (Yosys, nextpnr, openFPGALoader), Zig 0.17, `riscv64-*-objcopy`.

### Tests (Python/pytest via uv)

```sh
# Run all hardware tests (Amaranth simulation)
uv run pytest hardware

# Run single test file
uv run pytest hardware/systolic/test_*.py -v

# All offline tests (pytest + zig host tests)
just test

# Sim regression (Verilator + firmware, no board)
just gemm sim                      # 8×8×8 default
just gemm sim 4 16 4               # custom m k n

# Full MNIST model-binary path
just mnist sim
just mnist                         # on the board
```

Tests live in: `hardware/systolic/`, `hardware/memory/`, `hardware/epilogue/`, and `tests/`.
pytest configured in `pyproject.toml` to include these paths.

### Compiler / Model Binary

```sh
# Train/export the quantized MNIST ONNX if missing
just mnist-model

# Compile ONNX to a deploy .lmb for the current hardware CSR map
uv run loom-compile-model models/out/mnist_int8.onnx \
  -o build/mnist.lmb \
  --csr-json build/sipeed_tang_nano_20k/csr.json \
  --manifest

# Emit deterministic raw int8 input/expected files plus .lmb
uv run python -m tools.mnist_artifact \
  --onnx models/out/mnist_int8.onnx \
  --csr-json build/sipeed_tang_nano_20k/csr.json \
  --model build/mnist.lmb \
  --input build/mnist_input_i8.bin \
  --expected build/mnist_expected_i8.bin

# Run a precompiled .lmb on board or sim
uv run loom-run-model build/mnist.lmb build/mnist_input_i8.bin \
  --port /dev/ttyUSB1 \
  --expect-output build/mnist_expected_i8.bin \
  --tolerance 2 \
  --check-prediction
```

## Build Order (Critical)

When making changes:

1. **If RTL changes:** `just soc` (regenerates `top.v` and CSR addresses), then `just flash`
2. **If firmware changes:** `just up` (rebuilds firmware against `csr.json` and boots it)
3. **If hardware tests fail:** `uv run pytest hardware -v` (simulation, no board needed)
4. **If e2e fails:** `just up && just gemm`

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
- `firmware/interpreter.zig`: Firmware IR interpreter (executes instructions, drives DMA, sequencer).
- `tools/ir.py`: ~~Legacy IR bytecode builder~~ (deleted; functionality merged into `shared/ir.py`).
- `firmware/model.zig`: `.lmb` model-binary parser/executor (loads constants, calls generated RV32I functions).
- `tests/test_sim.py`: pytest-based Verilator sim tests (layer-by-layer GEMM, full ONNX→model-binary pipeline).

Changing IR requires sync across `shared/ir.zig`, `shared/ir.py`, and `firmware/interpreter.zig`.

Changing the `.lmb` model binary requires sync across `compiler/backend/model.py`, `compiler/backend/runtime.py`, `firmware/model.zig`, and host/transport tests.

## xDSL Compiler Pipeline

- `compiler/frontend/`: imports supported quantized ONNX ops into xDSL TOSA-like IR.
- `compiler/dialects/loom.py`: IRDL ops for GEMM candidates, planned GEMM, tiling, DMA loads, MMA, epilogue, and stores.
- `compiler/dialects/loom_cpu.py`: CPU-side model-binary ops such as activation packing.
- `compiler/dialects/loom_riscv.py`: target CUSTOM0/control ops before final RV32 encoding.
- `compiler/backend/rewrites.py`: xDSL `RewritePattern` and `ModulePass` lowering stages.
- `compiler/backend/model.py`: memory planning, constants, schedule packing, `.lmb` sections.
- `compiler/backend/riscv_ir.py`: affine GEMM tile schedules to structured xDSL RISC-V/SCF plus CUSTOM0 ops.
- `compiler/backend/riscv_ir_encode.py`: direct RV32I + CUSTOM0 machine-code encoder.
- `compiler/backend/cpu.py`: non-GEMM CPU functions emitted as RV32I; use LLVM/MLIR tooling where practical.
- `compiler/quant_utils.py`: Derives per-channel epilogue params (bias, multiplier, shift) from scale/zero-point.

The VexRiscv target has no hardware floating point. Generated firmware/model code must stay integer-only, and generated CPU helper code must stay RV32I unless the SoC/firmware contract is explicitly changed.

## Hardware Quirks

### Datapath Flow

Firmware receives commands over UART. KIR bytecode is interpreted directly; `.lmb` model binaries load constants, run generated RV32I functions on the VexRiscv CPU, and issue CUSTOM0 instructions for GEMM. The CFU uses DMA readers to fill double-buffered scratchpads, the sequencer drives the output-stationary systolic array, and the epilogue requantizes int32 accumulators to int8 results.

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
- **Host tests:** `zig build test` runs `firmware/model.zig` unit tests natively.
- **Linker script:** `firmware/linker.ld` (VexRISCV memory map).
- **Build options:** `-Dcfu-rows=N`, `-Dcfu-cols=N`, `-Dcfu-store-depth=D` (must match Verilog + Justfile).
- **Output:** `zig-out/bin/firmware.bin` (objcopy'd ELF → binary for serial boot).

## Common Pitfalls

1. **"csr.json not found"** → Run `just soc` before `zig build firmware`.
2. **Stale `top.v`** → Always run `just verilog` after RTL changes.
3. **Parameter mismatch** → If you change `cfu_rows`, update Justfile variables.
4. **Serial port collision** → Only one process can access `/dev/ttyUSB1`; close `litex_term` before `just gemm`.
5. **E2E test hangs** → Board may be in bad state; run `just up` to recover.

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

Parse the debug payload in the `ProtocolError` message when seeing errors.

### Common Firmware Errors

- **`BadMagic`** → IR program header mismatch (wrong magic or version)
- **`BadPayloadLen`** → Byte counting error in `readInstruction` (was using wrong subtraction)
- **`BadAddress`** → Tensor descriptor validation failed (dtype, bounds, offset out of range, etc.)
- **`IllegalInstruction`** → Unknown opcode in IR program

### Common Fixes Applied

1. **`tile_store` field order**: Python emitted `n_offset` before `m_offset`, but firmware struct expects `m_offset` first. Fixed in `shared/ir.py`.
2. **Input `k_offset`**: For `[K, tile]` layout, k_offset should be `k_base * tile` (byte offset within M-tile), not `k_base * 4`.
3. **Input tensor stride**: Should be `K * tile` bytes (size of one M-tile), matching the `[K, tile]` layout.
4. **Byte counting**: `readInstruction` was subtracting `@sizeOf(T)` instead of `@sizeOf(T) - 1`.
5. **Software N tiling**: Added N-tiling loop when N > 8, with proper `n_offset` and `n_count` for all operations.

### Verify IR Struct Alignment

Python `tile_store(m_offset, n_offset)` emits in the same order as Zig `TileStore{m_offset, n_offset}` expects. Always verify byte order matches between `shared/ir.py` and `shared/ir.zig`.

## Debugging Hardware

### LED Status Indicators

With `--with-cfu-led-debug`, board LEDs are driven by CFU hardware debug, not firmware prints:
- RGB blue means idle.
- RGB yellow/green-yellow means DMA active.
- RGB green means sequencer/compute active.
- RGB cyan means epilogue active.
- RGB white means done/response pending.
- RGB red means error.

The six debug LEDs are active-low on the pins but the wrapper inverts them for display. Logical `debug_word = {phase_code[2:0], cfu_instr_debug[2:0]}`. Depending on physical viewing direction, the visible six-bit string may need reversing. The low three bits latch the last CFU instruction, so an idle end state can still show `...100` after `READ_RESULT`; idle is indicated by RGB blue and phase `000`.

### Simulation Tests

Before testing on hardware, verify RTL with simulation:
```sh
uv run pytest hardware -v
```

### GEMM Test Variants

`loom-gemm` supports two test variants:
- **non-pipelined** → Load → DMA wait → Compute → Store (no overlap)
- **pipelined** → Full K-tiling with DMA/compute overlap

Test parameters:
- `--m`, `--k`, `--n` for matrix dimensions
- Supports software M and N tiling for matrices larger than 8x8

Use `--no-verify` to just run and check cycles, or `--verify-tolerance 1` for ±1 tolerance.

## Pre-commit / CI

No pre-commit hooks or CI workflows currently defined. Manual checks:
- `uv run pytest hardware` must pass (Amaranth simulation).
- `just all` must complete end-to-end on Tang Nano 20K.
- `just gemm-sweep` should pass on the board.
- `just gemm sim` should pass (GEMM regression on Verilator).
- Don't commit `top.v` if it's stale; regenerate before committing.
