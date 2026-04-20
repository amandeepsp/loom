# accel — RISC-V ML Accelerator on FPGA

A full-stack int8 inference accelerator targeting the **Sipeed Tang Nano 20K** (Gowin GW2AR-18).
Implements an output-stationary systolic array as a VexRiscv Custom Function Unit (CFU),
with a custom IR, bare-metal firmware, host driver, and TVM compiler integration.

## Stack

```
TVM / Python host
  ↓  ONNX → Relax IR → pattern match → codegen
Host driver (Zig → libaccel.so, C FFI)
  ↓  UART serial protocol
VexRiscv firmware (Zig, bare-metal RISC-V)
  ↓  Kernel bytecode interpreter
CFU hardware (Amaranth → Verilog → LiteX SoC)
  DMA → double-buffered scratchpads → 8×8 systolic array → epilogue (int32→int8)
```

## Key Design Points

- **Output-stationary 8×8 systolic array** — int8 inputs, int32 accumulators, configurable dimensions
- **Double-buffered scratchpads** — DMA fills one bank while the sequencer reads the other
- **Hardware requantization epilogue** — per-channel bias/multiplier/shift, SRDHM + RDBPOT (TFLite-compatible)
- **Custom bytecode IR (KIR)** — tile-oriented instructions: load-act, load-wgt, mma, store, set-epilogue, done
- **K-tiling with DMA/compute overlap** — pipelined execution for matrices larger than scratchpad depth
- **Software M/N tiling** — handles matrices larger than the array dimensions
- **TVM Relax integration** — pattern matching for quantized matmul composites, codegen to KIR, runtime bridge via `libaccel.so`

## Directory Map

| Directory   | Language        | Purpose                                                  |
|-------------|-----------------|----------------------------------------------------------|
| `hardware/` | Python/Amaranth | CFU RTL — systolic array, DMA scratchpads, sequencer, epilogue |
| `firmware/` | Zig             | Bare-metal RISC-V firmware: UART protocol, KIR interpreter, DMA/CFU drivers |
| `host/`     | Zig             | Host-side native driver + C API (`libaccel.so`)          |
| `shared/`   | Zig + Python    | Wire protocol, IR definitions (shared across firmware, host, and tools) |
| `tvm/`      | Python          | TVM Relax patterns, codegen, quantization utils, runtime bridge |
| `models/`   | Python          | Int8 MNIST training + static quantization (ONNX export)  |
| `tools/`    | Python          | E2E test harness, IR bytecode builder, LiteX flash utils |
| `docs/`     | Markdown        | Architecture Decision Records                            |
| `top.v`     | Verilog (gen.)  | Generated CFU Verilog — never edit directly              |

## Prerequisites

- [oss-cad-suite](https://github.com/YosysHQ/oss-cad-suite-build) (Yosys, nextpnr-gowin, openFPGALoader)
- [Zig](https://ziglang.org/) (0.13+)
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [just](https://github.com/casey/just) (task runner)
- Sipeed Tang Nano 20K board (for hardware testing; simulation tests need no board)

## Quick Start

```sh
# Run RTL simulation tests (no board needed)
uv run pytest hardware -v

# Full hardware flow: generate Verilog → build SoC → flash → build firmware → upload
just hw-all

# Run end-to-end GEMM test against the board
just hw-e2e-gemm
```

## Build Order

RTL changes and firmware builds have a strict dependency chain:

1. `just verilog` — regenerate `top.v` from Amaranth
2. `just hw-build` — build LiteX SoC + bitstream (generates `csr.json`)
3. `just hw-flash` — flash bitstream to board
4. `just hw-firmware` — build firmware (reads CSR addresses from `csr.json`)
5. `just hw-upload-once` — serial-boot firmware onto running board
6. `just hw-e2e-gemm` — end-to-end verification

Skipping step 2 before step 4 causes `csr.json not found` errors.

## Configuration

Array dimensions and scratchpad depth are configured in the `Justfile` and must match across Verilog generation, firmware build, and E2E tests:

| Parameter         | Default | Description                    |
|-------------------|---------|--------------------------------|
| `cfu_rows`        | 8       | Systolic array rows            |
| `cfu_cols`        | 8       | Systolic array columns         |
| `cfu_store_depth` | 512     | Scratchpad depth (32-bit words)|
| `cfu_in_width`    | 8       | Input element width (bits)     |
| `cfu_acc_width`   | 32      | Accumulator width (bits)       |

## Tests

```sh
# All RTL simulation tests
uv run pytest hardware -v

# Individual modules
uv run pytest hardware/systolic/ -v    # PE array
uv run pytest hardware/memory/ -v      # Scratchpad
uv run pytest hardware/epilogue/ -v    # Requantization
uv run pytest hardware/control/ -v     # Sequencer

# E2E on hardware (requires board + firmware)
just hw-e2e-gemm
just hw-e2e-gemm-large
```

## TVM Integration

The `tvm/` directory implements an out-of-tree backend for Apache TVM's Relax IR:

1. **Pattern matching** (`patterns.py`) — identifies quantized matmul composites (QDQ format) in imported ONNX models
2. **Codegen** (`codegen.py`) — lowers matched regions to `call_dps_packed` with extracted weights and quantization constants
3. **Runtime** (`runtime.py`) — bridges TVM packed functions to `libaccel.so`, handles memory layout, builds KIR programs, executes on hardware
4. **Quantization utils** (`quant_utils.py`) — derives per-channel epilogue parameters (bias, multiplier, shift) from ONNX scale/zero-point constants

```sh
# Train MNIST + export statically-quantized int8 ONNX
uv run python -m models.mnist

# Build the host shared library
just libaccel
```
