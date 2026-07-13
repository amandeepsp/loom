# loom — RISC-V ML Accelerator on FPGA

A full-stack int8 inference accelerator targeting the **Sipeed Tang Nano 20K** (Gowin GW2AR-18).
Implements an output-stationary systolic array as a VexRiscv Custom Function Unit (CFU),
with a model-binary runtime, bare-metal firmware, host driver, and xDSL compiler pipeline.

## Stack

The active compiler path lowers supported quantized ONNX models into xDSL/TOSA-like IR,
rewrites GEMM regions into loom dialect ops, lowers tiled affine schedules into
RV32I/CUSTOM0 machine code, and packages constants, code, and a call schedule
into a `.lmb` model binary. Firmware uploads constants, executes non-GEMM work
on the VexRiscv CPU, and uses CUSTOM0 instructions to drive the CFU for tiled
GEMM kernels.

## Key Design Points

- **Output-stationary 8×8 systolic array** — int8 inputs, int32 accumulators, configurable dimensions
- **Double-buffered scratchpads** — DMA fills one bank while the sequencer reads the other
- **Hardware requantization epilogue** — per-channel bias/multiplier/shift, SRDHM + RDBPOT (TFLite-compatible)
- **Model-binary runtime** — `.lmb` artifacts carry constants, RV32I functions, CUSTOM0 GEMM kernels, and the execution schedule
- **K-tiling with DMA/compute overlap** — pipelined execution for matrices larger than scratchpad depth
- **Software M/N tiling** — handles matrices larger than the array dimensions
- **xDSL rewrite pipeline** — frontend ops, IRDL dialects, greedy rewrite passes, affine loop lowering, and direct RV32I encoding

## Directory Map

| Directory   | Language        | Purpose                                                  |
|-------------|-----------------|----------------------------------------------------------|
| `hardware/` | Python/Amaranth | CFU RTL — systolic array, DMA scratchpads, sequencer, epilogue |
| `firmware/` | Zig             | Bare-metal RISC-V firmware: UART protocol, `.lmb` executor, KIR interpreter, DMA/CFU drivers |
| `host/`     | Zig             | Host-side native driver + C API (`libloom.so`)          |
| `shared/`   | Zig + Python    | Wire protocol, IR definitions (shared across firmware, host, and tools) |
| `compiler/` | Python          | xDSL frontend/dialects/rewrites, affine lowering, RV32I + CUSTOM0 backend, model runtime |
| `models/`   | Python          | Int8 MNIST training + static quantization (ONNX export)  |
| `tests/`    | Python          | E2E GEMM test harness, Verilator sim tests, model-binary pipeline tests |
| `tools/`    | Python          | LiteX flash utils, IR debug helpers                      |
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
just hw-gemm
```

## Build Order

RTL changes and firmware builds have a strict dependency chain:

1. `just verilog` — regenerate `top.v` from Amaranth
2. `just hw-build` — build LiteX SoC + bitstream (generates `csr.json`)
3. `just hw-flash` — flash bitstream to board
4. `just hw-firmware` — build firmware (reads CSR addresses from `csr.json`)
5. `just hw-upload-once` — serial-boot firmware onto running board
6. `just hw-gemm` — end-to-end verification

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
just hw-gemm
just hw-gemm-reset
```

## xDSL Model-Binary Compiler

The `compiler/` package lowers supported quantized ONNX GEMM and MobileNet-v1
regions through xDSL/TOSA and affine schedules into a loom model binary:

1. **Frontend** (`compiler/frontend/`) — imports ONNX quantized GEMM/linear patterns and canonicalizes QOperator MobileNet-v1 convolution, ReLU6, global-pool, and classifier regions.
2. **Dialects** (`compiler/dialects/`) — defines loom IR with `@irdl_op_definition` ops for GEMM, repack, DMA tile loads, tiled MMA, epilogue config, and stores.
3. **Backend** (`compiler/backend/model.py`) — plans tensor memory, constants, affine tiling, K-reduction, runtime schedule, and model-binary sections.
4. **Target lowering** (`compiler/backend/riscv_ir.py`, `compiler/backend/riscv_ir_encode.py`) — lowers affine loop schedules through xDSL RISC-V target IR into compact RV32I + CUSTOM0 code with async DMA setup/waits, tiled K reduction, epilogue writes, and result stores.
5. **Runtime** (`compiler/backend/runtime.py`) — uploads raw quantized int8 input plus the model binary and calls firmware `exec_model`; input/inter-layer packing runs as RV32I code in the model schedule.

MobileNet pointwise and spatial convolutions use the CFU GEMM path. Depthwise
weights are converted to sparse block-diagonal matrices, while im2col and tile
packing are generated as structured MLIR and lowered by LLVM to RV32I. The
runtime boundary for the supplied MobileNet-v1 0.25 model is signed int8 NCHW.

```sh
# Train MNIST + export statically-quantized int8 ONNX
uv run python -m models.mnist

# Compile a model binary
uv run loom-compile-model models/out/mnist_int8.onnx -o build/mnist.lmb \
  --csr-json build/sipeed_tang_nano_20k/csr.json --manifest
# The printed manifest includes deployment.model_addr for exec-model-bin.
# Add --debug-sections when you want embedded ASM/xDSL IR traces in the .lmb;
# the default artifact contains only manifest, constants, RISC-V code, and schedule.

# Deterministically quantize + compile MobileNet-v1 0.25 (32x32, 10 classes)
just mobilenet-model
# Outputs models/out/mobilenet_v1_025_int8.onnx and build/mobilenet_v1_025.lmb

# Execute a precompiled binary over serial or sim TCP with raw int8 input bytes
uv run loom-run-model build/mnist.lmb build/input_i8.bin --port /dev/ttyUSB1 \
  -o build/output_i8.bin --expect-output build/expected_i8.bin \
  --tolerance 2 --check-prediction

# Run the compiled model binary on Verilator sim and print/check logits
just sim-mnist

# Run the standalone .lmb artifact path through the loom-run-model CLI on sim
just sim-mnist-artifact

# Run the model binary on hardware after bitstream + firmware upload
just hw-mnist

# Run the standalone .lmb artifact path through the loom-run-model CLI on hardware
just hw-mnist-artifact
```
