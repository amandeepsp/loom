# loom — RISC-V ML Accelerator on FPGA

A full-stack int8 inference accelerator targeting the **Sipeed Tang Nano 20K** (Gowin GW2AR-18).
Implements an output-stationary systolic array as a VexRiscv Custom Function Unit (CFU),
with a model-binary runtime, bare-metal firmware, a Python host transport, and an xDSL compiler pipeline.

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
| `firmware/` | Zig             | Bare-metal RISC-V firmware: UART protocol, `.lmb` executor, CFU driver |
| `shared/`   | Zig + Python    | Wire protocol + Serial/TCP transports, IR definitions (shared by firmware and Python) |
| `compiler/` | Python          | xDSL frontend/dialects/rewrites, affine lowering, RV32I + CUSTOM0 backend, model runtime |
| `models/`   | Python          | Int8 MNIST training + static quantization (ONNX export)  |
| `tests/`    | Python          | E2E GEMM test harness, Verilator sim tests, model-binary pipeline tests |
| `tools/`    | Python          | LiteX flash utils, IR debug helpers                      |
| `docs/`     | Markdown        | Architecture Decision Records                            |
| `top.v`     | Verilog (gen.)  | Generated CFU Verilog — never edit directly              |

## Prerequisites

- [oss-cad-suite](https://github.com/YosysHQ/oss-cad-suite-build) (Yosys, nextpnr-gowin, openFPGALoader)
- [Zig](https://ziglang.org/) (0.17)
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [just](https://github.com/casey/just) (task runner)
- Sipeed Tang Nano 20K board (for hardware testing; simulation tests need no board)

## Quick Start

```sh
# Offline tests: RTL simulation, compiler, firmware host tests (no board needed)
just test

# Full hardware flow: Verilog → SoC/bitstream → flash → firmware → boot
just all

# Board already flashed? Just (re)boot the firmware
just up

# End-to-end GEMM on the board, or on Verilator with `sim`
just gemm
just gemm sim
```

Run/test recipes take `target=hw|sim` (default `hw`). Override the serial port
and toolchain location with `LOOM_PORT` / `OSS_CAD_BIN` (environment or `.env`).
```

## Build Order

RTL changes and firmware builds have a strict dependency chain:

1. `just soc` — regenerate `top.v`, build LiteX SoC + bitstream (generates `csr.json`)
2. `just flash` — flash bitstream to board
3. `just up` — build firmware against `csr.json`, reset, serial-boot, ping
4. `just gemm` — end-to-end verification

`just firmware` refuses to run until `just soc` has produced `csr.json`.

## Performance

MobileNet-v1 0.25 (32x32, CIFAR-10) runs in **18.1 ms (732K cycles at
40.5 MHz)** on the Tang Nano 20K, bit-exact against the int8 ONNX model, down
from 1022 ms. The big steps were CFU features (result LUT, CPU-written
activations, per-column params, hardware depthwise im2col "LineGen", result
write-back DMA),
dropping the SoC's tiny L2, and software pipelining/fusion in the compiler.
Details, the CFU op map and deferred work: `docs/architecture-improvements.md`
(local `docs/`).

## Showcase Analytics

The reproducible showcase dashboard separates hardware and software/compiler
metrics and compares theoretical ceilings with measured board performance:

```sh
# Refresh the counter-backed GEMM roofline sweep.
just showcase-benchmark

# Collect P&R, model, accuracy, cycle, arithmetic-intensity, and sweep data.
just showcase-metrics

# Export five square 1600x1600 Matplotlib PNGs, including native nextpnr
# placement and a routed-wire occupancy/connectivity view.
just showcase-pngs
```

See [`analytics/README.md`](analytics/README.md) for metric definitions and the
portable HTML build command. Hardware counters separate array-active, DMA, and
concurrent overlap cycles from complete compiled-model time. GEMM sweeps run
as compiled `.lmb` binaries, so their `mcycle` counts are on-chip only. The
Gowin nextpnr backend does not implement native wire/pip SVG graphics, so the
routing figure uses its routed JSON attributes rather than implying that
`--routed-svg` contains connections.

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

# E2E on hardware (requires `just up`)
just gemm
just gemm-sweep            # small GEMM + K-tiled GEMM
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

# Run real test images on the board, checked against a fresh ONNX reference
uv run loom board                                  # firmware / BIOS / silent?
uv run loom run build/mnist.lmb --count 20 --tolerance 1
uv run loom run build/mobilenet_v1_025.lmb --count 10

# Compile MNIST and run it on the board / sim in one step
just mnist
just mnist sim
```
