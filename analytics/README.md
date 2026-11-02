# loom showcase analytics

This package produces a source-backed hardware/software dashboard for the
MobileNet-v1 0.25 deployment. It separates architectural ceilings from board
measurements and never substitutes estimates for missing counters.

## Collect the dashboard data

The current FPGA build, compiled `.lmb`, tracked ONNX model, and reviewed board
measurements are required:

```sh
uv run python -m tools.showcase_metrics collect --output-dir build/showcase
```

This writes compact JSON datasets and a canonical Data Analytics
`artifact.json` under `build/showcase/`.

## Refresh the GEMM roofline counter sweep

With the board flashed and firmware running:

```sh
uv run python -m tools.showcase_metrics benchmark \
  --port /dev/ttyUSB1 \
  --output analytics/measurements/gemm_sweep.json
```

Then rerun the collection command. The benchmark covers generic and
MobileNet-shaped GEMMs, verifies each output within one INT8 LSB, and records
DMA transaction-active, sequencer-busy, and concurrent cycles from the CFU
counters. Its KIR instructions are read directly from UART while `mcycle` is
running, so streamed-KIR GOPS are reported separately from accelerator rates.

## Build the portable dashboard

Render and browser-verify the generated artifact:

```sh
just showcase-dashboard
```

The generated HTML is self-contained and can be shared without a local server.
It includes light/dark static SVGs as well as the interactive reader. Set
`DATA_ANALYTICS_SKILL_ROOT` if the Data Analytics plugin is installed elsewhere.

## Measurement definitions

- `issued operations` count every padded 8x8-array MAC as multiply plus add.
- `dense-useful operations` remove M/N tail padding but retain dense depthwise
  lowering.
- `semantic operations` count only nonzero depthwise connections.
- `theoretical peak` is `rows * cols * clock * 2` operations per second.
- `array-active throughput` divides issued work by sequencer-busy cycles.
- `CFU + DMA throughput` divides issued work by the union of sequencer and DMA
  active cycles.
- `end-to-end throughput` includes generated RV32 packing, control, and all
  other compiled-model execution.
- `roofline ceiling` is the minimum of peak compute and the 32-bit SDRAM
  interface bandwidth times algorithmic arithmetic intensity.

The measured counters derive exposed DMA/compute time and overlap efficiency.
Traffic and fixed sequencer phases come from the emitted schedule. Per-layer
achieved cycles and power/energy still require
additional profiling or an external power measurement; the dashboard reports
those fields as unavailable rather than claiming modeled values are measured.
