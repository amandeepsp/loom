# loom — RISC-V CFU accelerator task runner

set dotenv-load := false

oss_cad_bin := "/home/amandeeps/Projects/oss-cad-suite/bin"
cfu_rows := "8"
cfu_cols := "8"
cfu_store_depth := "512"
cfu_in_width := "8"
cfu_acc_width := "32"
port := "/dev/ttyUSB1"
sim_port := "21450"

# Default: list available recipes
default:
    @just --list

# Generate Verilog CFU from Amaranth
verilog:
    uv run python -m hardware.top \
        --cfu-rows {{ cfu_rows }} \
        --cfu-cols {{ cfu_cols }} \
        --cfu-store-depth {{ cfu_store_depth }} \
        --cfu-in-width {{ cfu_in_width }} \
        --cfu-acc-width {{ cfu_acc_width }}

# Build the LiteX SoC for Tang Nano 20K
hw-build: verilog
    env PATH={{ oss_cad_bin }}:{{ env_var('PATH') }} \
        uv run python -m litex_boards.targets.sipeed_tang_nano_20k \
        --build --toolchain apicula --pnr-seed 2 \
        --cpu-type vexriscv --cpu-variant full+cfu --cpu-cfu top.v \
        --cfu-rows {{ cfu_rows }} \
        --cfu-cols {{ cfu_cols }} \
        --cfu-store-depth {{ cfu_store_depth }} \
        --cfu-in-width {{ cfu_in_width }} \
        --with-cfu-led-debug

# Flash the bitstream to the board
hw-flash:
    env PATH={{ oss_cad_bin }}:{{ env_var('PATH') }} \
        uv run python -m litex_boards.targets.sipeed_tang_nano_20k --flash

# Reset the FPGA board
hw-reset:
    env PATH={{ oss_cad_bin }}:{{ env_var('PATH') }} \
        openFPGALoader --board tangnano20k --reset

# Build libloom.so for host Python binding
libloom:
    zig build host-lib -Dbuild-dir=build/sipeed_tang_nano_20k

# Train + export a quantized MNIST ONNX for compiler/hardware smoke tests (no-op if present).
mnist-model:
    @test -f models/out/mnist_int8.onnx || \
        uv run python -m models.mnist --epochs 10 --out-dir models/out

# Deterministically quantize and compile MobileNet-v1 0.25 (32x32, 10 classes).
mobilenet-model:
    uv run python -m models.mobilenet
    uv run loom-compile-model models/out/mobilenet_v1_025_int8.onnx \
        -o build/mobilenet_v1_025.lmb \
        --csr-json build/sipeed_tang_nano_20k/csr.json \
        --manifest

# Collect source-backed hardware/software showcase analytics.
showcase-metrics: mobilenet-model
    uv run python -m tools.showcase_metrics collect --output-dir build/showcase

# Render and browser-verify the portable showcase dashboard.
showcase-dashboard: showcase-metrics
    node tools/build_showcase_dashboard.mjs \
        build/showcase/artifact.json \
        build/showcase/loom-showcase.html

# Export nextpnr's native Gowin placement geometry for the showcase.
showcase-nextpnr:
    mkdir -p build/showcase/nextpnr
    cd build/sipeed_tang_nano_20k/gateware && \
        env PATH={{ oss_cad_bin }}:{{ env_var('PATH') }} \
        nextpnr-himbaechel \
            --json sipeed_tang_nano_20k.json \
            --top sipeed_tang_nano_20k \
            --device GW2AR-LV18QN88C8/I7 \
            --vopt family=GW2A-18C \
            --vopt cst=sipeed_tang_nano_20k.cst \
            --timing-allow-fail \
            --seed 2 \
            --no-route \
            --placed-svg ../../../build/showcase/nextpnr/placed.svg

# Export square Matplotlib PNGs for social posts and routing review.
showcase-pngs: showcase-metrics showcase-nextpnr
    uv run --with matplotlib python -m tools.export_showcase_pngs \
        --nextpnr-svg build/showcase/nextpnr/placed.svg \
        --output-dir build/showcase/png

# Refresh theoretical-vs-achieved GEMM and schedule-overlap measurements on hardware.
showcase-benchmark port=port:
    uv run python -m tools.showcase_metrics benchmark \
        --port {{ port }} \
        --output analytics/measurements/gemm_sweep.json

# Build firmware targeting the hardware SoC
hw-firmware build-dir="build/sipeed_tang_nano_20k":
    zig build firmware \
        -Dbuild-dir={{ build-dir }} \
        -Dcfu-rows={{ cfu_rows }} \
        -Dcfu-cols={{ cfu_cols }} \
        -Dcfu-store-depth={{ cfu_store_depth }}

# Upload firmware via serial boot
hw-upload:
    uv run litex_term {{ port }} --kernel zig-out/bin/firmware.bin

# Upload firmware and automatically release the serial port when the transfer completes
hw-upload-once:
    uv run litex-upload-once {{ port }} zig-out/bin/firmware.bin \
        --reset-command 'just hw-reset' \
        --post-boot-timeout 12

# === Hardware tests (require board + running firmware) =======================

# Run GEMM test against real hardware.  Tunable: m, k, n, variant, verify-tolerance.
hw-gemm m="8" k="8" n="8" variant="all" verify-tolerance="1": libloom hw-firmware
    uv run python -m tests.test_gemm {{ port }} {{ variant }} \
        --m {{ m }} --k {{ k }} --n {{ n }} \
        --cfu-word-bits $(({{ cfu_rows }} * {{ cfu_in_width }})) \
        --cfu-store-depth-words {{ cfu_store_depth }} \
        --verify-tolerance {{ verify-tolerance }}

# Run compiled MNIST model binary against real hardware (auto-builds the ONNX if needed).
hw-mnist port=port tolerance="2": mnist-model hw-firmware
    uv run python -m tests.test_hw_pipeline --port {{ port }} \
        --csr-json build/sipeed_tang_nano_20k/csr.json \
        --tolerance {{ tolerance }}

# Run a precompiled MNIST .lmb artifact against real hardware via loom-run-model.
hw-mnist-artifact port=port tolerance="2": mnist-model hw-firmware
    uv run python -m tools.mnist_artifact \
        --onnx models/out/mnist_int8.onnx \
        --csr-json build/sipeed_tang_nano_20k/csr.json \
        --model build/mnist.lmb \
        --input build/mnist_input_i8.bin \
        --expected build/mnist_expected_i8.bin
    uv run loom-run-model build/mnist.lmb build/mnist_input_i8.bin \
        --port {{ port }} \
        --expect-output build/mnist_expected_i8.bin \
        --tolerance {{ tolerance }} \
        --check-prediction \
        -o build/mnist_output_i8.bin

# JTAG reset → firmware upload → small GEMM → large GEMM (one after another).
hw-gemm-reset:
    just hw-reset
    sleep 2
    just hw-upload-once
    just hw-gemm
    just hw-gemm m={{ cfu_rows }} \
        k="$(( ({{ cfu_store_depth }} * 4 / {{ cfu_rows }}) + 128 ))" \
        n={{ cfu_cols }}

# Full hardware flow: build SoC → flash → build firmware → upload
hw-all: hw-build hw-flash hw-firmware hw-upload-once

# === Simulation (no board needed) ============================================

# Generate sim SoC (csr.json) without compiling gateware
sim-generate: verilog
    uv run python -m soc.sim \
        --no-compile-gateware \
        --cfu-rows {{ cfu_rows }} \
        --cfu-cols {{ cfu_cols }} \
        --cfu-store-depth {{ cfu_store_depth }} \
        --cfu-in-width {{ cfu_in_width }} \
        --l2-size 128 \
        --output-dir build/sim
    just _crt0-restore build/sim

# LiteX BIOS build deletes crt0.o as a cleanup step.  Restore it so the
# firmware link (which depends on it) doesn't fail.
_crt0_src := `uv run python -c "import litex; print(litex.__path__[0])" 2>/dev/null` + "/soc/cores/cpu/vexriscv/crt0.S"

_crt0-restore build-dir:
    @test -f {{ build-dir }}/software/bios/crt0.o || \
        riscv64-elf-gcc -march=rv32i2p0_m -mabi=ilp32 -D__vexriscv__ \
        -I{{ build-dir }}/software/bios/../include -c {{ _crt0_src }} \
        -o {{ build-dir }}/software/bios/crt0.o

# Build firmware targeting the simulation SoC
sim-firmware: sim-generate
    zig build firmware \
        -Dbuild-dir=build/sim \
        -Dcfu-rows={{ cfu_rows }} \
        -Dcfu-cols={{ cfu_cols }} \
        -Dcfu-store-depth={{ cfu_store_depth }} \
        -Ddebug-info=true

# Run full Verilator simulation with firmware (interactive)
sim-run: sim-firmware
    uv run python -m soc.sim \
        --cfu-rows {{ cfu_rows }} \
        --cfu-cols {{ cfu_cols }} \
        --cfu-store-depth {{ cfu_store_depth }} \
        --cfu-in-width {{ cfu_in_width }} \
        --l2-size 128 \
        --sdram-init zig-out/bin/firmware.bin \
        --output-dir build/sim

# Run Verilator sim non-interactively (for CI)
sim-run-ci: sim-firmware
    uv run python -m soc.sim \
        --cfu-rows {{ cfu_rows }} \
        --cfu-cols {{ cfu_cols }} \
        --cfu-store-depth {{ cfu_store_depth }} \
        --cfu-in-width {{ cfu_in_width }} \
        --l2-size 128 \
        --sdram-init zig-out/bin/firmware.bin \
        --output-dir build/sim \
        --non-interactive

# Shared sim-launch args forwarded to soc.sim (cfu config + firmware image).
_sim_args := "--sim-arg=--cfu-rows --sim-arg=" + cfu_rows + " " \
    + "--sim-arg=--cfu-cols --sim-arg=" + cfu_cols + " " \
    + "--sim-arg=--cfu-store-depth --sim-arg=" + cfu_store_depth + " " \
    + "--sim-arg=--cfu-in-width --sim-arg=" + cfu_in_width + " " \
    + "--sim-arg=--l2-size --sim-arg=128 " \
    + "--sim-arg=--sdram-init --sim-arg=zig-out/bin/firmware.bin " \
    + "--sim-arg=--output-dir --sim-arg=build/sim"

# GEMM regression test on Verilator simulation.  Tunable: m, k, n, variant, tolerance.
sim-gemm m="8" k="8" n="8" variant="all" verify-tolerance="1": sim-firmware
    uv run python -m tests.sim_run --port {{ sim_port }} {{ _sim_args }} -- \
        uv run python -m tests.test_gemm \
            tcp://127.0.0.1:{{ sim_port }} {{ variant }} \
            --m {{ m }} --k {{ k }} --n {{ n }} \
            --cfu-word-bits $(({{ cfu_rows }} * {{ cfu_in_width }})) \
            --cfu-store-depth-words {{ cfu_store_depth }} \
            --driver-timeout 1800 \
            --verify-tolerance {{ verify-tolerance }}

# Full compiled MNIST model binary on Verilator simulation.
sim-mnist tolerance="2": mnist-model sim-firmware
    uv run python -m tests.sim_run --port {{ sim_port }} {{ _sim_args }} -- \
        uv run python -m tests.test_hw_pipeline \
            --port tcp://127.0.0.1:{{ sim_port }} \
            --csr-json build/sim/csr.json \
            --tolerance {{ tolerance }}

# Full precompiled .lmb artifact through loom-run-model on Verilator simulation.
sim-mnist-artifact tolerance="2": mnist-model sim-firmware
    uv run python -m tools.mnist_artifact \
        --onnx models/out/mnist_int8.onnx \
        --csr-json build/sim/csr.json \
        --model build/mnist.lmb \
        --input build/mnist_input_i8.bin \
        --expected build/mnist_expected_i8.bin
    uv run python -m tests.sim_run --port {{ sim_port }} {{ _sim_args }} -- \
        uv run loom-run-model build/mnist.lmb build/mnist_input_i8.bin \
            --port tcp://127.0.0.1:{{ sim_port }} \
            --expect-output build/mnist_expected_i8.bin \
            --tolerance {{ tolerance }} \
            --check-prediction \
            -o build/mnist_output_i8.bin
