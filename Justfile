# accel — RISC-V CFU accelerator task runner

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
        --build --toolchain apicula \
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

# Build libaccel.so for host Python binding
libaccel:
    zig build libaccel -Dbuild-dir=build/sipeed_tang_nano_20k

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
hw-gemm m="8" k="8" n="8" variant="all" verify-tolerance="1": libaccel hw-firmware
    uv run python -m tools.test_gemm {{ port }} {{ variant }} \
        --m {{ m }} --k {{ k }} --n {{ n }} \
        --cfu-word-bits $(({{ cfu_rows }} * {{ cfu_in_width }})) \
        --cfu-store-depth-words {{ cfu_store_depth }} \
        --verify-tolerance {{ verify-tolerance }}

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
    uv run python -m tools.sim_run --port {{ sim_port }} {{ _sim_args }} -- \
        uv run python -m tools.test_gemm \
            tcp://127.0.0.1:{{ sim_port }} {{ variant }} \
            --m {{ m }} --k {{ k }} --n {{ n }} \
            --cfu-word-bits $(({{ cfu_rows }} * {{ cfu_in_width }})) \
            --cfu-store-depth-words {{ cfu_store_depth }} \
            --driver-timeout 1800 \
            --verify-tolerance {{ verify-tolerance }}

# TVM path MNIST inference on Verilator simulation.
sim-tvm verify-tolerance="1": sim-firmware
    uv run python -m tools.sim_run --port {{ sim_port }} {{ _sim_args }} -- \
        uv run python tools/tvm_sim_test.py \
            --tcp tcp://127.0.0.1:{{ sim_port }} \
            --verify-tolerance {{ verify-tolerance }} \
            --driver-timeout 1800

# Full TVM pipeline (ONNX → Relax → patterns → codegen → sim execution).
sim-tvm-pipeline: sim-firmware
    uv run python -m tools.sim_run --port {{ sim_port }} {{ _sim_args }} -- \
        uv run python tools/test_tvm_pipeline.py \
            --tcp tcp://127.0.0.1:{{ sim_port }} \
            --driver-timeout 1800
