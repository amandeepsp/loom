# loom — RISC-V CFU accelerator task runner
#
# Most run/test recipes take `target=hw|sim`. `hw` talks to the board over
# $LOOM_PORT; `sim` boots the Verilator SoC and bridges its UART over TCP.

set dotenv-load := true

oss_cad_bin := env_var_or_default("OSS_CAD_BIN", home_directory() / "Projects/oss-cad-suite/bin")
port := env_var_or_default("LOOM_PORT", "/dev/ttyUSB1")
sim_port := "21450"

# Array configuration. Must match across Verilog, SoC, firmware, and tests.
cfu_rows := "8"
cfu_cols := "8"
cfu_store_depth := "512"
cfu_in_width := "8"
cfu_acc_width := "32"

hw_dir := "build/sipeed_tang_nano_20k"
sim_dir := "build/sim"
fw_bin := "zig-out/bin/firmware.bin"
cad_env := "env PATH=" + oss_cad_bin + ":" + env_var("PATH")

_cfu_args := "--cfu-rows " + cfu_rows + " --cfu-cols " + cfu_cols \
    + " --cfu-store-depth " + cfu_store_depth + " --cfu-in-width " + cfu_in_width
_sim_soc_args := _cfu_args + " --l2-size 128 --output-dir " + sim_dir
_sim_launch := "uv run python -m tests.sim_run --port " + sim_port + " " \
    + replace_regex(_sim_soc_args + " --sdram-init " + fw_bin, '(\S+)', '--sim-arg=$1') + " --"

default:
    @just --list

# === Build ===================================================================

# Generate the CFU Verilog (top.v) from Amaranth.
verilog:
    uv run python -m hardware.top {{ _cfu_args }} --cfu-acc-width {{ cfu_acc_width }}

# Build the LiteX SoC + bitstream for the Tang Nano 20K (slow).
soc: verilog
    {{ cad_env }} uv run python -m litex_boards.targets.sipeed_tang_nano_20k \
        --build --toolchain apicula --pnr-seed 2 \
        --cpu-type vexriscv --cpu-variant full+cfu --cpu-cfu top.v \
        {{ _cfu_args }} --with-cfu-led-debug

# Generate the simulation SoC (csr.json, BIOS) without compiling gateware.
sim-soc: verilog
    uv run python -m soc.sim --no-compile-gateware {{ _sim_soc_args }}
    just _crt0-restore {{ sim_dir }}

# Build firmware for a target (hw needs `just soc` first; sim builds its SoC).
firmware target="hw": (_firmware-deps target)
    zig build firmware \
        -Dbuild-dir={{ if target == "sim" { sim_dir } else { hw_dir } }} \
        -Dcfu-rows={{ cfu_rows }} -Dcfu-cols={{ cfu_cols }} \
        -Dcfu-store-depth={{ cfu_store_depth }} \
        -Ddebug-info={{ if target == "sim" { "true" } else { "false" } }}

_firmware-deps target:
    @if [ "{{ target }}" = "sim" ]; then just sim-soc; \
     elif [ ! -f {{ hw_dir }}/csr.json ]; then echo "missing {{ hw_dir }}/csr.json: run 'just soc' first" >&2; exit 1; fi

# LiteX BIOS build deletes crt0.o as a cleanup step; the firmware link needs it.
_crt0_src := `uv run python -c "import litex; print(litex.__path__[0])" 2>/dev/null` + "/soc/cores/cpu/vexriscv/crt0.S"

_crt0-restore build-dir:
    @test -f {{ build-dir }}/software/bios/crt0.o || \
        riscv64-elf-gcc -march=rv32i2p0_m -mabi=ilp32 -D__vexriscv__ \
        -I{{ build-dir }}/software/bios/../include -c {{ _crt0_src }} \
        -o {{ build-dir }}/software/bios/crt0.o

# === Board ===================================================================

# Flash the bitstream to the board's SPI flash.
flash:
    {{ cad_env }} uv run python -m litex_boards.targets.sipeed_tang_nano_20k --flash

# Reset the FPGA (reloads the bitstream from flash).
reset:
    {{ cad_env }} openFPGALoader --board tangnano20k --reset

# Build firmware, reset the board, serial-boot the firmware, and ping it.
up: (firmware "hw")
    uv run litex-upload-once {{ port }} {{ fw_bin }} \
        --reset-command 'just reset' --post-boot-timeout 12
    uv run python -c "from shared.protocol import SerialTransport as T; T('{{ port }}', timeout_s=5).ping(); print('board ready')"

# Interactive LiteX terminal with firmware upload (manual alternative to `up`).
term:
    uv run litex_term {{ port }} --kernel {{ fw_bin }}

# Everything from RTL to a running board: soc → flash → up.
all: soc flash up

# Interactive Verilator simulation with firmware preloaded.
sim: (firmware "sim")
    uv run python -m soc.sim {{ _sim_soc_args }} --sdram-init {{ fw_bin }}

# === Models ==================================================================

# Train + export a quantized MNIST ONNX (no-op if present).
mnist-model:
    @test -f models/out/mnist_int8.onnx || \
        uv run python -m models.mnist --epochs 10 --out-dir models/out

# Deterministically quantize and compile MobileNet-v1 0.25 (32x32, 10 classes).
mobilenet-model:
    uv run python -m models.mobilenet
    uv run loom-compile-model models/out/mobilenet_v1_025_int8.onnx \
        -o build/mobilenet_v1_025.lmb --csr-json {{ hw_dir }}/csr.json --manifest

# === Tests ===================================================================

# Offline tests: RTL simulation (Amaranth) + compiler + firmware host tests.
test *args:
    uv run pytest {{ args }}
    zig build test

# End-to-end GEMM on hw or sim.  Tunable: m, k, n, variant, tolerance.
gemm target="hw" m="8" k="8" n="8" variant="all" tolerance="1": (_run-deps target)
    {{ if target == "sim" { _sim_launch } else { "" } }} \
    uv run python -m tests.test_gemm \
        {{ if target == "sim" { "tcp://127.0.0.1:" + sim_port + " --driver-timeout 1800" } else { port } }} \
        {{ variant }} --m {{ m }} --k {{ k }} --n {{ n }} \
        --cfu-word-bits $(({{ cfu_rows }} * {{ cfu_in_width }})) \
        --cfu-store-depth-words {{ cfu_store_depth }} \
        --verify-tolerance {{ tolerance }}

# Small GEMM, then one large enough to need K-tiling (board must be `up`).
gemm-sweep target="hw":
    just gemm {{ target }}
    just gemm {{ target }} {{ cfu_rows }} "$(( ({{ cfu_store_depth }} * 4 / {{ cfu_rows }}) + 128 ))" {{ cfu_cols }}

# Compile MNIST to a .lmb and run it on hw or sim via loom-run-model.
mnist target="hw" tolerance="2": mnist-model (_run-deps target)
    uv run python -m tools.mnist_artifact \
        --onnx models/out/mnist_int8.onnx \
        --csr-json {{ if target == "sim" { sim_dir } else { hw_dir } }}/csr.json \
        --model build/mnist.lmb --input build/mnist_input_i8.bin \
        --expected build/mnist_expected_i8.bin
    {{ if target == "sim" { _sim_launch } else { "" } }} \
    uv run loom-run-model build/mnist.lmb build/mnist_input_i8.bin \
        --port {{ if target == "sim" { "tcp://127.0.0.1:" + sim_port } else { port } }} \
        --expect-output build/mnist_expected_i8.bin --tolerance {{ tolerance }} \
        --check-prediction -o build/mnist_output_i8.bin

# sim needs fresh sim firmware; hw assumes the board is already `up`.
_run-deps target:
    @if [ "{{ target }}" = "sim" ]; then just firmware sim; fi

# === Showcase analytics ======================================================

# Collect source-backed hardware/software showcase analytics.
showcase-metrics: mobilenet-model
    uv run python -m tools.showcase_metrics collect --output-dir build/showcase

# Render and browser-verify the portable showcase dashboard.
showcase-dashboard: showcase-metrics
    node tools/build_showcase_dashboard.mjs build/showcase/artifact.json build/showcase/loom-showcase.html

# Export nextpnr's native Gowin placement geometry for the showcase.
showcase-nextpnr:
    mkdir -p build/showcase/nextpnr
    cd {{ hw_dir }}/gateware && {{ cad_env }} nextpnr-himbaechel \
        --json sipeed_tang_nano_20k.json --top sipeed_tang_nano_20k \
        --device GW2AR-LV18QN88C8/I7 --vopt family=GW2A-18C \
        --vopt cst=sipeed_tang_nano_20k.cst \
        --timing-allow-fail --seed 2 --no-route \
        --placed-svg ../../../build/showcase/nextpnr/placed.svg

# Export square Matplotlib PNGs for social posts and routing review.
showcase-pngs: showcase-metrics showcase-nextpnr
    uv run --with matplotlib python -m tools.export_showcase_pngs \
        --nextpnr-svg build/showcase/nextpnr/placed.svg --output-dir build/showcase/png

# Refresh GEMM roofline + overlap counter measurements on the board.
showcase-benchmark:
    uv run python -m tools.showcase_metrics benchmark \
        --port {{ port }} --output analytics/measurements/gemm_sweep.json
