# LiteX SoC

## SoC Simulator (litex_sim)

Test the SoC with CFU in simulation:
```sh
export PATH=~/Projects/oss-cad-suite:$PATH
uv run litex_sim --cpu-type vexriscv --cpu-variant full+cfu --cpu-cfu hardware/test_cfu_add.v --with-wishbone
```

More info: https://github.com/enjoy-digital/litex/wiki/SoC-Simulator

## Building the Bitstream

Build with CFU support for Sipeed Tang Nano 20K:
```sh
export PATH=~/Projects/oss-cad-suite:$PATH
uv run python -m litex_boards.targets.sipeed_tang_nano_20k --build --toolchain apicula --cpu-type vexriscv --cpu-variant full+cfu --cpu-cfu hardware/test_cfu_add.v
```

**Note**: The CFU Verilog module must be generated before building. See `hardware/test_cfu_add.v` and ensure it's a valid Verilog module named "Cfu".

## Flashing the Bitstream

```sh
uv run python -m litex_boards.targets.sipeed_tang_nano_20k --flash
```

## Serial Console

Connect and interact with the SoC:
```sh
uv run litex_term /dev/ttyUSB1
```

Test the CFU add instruction:
```
cfu_add 5 3
```

For more info on loading application code: https://github.com/enjoy-digital/litex/wiki/Load-Application-Code-To-CPU
