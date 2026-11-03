"""Shared Amaranth simulation helpers for hardware integration tests."""


def pack_int8(vals) -> int:
    """Pack list of int8 values into a little-endian 32-bit word."""
    word = 0
    for i, v in enumerate(vals):
        word |= (v & 0xFF) << (8 * i)
    return word


def to_signed8(val: int) -> int:
    """Convert unsigned 8-bit value to signed int8."""
    val = val & 0xFF
    return val - 256 if val >= 128 else val


async def dma_fill(ctx, dma_port, words):
    """Write a list of 32-bit words via a DMA port."""
    for addr, word in enumerate(words):
        ctx.set(dma_port.addr, addr)
        ctx.set(dma_port.data, word)
        ctx.set(dma_port.en, 1)
        await ctx.tick()
    ctx.set(dma_port.en, 0)
    await ctx.tick()


async def cfu_op(ctx, dut, funct3, funct7, in0, in1, *, max_cycles=5000):
    """Issue a CFU command and wait for the response."""
    ctx.set(dut.cmd_valid, 1)
    ctx.set(dut.cmd_function_id, {"funct3": funct3, "funct7": funct7})
    ctx.set(dut.cmd_in0, in0)
    ctx.set(dut.cmd_in1, in1)
    ctx.set(dut.rsp_ready, 1)
    single_cycle = ctx.get(dut.rsp_valid)  # responded in the handshake cycle
    await ctx.tick()
    if not single_cycle:
        # Like the CPU: the command was accepted (cmd_ready in WAIT_CMD), so
        # the payload is no longer valid while a multi-cycle op runs.
        # Scramble it so instructions that re-read funct7 / inputs after the
        # start cycle fail here instead of only on the SoC.
        ctx.set(dut.cmd_valid, 0)
        ctx.set(dut.cmd_function_id, {"funct3": funct3, "funct7": 0x55})
        ctx.set(dut.cmd_in0, 0xA5A5A5A5)
        ctx.set(dut.cmd_in1, 0x5A5A5A5A)

    for _ in range(max_cycles):
        if ctx.get(dut.rsp_valid):
            result = ctx.get(dut.rsp_out)
            ctx.set(dut.cmd_valid, 0)
            await ctx.tick()
            return result
        await ctx.tick()
    raise TimeoutError(f"CFU did not respond within {max_cycles} cycles")


async def write_per_channel_params(ctx, dut, biases, mults, shifts):
    """Write per-channel epilogue params via CFU config instructions."""
    for ch in range(len(biases)):
        await cfu_op(ctx, dut, funct3=2, funct7=0, in0=ch, in1=biases[ch])
        await cfu_op(ctx, dut, funct3=2, funct7=1, in0=ch, in1=mults[ch])
        await cfu_op(ctx, dut, funct3=2, funct7=2, in0=ch, in1=shifts[ch])


async def run_tile(ctx, dut, k, *, first, last):
    """Issue COMPUTE_START then COMPUTE_WAIT via CFU instructions."""
    flags = int(first) | (int(last) << 1)
    await cfu_op(ctx, dut, funct3=0, funct7=0, in0=flags, in1=k)
    await cfu_op(ctx, dut, funct3=1, funct7=0, in0=0, in1=0)
