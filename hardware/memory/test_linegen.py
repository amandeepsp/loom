import numpy as np
import pytest
from amaranth.sim import Simulator

from hardware.memory.linegen import LineGen


def reference_lines(planes, oy, ox0, stride, pad_t, pad_l, zp, ow=None):
    """Lines for the tile of 8 output pixels starting at (oy, ox0); with a
    4-wide output (ow=4) the tile spans output rows oy and oy + 1."""
    c, ih, iw = planes.shape
    lines = []
    for cl in range(8):
        for ky in range(3):
            for kx in range(3):
                b = []
                for r in range(8):
                    if ow == 4:
                        oyr, oxr = oy + r // 4, r % 4
                    else:
                        oyr, oxr = oy, ox0 + r
                    iy = oyr * stride + ky - pad_t
                    ix = oxr * stride + kx - pad_l
                    b.append(int(planes[cl, iy, ix]) & 0xFF if 0 <= iy < ih and 0 <= ix < iw
                             else zp & 0xFF)
                lines.append(int.from_bytes(bytes(b), "little"))
    return lines


@pytest.mark.parametrize("ih,iw,stride,pad,tiles,ow", [
    (8, 8, 1, 1, [(0, 0), (3, 0), (7, 0)], None),
    (32, 32, 1, 1, [(0, 0), (0, 24), (5, 8), (31, 24)], None),
    (32, 32, 2, 1, [(0, 0), (0, 8), (15, 8), (7, 0)], None),
    (16, 16, 2, 0, [(0, 0), (7, 0)], None),
    (16, 16, 1, 0, [(0, 8), (15, 0)], None),
    # Two-row mode (4-wide outputs): tiles start at even output rows.
    (4, 4, 1, 1, [(0, 0), (2, 0)], 4),
    (8, 8, 2, 1, [(0, 0), (2, 0)], 4),
])
def test_linegen_matches_im2col(ih, iw, stride, pad, tiles, ow):
    rng = np.random.default_rng(ih + iw + stride)
    planes = rng.integers(-128, 128, size=(8, ih, iw), dtype=np.int8)
    zp = -7
    plane = ih * iw
    buf = planes.reshape(-1).view(np.uint8)
    words = [int.from_bytes(buf[i:i + 8].tobytes(), "little") for i in range(0, len(buf), 8)]
    dut = LineGen()

    async def bench(ctx):
        ctx.set(dut.iw, iw)
        ctx.set(dut.ih, ih)
        ctx.set(dut.plane, plane)
        ctx.set(dut.stride2, stride == 2)
        ctx.set(dut.tworow, ow == 4)
        ctx.set(dut.zp, zp & 0xFF)
        for i, w in enumerate(words):
            ctx.set(dut.dma_addr, i)
            ctx.set(dut.dma_data, w)
            ctx.set(dut.dma_en, 1)
            await ctx.tick()
        ctx.set(dut.dma_en, 0)
        for oy, ox0 in tiles:
            iy0 = oy * stride - pad
            ctx.set(dut.iy0, iy0)
            ctx.set(dut.row0, iy0 * iw)
            ctx.set(dut.ix0, ox0 * stride - pad)
            ctx.set(dut.start, 1)
            await ctx.tick()
            ctx.set(dut.start, 0)
            got = {}
            for _ in range(400):
                if ctx.get(dut.wr_en):
                    got[ctx.get(dut.wr_addr)] = ctx.get(dut.wr_data)
                if not ctx.get(dut.busy):
                    break
                await ctx.tick()
            want = reference_lines(planes, oy, ox0, stride, pad, pad, zp, ow=ow)
            assert sorted(got) == list(range(72)), (oy, ox0, sorted(got)[:5], len(got))
            bad = [i for i in range(72) if got[i] != want[i]]
            assert not bad, f"tile {(oy, ox0)} lines {bad[:6]}: got {got[bad[0]]:016x} want {want[bad[0]]:016x}"

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(bench)
    sim.run()
