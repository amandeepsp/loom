import numpy as np
import pytest
from amaranth.sim import Simulator

from hardware.memory.linegen import LineGen


def reference_lines(planes, oy, ox0, stride, pad_t, pad_l, zp, ow=None, dense=False):
    """Lines for the tile of 8 output pixels m0 .. m0 + 7 (m0 = oy * ow + ox0
    for ow < 8; one output row starting at (oy, ox0) otherwise).
    Channel-major lines (c * 9 + tap) for the 8 planes, or with `dense`
    tap-major lines (tap * C + c) for all C planes."""
    c, ih, iw = planes.shape
    nch = c if dense else 8
    lines = [0] * (9 * nch)
    for cl in range(nch):
        for ky in range(3):
            for kx in range(3):
                b = []
                for r in range(8):
                    if ow is not None and ow < 8:
                        oyr, oxr = oy + r // ow, r % ow
                    else:
                        oyr, oxr = oy, ox0 + r
                    iy = oyr * stride + ky - pad_t
                    ix = oxr * stride + kx - pad_l
                    b.append(int(planes[cl, iy, ix]) & 0xFF if 0 <= iy < ih and 0 <= ix < iw
                             else zp & 0xFF)
                tap = ky * 3 + kx
                line = tap * nch + cl if dense else cl * 9 + tap
                lines[line] = int.from_bytes(bytes(b), "little")
    return lines


@pytest.mark.parametrize("ih,iw,stride,pad,tiles,ow,dense", [
    (8, 8, 1, 1, [(0, 0), (3, 0), (7, 0)], None, 0),
    (32, 32, 1, 1, [(0, 0), (0, 24), (5, 8), (31, 24)], None, 0),
    (32, 32, 2, 1, [(0, 0), (0, 8), (15, 8), (7, 0)], None, 0),
    (16, 16, 2, 0, [(0, 0), (7, 0)], None, 0),
    (16, 16, 1, 0, [(0, 8), (15, 0)], None, 0),
    # Two-row mode (4-wide outputs): tiles start at even output rows.
    (4, 4, 1, 1, [(0, 0), (2, 0)], 4, 0),
    (8, 8, 2, 1, [(0, 0), (2, 0)], 4, 0),
    # 2-wide outputs: a tile is 4 output rows (rows past M are don't-care).
    (2, 2, 1, 1, [(0, 0)], 2, 0),
    (4, 4, 2, 1, [(0, 0)], 2, 0),
    # Dense 3x3 conv over 3 planes, tap-major lines (the first layer).
    (32, 32, 1, 1, [(0, 0), (5, 16), (31, 24)], None, 3),
])
def test_linegen_matches_im2col(ih, iw, stride, pad, tiles, ow, dense):
    rng = np.random.default_rng(ih + iw + stride)
    planes = rng.integers(-128, 128, size=(dense or 8, ih, iw), dtype=np.int8)
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
        wide = ow is None or ow >= 8
        ctx.set(dut.rowwise, not wide)
        ctx.set(dut.reads_log2, (1 if stride == 2 else 0) if wide else (1 if ow == 4 else 2))
        ctx.set(dut.dense, bool(dense))
        ctx.set(dut.nch, dense if dense else 0)
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
            for _ in range(1200):
                if ctx.get(dut.wr_en):
                    got[ctx.get(dut.wr_addr)] = ctx.get(dut.wr_data)
                if not ctx.get(dut.busy):
                    break
                await ctx.tick()
            want = reference_lines(planes, oy, ox0, stride, pad, pad, zp, ow=ow, dense=bool(dense))
            n_lines = len(want)
            assert sorted(got) == list(range(n_lines)), (oy, ox0, sorted(got)[:5], len(got))
            # 2-wide tiles: output rows past M (r >= 4 when M = 4) are don't-care.
            mask = (1 << 32) - 1 if ow == 2 and oy + 2 >= ih // stride else (1 << 64) - 1
            bad = [i for i in range(n_lines) if (got[i] ^ want[i]) & mask]
            assert not bad, f"tile {(oy, ox0)} lines {bad[:6]}: got {got[bad[0]]:016x} want {want[bad[0]]:016x}"

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(bench)
    sim.run()
