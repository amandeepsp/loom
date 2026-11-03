"""dwpack.c (CHW depthwise im2col) vs a direct numpy model, compiled for the host."""

import ctypes
import shutil
import subprocess

import numpy as np
import pytest

from compiler.backend.cpu import KERNEL_DIR

CASES = [
    # (C, IH, IW, stride, pad_t, pad_l)
    (8, 32, 32, 1, 1, 1),
    (16, 32, 32, 2, 1, 1),
    (32, 16, 16, 1, 1, 1),
    (32, 16, 16, 2, 0, 0),
    (64, 8, 8, 2, 1, 1),
    (128, 4, 4, 1, 1, 1),
    (8, 8, 8, 1, 0, 0),
]
TAP_MAJOR_CASES = [(3, 32, 32, 1, 1, 1), (3, 16, 16, 2, 1, 1)]


def reference(img, oh, ow, stride, pad_t, pad_l, zp, tile=8, channel_major=True):
    c, ih, iw = img.shape
    k = 9 * c
    a = np.full((oh * ow, k), zp, np.int8)
    for m in range(oh * ow):
        oy, ox = divmod(m, ow)
        for ky in range(3):
            for kx in range(3):
                y, x = oy * stride + ky - pad_t, ox * stride + kx - pad_l
                if 0 <= y < ih and 0 <= x < iw:
                    tap = ky * 3 + kx
                    rows = np.arange(c) * 9 + tap if channel_major else tap * c + np.arange(c)
                    a[m, rows] = img[:, y, x]
    return a.reshape(-1, tile, k).transpose(0, 2, 1).reshape(-1)


@pytest.mark.skipif(shutil.which("clang") is None, reason="needs clang")
@pytest.mark.parametrize("c,ih,iw,stride,pad_t,pad_l,channel_major",
                         [(*x, True) for x in CASES] + [(*x, False) for x in TAP_MAJOR_CASES])
def test_dwpack_matches_reference(tmp_path, c, ih, iw, stride, pad_t, pad_l, channel_major):
    oh = (ih + 2 * pad_t - 3) // stride + 1 if pad_t else (ih - 3 + 1 + stride - 1) // stride
    ow = (iw + 2 * pad_l - 3) // stride + 1 if pad_l else (iw - 3 + 1 + stride - 1) // stride
    if pad_l == 0 and stride == 1:
        oh, ow = ih, iw  # SAME-style (0, 2) padding, all right/bottom
    if pad_l == 0 and stride == 2:
        oh, ow = ih // 2, iw // 2  # SAME-style (0, 1) padding
    zp = -7
    defines = dict(C=c, IH=ih, IW=iw, OH=oh, OW=ow, STRIDE=stride, PAD_T=pad_t, PAD_L=pad_l,
                   ZP=zp, PLANE=ih * iw, K=9 * c, DWPACK_HOST_TEST=1,
                   CHANNEL_MAJOR=int(channel_major))
    lib = tmp_path / "dwpack.so"
    subprocess.run(["clang", "-O2", "-shared", "-fPIC", *[f"-D{k}={v}" for k, v in defines.items()],
                    str(KERNEL_DIR / "dwpack.c"), "-o", str(lib)], check=True)
    fn = ctypes.CDLL(str(lib)).dwpack_host
    rng = np.random.default_rng(c * ih + stride)
    img = rng.integers(-128, 128, size=(c, ih, iw), dtype=np.int8)
    want = reference(img, oh, ow, stride, pad_t, pad_l, zp, channel_major=channel_major)
    src = np.ascontiguousarray(img).copy()
    dst = np.zeros_like(want)
    fn(src.ctypes.data_as(ctypes.c_void_p), dst.ctypes.data_as(ctypes.c_void_p))
    assert np.array_equal(dst, want), f"{int((dst != want).sum())} wrong bytes"
