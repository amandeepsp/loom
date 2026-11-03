/* Shared 3x3 im2col word builder over a channel-major (CHW) source.
 *
 * chw3x3_row() returns, for one source plane, output row oy, 4-pixel quad
 * starting at output column ox0 and kernel row ky, the three tap words
 * (kx = 0, 1, 2): word byte r holds input pixel (oy*S + ky - PAD_T,
 * (ox0 + r)*S + kx - PAD_L), or the zero point when that is padding.
 *
 * Built from aligned word loads plus a byte funnel shift (stride 1) or
 * even/odd byte extraction (stride 2). Needs -D IH, IW, STRIDE (1|2),
 * PAD_T, PAD_L (0|1), ZP; IW % 4 == 0 and a word-aligned plane.
 */
#ifndef LOOM_CHW3X3_H
#define LOOM_CHW3X3_H

#include <stdint.h>

typedef uint8_t u8;
typedef uint32_t u32;
typedef uint32_t __attribute__((may_alias)) u32a;

#define ZPW ((u32)(u8)(ZP) * 0x01010101u)

static inline u32 even_bytes(u32 a, u32 b) {
    return (a & 0xffu) | ((a >> 8) & 0xff00u) | ((b & 0xffu) << 16) | ((b << 8) & 0xff000000u);
}

static inline u32 odd_bytes(u32 a, u32 b) {
    return ((a >> 8) & 0xffu) | ((a >> 16) & 0xff00u) | ((b << 8) & 0xff0000u) | (b & 0xff000000u);
}

static inline __attribute__((always_inline)) void chw3x3_row(
    const u8 *plane, int oy, int ox0, int ky, u32 *w0, u32 *w1, u32 *w2) {
    int iy = oy * STRIDE + ky - PAD_T;
    *w0 = *w1 = *w2 = ZPW;
    if ((unsigned)iy >= (unsigned)IH) return;
    const u32a *row = (const u32a *)(plane + iy * IW);
#if STRIDE == 1
    int i = ox0 >> 2;
    u32 cur = row[i];
    u32 next = ox0 + 4 < IW ? row[i + 1] : ZPW;
#if PAD_L == 1
    u32 prev = i > 0 ? row[i - 1] : ZPW;
    *w0 = (prev >> 24) | (cur << 8);
    *w1 = cur;
    *w2 = (cur >> 8) | (next << 24);
#else
    *w0 = cur;
    *w1 = (cur >> 8) | (next << 24);
    *w2 = (cur >> 16) | (next << 16);
#endif
#else /* STRIDE == 2: input bytes 2*ox0 .. 2*ox0 + 7 live in words i, i + 1 */
    int i = ox0 >> 1;
    u32 a = row[i], b = row[i + 1];
    u32 e = even_bytes(a, b), o = odd_bytes(a, b);
#if PAD_L == 1
    u32 prev = i > 0 ? row[i - 1] : ZPW;
    *w0 = (prev >> 24) | (o << 8);
    *w1 = e;
    *w2 = o;
#else
    u32 next = 2 * ox0 + 8 < IW ? row[i + 2] : ZPW;
    *w0 = e;
    *w1 = o;
    *w2 = (e >> 8) | (next << 24);
#endif
#endif
}

/* Both 4-pixel halves of an 8-pixel tile on one output row (OW % 8 == 0):
 * a* for pixels ox0..ox0+3, b* for ox0+4..ox0+7. The halves share source
 * words, so this loads 4 (stride 1) or 5 (stride 2) words instead of 6. */
static inline __attribute__((always_inline)) void chw3x3_row8(
    const u8 *plane, int oy, int ox0, int ky,
    u32 *a0, u32 *a1, u32 *a2, u32 *b0, u32 *b1, u32 *b2) {
    int iy = oy * STRIDE + ky - PAD_T;
    if ((unsigned)iy >= (unsigned)IH) {
        *a0 = *a1 = *a2 = *b0 = *b1 = *b2 = ZPW;
        return;
    }
    const u32a *row = (const u32a *)(plane + iy * IW);
#if STRIDE == 1
    int i = ox0 >> 2;
    u32 c0 = row[i], c1 = row[i + 1];
    u32 next = ox0 + 8 < IW ? row[i + 2] : ZPW;
#if PAD_L == 1
    u32 prev = i > 0 ? row[i - 1] : ZPW;
    *a0 = (prev >> 24) | (c0 << 8);
    *a1 = c0;
    *a2 = (c0 >> 8) | (c1 << 24);
    *b0 = (c0 >> 24) | (c1 << 8);
    *b1 = c1;
    *b2 = (c1 >> 8) | (next << 24);
#else
    *a0 = c0;
    *a1 = (c0 >> 8) | (c1 << 24);
    *a2 = (c0 >> 16) | (c1 << 16);
    *b0 = c1;
    *b1 = (c1 >> 8) | (next << 24);
    *b2 = (c1 >> 16) | (next << 16);
#endif
#else /* STRIDE == 2: bytes 2*ox0 .. 2*ox0 + 15 in words i .. i + 3 */
    int i = ox0 >> 1;
    u32 w0 = row[i], w1 = row[i + 1], w2 = row[i + 2], w3 = row[i + 3];
    u32 ea = even_bytes(w0, w1), oa = odd_bytes(w0, w1);
    u32 eb = even_bytes(w2, w3), ob = odd_bytes(w2, w3);
#if PAD_L == 1
    u32 prev = i > 0 ? row[i - 1] : ZPW;
    *a0 = (prev >> 24) | (oa << 8);
    *a1 = ea;
    *a2 = oa;
    *b0 = (oa >> 24) | (ob << 8);
    *b1 = eb;
    *b2 = ob;
#else
    u32 next = 2 * ox0 + 16 < IW ? row[i + 4] : ZPW;
    *a0 = ea;
    *a1 = oa;
    *a2 = (ea >> 8) | (eb << 24);
    *b0 = eb;
    *b1 = ob;
    *b2 = (eb >> 8) | (next << 24);
#endif
#endif
}

#endif
