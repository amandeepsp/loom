/* Depthwise / first-layer 3x3 im2col from a channel-major (CHW) source.
 *
 * Destination is loom's packed tile layout: for M tile t and K row kk, the
 * 8 bytes at dst + t * K * 8 + kk * 8 hold that row's input for the tile's
 * 8 output pixels. With a CHW source those 8 pixels are (nearly) contiguous
 * bytes of one input row, so each 4-pixel half comes from chw3x3_row()
 * (aligned loads + funnel shift / even-odd extraction) instead of strided
 * byte gathers. Padding uses the input zero point (ZP), already in the
 * LUT-mapped domain when the producer stored LUT-mapped values.
 *
 * Constants (-D): C, IH, IW, OH, OW, STRIDE (1|2), PAD_T, PAD_L (0|1), ZP,
 * PLANE (source bytes per channel), K (= 9 * C), SRC_ADDR, DST_ADDR,
 * optional CHANNEL_MAJOR (default 1).
 * Requires OW % 4 == 0, IW % 4 == 0, PLANE % 4 == 0, (OH * OW) % 8 == 0.
 */
#include "chw3x3.h"

/* K order: channel-major kk = c * 9 + tap (depthwise, the default), or
 * tap-major kk = tap * C + c (dense 3x3 conv, e.g. the first layer). */
#ifndef CHANNEL_MAJOR
#define CHANNEL_MAJOR 1
#endif
#if CHANNEL_MAJOR
#define KROW(c, tap) ((c) * 9 + (tap))
#else
#define KROW(c, tap) ((tap) * C + (c))
#endif

static inline void dwpack(const u8 *src, u8 *dst) {
    for (int c = 0; c < C; ++c) {
        const u8 *plane = src + c * PLANE;
        for (int oy = 0; oy < OH; ++oy) {
            for (int ox0 = 0; ox0 < OW; ox0 += 4) {
                int q = (oy * OW + ox0) >> 2; /* 4-pixel quad; two per M tile */
                u8 *out = dst + (q >> 1) * (K * 8) + (q & 1) * 4;
                for (int ky = 0; ky < 3; ++ky) {
                    u32 w0, w1, w2;
                    chw3x3_row(plane, oy, ox0, ky, &w0, &w1, &w2);
                    *(u32a *)(out + KROW(c, ky * 3 + 0) * 8) = w0;
                    *(u32a *)(out + KROW(c, ky * 3 + 1) * 8) = w1;
                    *(u32a *)(out + KROW(c, ky * 3 + 2) * 8) = w2;
                }
            }
        }
    }
}

#ifndef DWPACK_HOST_TEST
void loom_dwpack(void) { dwpack((const u8 *)SRC_ADDR, (u8 *)DST_ADDR); }
#else
void dwpack_host(const u8 *src, u8 *dst) { dwpack(src, dst); }
#endif
