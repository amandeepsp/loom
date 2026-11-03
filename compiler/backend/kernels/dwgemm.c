/* Fused 3x3 conv layer (depthwise, or the dense first layer) on the CFU,
 * with the CPU streaming im2col lines straight into the activation
 * scratchpad (WRITE_ACT, `act_write` CFU) instead of through a packed DRAM
 * buffer and the activation DMA.
 *
 * For each group of 8 output channels: upload epilogue params, DMA the
 * group's weights once, then for each M tile of 8 output pixels write the
 * KG activation lines (8 bytes = the tile's 8 pixels for one K row), run
 * one compute (keeping the weight bank after the first tile), and store the
 * results with column reads (rows 0..3 / 4..7 of a column per READ_RESULT;
 * already mapped through the next layer's LUT by the CFU when CFU_LUT_ADDR).
 *
 * Output element (m, n) goes to OUT_ADDR + m * MSTRIDE + n * CSTRIDE with
 * the 8 rows of a column contiguous: the next layer's packed layout
 * (MSTRIDE = N, CSTRIDE = 8) or channel-major (MSTRIDE = 1, CSTRIDE = plane).
 *
 * Constants (-D): source/shape as chw3x3.h plus C, OH, OW, PLANE, SRC_ADDR,
 * CHANNEL_MAJOR (1: depthwise, group g reads channels 8g..8g+7, K row
 * c_local * 9 + tap; 0: dense, one group, K row tap * C + c), N, KG,
 * PARAM_ROWS (8, or 1 with per-column params), WGT_ADDR, BIAS_ADDR, MULT_ADDR, SHIFT_ADDR, OUT_ZP, ACT_MIN, ACT_MAX,
 * CFU_LUT_ADDR, OUT_ADDR, MSTRIDE, CSTRIDE, and the weight DMA CSRs
 * WGT_CSR_BASE, WGT_CSR_LENGTH, WGT_CSR_ENABLE, WGT_CSR_LOOP, WGT_CSR_DRAINED.
 */
#include "chw3x3.h"

typedef int32_t i32;

#ifdef DWGEMM_HOST_TEST
/* Host harness: CFU ops and MMIO go to functions the test provides. */
void host_cfu(int f3, int f7, u32 a, u32 b);
u32 host_cfu_rd(int f3, int f7, u32 a);
volatile u32 *host_mmio(unsigned long addr);
extern u8 host_src[], host_out[];
extern int32_t host_params[];
#define CFU_W(f3, f7, a, b) host_cfu(f3, f7, (u32)(a), (u32)(b))
#define CFU_RD(f3, f7, a) host_cfu_rd(f3, f7, (u32)(a))
#define MMIO(addr) (*host_mmio((unsigned long)(addr)))
#else
#define CFU_W(f3, f7, a, b) \
    asm volatile(".insn r 0x0b, " #f3 ", " #f7 ", x0, %0, %1" ::"r"(a), "r"(b) : "memory")

#define CFU_RD(f3, f7, a)                                                              \
    ({                                                                                 \
        u32 _r;                                                                        \
        asm volatile(".insn r 0x0b, " #f3 ", " #f7 ", %0, %1, x0" : "=r"(_r) : "r"(a)); \
        _r;                                                                            \
    })

#define MMIO(addr) (*(volatile u32 *)(addr))
#endif


/* Gather mode: input byte for output pixel m, tap; ZP for padding or m >= M.
 * Fully unrolled per tile, so single-tile layers fold every bound check and
 * offset to a constant. */
static inline __attribute__((always_inline)) u8 gather_byte(const u8 *plane, int m, int tap) {
    int oy = m / OW, ox = m % OW, ky = tap / 3, kx = tap % 3;
    int iy = oy * STRIDE + ky - PAD_T, ix = ox * STRIDE + kx - PAD_L;
    if (m >= OH * OW || (unsigned)iy >= (unsigned)IH || (unsigned)ix >= (unsigned)IW)
        return (u8)(ZP);
    return plane[iy * IW + ix];
}

static inline __attribute__((always_inline)) void put_line(int line, u32 lo, u32 hi) {
#if !CHANNEL_MAJOR
    CFU_W(6, 1, line, 0); /* tap-major lines are not written in order */
#else
    (void)line;
#endif
    CFU_W(6, 0, lo, hi);
}

void loom_dwgemm(void) {
    const u8 *src = (const u8 *)SRC_ADDR;
    const i32 *bias = (const i32 *)BIAS_ADDR;
    const i32 *mult = (const i32 *)MULT_ADDR;
    const i32 *shift = (const i32 *)SHIFT_ADDR;

    CFU_W(3, 0, 0, OUT_ZP);
    CFU_W(3, 1, 0, ACT_MIN);
    CFU_W(3, 2, 0, ACT_MAX);
#if CFU_LUT_ADDR
    for (int i = 0; i < 64; ++i) CFU_W(3, 3, i, ((const u32a *)CFU_LUT_ADDR)[i]);
    CFU_W(3, 4, 0, 1);
#else
    CFU_W(3, 4, 0, 0);
#endif

    for (int g = 0; g < N / 8; ++g) {
        /* Epilogue params: one entry per (row, col) result slot. */
        for (int col = 0; col < 8; ++col) {
            int n = g * 8 + col;
            i32 b = bias[n], mu = mult[n], sh = shift[n];
            for (int r = 0; r < PARAM_ROWS; ++r) { /* 1 on col_params CFUs */
                CFU_W(2, 0, r * 8 + col, b);
                CFU_W(2, 1, r * 8 + col, mu);
                CFU_W(2, 2, r * 8 + col, sh);
            }
        }
        /* This group's weights, once; they stay in the bank for every tile. */
        MMIO(WGT_CSR_ENABLE) = 0;
        MMIO(WGT_CSR_LOOP) = 0;
        MMIO(WGT_CSR_BASE) = WGT_ADDR + g * (KG * 8);
        MMIO(WGT_CSR_LENGTH) = KG * 8;
        MMIO(WGT_CSR_ENABLE) = 1;
        while (!MMIO(WGT_CSR_DRAINED)) {
        }

#ifndef GATHER
#define GATHER 0
#endif
        /* Software pipeline: tile t's lines go into the fill bank while
         * tile t - 1 computes on the other bank. */
        const int T = (OH * OW + 7) / 8;
        for (int t = 0; t <= T; ++t) {
            if (t > 0) {
                /* Tile t - 1: first | last, keep the weight bank after the
                 * group's first tile. */
                CFU_W(0, 0, 3 | (t > 1 ? 4 : 0), KG);
                /* The start pulse is registered a cycle late: give it time to
                 * reach PRIME (bank swap) before the next WRITE_ACT, and keep
                 * COMPUTE_WAIT off the previous tile's latched done. */
#ifndef DWGEMM_HOST_TEST
                asm volatile("nop\n\tnop\n\tnop\n\tnop");
#endif
            }
            if (t < T) {
            int m0 = t * 8;
#if GATHER
            CFU_W(6, 1, 0, 0);
            for (int cl = 0; cl < 8; ++cl) {
                const u8 *plane = src + (g * 8 + cl) * PLANE;
#pragma clang loop unroll(full)
                for (int tap = 0; tap < 9; ++tap) {
                    u32 lo = 0, hi = 0;
#pragma clang loop unroll(full)
                    for (int r = 0; r < 4; ++r) {
                        lo |= (u32)gather_byte(plane, m0 + r, tap) << (8 * r);
                        hi |= (u32)gather_byte(plane, m0 + 4 + r, tap) << (8 * r);
                    }
                    CFU_W(6, 0, lo, hi);
                }
            }
#else
            int oy0 = m0 / OW, ox0 = m0 % OW;            /* rows 0..3 */
            int oy1 = (m0 + 4) / OW, ox1 = (m0 + 4) % OW; /* rows 4..7 */
            CFU_W(6, 1, 0, 0);
#if CHANNEL_MAJOR
            for (int cl = 0; cl < 8; ++cl) {
                const u8 *plane = src + (g * 8 + cl) * PLANE;
#else
            for (int cl = 0; cl < C; ++cl) {
                const u8 *plane = src + cl * PLANE;
#endif
                for (int ky = 0; ky < 3; ++ky) {
                    u32 a0, a1, a2, b0, b1, b2;
                    chw3x3_row(plane, oy0, ox0, ky, &a0, &a1, &a2);
                    chw3x3_row(plane, oy1, ox1, ky, &b0, &b1, &b2);
#if CHANNEL_MAJOR
#define LINE(kx) (cl * 9 + ky * 3 + (kx))
#else
#define LINE(kx) ((ky * 3 + (kx)) * C + cl)
#endif
                    put_line(LINE(0), a0, b0);
                    put_line(LINE(1), a1, b1);
                    put_line(LINE(2), a2, b2);
#undef LINE
                }
            }
#endif /* GATHER */
            }
            if (t == 0) continue;
            (void)CFU_RD(1, 0, 0);

            u8 *dst = (u8 *)OUT_ADDR + (t - 1) * 8 * MSTRIDE + g * 8 * CSTRIDE;
            for (int col = 0; col < 8; ++col) {
                u32a *p = (u32a *)(dst + col * CSTRIDE);
                p[0] = CFU_RD(4, 2, col);
                p[1] = CFU_RD(4, 2, 32 + col);
            }
        }
    }
}
