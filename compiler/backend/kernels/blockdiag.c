/* Block-diagonal GEMM on the CPU (global average pool and similar).
 *
 * The plan layer is a GEMM over K = TAPS * C whose weight is nonzero only
 * where (k % C) == output channel, with tap-major K (k = t * C + c). For
 * each of M rows:
 *   acc   = sum_t lut(src[row * K + t * C + c]) * w[t][c]
 *   out   = clamp(RDBPOT(SRDHM(acc + bias[c], mult[c]), shift[c]) + OUT_ZP)
 * which is bit-identical to packing + the zero-skipping CFU GEMM, without
 * multiplying 255 of every 256 weights by zero. Compiled per layer with -D
 * constants into relocation-free RV32IM .text.
 */
#include <stdint.h>

typedef int8_t i8;
typedef int32_t i32;
typedef int64_t i64;
typedef uint32_t u32;

static inline i32 srdhm(i32 a, i32 b) {
    if (a == INT32_MIN && b == INT32_MIN) return INT32_MAX;
    i64 ab = (i64)a * (i64)b;
    i64 nudged = ab + (ab >= 0 ? ((i64)1 << 30) : (1 - ((i64)1 << 30)));
    i64 q = nudged >> 31;  /* divide by 2^31 truncating toward zero */
    if (nudged < 0 && (nudged & 0x7fffffff)) q += 1;
    return (i32)q;
}

static inline i32 rdbpot(i32 x, i32 exponent) {
    if (exponent == 0) return x;
    i32 mask = (i32)(((u32)1 << exponent) - 1);
    i32 remainder = x & mask;
    i32 threshold = (mask >> 1) + (x < 0);
    return (x >> exponent) + (remainder > threshold);
}

void loom_blockdiag(void) {
    const i8 *src = (const i8 *)SRC_ADDR;          /* [M][TAPS * C] */
    i8 *dst = (i8 *)DST_ADDR;                      /* [M][C] */
    const i8 *wgt = (const i8 *)WGT_ADDR;          /* [TAPS][C] */
    const i32 *bias = (const i32 *)BIAS_ADDR;
    const i32 *mult = (const i32 *)MULT_ADDR;
    const i32 *shift = (const i32 *)SHIFT_ADDR;
#if LUT_ADDR
    const i8 *lut = (const i8 *)LUT_ADDR + 128;   /* lut[x] for x in [-128, 127] */
#endif
    for (int row = 0; row < M; ++row) {
        const i8 *in = src + row * TAPS * C;
        for (int c = 0; c < C; ++c) {
            i32 acc = 0;
            for (int t = 0; t < TAPS; ++t) {
                i32 x = in[t * C + c];
#if LUT_ADDR
                x = lut[x];
#endif
                acc += x * wgt[t * C + c];
            }
            i32 v = rdbpot(srdhm(acc + bias[c], mult[c]), shift[c]) + OUT_ZP;
            dst[row * C + c] = (i8)(v < ACT_MIN ? ACT_MIN : v > ACT_MAX ? ACT_MAX : v);
        }
    }
}
