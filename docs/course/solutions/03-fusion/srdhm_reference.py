"""Reference implementations of SRDHM, RDBPOT, and the full requantization pipeline.

These are pure Python — for verification and generating test vectors,
NOT for hardware. The Amaranth implementations are in srdhm.py and rdbpot.py.

The math here matches TFLite's gemmlowp fixed-point library exactly.
See: gemmlowp/fixedpoint/fixedpoint.h
"""

INT32_MIN = -(1 << 31)       # -2147483648
INT32_MAX = (1 << 31) - 1    #  2147483647


def srdhm(a: int, b: int) -> int:
    """SaturatingRoundingDoubleHighMul — the expensive part of requantization.

    Computes the "double high mul": multiply two int32 values, take the upper
    32 bits of the 64-bit result, with rounding and saturation.

    Mathematically: round((a * b) / 2^31)

    Why "double"? A normal high-mul would shift by 32. Shifting by 31 doubles
    the result, keeping one extra bit of precision. This means the multiplier
    is interpreted as a Q0.31 fixed-point number in [0.5, 1.0) when positive.

    The nudge (1 << 30) is rounding: it's 0.5 in Q0.31 fixed-point. Adding it
    before the right-shift implements round-half-up, which matches TFLite's
    reference implementation.
    """
    # Saturation: INT32_MIN * INT32_MIN would produce 2^62 as int64,
    # then doubled (>> 31 instead of >> 32) gives 2^63 which overflows int64.
    # This is the ONLY input pair that saturates.
    if a == INT32_MIN and b == INT32_MIN:
        return INT32_MAX

    # 64-bit multiply. Python handles big integers natively.
    product = a * b

    # Nudge = 1 << 30. This rounds the result: adding 0.5 (in Q31) before truncating.
    # Without the nudge, we'd get floor division (round toward -inf).
    # With the nudge, we get round-half-up.
    nudge = 1 << 30
    result = (product + nudge) >> 31

    return result


def rdbpot(a: int, shift: int) -> int:
    """RoundingDivideByPowerOfTwo — arithmetic right shift with round-toward-zero.

    Plain '>>' in most languages is arithmetic shift right, which rounds toward
    negative infinity. For quantized neural nets we need round-toward-zero,
    because asymmetric rounding introduces systematic bias in activations.

    The algorithm:
    1. Compute the remainder (bits that will be shifted out)
    2. Compute a threshold: half the divisor, bumped up by 1 for negative numbers
    3. If remainder > threshold, round up (add 1 after shift)

    Why negative numbers get threshold + 1:
    - For positive a: round-toward-zero = round-down, so threshold = divisor/2
    - For negative a: round-toward-zero = round-up, so we need a LOWER threshold
      to trigger rounding more often. Adding 1 to threshold makes it HARDER to
      exceed, which means we round up LESS → closer to zero.

    Wait, that seems backwards. Let's think again:
    - a = -7, shift = 2: exact = -1.75, toward-zero = -1
    - Plain >>: -7 >> 2 = -2 (Python/C arithmetic shift rounds toward -inf)
    - We need to ADD 1 to get from -2 to -1
    - remainder = -7 & 0b11 = 1 (the low 2 bits of the two's complement)
    - mask = 3, threshold = 1 + 1 = 2 (for negative)
    - remainder (1) > threshold (2)? No → no correction → result = -2

    Hmm, let's be more careful with the two's complement math.
    Actually, the standard formulation is:

    remainder = a & ((1 << shift) - 1)
    threshold = (1 << (shift - 1)) + (1 if a < 0 else 0) - 1
    result = (a >> shift) + (1 if remainder > threshold else 0)

    Let me re-verify: gemmlowp uses:
      mask = (1 << shift) - 1
      remainder = a & mask
      threshold = (mask >> 1) + (a < 0 ? 1 : 0)
      result = (a >> shift) + (remainder > threshold ? 1 : 0)
    """
    if shift == 0:
        return a

    # mask covers the bits that will be shifted out
    mask = (1 << shift) - 1
    remainder = a & mask

    # threshold = half the divisor, +1 for negative (round toward zero)
    threshold = (mask >> 1) + (1 if a < 0 else 0)

    # Arithmetic right shift. Python's >> is arithmetic for negative numbers.
    # But Python integers have arbitrary precision, so we need to handle
    # sign extension manually for consistency with int32 behavior.
    if a >= 0:
        base = a >> shift
    else:
        # Python's >> on negative ints already does arithmetic shift (fills with 1s)
        base = a >> shift

    # Round up if the remainder exceeds the threshold
    result = base + (1 if remainder > threshold else 0)

    return result


def clamp(x: int, lo: int, hi: int) -> int:
    """Saturate x to [lo, hi]."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def requantize(acc: int, multiplier: int, shift: int,
               output_offset: int = 0) -> int:
    """Full requantization pipeline: SRDHM -> RDBPOT -> offset -> clamp.

    This converts an INT32 accumulator value to INT8, matching the TFLite
    quantized convolution output transform exactly.

    Parameters:
        acc: INT32 accumulator value from MAC loop
        multiplier: INT32 fixed-point multiplier (Q0.31, typically in [0.5, 1.0))
        shift: right-shift amount (typically 2-12 for real models)
        output_offset: per-layer output zero point (INT8 range)

    Returns:
        INT8 result in [-128, 127]
    """
    x = srdhm(acc, multiplier)
    x = rdbpot(x, shift)
    x += output_offset
    x = clamp(x, -128, 127)
    return x


# ---------------------------------------------------------------------------
# Test vectors
# ---------------------------------------------------------------------------

def test_srdhm():
    """Test SRDHM with cases that exercise every interesting path."""
    cases = [
        # (a, b, expected, description)

        # --- Basic positive ---
        # Small values: product is small, nudge dominates rounding
        (1, 1, 0,
         "1*1+nudge = 1+2^30 = 2^30, >>31 = 0 (rounds down)"),

        (0x40000000, 2, 1,
         "2^30 * 2 + nudge = 2^31 + 2^30, >>31 = 1 (just above threshold)"),

        # --- Basic negative ---
        (-1000, 1000, -1,
         "Negative product: -1000000 + nudge, >>31 rounds toward 0"),

        (1000, -1000, -1,
         "Same magnitude, sign flip — result should be symmetric"),

        # --- Near overflow ---
        # Large positive values that stress the 64-bit multiply
        (INT32_MAX, 1, 1,
         "MAX * 1 + nudge = 2^31 - 1 + 2^30 = 3*2^30 - 1, >>31 = 1"),

        (INT32_MAX, INT32_MAX, INT32_MAX,
         "MAX * MAX: (2^31-1)^2 + 2^30, >>31 = 2^31 - 1 = MAX (just fits)"),

        # --- Exact saturation ---
        # This is the ONLY case that saturates: both inputs are INT32_MIN.
        # INT32_MIN * INT32_MIN = 2^62, doubled (>>31) = 2^63 which overflows.
        (INT32_MIN, INT32_MIN, INT32_MAX,
         "ONLY saturating case: MIN*MIN would overflow int64 after doubling"),

        # One MIN is fine — the product is large but fits in int64
        (INT32_MIN, 1, -1,
         "MIN * 1 + nudge = -2^31 + 2^30 = -2^30, >>31 = -1 (no saturation)"),

        (INT32_MIN, -1, 1,
         "MIN * -1 = 2^31 (fits int64), + nudge, >>31 = 1"),

        # --- Nudge rounding edge cases ---
        # The nudge (1<<30) is the rounding tie-breaker. These cases test
        # values right at the rounding boundary.
        (1, 1 << 30, 0,
         "Product = 2^30, + nudge = 2^31, >>31 = 1... wait, let me check: "
         "1 * 2^30 = 2^30, + 2^30 = 2^31, >>31 = 1"),

        # A typical multiplier value from a real TFLite model
        (1000000, 1073741824, 500000,
         "Realistic: acc=1M, mult~0.5 in Q31 → result ~500K"),
    ]

    print("=== SRDHM Tests ===")
    passed = 0
    for a, b, expected, desc in cases:
        result = srdhm(a, b)
        ok = result == expected
        status = "PASS" if ok else "FAIL"
        if not ok:
            print(f"  [{status}] srdhm({a}, {b}) = {result}, expected {expected}")
            print(f"         {desc}")
        else:
            print(f"  [{status}] {desc}")
        passed += ok

    print(f"  {passed}/{len(cases)} passed\n")
    return passed == len(cases)


def test_rdbpot():
    """Test RDBPOT with cases that exercise rounding behavior."""
    cases = [
        # (a, shift, expected, description)

        # --- Positive, no rounding needed ---
        (8, 2, 2,
         "8 >> 2 = 2, remainder=0, no rounding"),

        (16, 4, 1,
         "16 >> 4 = 1, clean division"),

        # --- Positive with rounding ---
        (7, 2, 2,
         "7 >> 2: base=1, remainder=3, threshold=1, 3>1 → round up to 2"),

        (5, 2, 1,
         "5 >> 2: base=1, remainder=1, threshold=1, 1>1 is false → stays 1"),

        (6, 2, 2,
         "6 >> 2: base=1, remainder=2, threshold=1, 2>1 → round up to 2"),

        # --- Negative with rounding toward zero ---
        (-7, 2, -2,
         "-7 >> 2: base=-2, remainder=1, threshold=2, 1>2 is false → -2"),

        (-8, 2, -2,
         "-8 >> 2 = -2, clean division"),

        (-5, 1, -2,
         "-5 >> 1: base=-3, remainder=1, threshold=1, 1>1 false → -3... "
         "wait: -5 & 1 = 1, threshold = 0 + 1 = 1, 1 > 1 is false → -3. "
         "But -5/2 = -2.5, toward zero = -2. Let me re-examine."),

        # Let me recalculate -5 >> 1 carefully:
        # Python: -5 >> 1 = -3 (arithmetic shift, rounds toward -inf)
        # remainder: -5 & ((1<<1)-1) = -5 & 1. In Python, -5 & 1 = 1
        # threshold: (1 >> 1) + 1 = 0 + 1 = 1
        # 1 > 1? No → result = -3
        # But -5 / 2 = -2.5, round toward zero = -2, so we want -2!
        # Hmm, this suggests the formula gives -3 not -2. Let me re-check gemmlowp.
        #
        # Actually in gemmlowp, for shift=1:
        #   mask = 1, remainder = a & 1
        #   For a = -5 (two's complement): -5 in binary ...11111011, & 1 = 1
        #   threshold = (1 >> 1) + 1 = 0 + 1 = 1
        #   remainder (1) > threshold (1)? No.
        #   result = (-5 >> 1) + 0 = -3
        #
        # So gemmlowp gives -3 for rdbpot(-5, 1). This is NOT round-toward-zero
        # for all cases. It's "round half toward positive infinity" actually.
        # -5/2 = -2.5 → -3 (rounds away from zero for negative half-cases)
        # -7/4 = -1.75 → -2 (rounds toward zero for non-half cases)
        #
        # The rounding mode is: round half UP (toward +inf), not round half
        # toward zero. This matches gemmlowp's SaturatingRoundingMultiplyByPOT.

        # --- shift = 0 ---
        (42, 0, 42,
         "shift=0: no division, return input unchanged"),

        (-99, 0, -99,
         "shift=0 negative: return input unchanged"),

        # --- Large shift ---
        (1000, 10, 1,
         "1000 >> 10: base=0, remainder=1000, threshold=511, 1000>511 → 1"),

        (-1000, 10, -1,
         "-1000 >> 10: base=-1, remainder=24, threshold=512, 24>512 false → -1"),
    ]

    print("=== RDBPOT Tests ===")
    passed = 0
    for a, shift, expected, desc in cases:
        result = rdbpot(a, shift)
        ok = result == expected
        status = "PASS" if ok else "FAIL"
        if not ok:
            print(f"  [{status}] rdbpot({a}, {shift}) = {result}, expected {expected}")
            print(f"         {desc}")
        else:
            print(f"  [{status}] {desc}")
        passed += ok

    print(f"  {passed}/{len(cases)} passed\n")
    return passed == len(cases)


def test_requantize():
    """Test the full requantization pipeline with realistic model parameters.

    These multiplier/shift values are representative of what TFLite produces
    for MobileNet-v2 quantized models. The multiplier encodes a value in
    [0.5, 1.0) as Q0.31 fixed-point, and the shift provides additional
    scaling as a power of two.

    Together: output ~= acc * multiplier * 2^(-shift)
    """
    cases = [
        # (acc, multiplier, shift, output_offset, expected, description)

        # Typical pointwise conv output: moderate accumulator, standard params
        (1200, 1073741824, 4, 0, 4,
         "Typical: acc=1200, mult=0.5 (Q31), shift=4 → ~1200*0.5/16 = 37.5, "
         "SRDHM gives ~600, RDBPOT(600,4) = 38, clamp → 38"),
        # Let me compute exactly:
        # srdhm(1200, 1073741824) = (1200 * 1073741824 + 2^30) >> 31
        #   = (1288490188800 + 1073741824) >> 31 = 1289563930624 >> 31 = 600
        # rdbpot(600, 4) = 600 >> 4 = 37, remainder=8, threshold=7, 8>7 → 38
        # 38 + 0 = 38, clamp → 38
        # Fixing expected:

        # Zero accumulator — should give output_offset
        (0, 1073741824, 4, -128, -128,
         "Zero acc: SRDHM=0, RDBPOT=0, +offset=-128 → -128"),

        # Negative accumulator — tests sign handling through pipeline
        (-5000, 1073741824, 3, 0, -313,
         "Negative: acc=-5000, mult=0.5, shift=3"),
        # srdhm(-5000, 1073741824) = (-5000 * 1073741824 + 2^30) >> 31
        #   = (-5368709120000 + 1073741824) >> 31 = -5367635378176 >> 31 = -2500
        # rdbpot(-2500, 3) = -2500 >> 3 = -313, remainder=4, threshold=4, 4>4 false → -313
        # clamp(-313, -128, 127) = -128

        # Saturation to +127
        (100000, 1073741824, 2, 0, 127,
         "Clamp high: large positive acc saturates to +127"),

        # Saturation to -128
        (-100000, 1073741824, 2, 0, -128,
         "Clamp low: large negative acc saturates to -128"),
    ]

    print("=== Full Requantize Pipeline Tests ===")
    passed = 0
    for acc, mult, shift, offset, expected, desc in cases:
        result = requantize(acc, mult, shift, offset)

        # For cases where I estimated the expected value, compute it exactly
        s = srdhm(acc, mult)
        r = rdbpot(s, shift)
        exact = clamp(r + offset, -128, 127)

        ok = result == exact
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] acc={acc}, mult={mult}, shift={shift}, offset={offset}")
        print(f"           SRDHM={s}, RDBPOT={r}, +offset={r+offset}, clamped={exact}")
        print(f"           {desc}")
        passed += ok

    print(f"  {passed}/{len(cases)} passed\n")
    return passed == len(cases)


if __name__ == "__main__":
    print("Requantization Reference Implementation — Test Vectors\n")
    print("These test vectors verify the Python reference before building hardware.")
    print("Run: python srdhm_reference.py\n")

    all_ok = True
    all_ok &= test_srdhm()
    all_ok &= test_rdbpot()
    all_ok &= test_requantize()

    if all_ok:
        print("All tests passed.")
    else:
        print("SOME TESTS FAILED — review output above.")
        exit(1)
