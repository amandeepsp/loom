"""PE — Single Processing Element for a weight-stationary systolic array.

This is one cell in the systolic array grid. Multiple PEs are tiled into
rows and columns to form the full array (e.g., 4x4 = 16 PEs).

Dataflow (weight-stationary):
    - Weight: loaded once via weight_load signal, stays in the PE's register
    - Activation: flows LEFT → RIGHT (passthrough to neighbor)
    - Partial sum: flows TOP → BOTTOM (accumulated through the column)

Connection to neighbors:
    ┌──────────────────────────────────────────────────────────┐
    │                                                          │
    │  PE[row, col-1].act_out ──► act_in    act_out ──► PE[row, col+1]
    │                                                          │
    │  PE[row-1, col].psum_out ─► psum_in   psum_out ─► PE[row+1, col]
    │                                                          │
    └──────────────────────────────────────────────────────────┘

    For the top row: psum_in = 0 (no incoming partial sum)
    For the bottom row: psum_out feeds into the post-processing pipeline
    For the leftmost column: act_in comes from the activation BSRAM
    For the rightmost column: act_out is unused (drops off the edge)

Weight-stationary dataflow explained:
    In weight-stationary mode, each PE "owns" one weight value. During
    computation, activations stream horizontally through all PEs in a row.
    Each PE multiplies its stored weight by the passing activation and adds
    the result to the partial sum flowing vertically.

    After K cycles (where K is the reduction dimension), the bottom row's
    psum_out contains the complete dot product for each column's output channel.

    Alternative dataflows:
    - Output-stationary: accumulator stays, both weights and activations flow
      (used by hps_accel — see appendix-prior-art.md)
    - Row-stationary (Eyeriss): minimizes total data movement energy

    Weight-stationary is simplest to implement and debug, which is why we
    start here. The key insight from the appendix is that the dataflow choice
    matters LESS than getting the CPU off the data path.

INPUT_OFFSET (128):
    Same trick as in SimdMac4. TFLite INT8 quantization uses "asymmetric"
    quantization where the zero point can be non-zero. For the input
    activation, the zero point is typically -128, meaning:
        real_value = (quantized_uint8_value - 128) * scale
    Or equivalently:
        real_value = (quantized_int8_value + 128) * scale

    Adding 128 converts the signed int8 [-128, 127] to unsigned [0, 255],
    which is what the weight was quantized against. This offset is applied
    BEFORE the multiply, not after — it's part of the mathematical identity:
        sum((act + offset) * weight) = sum(act * weight) + offset * sum(weight)

    The hardware does (act + 128) * weight per PE, which is mathematically
    correct for TFLite's quantization scheme.

DSP mapping on Gowin GW2AR-18C:
    Each PE needs one INT8 x INT8 multiply. The Gowin pDSP in MULT9X9 mode
    handles one 9x9 signed multiply — perfect for (int8 + offset) * int8
    where the offset makes the activation effectively 9 bits.
    So: 1 PE = 1 DSP block. A 4x4 array = 16 DSPs (33% of the 48 available).
"""

from amaranth import Module, Signal, signed

from cfu import Instruction


class PE(Instruction):
    """Single processing element for the systolic array.

    Note: PE extends Instruction for integration with the CFU test harness,
    but in the actual systolic array it would be a plain Elaboratable wired
    into the array grid. The Instruction interface (in0, in1, output, done)
    is used here for standalone unit testing only.

    For standalone testing via CFU:
        in0[7:0]  = activation input (int8)
        in0[8]    = weight_load signal
        in0[16:9] = weight_data (int8, only used when weight_load=1)
        in1       = partial_sum_in (int32)
        output    = partial_sum_out (int32)
    """

    INPUT_OFFSET = 128

    def __init__(self):
        super().__init__()

        # --- Systolic array interface (active when wired into the grid) ---

        # Activation flows left-to-right: arrives from left neighbor, passes
        # through to right neighbor unchanged (1 cycle delay for pipelining).
        self.act_in = Signal(8)
        self.act_out = Signal(8)

        # Weight is loaded once, then stays in the register.
        self.weight_load = Signal()
        self.weight_data = Signal(signed(8))

        # Partial sum flows top-to-bottom: arrives from above, we add our
        # product, pass the updated sum downward.
        self.psum_in = Signal(signed(32))
        self.psum_out = Signal(signed(32))

    def elaborate(self, platform):
        m = Module()

        # --- Internal weight register ---
        # Loaded once when weight_load is asserted. Stays until next load.
        # In weight-stationary mode, this happens once per tile (when new
        # weights are distributed to the array).
        weight_reg = Signal(signed(8))

        with m.If(self.weight_load):
            m.d.sync += weight_reg.eq(self.weight_data)

        # --- Activation passthrough ---
        # The activation passes through unchanged to the right neighbor,
        # delayed by one cycle. This one-cycle delay is what creates the
        # "skew" in the systolic array — row i sees the activation i cycles
        # after row 0. The skew ensures that each PE sees the correct
        # activation at the correct time.
        m.d.sync += self.act_out.eq(self.act_in)

        # --- Multiply-accumulate ---
        # product = (activation + INPUT_OFFSET) * stored_weight
        # The INPUT_OFFSET converts signed int8 to unsigned for correct
        # TFLite quantization math (see module docstring).
        act_with_offset = Signal(9)  # 8-bit + offset needs 9 bits
        m.d.comb += act_with_offset.eq(self.act_in + self.INPUT_OFFSET)

        # 9-bit unsigned × 8-bit signed = 17-bit signed product
        # This fits in one Gowin MULT9X9 (9×9 signed multiplier).
        product = Signal(signed(32))
        m.d.comb += product.eq(act_with_offset * weight_reg)

        # Partial sum: add our product to the incoming partial sum from above.
        # For the top row of the array, psum_in is tied to 0.
        m.d.comb += self.psum_out.eq(self.psum_in + product)

        # --- CFU test interface ---
        # When testing standalone (not in systolic grid), wire the CFU
        # signals to the PE's systolic ports.
        #
        # Decode in0:
        #   [7:0]  = activation
        #   [8]    = weight_load
        #   [16:9] = weight_data
        m.d.comb += [
            self.act_in.eq(self.in0[:8]),
            self.weight_load.eq(self.in0[8]),
            self.weight_data.eq(self.in0[9:17]),
        ]
        m.d.comb += self.psum_in.eq(self.in1)
        m.d.comb += self.output.eq(self.psum_out)
        m.d.comb += self.done.eq(1)

        return m
