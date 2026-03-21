/// CFU (Custom Function Unit) interface for VexRiscv.
///
/// Issues CUSTOM_0 (opcode 0x0B) R-type instructions:
///   31:25  funct7  |  24:20  rs2  |  19:15  rs1  |  14:12  funct3  |  11:7  rd  |  6:0  opcode
///
/// funct3 selects the instruction (0–7), funct7 provides control bits.
inline fn cfu_op(comptime funct3: u3, comptime funct7: u7, rs1: i32, rs2: i32) i32 {
    return asm volatile (
        \\.insn r CUSTOM_0, %[f3], %[f7], %[rd], %[rs1], %[rs2]
        : [rd] "=r" (-> i32),
        : [rs1] "r" (rs1),
          [rs2] "r" (rs2),
          [f3] "i" (funct3),
          [f7] "i" (funct7),
    );
}

/// MAC4 accumulate: acc += Σ(a[i]+offset)*b[i] for i in 0..3.
/// Hardware does not produce an output — the result lives in the
/// accumulator and must be read separately via read_accumulator().
pub inline fn mac4(a: i32, b: i32) void {
    _ = cfu_op(0, 0, a, b);
}

pub inline fn set_input_offset(offset: i32) void {
    _ = cfu_op(2, 0, 0, offset);
}

pub fn reset_accumulator() void {
    _ = cfu_op(2, 0, 0x1, 0);
}

pub fn read_accumulator() i32 {
    return cfu_op(1, 0, 0x1, 0);
}
