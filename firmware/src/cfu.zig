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
          [f3] "i" (@as(u32, funct3)),
          [f7] "i" (@as(u32, funct7)),
    );
}

/// MAC4 accumulate: acc += Σ(a[i]+128)*b[i] for i in 0..3.
/// Returns the running accumulator value.
pub inline fn mac4(a: i32, b: i32) i32 {
    return cfu_op(0, 0, a, b);
}

/// MAC4 reset + compute: acc = Σ(a[i]+128)*b[i].
/// Clears the accumulator then computes. Use for the first op in a sequence.
pub inline fn mac4_first(a: i32, b: i32) i32 {
    return cfu_op(0, 1, a, b);
}
