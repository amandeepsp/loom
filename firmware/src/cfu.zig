// CFU Custom Instructions

inline fn cfu_call(comptime funct3: u3, comptime funct7: u7, rs1: i32, rs2: i32) i32 {
    return asm volatile (".insn r CUSTOM_0, %[f3], %[f7], %[rd], %[rs1], %[rs2]"
        : [rd] "=r" (-> i32),
        : [rs1] "r" (rs1),
          [rs2] "r" (rs2),
          [f3] "i" (@as(u32, funct3)),
          [f7] "i" (@as(u32, funct7)),
        : .{ .memory = true });
}

pub inline fn mac4(acc: i32, a: i32, b: i32) i32 {
    return acc + cfu_call(0x0, 0x00, a, b);
}
