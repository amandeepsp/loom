/// CFU (Custom Function Unit) interface for the loom ISA in docs/isa.md.
///
/// All instructions use CUSTOM_0 (opcode 0x0B) with R-type encoding:
///   31:25 funct7 | 24:20 rs2 | 19:15 rs1 | 14:12 funct3 | 11:7 rd | 6:0 opcode
///
/// funct3 selects the instruction group:
///   0 = COMPUTE_START
///   1 = COMPUTE_WAIT
///   2 = EPI_PARAM
///   3 = CONFIG
///   4 = READ_RESULT
///   5 = PERF_COUNTER (funct7=0 reset, 1..3 read)
pub const EpiParamField = enum(u3) {
    bias = 0,
    multiplier = 1,
    shift = 2,
};

pub const ConfigRegister = enum(u3) {
    output_offset = 0,
    activation_min = 1,
    activation_max = 2,
};

pub const PerfCounter = enum(u7) {
    seq_busy_cycles = 1,
    dma_active_cycles = 2,
    overlap_cycles = 3,
};

pub inline fn op(comptime funct3: u3, comptime funct7: u7, rs1: i32, rs2: i32) i32 {
    return asm volatile (
        \\.insn r CUSTOM_0, %[f3], %[f7], %[rd], %[rs1], %[rs2]
        : [rd] "=r" (-> i32),
        : [rs1] "r" (rs1),
          [rs2] "r" (rs2),
          [f3] "i" (@as(u32, funct3)),
          [f7] "i" (@as(u32, funct7)),
    );
}

pub inline fn computeStart(first: bool, last: bool, k_count: i32) void {
    const flags: i32 = @intFromBool(first) | (@as(i32, @intFromBool(last)) << 1);
    _ = op(0, 0, flags, k_count);
}

pub inline fn computeWait() void {
    _ = op(1, 0, 0, 0);
}

pub inline fn writeEpiParam(comptime field: EpiParamField, channel: i32, value: i32) void {
    _ = op(2, @intFromEnum(field), channel, value);
}

pub inline fn writeConfig(comptime reg: ConfigRegister, value: i32) void {
    _ = op(3, @intFromEnum(reg), 0, value);
}

pub inline fn setOutputOffset(value: i32) void {
    writeConfig(.output_offset, value);
}

pub inline fn setActivationMin(value: i32) void {
    writeConfig(.activation_min, value);
}

pub inline fn setActivationMax(value: i32) void {
    writeConfig(.activation_max, value);
}

pub inline fn readResult(index: i32) i32 {
    return op(4, 0, index, 0);
}

pub inline fn resetPerfCounters() void {
    _ = op(5, 0, 0, 0);
}

pub inline fn readPerfCounter(comptime counter: PerfCounter) u32 {
    return @bitCast(op(5, @intFromEnum(counter), 0, 0));
}
