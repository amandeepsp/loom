const csr = @import("csr");
const std = @import("std");
const builtin = @import("builtin");
const cpu_csr = @import("cpu_csr.zig");
const memory = @import("memory.zig");

const model_magic: u32 = 0x30424D4C; // "LMB0"
const model_version: u16 = 1;

const section_constants: u16 = 2;
const section_riscv: u16 = 5;
const section_schedule: u16 = 8;

const sched_call_riscv: u8 = 1;

pub var debug_buf: [8]u8 = @splat(0);

/// Per-step mcycle counts from the last exec_model: [0] = constant loading,
/// [1 + i] = schedule entry i (same order as the manifest's riscv_functions).
pub const max_profile = 128;
pub var profile: [max_profile]u32 = @splat(0);
pub var profile_len: u32 = 0;

/// Constants are read-only while a model runs, so they only need copying out
/// of the blob once per model. `resident` remembers which blob was loaded and
/// the address span its constants cover; any host write overlapping either
/// (see `noteWrite`, called from memory.writeMem) forces a reload.
const Resident = struct { addr: u32, len: u32, lo: u32, hi: u32 };
var resident: ?Resident = null;

pub fn noteWrite(addr: u32, len: u32) void {
    const r = resident orelse return;
    const end = addr +% len;
    const hits_blob = addr < r.addr + r.len and end > r.addr;
    const hits_consts = addr < r.hi and end > r.lo;
    if (hits_blob or hits_consts) resident = null;
}

/// mcycle (low 32 bits); host-side unit tests have no RISC-V CSRs.
fn now() u32 {
    if (builtin.is_test) return 0;
    return @truncate(cpu_csr.mcycle.read());
}

fn isResident(addr: u32, len: u32) bool {
    const r = resident orelse return false;
    return r.addr == addr and r.len == len;
}

pub const ExecError = error{
    BadMagic,
    BadPayloadLen,
    BadAddress,
};

const Range = struct {
    off: u32 = 0,
    len: u32 = 0,

    fn slice(self: Range, data: []const u8) ExecError![]const u8 {
        const end = addU32(self.off, self.len) catch return error.BadPayloadLen;
        if (end > data.len) return error.BadPayloadLen;
        return data[self.off..end];
    }
};

const Sections = struct {
    constants: ?Range = null,
    riscv: ?Range = null,
    schedule: ?Range = null,
};

pub fn execute(addr: u32, len: u32) ExecError!u32 {
    clearDebug();
    if (!memory.rangeValid(addr, len)) return error.BadAddress;
    if (len < 12) return error.BadPayloadLen;
    // Lines cached from a previous inference may predate DMA write-back.
    invalidateDcache();

    const ptr: [*]const u8 = @ptrFromInt(addr);
    const data = ptr[0..len];
    const sections = parseSections(data) catch |err| {
        if (debug_buf[0] == 0) setDebug(1, 0, 0, len);
        return err;
    };

    const constants = sections.constants orelse return error.BadPayloadLen;
    const riscv = sections.riscv orelse return error.BadPayloadLen;
    const schedule = sections.schedule orelse return error.BadPayloadLen;

    const cycle_start = cpu_csr.mcycle.read();
    profile_len = 0;
    if (!isResident(addr, len)) {
        resident = null;
        const span = loadConstants(data, constants) catch |err| {
            if (debug_buf[0] == 0) setDebug(2, 0, 0, constants.off);
            return err;
        };
        resident = .{ .addr = addr, .len = len, .lo = span.lo, .hi = span.hi };
    }
    profile[0] = @truncate(cpu_csr.mcycle.read() -% cycle_start);
    profile_len = 1;
    flushDataCaches();
    runSchedule(addr, data, riscv, schedule) catch |err| {
        if (debug_buf[0] == 0) setDebug(3, 0, 0, schedule.off);
        return err;
    };
    return cpu_csr.mcycle.read() -% cycle_start;
}

fn parseSections(data: []const u8) ExecError!Sections {
    const magic = readU32(data[0..4]);
    if (magic != model_magic) {
        setDebug(1, 1, 0, magic);
        return error.BadMagic;
    }
    const version = readU16(data[4..6]);
    if (version != model_version) {
        setDebug(1, 2, version, model_version);
        return error.BadPayloadLen;
    }

    const count = readU32(data[8..12]);
    if (count > 64) {
        setDebug(1, 3, 0, count);
        return error.BadPayloadLen;
    }
    var cursor: u32 = 12;
    var sections = Sections{};

    for (0..count) |idx| {
        if (cursor + 12 > data.len) {
            setDebug(1, 4, @truncate(idx), cursor);
            return error.BadPayloadLen;
        }
        const entry = data[cursor .. cursor + 12];
        const typ = readU16(entry[0..2]);
        const off = readU32(entry[4..8]);
        const len = readU32(entry[8..12]);
        const end = addU32(off, len) catch {
            setDebug(1, 5, @truncate(idx), off);
            return error.BadPayloadLen;
        };
        if (end > data.len) {
            setDebug(1, 6, @truncate(idx), end);
            return error.BadPayloadLen;
        }
        const range = Range{ .off = off, .len = len };
        switch (typ) {
            section_constants => sections.constants = range,
            section_riscv => sections.riscv = range,
            section_schedule => sections.schedule = range,
            else => {},
        }
        cursor += 12;
    }
    return sections;
}

const Span = struct { lo: u32 = std.math.maxInt(u32), hi: u32 = 0 };

fn loadConstants(data: []const u8, range: Range) ExecError!Span {
    const sec = try range.slice(data);
    if (sec.len < 4) return error.BadPayloadLen;
    const count = readU32(sec[0..4]);
    var cursor: u32 = 4;
    var span: Span = .{};
    for (0..count) |_| {
        if (cursor + 12 > sec.len) return error.BadPayloadLen;
        const hdr = sec[cursor .. cursor + 12];
        const dst_addr = readU32(hdr[0..4]);
        const data_len = readU32(hdr[4..8]);
        const name_len = readU16(hdr[8..10]);
        cursor += 12;
        cursor = align4(addU32(cursor, name_len) catch return error.BadPayloadLen);
        const end = addU32(cursor, data_len) catch return error.BadPayloadLen;
        if (end > sec.len) return error.BadPayloadLen;
        if (!memory.rangeValid(dst_addr, data_len)) return error.BadAddress;

        const dst: [*]u8 = @ptrFromInt(dst_addr);
        std.mem.copyForwards(u8, dst[0..data_len], sec[cursor..end]);
        span.lo = @min(span.lo, dst_addr);
        span.hi = @max(span.hi, dst_addr + data_len);
        cursor = align4(end);
    }
    return span;
}

fn runSchedule(model_addr: u32, data: []const u8, riscv_range: Range, schedule_range: Range) ExecError!void {
    const sec = try schedule_range.slice(data);
    if (sec.len < 4) return error.BadPayloadLen;
    const count = readU32(sec[0..4]);
    var cursor: u32 = 4;
    for (0..count) |_| {
        const step_start = now();
        defer if (profile_len < max_profile) {
            profile[profile_len] = now() -% step_start;
            profile_len += 1;
        };
        if (cursor + 4 > sec.len) return error.BadPayloadLen;
        const op = sec[cursor];
        setDebug(3, op, @truncate(cursor), count);
        switch (op) {
            sched_call_riscv => {
                if (cursor + 8 > sec.len) return error.BadPayloadLen;
                const index = readU32(sec[cursor + 4 .. cursor + 8]);
                const code = try findRiscvCode(data, riscv_range, index);
                try callCode(model_addr + code.off, code.len);
                cursor += 8;
            },
            else => return error.BadPayloadLen,
        }
    }
}

fn findRiscvCode(data: []const u8, range: Range, target_index: u32) ExecError!Range {
    setDebug(4, @truncate(target_index), 0, range.off);
    const sec = try range.slice(data);
    if (sec.len < 4) return error.BadPayloadLen;
    const count = readU32(sec[0..4]);
    if (target_index >= count) return error.BadPayloadLen;
    var cursor: u32 = 4;
    for (0..count) |idx| {
        if (cursor + 8 > sec.len) return error.BadPayloadLen;
        const code_len = readU32(sec[cursor .. cursor + 4]);
        const name_len = readU16(sec[cursor + 4 .. cursor + 6]);
        cursor += 8;
        cursor = align4(addU32(cursor, name_len) catch return error.BadPayloadLen);
        if (code_len == 0 or (code_len & 0x3) != 0) return error.BadPayloadLen;
        const code_off_in_sec = cursor;
        const end = addU32(cursor, code_len) catch return error.BadPayloadLen;
        if (end > sec.len) return error.BadPayloadLen;
        if (idx == target_index) {
            return .{
                .off = range.off + code_off_in_sec,
                .len = code_len,
            };
        }
        cursor = align4(end);
    }
    return error.BadPayloadLen;
}

fn callCode(addr: u32, len: u32) ExecError!void {
    setDebug(5, 0, @truncate(len), addr);
    if (len == 0 or (addr & 0x3) != 0) return error.BadPayloadLen;
    if (!memory.rangeValid(addr, len)) return error.BadAddress;
    if (builtin.is_test) return error.BadAddress;
    flushDataCaches();
    asm volatile ("fence.i");
    const Entry = *const fn () void;
    const entry: Entry = @ptrFromInt(addr);
    entry();
    drainResultWriteback();
    flushDataCaches();
}

/// Wait until every CFU result write-back beat has reached DRAM, so the next
/// step (DMA or CPU) reads the final data. No-op on bitstreams without it.
fn drainResultWriteback() void {
    if (@hasDecl(csr, "wb_dma_pending")) {
        const pending: *const volatile u32 = @ptrFromInt(csr.wb_dma_pending);
        while (pending.* != 0) {}
    }
}

/// Drop dcache lines: needed before the CPU reads memory the result
/// write-back DMA may have written (it bypasses the write-through dcache).
pub fn invalidateDcache() void {
    if (@hasDecl(csr, "wb_dma_pending")) asm volatile (".word(0x500F)");
}

/// Make CPU writes visible to the CFU DMA readers, which read DRAM directly.
///
/// The VexRiscv dcache is write-through and nothing but the CPU writes
/// memory (the DMAs only read), so the dcache never holds stale data and is
/// not invalidated: that only threw away warm constants every step. The
/// write-back L2 (direct-mapped) is flushed by reading one L2-size window,
/// which evicts every dirty line; with no L2 there is nothing to do.
fn flushDataCaches() void {
    flushL2Cache();
}

// LiteX omits config_l2_size when the SoC has no L2.
const config_l2_size: usize = if (@hasDecl(csr, "config_l2_size")) @intCast(csr.config_l2_size) else 0;

fn flushL2Cache() void {
    if (config_l2_size == 0) return;
    const base: [*]volatile u32 = @ptrFromInt(0x40000000);
    for (0..(2 * config_l2_size / @sizeOf(u32))) |idx| {
        _ = base[idx];
    }
}

fn align4(value: u32) u32 {
    return (value + 3) & ~@as(u32, 3);
}

fn addU32(lhs: anytype, rhs: anytype) ExecError!u32 {
    return std.math.add(u32, lhs, rhs) catch error.BadPayloadLen;
}

fn mulU32(lhs: anytype, rhs: anytype) ExecError!u32 {
    return std.math.mul(u32, lhs, rhs) catch error.BadPayloadLen;
}

fn readU16(bytes: []const u8) u16 {
    return @as(u16, bytes[0]) | (@as(u16, bytes[1]) << 8);
}

fn readU32(bytes: []const u8) u32 {
    return @as(u32, bytes[0]) |
        (@as(u32, bytes[1]) << 8) |
        (@as(u32, bytes[2]) << 16) |
        (@as(u32, bytes[3]) << 24);
}

fn clearDebug() void {
    @memset(&debug_buf, 0);
}

fn setDebug(stage: u8, arg0: u8, arg1: u16, arg2: u32) void {
    debug_buf[0] = stage;
    debug_buf[1] = arg0;
    debug_buf[2] = @truncate(arg1);
    debug_buf[3] = @truncate(arg1 >> 8);
    debug_buf[4] = @truncate(arg2);
    debug_buf[5] = @truncate(arg2 >> 8);
    debug_buf[6] = @truncate(arg2 >> 16);
    debug_buf[7] = @truncate(arg2 >> 24);
}

test "model section parser accepts required sections and ignores debug sections" {
    var blob: [96]u8 = @splat(0);
    writeHeader(blob[0..], 4);
    writeSectionEntry(blob[12..], section_constants, 64, 4);
    writeSectionEntry(blob[24..], section_riscv, 68, 4);
    writeSectionEntry(blob[36..], section_schedule, 72, 4);
    writeSectionEntry(blob[48..], 9, 76, 4);

    const sections = try parseSections(blob[0..]);

    try std.testing.expect(sections.constants != null);
    try std.testing.expect(sections.riscv != null);
    try std.testing.expect(sections.schedule != null);
    try std.testing.expectEqual(@as(u32, 64), sections.constants.?.off);
    try std.testing.expectEqual(@as(u32, 68), sections.riscv.?.off);
    try std.testing.expectEqual(@as(u32, 72), sections.schedule.?.off);
}

test "model section parser rejects bad magic version count and ranges" {
    var blob: [64]u8 = @splat(0);
    writeHeader(blob[0..], 0);
    blob[0] = 'X';
    try std.testing.expectError(error.BadMagic, parseSections(blob[0..]));

    writeHeader(blob[0..], 0);
    writeU16(blob[4..6], model_version + 1);
    try std.testing.expectError(error.BadPayloadLen, parseSections(blob[0..]));

    writeHeader(blob[0..], 65);
    try std.testing.expectError(error.BadPayloadLen, parseSections(blob[0..]));

    writeHeader(blob[0..], 1);
    writeSectionEntry(blob[12..], section_riscv, 60, 8);
    try std.testing.expectError(error.BadPayloadLen, parseSections(blob[0..]));
}

test "RISC-V section lookup validates index length and alignment" {
    var sec: [32]u8 = @splat(0);
    writeU32(sec[0..4], 1);
    writeU32(sec[4..8], 4);
    writeU16(sec[8..10], 1);
    sec[12] = 'f';
    sec[16] = 0x13;

    const code = try findRiscvCode(sec[0..20], .{ .off = 0, .len = 20 }, 0);
    try std.testing.expectEqual(@as(u32, 16), code.off);
    try std.testing.expectEqual(@as(u32, 4), code.len);
    try std.testing.expectError(error.BadPayloadLen, findRiscvCode(sec[0..20], .{ .off = 0, .len = 20 }, 1));

    writeU32(sec[4..8], 2);
    try std.testing.expectError(error.BadPayloadLen, findRiscvCode(sec[0..20], .{ .off = 0, .len = 20 }, 0));
}

test "schedule parser rejects malformed commands before calling code" {
    var data: [32]u8 = @splat(0);
    writeU32(data[0..4], 0);
    writeU32(data[8..12], 1);
    data[12] = sched_call_riscv;
    writeU32(data[16..20], 0);
    try std.testing.expectError(
        error.BadPayloadLen,
        runSchedule(0x4001_0000, data[0..20], .{ .off = 0, .len = 4 }, .{ .off = 8, .len = 12 }),
    );

    data[12] = 0xFF;
    try std.testing.expectError(
        error.BadPayloadLen,
        runSchedule(0x4001_0000, data[0..20], .{ .off = 0, .len = 4 }, .{ .off = 8, .len = 12 }),
    );
}

fn writeHeader(buf: []u8, section_count: u32) void {
    writeU32(buf[0..4], model_magic);
    writeU16(buf[4..6], model_version);
    writeU32(buf[8..12], section_count);
}

fn writeSectionEntry(buf: []u8, section_type: u16, off: u32, len: u32) void {
    writeU16(buf[0..2], section_type);
    writeU32(buf[4..8], off);
    writeU32(buf[8..12], len);
}

fn writeU16(buf: []u8, value: u16) void {
    std.mem.writeInt(u16, buf[0..2], value, .little);
}

fn writeU32(buf: []u8, value: u32) void {
    std.mem.writeInt(u32, buf[0..4], value, .little);
}
