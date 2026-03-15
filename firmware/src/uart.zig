const CSR_BASE: usize = 0x12000000; // from generated/csr.zig CSR_BASE

inline fn rd(offset: usize) u32 {
    return @as(*volatile u32, @ptrFromInt(CSR_BASE + offset)).*;
}
inline fn wr(offset: usize, v: u32) void {
    @as(*volatile u32, @ptrFromInt(CSR_BASE + offset)).* = v;
}

pub inline fn rxtx_read() u32 {
    return rd(0x3000);
}
pub inline fn rxtx_write(v: u32) void {
    wr(0x3000, v);
}
pub inline fn txfull_read() u32 {
    return rd(0x3004);
}
pub inline fn rxempty_read() u32 {
    return rd(0x3008);
}
pub inline fn txempty_read() u32 {
    return rd(0x3018);
}
pub inline fn rxfull_read() u32 {
    return rd(0x301c);
}
pub inline fn ev_pending_write(v: u32) void {
    wr(0x3010, v);
}

// Higher-level helpers used by link.zig
pub fn write_byte(b: u8) void {
    while (txfull_read() != 0) {}
    rxtx_write(b);
}

pub fn read_byte_blocking() u8 {
    while (rxempty_read() != 0) {}
    return @truncate(rxtx_read());
}

pub fn write_bytes(buf: []const u8) void {
    for (buf) |b| write_byte(b);
}

pub fn read_bytes(buf: []u8) void {
    for (buf) |*b| b.* = read_byte_blocking();
}
