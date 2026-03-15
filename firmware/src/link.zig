const uart = @import("uart.zig");
const std = @import("std");

pub const MAGIC_REQ: u8 = 0xCF;
pub const MAGIC_RESP: u8 = 0xFC;

pub const OpType = enum(u8) {
    mac4 = 0x1,
    _,
};

pub const Header = extern struct {
    magic: u8,
    op: OpType,
    payload_len: u16,
    seq_id: u16,
    _reserved: u16,
};

pub const Response = extern struct {
    magic: u8 = MAGIC_RESP,
    status: u8,
    payload_len: u16,
    seq_id: u16,
    cycles_lo: u16,
};

pub const LinkError = error{ BadMagic, Timeout };

pub fn recv_header() LinkError!Header {
    var header: Header = undefined;
    const buf: []u8 = std.mem.asBytes(&header);
    uart.read_bytes(buf);
    if (header.magic != MAGIC_REQ) return error.BadMagic;
    return header;
}

pub fn send_response(seq_id: u16, status: u8, data: []const u8, cycles: u16) void {
    const rsp = Response{
        .status = status,
        .payload_len = @intCast(data.len),
        .seq_id = seq_id,
        .cycles_lo = cycles,
    };
    uart.write_bytes(std.mem.asBytes(&rsp));
    uart.write_bytes(data);
}

pub fn send_ok(seq_id: u16, data: []const u8, cycles: u16) void {
    send_response(seq_id, 0x00, data, cycles);
}

pub fn send_error(seq_id: u16, code: u8) void {
    send_response(seq_id, code, &.{}, 0);
}
