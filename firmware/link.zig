const uart = @import("uart.zig");
const std = @import("std");
const protocol = @import("protocol");

pub const MAGIC_REQ = protocol.MAGIC_REQ;
pub const MAGIC_RESP = protocol.MAGIC_RESP;
pub const OpType = protocol.OpType;
pub const StatusCode = protocol.StatusCode;
pub const Header = protocol.RequestHeader;

pub const LinkError = error{ BadMagic, Timeout };

pub fn recv_header() LinkError!protocol.RequestHeader {
    // Sync: skip bytes until we find the magic request byte.
    while (true) {
        const b = uart.read_byte_blocking();
        if (b == MAGIC_REQ) break;
    }
    // Read remaining 7 bytes of the header.
    var header: protocol.RequestHeader = undefined;
    const buf: []u8 = std.mem.asBytes(&header);
    buf[0] = MAGIC_REQ;
    uart.read_bytes(buf[1..]);
    return header;
}

pub fn send_response(seq_id: u16, status: StatusCode, data: []const u8, cycles: u16) void {
    const rsp = protocol.ResponseHeader{
        .status = status,
        .payload_len = @intCast(data.len),
        .seq_id = seq_id,
        .cycles_lo = cycles,
    };
    uart.write_bytes(std.mem.asBytes(&rsp));
    uart.write_bytes(data);
}

pub fn send_ok(seq_id: u16, data: []const u8, cycles: u16) void {
    send_response(seq_id, .ok, data, cycles);
}

pub fn send_error(seq_id: u16, code: StatusCode) void {
    send_response(seq_id, code, &.{}, 0);
}
