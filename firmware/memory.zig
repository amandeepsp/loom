const std = @import("std");
const link = @import("link.zig");
const model = @import("model.zig");
const protocol = @import("protocol");
const uart = @import("uart.zig");

const mem_start: u32 = 0x4001_0000;
const mem_end: u32 = 0x4080_0000;

pub fn addrValid(addr: u32) bool {
    return mem_start <= addr and addr < mem_end;
}

pub fn rangeValid(addr: u32, len: u32) bool {
    if (!addrValid(addr)) return false;
    const end = std.math.add(u32, addr, len) catch return false;
    return end <= mem_end;
}

pub fn writeMem(header: link.Header) void {
    if (header.payload_len < @sizeOf(protocol.WriteMem.ReqHeader)) {
        link.drainPayload(header.payload_len);
        link.sendError(header.seq_id, .bad_payload_len, &.{});
        return;
    }

    var req_hdr: protocol.WriteMem.ReqHeader = undefined;
    uart.readBytes(std.mem.asBytes(&req_hdr));

    const data_len = header.payload_len - @as(u16, @sizeOf(protocol.WriteMem.ReqHeader));
    if (!rangeValid(req_hdr.addr, @as(u32, data_len))) {
        link.drainPayload(data_len);
        link.sendError(header.seq_id, .bad_address, &.{});
        return;
    }

    model.noteWrite(req_hdr.addr, data_len);
    var dst: [*]u8 = @ptrFromInt(req_hdr.addr);

    var offset: usize = 0;
    var buf: [64]u8 = undefined;

    while (offset < data_len) {
        const chunk_len = @min(buf.len, @as(usize, data_len) - offset);
        uart.readBytes(buf[0..chunk_len]);
        @memcpy(dst[offset .. offset + chunk_len], buf[0..chunk_len]);
        offset += chunk_len;
    }

    link.sendOk(header.seq_id, &.{}, 0);
}

pub fn readMem(header: link.Header) void {
    if (header.payload_len != @sizeOf(protocol.ReadMem.Req)) {
        link.drainPayload(header.payload_len);
        link.sendError(header.seq_id, .bad_payload_len, &.{});
        return;
    }

    var req: protocol.ReadMem.Req = undefined;
    uart.readBytes(std.mem.asBytes(&req));

    const data_len: u32 = req.len;
    if (data_len > std.math.maxInt(u16) or !rangeValid(req.addr, data_len)) {
        link.sendError(header.seq_id, .bad_address, &.{});
        return;
    }

    const src: [*]const u8 = @ptrFromInt(req.addr);
    link.sendResponseHeader(header.seq_id, .ok, @intCast(data_len), 0);
    uart.writeBytes(src[0..data_len]);
}
