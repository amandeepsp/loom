const std = @import("std");
const cfu = @import("cfu.zig");
const link = @import("link.zig");
const memory = @import("memory.zig");
const model = @import("model.zig");
const protocol = @import("protocol");
const uart = @import("uart.zig");

pub fn dispatch(header: link.Header) void {
    switch (header.op) {
        .ping => ping(header),
        .read => memory.readMem(header),
        .write => memory.writeMem(header),
        .checksum => checksum(header),
        .profile => profileOp(header),
        .exec_model => execModel(header),
        .perf => perf(header),
        else => link.sendError(header.seq_id, .unknown_op, &.{}),
    }
}

fn perf(header: link.Header) void {
    if (header.payload_len != @sizeOf(protocol.Perf.Req)) {
        link.drainPayload(header.payload_len);
        link.sendError(header.seq_id, .bad_payload_len, &.{});
        return;
    }

    var req: protocol.Perf.Req = undefined;
    uart.readBytes(std.mem.asBytes(&req));
    if (req.reset != 0) cfu.resetPerfCounters();

    const counters = protocol.Perf.Counters{
        .seq_busy_cycles = cfu.readPerfCounter(.seq_busy_cycles),
        .dma_active_cycles = cfu.readPerfCounter(.dma_active_cycles),
        .overlap_cycles = cfu.readPerfCounter(.overlap_cycles),
    };
    link.sendResponse(header.seq_id, .ok, std.mem.asBytes(&counters), 0);
}

fn ping(header: link.Header) void {
    if (header.payload_len != 0) {
        link.drainPayload(header.payload_len);
        link.sendError(header.seq_id, .bad_payload_len, &.{});
        return;
    }

    link.sendOk(header.seq_id, &.{}, 0);
}

/// Response: u32 count, then `count` u32 cycle counts (see model.profile).
fn profileOp(header: link.Header) void {
    if (header.payload_len != 0) {
        link.drainPayload(header.payload_len);
        link.sendError(header.seq_id, .bad_payload_len, &.{});
        return;
    }
    var buf: [4 + 4 * model.max_profile]u8 = undefined;
    const n = model.profile_len;
    std.mem.writeInt(u32, buf[0..4], n, .little);
    for (0..n) |i| std.mem.writeInt(u32, buf[4 + 4 * i ..][0..4], model.profile[i], .little);
    link.sendResponse(header.seq_id, .ok, buf[0 .. 4 + 4 * n], 0);
}

fn checksum(header: link.Header) void {
    if (header.payload_len != @sizeOf(protocol.Checksum.Req)) {
        link.drainPayload(header.payload_len);
        link.sendError(header.seq_id, .bad_payload_len, &.{});
        return;
    }

    var req: protocol.Checksum.Req = undefined;
    uart.readBytes(std.mem.asBytes(&req));
    if (!memory.rangeValid(req.addr, req.len)) {
        link.sendError(header.seq_id, .bad_address, &.{});
        return;
    }
    const ptr: [*]const u8 = @ptrFromInt(req.addr);
    const crc: u32 = std.hash.Crc32.hash(ptr[0..req.len]);
    link.sendResponse(header.seq_id, .ok, std.mem.asBytes(&crc), 0);
}

fn execModel(header: link.Header) void {
    if (header.payload_len != @sizeOf(protocol.ExecModel.Req)) {
        link.drainPayload(header.payload_len);
        link.sendError(header.seq_id, .bad_payload_len, &.{});
        return;
    }

    var req: protocol.ExecModel.Req = undefined;
    uart.readBytes(std.mem.asBytes(&req));
    const cycles = model.execute(req.addr, req.len) catch |err| {
        return sendModelError(header.seq_id, err);
    };
    link.sendResponse(header.seq_id, .ok, std.mem.asBytes(&cycles), @truncate(cycles));
}

fn sendModelError(seq_id: u16, err: model.ExecError) void {
    const code: link.StatusCode = switch (err) {
        error.BadMagic => .bad_magic,
        error.BadPayloadLen => .bad_payload_len,
        error.BadAddress => .bad_address,
    };
    link.sendError(seq_id, code, &model.debug_buf);
}
