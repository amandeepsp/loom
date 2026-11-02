const std = @import("std");
const config = @import("config");
const cfu = @import("cfu.zig");
const interpreter = @import("interpreter.zig");
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
        .exec => exec(header),
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

fn exec(header: link.Header) void {
    const cycles = if (config.debug_info) blk: {
        var debug_buf: [8]u8 = undefined;
        break :blk interpreter.execute(header.payload_len, &debug_buf) catch |err| {
            return sendExecError(header.seq_id, err, &debug_buf);
        };
    } else blk: {
        break :blk interpreter.execute(header.payload_len, null) catch |err| {
            return sendExecError(header.seq_id, err, &.{});
        };
    };

    link.sendResponse(header.seq_id, .ok, std.mem.asBytes(&cycles), @truncate(cycles));
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

fn sendExecError(seq_id: u16, err: interpreter.ExecError, debug_buf: []const u8) void {
    const code: link.StatusCode = switch (err) {
        error.BadMagic => .bad_magic,
        error.BadPayloadLen => .bad_payload_len,
        error.BadAddress => .bad_address,
    };
    link.sendError(seq_id, code, debug_buf);
}

fn sendModelError(seq_id: u16, err: model.ExecError) void {
    const code: link.StatusCode = switch (err) {
        error.BadMagic => .bad_magic,
        error.BadPayloadLen => .bad_payload_len,
        error.BadAddress => .bad_address,
    };
    link.sendError(seq_id, code, &model.debug_buf);
}
