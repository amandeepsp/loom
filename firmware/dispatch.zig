const std = @import("std");
const config = @import("config");
const interpreter = @import("interpreter.zig");
const link = @import("link.zig");
const memory = @import("memory.zig");

pub fn dispatch(header: link.Header) void {
    switch (header.op) {
        .ping => ping(header),
        .read => memory.readMem(header),
        .write => memory.writeMem(header),
        .exec => exec(header),
        else => link.sendError(header.seq_id, .unknown_op, &.{}),
    }
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

fn sendExecError(seq_id: u16, err: interpreter.ExecError, debug_buf: []const u8) noreturn {
    const code: link.StatusCode = switch (err) {
        error.BadMagic => .bad_magic,
        error.BadPayloadLen => .bad_payload_len,
        error.BadAddress => .bad_address,
    };
    link.sendError(seq_id, code, debug_buf);
}
