const std = @import("std");
const serial = @import("serial");
const protocol = @import("protocol");

const log = std.log.scoped(.accel);

pub const AccelError = error{
    PayloadTooLarge,
    BadResponse,
    BadMagic,
    UnknownOp,
    BadPayloadLen,
    IllegalInstruction,
    TrapFault,
    DeviceError,
};

pub const Driver = struct {
    port: std.fs.File,
    last_cycles: u16 = 0,

    pub fn init(port_path: []const u8, baud_rate: u32) !Driver {
        const port = try std.fs.openFileAbsolute(port_path, .{ .mode = .read_write });
        errdefer port.close();

        try serial.configureSerialPort(port, .{
            .baud_rate = baud_rate,
            .word_size = .eight,
        });

        try serial.flushSerialPort(port, .both);

        return .{
            .port = port,
        };
    }

    pub fn deinit(self: *Driver) void {
        self.port.close();
    }

    fn issue(
        self: *Driver,
        comptime op: protocol.OpType,
        req: protocol.ReqType(op),
        resp: *protocol.RespType(op),
    ) !void {
        const payload_len = @sizeOf(protocol.ReqType(op));
        if (payload_len > std.math.maxInt(u16)) return error.PayloadTooLarge;

        const header = protocol.RequestHeader.init(op, payload_len, 0);
        try self.port.writeAll(header.as_bytes());
        std.debug.print(">> req header = {}\n", .{header});
        if (payload_len > 0) {
            try self.port.writeAll(std.mem.asBytes(&req));
            std.debug.print(">> req payload = {}\n", .{req});
        }

        var resp_header_buf: [@sizeOf(protocol.ResponseHeader)]u8 = undefined;
        const read_len = try self.port.readAll(&resp_header_buf);
        if (read_len < @sizeOf(protocol.ResponseHeader)) {
            return error.BadResponse;
        }

        const resp_header = protocol.ResponseHeader.from_bytes(&resp_header_buf);
        std.debug.print("<< resp header = {}\n", .{resp_header});
        if (resp_header.magic != protocol.MAGIC_RESP) {
            return error.BadMagic;
        }

        // Check status code from firmware.
        if (!resp_header.status.isOk()) {
            log.err("device returned error: {s} (0x{X:0>2})", .{
                resp_header.status.describe(),
                @intFromEnum(resp_header.status),
            });
            // Drain any error payload the firmware sent.
            if (resp_header.payload_len > 0) {
                var drain_buf: [256]u8 = undefined;
                const drain_len = @min(resp_header.payload_len, drain_buf.len);
                _ = try self.port.readAll(drain_buf[0..drain_len]);
            }
            return statusToError(resp_header.status);
        }

        self.last_cycles = resp_header.cycles_lo;

        const resp_bytes = std.mem.asBytes(resp);
        if (resp_bytes.len > 0 and resp_header.payload_len > 0) {
            const read_payload_len = try self.port.readAll(resp_bytes);
            if (read_payload_len < resp_bytes.len) {
                return error.BadResponse;
            }

            std.debug.print("<< resp payload = {}\n", .{resp});
        }

        std.debug.print(":: took {} cycles\n", .{resp_header.cycles_lo});
    }

    pub fn ping(self: *Driver) !void {
        var resp: protocol.Ping.Resp = undefined;
        try self.issue(.ping, .{}, &resp);
    }

    pub fn mac4(self: *Driver, as: [4]i8, bs: [4]i8) !i32 {
        var resp: protocol.Mac4.Resp = undefined;
        try self.issue(.mac4, protocol.Mac4.Req{ .as = as, .bs = bs }, &resp);
        return resp.result;
    }
};

fn statusToError(status: protocol.StatusCode) AccelError {
    return switch (status) {
        .ok => unreachable,
        .unknown_op => error.UnknownOp,
        .bad_payload_len => error.BadPayloadLen,
        .bad_magic => error.BadMagic,
        .timeout => error.DeviceError,
        .illegal_instruction => error.IllegalInstruction,
        .trap_fault => error.TrapFault,
    };
}
