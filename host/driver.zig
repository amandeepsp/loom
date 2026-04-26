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
    BadAddress,
    IllegalInstruction,
    TrapFault,
    DeviceError,
};

pub const Driver = struct {
    transport: Transport,
    last_cycles: u16 = 0,

    pub fn init(port_path: []const u8, baud_rate: u32) !Driver {
        if (std.mem.startsWith(u8, port_path, "tcp://")) {
            return .{ .transport = try Transport.initTcp(port_path["tcp://".len..]) };
        }

        const port = try std.fs.openFileAbsolute(port_path, .{ .mode = .read_write });
        errdefer port.close();

        try serial.configureSerialPort(port, .{
            .baud_rate = baud_rate,
            .word_size = .eight,
        });
        try serial.flushSerialPort(port, .both);

        return .{ .transport = .{ .serial = port } };
    }

    pub fn deinit(self: *Driver) void {
        self.transport.close();
    }

    fn drainBytes(self: *Driver, len: usize) !void {
        var remaining = len;
        var buf: [256]u8 = undefined;
        while (remaining > 0) {
            const chunk_len = @min(remaining, buf.len);
            const read_len = try self.transport.readAll(buf[0..chunk_len]);
            if (read_len != chunk_len) return error.BadResponse;
            remaining -= chunk_len;
        }
    }

    fn issuePayloadParts(
        self: *Driver,
        op: protocol.OpType,
        payload_a: []const u8,
        payload_b: []const u8,
        response: []u8,
    ) !void {
        const payload_len = payload_a.len + payload_b.len;
        if (payload_len > std.math.maxInt(u16)) return error.PayloadTooLarge;

        const header = protocol.RequestHeader.init(op, @intCast(payload_len), 0);
        try self.transport.writeAll(header.as_bytes());
        try self.transport.writeAll(payload_a);
        try self.transport.writeAll(payload_b);

        var resp_header_buf: [@sizeOf(protocol.ResponseHeader)]u8 = undefined;
        while (true) {
            const read_len = try self.transport.readAll(resp_header_buf[0..1]);
            if (read_len != 1) return error.BadResponse;
            if (resp_header_buf[0] == protocol.MAGIC_RESP) break;
        }
        const read_len = try self.transport.readAll(resp_header_buf[1..]);
        if (read_len != resp_header_buf.len - 1) return error.BadResponse;

        const resp_header = protocol.ResponseHeader.from_bytes(&resp_header_buf);

        if (!resp_header.status.isOk()) {
            log.err("device returned error: {s} (0x{X:0>2})", .{
                resp_header.status.describe(),
                @intFromEnum(resp_header.status),
            });
            if (resp_header.payload_len > 0) {
                var debug_buf: [256]u8 = undefined;
                const debug_len = @min(resp_header.payload_len, debug_buf.len);
                const debug_read = try self.transport.readAll(debug_buf[0..debug_len]);
                if (debug_read > 0) {
                    log.err("debug payload ({d} bytes): {any}", .{
                        debug_read,
                        debug_buf[0..debug_read],
                    });
                }
                // Drain remaining bytes
                if (resp_header.payload_len > debug_read) {
                    try self.drainBytes(resp_header.payload_len - debug_read);
                }
            }
            return statusToError(resp_header.status);
        }

        self.last_cycles = resp_header.cycles_lo;

        if (resp_header.payload_len != response.len) {
            if (resp_header.payload_len > 0) {
                try self.drainBytes(resp_header.payload_len);
            }
            return error.BadResponse;
        }

        if (response.len > 0) {
            const read_payload_len = try self.transport.readAll(response);
            if (read_payload_len != response.len) return error.BadResponse;
        }
    }

    fn issuePayload(
        self: *Driver,
        op: protocol.OpType,
        payload: []const u8,
        response: []u8,
    ) !void {
        try self.issuePayloadParts(op, payload, &.{}, response);
    }

    pub fn ping(self: *Driver) !void {
        var response: [0]u8 = .{};
        try self.issuePayload(.ping, &.{}, response[0..]);
    }

    pub fn writeMem(self: *Driver, addr: u32, data: []const u8) !void {
        const chunk_max = std.math.maxInt(u16) - @sizeOf(protocol.WriteMem.ReqHeader);
        var offset: usize = 0;

        while (offset < data.len) {
            const chunk_len = @min(chunk_max, data.len - offset);
            const req = protocol.WriteMem.ReqHeader{
                .addr = addr + @as(u32, @intCast(offset)),
            };
            var response: [0]u8 = .{};
            try self.issuePayloadParts(
                .write,
                std.mem.asBytes(&req),
                data[offset .. offset + chunk_len],
                response[0..],
            );
            offset += chunk_len;
        }
    }

    pub fn readMem(self: *Driver, addr: u32, buf: []u8) !void {
        const chunk_max = std.math.maxInt(u16);
        var offset: usize = 0;

        while (offset < buf.len) {
            const chunk_len = @min(chunk_max, buf.len - offset);
            const req = protocol.ReadMem.Req{
                .addr = addr + @as(u32, @intCast(offset)),
                .len = @intCast(chunk_len),
            };
            try self.issuePayload(.read, std.mem.asBytes(&req), buf[offset .. offset + chunk_len]);
            offset += chunk_len;
        }
    }

    pub fn exec(self: *Driver, program: []const u8) !u32 {
        var response: [4]u8 = undefined;
        try self.issuePayload(.exec, program, &response);
        return std.mem.readInt(u32, &response, .little);
    }
};

const Transport = union(enum) {
    serial: std.fs.File,
    tcp: std.net.Stream,

    fn initTcp(spec: []const u8) !Transport {
        const colon = std.mem.lastIndexOfScalar(u8, spec, ':') orelse return error.InvalidArgument;
        const host = spec[0..colon];
        const port_text = spec[colon + 1 ..];
        if (host.len == 0 or port_text.len == 0) return error.InvalidArgument;

        const port = try std.fmt.parseInt(u16, port_text, 10);
        const stream = try std.net.tcpConnectToHost(std.heap.page_allocator, host, port);
        return .{ .tcp = stream };
    }

    fn close(self: Transport) void {
        switch (self) {
            .serial => |file| file.close(),
            .tcp => |stream| stream.close(),
        }
    }

    fn readAll(self: Transport, buf: []u8) !usize {
        return switch (self) {
            .serial => |file| try file.readAll(buf),
            .tcp => |stream| try stream.readAtLeast(buf, buf.len),
        };
    }

    fn writeAll(self: Transport, data: []const u8) !void {
        switch (self) {
            .serial => |file| try file.writeAll(data),
            .tcp => |stream| try stream.writeAll(data),
        }
    }
};

fn statusToError(status: protocol.StatusCode) AccelError {
    return switch (status) {
        .ok => unreachable,
        .unknown_op => error.UnknownOp,
        .bad_payload_len => error.BadPayloadLen,
        .bad_address => error.BadAddress,
        .bad_magic => error.BadMagic,
        .timeout => error.DeviceError,
        .illegal_instruction => error.IllegalInstruction,
        .trap_fault => error.TrapFault,
    };
}
