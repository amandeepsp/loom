const std = @import("std");
const serial = @import("serial");
const protocol = @import("protocol");

const log = std.log.scoped(.loom);

pub const LoomError = error{
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
        return initPosix(port_path, baud_rate);
    }

    pub fn initIo(io: std.Io, port_path: []const u8, baud_rate: u32) !Driver {
        if (std.mem.startsWith(u8, port_path, "tcp://")) {
            return .{ .transport = try Transport.initTcp(port_path["tcp://".len..]) };
        }

        var port = try std.Io.Dir.openFileAbsolute(io, port_path, .{ .mode = .read_write });
        errdefer port.close(io);

        try serial.configureSerialPort(port, .{
            .baud_rate = baud_rate,
            .word_size = .eight,
            .parity = .none,
            .stop_bits = .one,
            .handshake = .none,
        });

        return .{ .transport = .{ .serial = .{ .file = port, .io = io } } };
    }

    fn initPosix(port_path: []const u8, baud_rate: u32) !Driver {
        if (std.mem.startsWith(u8, port_path, "tcp://")) {
            return .{ .transport = try Transport.initTcp(port_path["tcp://".len..]) };
        }

        const fd = try std.posix.openat(std.posix.AT.FDCWD, port_path, .{
            .ACCMODE = .RDWR,
            .NOCTTY = true,
        }, 0);
        errdefer _ = std.posix.system.close(fd);

        const port = std.Io.File{
            .handle = fd,
            .flags = .{ .nonblocking = false },
        };

        try serial.configureSerialPort(port, .{
            .baud_rate = baud_rate,
            .word_size = .eight,
            .parity = .none,
            .stop_bits = .one,
            .handshake = .none,
        });
        try serial.flushSerialPort(port, .both);

        return .{ .transport = .{ .serial = .{ .file = port, .io = null } } };
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
        const safe_chunk_max = 1024;
        const chunk_max = @min(safe_chunk_max, std.math.maxInt(u16) - @sizeOf(protocol.WriteMem.ReqHeader));
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

    pub fn execModel(self: *Driver, addr: u32, len: u32) !u32 {
        const req = protocol.ExecModel.Req{
            .addr = addr,
            .len = len,
        };
        var response: [4]u8 = undefined;
        try self.issuePayload(.exec_model, std.mem.asBytes(&req), &response);
        return std.mem.readInt(u32, &response, .little);
    }

    pub fn perfCounters(self: *Driver, reset: bool) !protocol.Perf.Counters {
        const req = protocol.Perf.Req{ .reset = @intFromBool(reset) };
        var response: protocol.Perf.Counters = undefined;
        try self.issuePayload(.perf, std.mem.asBytes(&req), std.mem.asBytes(&response));
        return response;
    }
};

const Transport = union(enum) {
    serial: SerialPort,
    tcp: std.posix.fd_t,

    fn initTcp(spec: []const u8) !Transport {
        const colon = std.mem.lastIndexOfScalar(u8, spec, ':') orelse return error.InvalidArgument;
        const host = spec[0..colon];
        const port_text = spec[colon + 1 ..];
        if (host.len == 0 or port_text.len == 0) return error.InvalidArgument;

        const port = try std.fmt.parseInt(u16, port_text, 10);
        const fd = try connectTcp4(host, port);
        return .{ .tcp = fd };
    }

    fn close(self: Transport) void {
        switch (self) {
            .serial => |port| {
                if (port.io) |io| {
                    port.file.close(io);
                } else {
                    _ = std.posix.system.close(port.file.handle);
                }
            },
            .tcp => |fd| _ = std.posix.system.close(fd),
        }
    }

    fn readAll(self: Transport, buf: []u8) !usize {
        return switch (self) {
            .serial => |port| if (port.io) |io|
                try readSerialAll(port.file, io, buf)
            else
                try readFdAll(port.file.handle, buf),
            .tcp => |fd| try readFdAll(fd, buf),
        };
    }

    fn writeAll(self: Transport, data: []const u8) !void {
        switch (self) {
            .serial => |port| if (port.io) |io| {
                var writer = port.file.writer(io, &.{});
                try writer.interface.writeAll(data);
            } else {
                try writeFdAll(port.file.handle, data);
            },
            .tcp => |fd| try writeFdAll(fd, data),
        }
    }
};

const SerialPort = struct {
    file: std.Io.File,
    io: ?std.Io,
};

fn connectTcp4(host: []const u8, port: u16) !std.posix.fd_t {
    const fd = try socketTcp4();
    errdefer _ = std.posix.system.close(fd);

    var addr = std.posix.sockaddr.in{
        .port = std.mem.nativeToBig(u16, port),
        .addr = std.mem.nativeToBig(u32, try parseIpv4(host)),
    };

    while (true) {
        const rc = std.posix.system.connect(
            fd,
            &addr,
            @intCast(@sizeOf(std.posix.sockaddr.in)),
        );
        switch (std.posix.errno(rc)) {
            .SUCCESS => return fd,
            .INTR => continue,
            .CONNREFUSED => return error.ConnectionRefused,
            .TIMEDOUT => return error.ConnectionTimedOut,
            .HOSTUNREACH, .NETUNREACH => return error.NetworkUnreachable,
            .ACCES => return error.AccessDenied,
            else => |err| return std.posix.unexpectedErrno(err),
        }
    }
}

fn socketTcp4() !std.posix.fd_t {
    const rc = std.posix.system.socket(
        std.posix.AF.INET,
        std.posix.SOCK.STREAM | std.posix.SOCK.CLOEXEC,
        std.posix.IPPROTO.TCP,
    );
    switch (std.posix.errno(rc)) {
        .SUCCESS => return @intCast(rc),
        .ACCES => return error.AccessDenied,
        .MFILE => return error.ProcessFdQuotaExceeded,
        .NFILE => return error.SystemFdQuotaExceeded,
        .NOMEM => return error.SystemResources,
        .PROTONOSUPPORT => return error.ProtocolUnsupported,
        else => |err| return std.posix.unexpectedErrno(err),
    }
}

fn parseIpv4(host: []const u8) !u32 {
    if (std.mem.eql(u8, host, "localhost")) return 0x7f000001;

    var iter = std.mem.splitScalar(u8, host, '.');
    var addr: u32 = 0;
    var count: usize = 0;
    while (iter.next()) |part| {
        if (part.len == 0) return error.InvalidArgument;
        const octet = try std.fmt.parseInt(u8, part, 10);
        addr = (addr << 8) | octet;
        count += 1;
    }
    if (count != 4) return error.InvalidArgument;
    return addr;
}

fn readSerialAll(file: std.Io.File, io: std.Io, buf: []u8) !usize {
    var reader_buf: [256]u8 = undefined;
    var reader = file.reader(io, &reader_buf);
    var offset: usize = 0;
    while (offset < buf.len) {
        const n = try reader.interface.readSliceShort(buf[offset..]);
        if (n == 0) break;
        offset += n;
    }
    return offset;
}

fn readFdAll(fd: std.posix.fd_t, buf: []u8) !usize {
    var offset: usize = 0;
    while (offset < buf.len) {
        const rc = std.posix.system.read(fd, buf[offset..].ptr, buf.len - offset);
        switch (std.posix.errno(rc)) {
            .SUCCESS => {
                const n: usize = @intCast(rc);
                if (n == 0) break;
                offset += n;
            },
            .INTR => continue,
            .AGAIN => return error.WouldBlock,
            .CONNRESET => return error.ConnectionResetByPeer,
            else => |err| return std.posix.unexpectedErrno(err),
        }
    }
    return offset;
}

fn writeFdAll(fd: std.posix.fd_t, data: []const u8) !void {
    var offset: usize = 0;
    while (offset < data.len) {
        const rc = std.posix.system.write(fd, data[offset..].ptr, data.len - offset);
        switch (std.posix.errno(rc)) {
            .SUCCESS => {
                const n: usize = @intCast(rc);
                if (n == 0) return error.BrokenPipe;
                offset += n;
            },
            .INTR => continue,
            .AGAIN => return error.WouldBlock,
            .PIPE => return error.BrokenPipe,
            .CONNRESET => return error.ConnectionResetByPeer,
            else => |err| return std.posix.unexpectedErrno(err),
        }
    }
}

fn statusToError(status: protocol.StatusCode) LoomError {
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
