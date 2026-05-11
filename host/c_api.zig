const std = @import("std");
const driver = @import("driver");

pub const LoomStatus = enum(c_int) {
    ok = 0,
    invalid_argument = 1,
    payload_too_large = 2,
    bad_response = 3,
    bad_magic = 4,
    unknown_op = 5,
    bad_payload_len = 6,
    bad_address = 7,
    illegal_instruction = 8,
    trap_fault = 9,
    device_error = 10,
    out_of_memory = 11,
    io_error = 12,

    pub fn describe(self: LoomStatus) [*:0]const u8 {
        return switch (self) {
            .ok => "ok",
            .invalid_argument => "invalid argument",
            .payload_too_large => "payload too large",
            .bad_response => "bad response",
            .bad_magic => "bad magic",
            .unknown_op => "unknown op",
            .bad_payload_len => "bad payload length",
            .bad_address => "bad address",
            .illegal_instruction => "illegal instruction",
            .trap_fault => "trap fault",
            .device_error => "device error",
            .out_of_memory => "out of memory",
            .io_error => "io error",
        };
    }

    pub fn fromError(err: anyerror) LoomStatus {
        return switch (err) {
            error.InvalidDimensions => .invalid_argument,
            error.PayloadTooLarge => .payload_too_large,
            error.BadResponse => .bad_response,
            error.BadMagic => .bad_magic,
            error.UnknownOp => .unknown_op,
            error.BadPayloadLen => .bad_payload_len,
            error.BadAddress => .bad_address,
            error.IllegalInstruction => .illegal_instruction,
            error.TrapFault => .trap_fault,
            error.DeviceError => .device_error,
            error.OutOfMemory => .out_of_memory,
            else => .io_error,
        };
    }
};

inline fn require(comptime T: type, opt: ?T) union(enum) { ok: T, err: c_int } {
    if (opt) |v| return .{ .ok = v };
    return .{ .err = @intFromEnum(LoomStatus.invalid_argument) };
}

inline fn mapError(err: anyerror) c_int {
    return @intFromEnum(LoomStatus.fromError(err));
}

inline fn unwrapHandle(handle: ?*LoomHandle) *LoomHandle {
    return switch (require(*LoomHandle, handle)) {
        .ok => |v| v,
        .err => |e| return e,
    };
}

inline fn unwrapPtr(comptime T: type, opt: ?T) T {
    return switch (require(T, opt)) {
        .ok => |v| v,
        .err => |e| return e,
    };
}

pub const LoomHandle = struct {
    driver: driver.Driver,
};

pub export fn loom_open(
    port_path: ?[*:0]const u8,
    baud_rate: u32,
    out_handle: ?*?*LoomHandle,
) c_int {
    const path = unwrapPtr([*:0]const u8, port_path);
    const handle_ptr = unwrapPtr(*?*LoomHandle, out_handle);

    const handle = std.heap.page_allocator.create(LoomHandle) catch {
        handle_ptr.* = null;
        return @intFromEnum(LoomStatus.out_of_memory);
    };
    errdefer std.heap.page_allocator.destroy(handle);

    handle.driver = driver.Driver.init(std.mem.span(path), baud_rate) catch |err| {
        handle_ptr.* = null;
        return mapError(err);
    };

    handle_ptr.* = handle;
    return @intFromEnum(LoomStatus.ok);
}

pub export fn loom_close(handle: ?*LoomHandle) void {
    if (handle) |h| {
        h.driver.deinit();
        std.heap.page_allocator.destroy(h);
    }
}

pub export fn loom_ping(handle: ?*LoomHandle) c_int {
    const h = unwrapHandle(handle);
    h.driver.ping() catch |err| return mapError(err);
    return @intFromEnum(LoomStatus.ok);
}

pub export fn loom_last_cycles(handle: ?*LoomHandle) u16 {
    const h = handle orelse return 0;
    return h.driver.last_cycles;
}

pub export fn loom_write_mem(
    handle: ?*LoomHandle,
    addr: u32,
    data: ?[*]const u8,
    len: usize,
) c_int {
    const h = unwrapHandle(handle);
    const ptr = unwrapPtr([*]const u8, data);
    h.driver.writeMem(addr, ptr[0..len]) catch |err| return mapError(err);
    return @intFromEnum(LoomStatus.ok);
}

pub export fn loom_read_mem(
    handle: ?*LoomHandle,
    addr: u32,
    buf: ?[*]u8,
    len: usize,
) c_int {
    const h = unwrapHandle(handle);
    const ptr = unwrapPtr([*]u8, buf);
    h.driver.readMem(addr, ptr[0..len]) catch |err| return mapError(err);
    return @intFromEnum(LoomStatus.ok);
}

pub export fn loom_exec(
    handle: ?*LoomHandle,
    program: ?[*]const u8,
    program_len: usize,
    out_cycles: ?*u32,
) c_int {
    const h = unwrapHandle(handle);
    const ptr = unwrapPtr([*]const u8, program);
    const cycles_ptr = unwrapPtr(*u32, out_cycles);

    cycles_ptr.* = h.driver.exec(ptr[0..program_len]) catch |err| return mapError(err);
    return @intFromEnum(LoomStatus.ok);
}

pub export fn loom_status_string(code: c_int) [*:0]const u8 {
    const status: LoomStatus = switch (code) {
        @intFromEnum(LoomStatus.ok) => .ok,
        @intFromEnum(LoomStatus.invalid_argument) => .invalid_argument,
        @intFromEnum(LoomStatus.io_error) => .io_error,
        @intFromEnum(LoomStatus.protocol_error) => .protocol_error,
        @intFromEnum(LoomStatus.device_error) => .device_error,
        else => return "unknown status",
    };
    return status.describe();
}
