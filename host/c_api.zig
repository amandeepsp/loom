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

inline fn mapError(err: anyerror) c_int {
    return @intFromEnum(LoomStatus.fromError(err));
}

pub const LoomHandle = struct {
    driver: driver.Driver,
};

pub export fn loom_open(
    port_path: ?[*:0]const u8,
    baud_rate: u32,
    out_handle: ?*?*LoomHandle,
) c_int {
    const path = port_path orelse return @intFromEnum(LoomStatus.invalid_argument);
    const handle_ptr = out_handle orelse return @intFromEnum(LoomStatus.invalid_argument);

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
    const h = handle orelse return @intFromEnum(LoomStatus.invalid_argument);
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
    const h = handle orelse return @intFromEnum(LoomStatus.invalid_argument);
    const ptr = data orelse return @intFromEnum(LoomStatus.invalid_argument);
    h.driver.writeMem(addr, ptr[0..len]) catch |err| return mapError(err);
    return @intFromEnum(LoomStatus.ok);
}

pub export fn loom_read_mem(
    handle: ?*LoomHandle,
    addr: u32,
    buf: ?[*]u8,
    len: usize,
) c_int {
    const h = handle orelse return @intFromEnum(LoomStatus.invalid_argument);
    const ptr = buf orelse return @intFromEnum(LoomStatus.invalid_argument);
    h.driver.readMem(addr, ptr[0..len]) catch |err| return mapError(err);
    return @intFromEnum(LoomStatus.ok);
}

pub export fn loom_exec(
    handle: ?*LoomHandle,
    program: ?[*]const u8,
    program_len: usize,
    out_cycles: ?*u32,
) c_int {
    const h = handle orelse return @intFromEnum(LoomStatus.invalid_argument);
    const ptr = program orelse return @intFromEnum(LoomStatus.invalid_argument);
    const cycles_ptr = out_cycles orelse return @intFromEnum(LoomStatus.invalid_argument);

    cycles_ptr.* = h.driver.exec(ptr[0..program_len]) catch |err| return mapError(err);
    return @intFromEnum(LoomStatus.ok);
}

pub export fn loom_exec_model(
    handle: ?*LoomHandle,
    model_addr: u32,
    model_len: u32,
    out_cycles: ?*u32,
) c_int {
    const h = handle orelse return @intFromEnum(LoomStatus.invalid_argument);
    const cycles_ptr = out_cycles orelse return @intFromEnum(LoomStatus.invalid_argument);

    cycles_ptr.* = h.driver.execModel(model_addr, model_len) catch |err| return mapError(err);
    return @intFromEnum(LoomStatus.ok);
}

pub export fn loom_exec_model_bytes(
    handle: ?*LoomHandle,
    model_addr: u32,
    model: ?[*]const u8,
    model_len: usize,
    out_cycles: ?*u32,
) c_int {
    const h = handle orelse return @intFromEnum(LoomStatus.invalid_argument);
    const ptr = model orelse return @intFromEnum(LoomStatus.invalid_argument);
    const cycles_ptr = out_cycles orelse return @intFromEnum(LoomStatus.invalid_argument);
    if (model_len > std.math.maxInt(u32)) {
        return @intFromEnum(LoomStatus.payload_too_large);
    }

    h.driver.writeMem(model_addr, ptr[0..model_len]) catch |err| return mapError(err);
    cycles_ptr.* = h.driver.execModel(model_addr, @intCast(model_len)) catch |err| return mapError(err);
    return @intFromEnum(LoomStatus.ok);
}

pub export fn loom_status_string(code: c_int) [*:0]const u8 {
    const status: LoomStatus = switch (code) {
        @intFromEnum(LoomStatus.ok) => .ok,
        @intFromEnum(LoomStatus.invalid_argument) => .invalid_argument,
        @intFromEnum(LoomStatus.payload_too_large) => .payload_too_large,
        @intFromEnum(LoomStatus.bad_response) => .bad_response,
        @intFromEnum(LoomStatus.bad_magic) => .bad_magic,
        @intFromEnum(LoomStatus.unknown_op) => .unknown_op,
        @intFromEnum(LoomStatus.bad_payload_len) => .bad_payload_len,
        @intFromEnum(LoomStatus.bad_address) => .bad_address,
        @intFromEnum(LoomStatus.illegal_instruction) => .illegal_instruction,
        @intFromEnum(LoomStatus.trap_fault) => .trap_fault,
        @intFromEnum(LoomStatus.device_error) => .device_error,
        @intFromEnum(LoomStatus.out_of_memory) => .out_of_memory,
        @intFromEnum(LoomStatus.io_error) => .io_error,
        else => return "unknown status",
    };
    return status.describe();
}
