/// Wire protocol shared between firmware and host driver.
///
/// All structs are `extern` so the layout is fixed and matches
/// what goes over the UART byte-for-byte.
const std = @import("std");

pub const MAGIC_REQ: u8 = 0xCF;
pub const MAGIC_RESP: u8 = 0xFC;

pub const OpType = enum(u8) {
    ping = 0x0,
    mac4 = 0x1,
    _,
};

/// Status codes returned in ResponseHeader.status.
/// 0x00 = success, 0x01–0x0F = dispatch errors, 0xE0–0xEF = link errors,
/// 0xF0–0xFF = trap/fatal errors.
pub const StatusCode = enum(u8) {
    ok = 0x00,
    unknown_op = 0x01,
    bad_payload_len = 0x02,
    bad_magic = 0xE0,
    timeout = 0xE1,
    illegal_instruction = 0xF0,
    trap_fault = 0xF1,

    pub fn isOk(self: StatusCode) bool {
        return self == .ok;
    }

    pub fn describe(self: StatusCode) []const u8 {
        return switch (self) {
            .ok => "ok",
            .unknown_op => "unknown opcode",
            .bad_payload_len => "payload length mismatch",
            .bad_magic => "bad magic byte",
            .timeout => "timeout",
            .illegal_instruction => "illegal instruction trap",
            .trap_fault => "unhandled trap",
        };
    }
};

pub const RequestHeader = extern struct {
    magic: u8,
    op: OpType,
    payload_len: u16,
    seq_id: u16,
    _reserved: u16,

    pub fn init(op: OpType, payload_len: u16, seq_id: u16) RequestHeader {
        return .{
            .magic = MAGIC_REQ,
            .op = op,
            .payload_len = payload_len,
            .seq_id = seq_id,
            ._reserved = 0,
        };
    }

    pub fn as_bytes(self: *const RequestHeader) *const [8]u8 {
        return @ptrCast(self);
    }
};

pub const ResponseHeader = extern struct {
    magic: u8 = MAGIC_RESP,
    status: StatusCode,
    payload_len: u16,
    seq_id: u16,
    /// Lower 16 bits of the cycles count from CSR, can be inaccurate.
    cycles_lo: u16,

    pub fn as_bytes(self: *const ResponseHeader) *const [8]u8 {
        return @ptrCast(self);
    }

    pub fn from_bytes(self: *const [8]u8) *const ResponseHeader {
        return @ptrCast(@alignCast(self));
    }
};

pub const Ping = struct {
    pub const Req = extern struct {};
    pub const Resp = extern struct {};
};

pub const Mac4 = struct {
    pub const Req = extern struct {
        as: [4]i8,
        bs: [4]i8,
    };
    pub const Resp = extern struct {
        result: i32,
    };
};

pub fn ReqType(comptime op: OpType) type {
    return switch (op) {
        .ping => Ping.Req,
        .mac4 => Mac4.Req,
        else => @compileError("uknown op type"),
    };
}

pub fn RespType(comptime op: OpType) type {
    return switch (op) {
        .ping => Ping.Resp,
        .mac4 => Mac4.Resp,
        else => @compileError("unknown op type"),
    };
}
