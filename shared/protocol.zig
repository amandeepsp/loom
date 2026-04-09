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
    srdhm = 0x2,
    rdbpot = 0x3,
    mma = 0x4,
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

pub const Srdhm = struct {
    pub const Req = extern struct {
        a: i32,
        b: i32,
    };
    pub const Resp = extern struct {
        result: i32,
    };
};

pub const Rdbpot = struct {
    pub const Req = extern struct {
        x: i32,
        exponent: i32,
    };
    pub const Resp = extern struct {
        result: i32,
    };
};

pub const Mma = struct {
    pub const store_depth_words: usize = 512;
    // The current firmware keeps one accumulator row in SRAM while it drives
    // mac4_next. Leave two i32 slots of headroom for the rest of .bss.
    pub const max_n: usize = store_depth_words * 4 - 2;

    /// One partial-GEMM row update using the current CPU-as-sequencer engine.
    ///
    /// Payload layout:
    /// - ReqHeader
    /// - packed lhs row: k_words x i32
    /// - packed rhs tile in filter-stream order: (n * k_words) x i32
    /// - accumulator row: n x i32
    ///
    /// Response payload:
    /// - updated accumulator row: n x i32
    pub const ReqHeader = extern struct {
        n: u16,
        k: u16,
        input_offset: i32,
    };

    pub fn kWords(k: u16) usize {
        return (@as(usize, k) + 3) / 4;
    }

    pub fn lhsWords(req: ReqHeader) usize {
        return kWords(req.k);
    }

    pub fn rhsWords(req: ReqHeader) usize {
        return lhsWords(req) * @as(usize, req.n);
    }

    pub fn accumWords(req: ReqHeader) usize {
        return @as(usize, req.n);
    }

    pub fn payloadBytes(req: ReqHeader) usize {
        return @sizeOf(ReqHeader) +
            @sizeOf(i32) * (lhsWords(req) + rhsWords(req) + accumWords(req));
    }

    pub fn responseBytes(req: ReqHeader) usize {
        return @sizeOf(i32) * accumWords(req);
    }

    pub fn fitsCurrentEngine(req: ReqHeader) bool {
        if (req.n == 0 or req.k == 0) return false;
        if (req.n > max_n) return false;

        const k_words = lhsWords(req);
        return k_words <= store_depth_words and rhsWords(req) <= store_depth_words;
    }

    pub fn checkedPayloadBytes(req: ReqHeader) ?u16 {
        if (!fitsCurrentEngine(req)) return null;

        const payload_len = payloadBytes(req);
        if (payload_len > std.math.maxInt(u16)) return null;
        return @intCast(payload_len);
    }

    pub fn checkedResponseBytes(req: ReqHeader) ?u16 {
        if (!fitsCurrentEngine(req)) return null;

        const payload_len = responseBytes(req);
        if (payload_len > std.math.maxInt(u16)) return null;
        return @intCast(payload_len);
    }
};

pub fn ReqType(comptime op: OpType) type {
    return switch (op) {
        .ping => Ping.Req,
        .mac4 => Mac4.Req,
        .srdhm => Srdhm.Req,
        .rdbpot => Rdbpot.Req,
        else => @compileError("unknown op type"),
    };
}

pub fn RespType(comptime op: OpType) type {
    return switch (op) {
        .ping => Ping.Resp,
        .mac4 => Mac4.Resp,
        .srdhm => Srdhm.Resp,
        .rdbpot => Rdbpot.Resp,
        else => @compileError("unknown op type"),
    };
}
