/// Wire protocol shared between firmware and host driver.
///
/// All structs are `extern` so the layout is fixed and matches
/// what goes over the UART byte-for-byte.
pub const MAGIC_REQ: u8 = 0xCF;
pub const MAGIC_RESP: u8 = 0xFC;

pub const OpType = enum(u8) {
    ping = 0x00,
    read = 0x10,
    write = 0x11,
    // 0x12 was `exec` (streamed KIR bytecode), removed in favour of `.lmb` models.
    exec_model = 0x13,
    perf = 0x14,
    checksum = 0x15,
    profile = 0x16,
    _,
};

/// Status codes returned in ResponseHeader.status.
/// 0x00 = success, 0x01–0x0F = dispatch errors, 0xE0–0xEF = link errors,
/// 0xF0–0xFF = trap/fatal errors.
pub const StatusCode = enum(u8) {
    ok = 0x00,
    unknown_op = 0x01,
    bad_payload_len = 0x02,
    bad_address = 0x03,
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
            .bad_address => "address out of range",
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

pub const WriteMem = struct {
    pub const ReqHeader = extern struct {
        addr: u32,
    };
};

pub const ReadMem = struct {
    pub const Req = extern struct {
        addr: u32,
        len: u32,
    };
};

pub const ExecModel = struct {
    pub const Req = extern struct {
        addr: u32,
        len: u32,
    };
};

/// CRC-32 (IEEE, same as zlib.crc32) of a memory range; lets the host skip
/// re-uploading a model binary that is already resident.
pub const Checksum = struct {
    pub const Req = extern struct {
        addr: u32,
        len: u32,
    };
};

pub const Perf = struct {
    pub const Req = extern struct {
        reset: u32,
    };

    pub const Counters = extern struct {
        seq_busy_cycles: u32,
        dma_active_cycles: u32,
        overlap_cycles: u32,
    };
};
