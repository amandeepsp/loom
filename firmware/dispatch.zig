const std = @import("std");
const link = @import("link.zig");
const uart = @import("uart.zig");
const mmio = @import("mmio.zig");
const cfu = @import("cfu");

const McycleCsr = mmio.Csr(0xC00);

/// Read the low 16 bits of the cycle counter.
inline fn read_cycles() u16 {
    return @truncate(McycleCsr.read());
}

pub fn dispatch(header: link.Header) void {
    switch (header.op) {
        .ping => handle_ping(header),
        .mac4 => handle_mac4(header),
        else => link.send_error(header.seq_id, .unknown_op),
    }
}

fn handle_ping(header: link.Header) void {
    if (header.payload_len != 0) {
        link.send_error(header.seq_id, .bad_payload_len);
        return;
    }
    link.send_ok(header.seq_id, &.{}, 0);
}

fn handle_mac4(header: link.Header) void {
    if (header.payload_len != 8) {
        link.send_error(header.seq_id, .bad_payload_len);
        return;
    }

    // Read the 8-byte payload: 4 x i8 'as' + 4 x i8 'bs'.
    var payload: [8]u8 = undefined;
    uart.read_bytes(&payload);

    const as: i32 = @bitCast(payload[0..4].*);
    const bs: i32 = @bitCast(payload[4..8].*);

    const t0 = read_cycles();
    cfu.reset_accumulator();
    cfu.mac4(as, bs);
    const result = cfu.read_accumulator();
    const t1 = read_cycles();

    const result_bytes = std.mem.asBytes(&result);
    link.send_ok(header.seq_id, result_bytes, t1 -% t0);
}
