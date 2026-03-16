const std = @import("std");
const link = @import("link.zig");
const uart = @import("uart.zig");
const cfu = @import("cfu.zig");

pub fn dispatch(header: link.Header) void {
    switch (header.op) {
        .mac4 => handle_mac4(header),
        else => link.send_error(header.seq_id, 0x01),
    }
}

fn handle_mac4(header: link.Header) void {
    if (header.payload_len < 8) {
        link.send_error(header.seq_id, 0x02);
        return;
    }

    var payload: [8]u8 = undefined;
    uart.read_bytes(&payload);

    const a = std.mem.readInt(i32, payload[0..4], .little);
    const b = std.mem.readInt(i32, payload[4..8], .little);

    const result = cfu.mac4_first(a, b);

    var response: [4]u8 = undefined;
    std.mem.writeInt(i32, response[0..4], result, .little);
    link.send_ok(header.seq_id, &response, 0);
}
