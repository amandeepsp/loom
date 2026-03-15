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
    // MAC4 payload is 8 bytes: 2x 32-bit values (in0, in1)
    // Each 32-bit value contains 4 packed 8-bit signed values
    if (header.payload_len < 8) {
        link.send_error(header.seq_id, 0x02); // Invalid payload length
        return;
    }

    // Read payload
    var payload: [8]u8 = undefined;
    uart.read_bytes(&payload);

    // Parse input values (little-endian) as packed 4x8-bit signed values
    const a_packed = std.mem.readInt(i32, payload[0..4], .little);
    const b_packed = std.mem.readInt(i32, payload[4..8], .little);

    // Execute MAC4 instruction using CFU (4-element dot product)
    // mac4(acc, a, b) where a and b are each 4 packed 8-bit signed values
    // Computes: acc + (a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3])
    const result = cfu.mac4(0, a_packed, b_packed);

    // Send result back
    var response: [4]u8 = undefined;
    std.mem.writeInt(i32, response[0..4], result, .little);
    link.send_ok(header.seq_id, &response, 0);
}
