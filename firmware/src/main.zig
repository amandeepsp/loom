const std = @import("std");
const link = @import("link.zig");
const dispatch = @import("dispatch.zig");

export fn main() void {
    link.send_ok(0, &.{}, 0);

    while (true) {
        const header = link.recv_header() catch |err| {
            link.send_error(0, switch (err) {
                error.BadMagic => 0x03,
                error.Timeout => 0x04,
            });
            continue;
        };
        dispatch.dispatch(header);
    }
}

export fn isr() void {}
