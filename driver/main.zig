const std = @import("std");
const driver = @import("driver");

pub fn main() !void {
    var dgb_alloc = std.heap.DebugAllocator(.{}){};
    defer _ = dgb_alloc.deinit();

    var args = std.process.args();
    _ = args.next(); // skip argv[0]
    const port = args.next() orelse "/dev/ttyUSB1";

    var drv = try driver.Driver.init(port, 115200);
    defer drv.deinit();

    try drv.ping();

    // Test mac4: [1,2,3,4] · [5,6,7,8] = 1*5 + 2*6 + 3*7 + 4*8 = 70
    const result = try drv.mac4(.{ 1, 2, 3, 4 }, .{ 5, 6, 7, 8 });
    std.debug.print("mac4 result = {} (expected 70)\n", .{result});
}
