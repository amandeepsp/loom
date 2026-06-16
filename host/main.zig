const std = @import("std");
const driver = @import("driver");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = std.process.Args.Iterator.init(init.minimal.args);
    defer args.deinit();

    _ = args.skip(); // skip argv[0]
    const port = args.next() orelse "/dev/ttyUSB1";
    const command = args.next() orelse "ping";

    var drv = try driver.Driver.initIo(init.io, port, 115200);
    defer drv.deinit();

    if (std.mem.eql(u8, command, "ping")) {
        try drv.ping();
        std.debug.print("ping ok, last_cycles={}\n", .{drv.last_cycles});
        return;
    }

    if (std.mem.eql(u8, command, "exec-bin")) {
        const path = args.next() orelse return error.InvalidArgument;
        const program = try readFileAlloc(init.io, allocator, path);
        defer allocator.free(program);

        const cycles = try drv.exec(program);
        std.debug.print("exec-bin ok, cycles={}, last_cycles={}\n", .{ cycles, drv.last_cycles });
        return;
    }

    if (std.mem.eql(u8, command, "exec-model")) {
        const addr = try parseAddr(args.next() orelse return error.InvalidArgument);
        const len = try parseU32(args.next() orelse return error.InvalidArgument);

        const cycles = try drv.execModel(addr, len);
        std.debug.print("exec-model ok, addr=0x{x}, bytes={}, cycles={}, last_cycles={}\n", .{
            addr,
            len,
            cycles,
            drv.last_cycles,
        });
        return;
    }

    if (std.mem.eql(u8, command, "exec-model-bin")) {
        const addr = try parseAddr(args.next() orelse return error.InvalidArgument);
        const path = args.next() orelse return error.InvalidArgument;
        const data = try readFileAlloc(init.io, allocator, path);
        defer allocator.free(data);

        try drv.writeMem(addr, data);
        const cycles = try drv.execModel(addr, @intCast(data.len));
        std.debug.print("exec-model-bin ok, addr=0x{x}, bytes={}, cycles={}, last_cycles={}\n", .{
            addr,
            data.len,
            cycles,
            drv.last_cycles,
        });
        return;
    }

    if (std.mem.eql(u8, command, "write-file")) {
        const addr = try parseAddr(args.next() orelse return error.InvalidArgument);
        const path = args.next() orelse return error.InvalidArgument;
        const data = try readFileAlloc(init.io, allocator, path);
        defer allocator.free(data);

        try drv.writeMem(addr, data);
        std.debug.print("write-file ok, addr=0x{x}, bytes={}\n", .{ addr, data.len });
        return;
    }

    if (std.mem.eql(u8, command, "read-file")) {
        const addr = try parseAddr(args.next() orelse return error.InvalidArgument);
        const len = try parseLen(args.next() orelse return error.InvalidArgument);
        const path = args.next() orelse return error.InvalidArgument;

        const buf = try allocator.alloc(u8, len);
        defer allocator.free(buf);

        try drv.readMem(addr, buf);
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = path, .data = buf });
        std.debug.print("read-file ok, addr=0x{x}, bytes={}, path={s}\n", .{ addr, len, path });
        return;
    }

    std.debug.print("unknown command: {s}\n", .{command});
    return error.InvalidArgument;
}

fn parseAddr(text: []const u8) !u32 {
    return try std.fmt.parseInt(u32, text, 0);
}

fn parseLen(text: []const u8) !usize {
    return try std.fmt.parseInt(usize, text, 0);
}

fn parseU32(text: []const u8) !u32 {
    return try std.fmt.parseInt(u32, text, 0);
}

fn readFileAlloc(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    return try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
}
