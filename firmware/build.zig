const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{
        .default_target = .{
            .cpu_arch = .riscv32,
            .os_tag = .freestanding,
            .abi = .ilp32,
            .cpu_features_add = std.Target.riscv.featureSet(&.{.m}),
            .cpu_features_sub = std.Target.riscv.featureSet(&.{
                .f, .d, .c, .zca, .a, .zaamo, .zalrsc,
            }),
        },
    });

    const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseSmall });

    // --- LiteX build directory (provides crt0.o and csr.json) ---
    const build_dir = b.option([]const u8, "build-dir", "LiteX build directory (default: ../build/sim)") orelse "../build/sim";

    const crt0_path = std.fmt.allocPrint(b.allocator, "{s}/software/bios/crt0.o", .{build_dir}) catch @panic("OOM");
    const csr_json_path = std.fmt.allocPrint(b.allocator, "{s}/csr.json", .{build_dir}) catch @panic("OOM");

    // --- CSR definitions from csr.json ---
    const csr_options = b.addOptions();
    const csr_json_contents = std.fs.cwd().readFileAlloc(b.allocator, csr_json_path, 4 * 1024 * 1024) catch |e| {
        std.debug.print("error: cannot read {s}: {}\n", .{ csr_json_path, e });
        std.debug.print("hint: run the LiteX SoC build first, then pass -Dbuild-dir=<path>\n", .{});
        std.process.exit(1);
    };
    const parsed = std.json.parseFromSlice(std.json.Value, b.allocator, csr_json_contents, .{}) catch |e| {
        std.debug.print("error: failed to parse {s}: {}\n", .{ csr_json_path, e });
        std.process.exit(1);
    };
    const regs = parsed.value.object.get("csr_registers").?.object;
    var it = regs.iterator();
    while (it.next()) |entry| {
        const addr: usize = @intCast(entry.value_ptr.object.get("addr").?.integer);
        csr_options.addOption(usize, entry.key_ptr.*, addr);
    }

    // --- Build firmware ELF ---
    const root_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    root_module.addImport("csr", csr_options.createModule());

    const exe = b.addExecutable(.{
        .name = "firmware",
        .root_module = root_module,
    });

    exe.addObjectFile(.{ .cwd_relative = crt0_path });
    exe.setLinkerScript(b.path("linker.ld"));

    b.installArtifact(exe);

    // ELF → flat binary (GNU objcopy; Zig 0.15's built-in is buggy, pads the binary with zeros)
    const objcopy_prog = b.findProgram(
        &.{ "riscv64-elf-objcopy", "riscv64-unknown-elf-objcopy", "riscv64-linux-gnu-objcopy" },
        &.{"/usr/bin"},
    ) catch @panic("no riscv objcopy found in PATH");
    const objcopy = b.addSystemCommand(&.{
        objcopy_prog, "-O", "binary",
    });
    objcopy.addArtifactArg(exe);
    const bin_output = objcopy.addOutputFileArg("firmware.bin");
    objcopy.step.dependOn(&exe.step);

    const install_bin = b.addInstallBinFile(bin_output, "firmware.bin");
    b.getInstallStep().dependOn(&install_bin.step);
}
