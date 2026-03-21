const std = @import("std");

pub fn build(b: *std.Build) void {
    // ── Shared protocol module (used by both firmware and host) ──────
    const protocol = b.createModule(.{
        .root_source_file = b.path("shared/protocol.zig"),
    });

    // ── Driver (native target) ────────────────────────────────────
    const drv_target = b.standardTargetOptions(.{});
    const drv_optimize = b.standardOptimizeOption(.{});

    const drv_lib = b.addLibrary(.{
        .name = "driver",
        .linkage = .static,
        .root_module = b.createModule(.{
            .root_source_file = b.path("driver/driver.zig"),
            .target = drv_target,
            .optimize = drv_optimize,
        }),
    });

    const serial_dep = b.dependency("serial", .{
        .target = drv_target,
        .optimize = drv_optimize,
    });
    drv_lib.root_module.addImport("serial", serial_dep.module("serial"));

    drv_lib.root_module.addImport("protocol", protocol);

    const drv_exe = b.addExecutable(.{
        .name = "driver",
        .root_module = b.createModule(.{
            .root_source_file = b.path("driver/main.zig"),
            .target = drv_target,
            .optimize = drv_optimize,
        }),
    });
    drv_exe.root_module.linkLibrary(drv_lib);
    drv_exe.root_module.addImport("driver", drv_lib.root_module);
    b.installArtifact(drv_exe);

    const run_drv = b.addRunArtifact(drv_exe);
    run_drv.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_drv.addArgs(args);

    const run_step = b.step("run", "Run the driver");
    run_step.dependOn(&run_drv.step);

    // ── Driver tests ────────────────────────────────────────────────
    const drv_tests = b.addTest(.{ .root_module = drv_lib.root_module });
    const run_drv_tests = b.addRunArtifact(drv_tests);

    const test_step = b.step("test", "Run driver tests");
    test_step.dependOn(&run_drv_tests.step);

    // ── Firmware (RISC-V freestanding target, always ReleaseSmall) ──
    const fw_query: std.Target.Query = .{
        .cpu_arch = .riscv32,
        .os_tag = .freestanding,
        .abi = .ilp32,
        .cpu_features_add = std.Target.riscv.featureSet(&.{.m}),
        .cpu_features_sub = std.Target.riscv.featureSet(&.{
            .f, .d, .c, .zca, .a, .zaamo, .zalrsc,
        }),
    };
    const fw_target = b.resolveTargetQuery(fw_query);

    const build_dir = b.option(
        []const u8,
        "build-dir",
        "LiteX build directory (default: build/sim)",
    ) orelse "build/sim";

    // Resolve build-dir relative to the project root (not CWD).
    const root_dir = b.build_root.handle;
    const crt0_path = std.fmt.allocPrint(
        b.allocator,
        "{s}/software/bios/crt0.o",
        .{build_dir},
    ) catch @panic("OOM");
    const csr_json_path = std.fmt.allocPrint(
        b.allocator,
        "{s}/csr.json",
        .{build_dir},
    ) catch @panic("OOM");

    // Parse CSR addresses from LiteX csr.json
    const csr_options = b.addOptions();
    const csr_json_contents = root_dir.readFileAlloc(
        b.allocator,
        csr_json_path,
        4 * 1024 * 1024,
    ) catch |e| {
        std.debug.print("error: cannot read {s}: {}\n", .{ csr_json_path, e });
        std.debug.print("hint: run the LiteX SoC build first, then pass -Dbuild-dir=<path>\n", .{});
        std.process.exit(1);
    };
    const parsed = std.json.parseFromSlice(
        std.json.Value,
        b.allocator,
        csr_json_contents,
        .{},
    ) catch |e| {
        std.debug.print("error: failed to parse {s}: {}\n", .{ csr_json_path, e });
        std.process.exit(1);
    };
    const regs = parsed.value.object.get("csr_registers").?.object;
    var it = regs.iterator();
    while (it.next()) |entry| {
        const addr: usize = @intCast(entry.value_ptr.object.get("addr").?.integer);
        csr_options.addOption(usize, entry.key_ptr.*, addr);
    }

    const fw_mod = b.createModule(.{
        .root_source_file = b.path("firmware/main.zig"),
        .target = fw_target,
        .optimize = .ReleaseSmall,
    });
    fw_mod.addImport("csr", csr_options.createModule());
    fw_mod.addImport("protocol", protocol);
    fw_mod.addImport("cfu", b.createModule(.{
        .root_source_file = b.path("shared/cfu.zig"),
        .target = fw_target,
        .optimize = .ReleaseSmall,
    }));

    const fw_exe = b.addExecutable(.{
        .name = "firmware",
        .root_module = fw_mod,
    });
    fw_exe.addObjectFile(b.path(crt0_path));
    fw_exe.setLinkerScript(b.path("firmware/linker.ld"));
    b.installArtifact(fw_exe);

    // ELF → flat binary
    const objcopy_prog = b.findProgram(
        &.{ "riscv64-elf-objcopy", "riscv64-unknown-elf-objcopy", "riscv64-linux-gnu-objcopy" },
        &.{"/usr/bin"},
    ) catch @panic("no riscv objcopy found in PATH");
    const objcopy = b.addSystemCommand(&.{ objcopy_prog, "-O", "binary" });
    objcopy.addArtifactArg(fw_exe);
    const bin_output = objcopy.addOutputFileArg("firmware.bin");
    objcopy.step.dependOn(&fw_exe.step);

    const install_bin = b.addInstallBinFile(bin_output, "firmware.bin");
    b.getInstallStep().dependOn(&install_bin.step);

    // Top-level step to build only firmware (ELF + flat binary)
    const install_elf = b.addInstallArtifact(fw_exe, .{});
    const fw_step = b.step("firmware", "Build firmware only");
    fw_step.dependOn(&install_bin.step);
    fw_step.dependOn(&install_elf.step);

    // ── Simulation (Renode + Verilated CFU) ────────────────────────
    const renode = b.findProgram(&.{"renode"}, &.{"/opt/renode"}) catch
        @panic("renode not found in PATH");
    const sim_cmd = b.addSystemCommand(&.{
        renode,
        "--disable-xwt",
        "--console",
        "-e",
        "include @sim/accel.resc; start",
    });
    sim_cmd.step.dependOn(&fw_exe.step);

    const sim_step = b.step("sim", "Run Renode simulation");
    sim_step.dependOn(&sim_cmd.step);
}
