const std = @import("std");

pub fn build(b: *std.Build) void {

    // Common modules
    const protocol = b.createModule(.{
        .root_source_file = b.path("shared/protocol.zig"),
    });
    const ir = b.createModule(.{
        .root_source_file = b.path("shared/ir.zig"),
    });

    // Config Options
    const cfu_rows = b.option(
        usize,
        "cfu-rows",
        "CFU array row count (default: 8)",
    ) orelse 8;
    const cfu_cols = b.option(
        usize,
        "cfu-cols",
        "CFU array column count (default: 8)",
    ) orelse 8;
    const cfu_store_depth = b.option(
        usize,
        "cfu-store-depth",
        "CFU scratchpad depth in words (default: 512)",
    ) orelse 512;
    const debug_info = b.option(bool, "debug-info", "Should include debug info") orelse false;

    const config = b.addOptions();
    config.addOption(usize, "cfu_rows", cfu_rows);
    config.addOption(usize, "cfu_cols", cfu_cols);
    config.addOption(usize, "cfu_store_depth", cfu_store_depth);
    config.addOption(bool, "debug_info", debug_info);

    // Host Driver
    const drv_target = b.standardTargetOptions(.{});
    const drv_optimize = b.standardOptimizeOption(.{});

    const serial_dep = b.dependency("serial", .{
        .target = drv_target,
        .optimize = drv_optimize,
    });

    const driver_mod = b.createModule(.{
        .root_source_file = b.path("host/driver.zig"),
        .target = drv_target,
        .optimize = drv_optimize,
    });
    driver_mod.addImport("serial", serial_dep.module("serial"));
    driver_mod.addImport("protocol", protocol);
    driver_mod.addImport("ir", ir);
    driver_mod.addImport("config", config.createModule());

    const drv_lib = b.addLibrary(.{
        .name = "driver",
        .linkage = .static,
        .root_module = driver_mod,
    });

    const drv_exe = b.addExecutable(.{
        .name = "driver",
        .root_module = b.createModule(.{
            .root_source_file = b.path("host/main.zig"),
            .target = drv_target,
            .optimize = drv_optimize,
        }),
    });
    drv_exe.root_module.linkLibrary(drv_lib);
    drv_exe.root_module.addImport("driver", driver_mod);
    drv_exe.root_module.addImport("ir", ir);
    drv_exe.root_module.addImport("config", config.createModule());
    b.installArtifact(drv_exe);

    const drv_c_api_mod = b.createModule(.{
        .root_source_file = b.path("host/c_api.zig"),
        .target = drv_target,
        .optimize = drv_optimize,
    });
    drv_c_api_mod.addImport("driver", driver_mod);

    const drv_c_api_lib = b.addLibrary(.{
        .name = "accel",
        .linkage = .dynamic,
        .root_module = drv_c_api_mod,
    });

    _ = b.addInstallArtifact(drv_c_api_lib, .{});

    const libaccel_step = b.step("host-lib", "Build libaccel.so");
    libaccel_step.dependOn(&drv_c_api_lib.step);

    const run_drv = b.addRunArtifact(drv_exe);
    run_drv.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_drv.addArgs(args);

    const run_step = b.step("run", "Run the driver");
    run_step.dependOn(&run_drv.step);

    const drv_tests = b.addTest(.{ .root_module = driver_mod });
    const run_drv_tests = b.addRunArtifact(drv_tests);

    const test_step = b.step("test", "Run driver tests");
    test_step.dependOn(&run_drv_tests.step);

    // Device firmware
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

    // Parse CSR addresses from LiteX csr.json (optional, only needed for firmware)
    const csr_options = b.addOptions();
    const maybe_contents: ?[]u8 = root_dir.readFileAlloc(b.allocator, csr_json_path, 4 * 1024 * 1024) catch null;
    if (maybe_contents) |contents| {
        const parsed = std.json.parseFromSlice(std.json.Value, b.allocator, contents, .{}) catch return;
        if (parsed.value.object.get("csr_registers")) |regs| {
            var it = regs.object.iterator();
            while (it.next()) |entry| {
                if (entry.value_ptr.object.get("addr")) |addr_val| {
                    const addr: usize = @intCast(addr_val.integer);
                    csr_options.addOption(usize, entry.key_ptr.*, addr);
                }
            }
        }
    }

    const fw_mod = b.createModule(.{
        .root_source_file = b.path("firmware/main.zig"),
        .target = fw_target,
        .optimize = .ReleaseSmall,
    });
    fw_mod.addImport("csr", csr_options.createModule());
    fw_mod.addImport("protocol", protocol);
    fw_mod.addImport("ir", ir);
    fw_mod.addImport("config", config.createModule());

    const fw_exe = b.addExecutable(.{
        .name = "firmware",
        .root_module = fw_mod,
    });
    fw_exe.addObjectFile(b.path(crt0_path));
    fw_exe.setLinkerScript(b.path("firmware/linker.ld"));
    b.installArtifact(fw_exe);

    // ELF → flat binary
    // Zig's inbuilt objcopy adds zero padding, see: https://github.com/ziglang/zig/issues/25653
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

    const install_elf = b.addInstallArtifact(fw_exe, .{});
    const fw_step = b.step("firmware", "Build firmware only");
    fw_step.dependOn(&install_bin.step);
    fw_step.dependOn(&install_elf.step);
}
