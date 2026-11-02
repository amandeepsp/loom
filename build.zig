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

    // Host-side unit tests for firmware modules
    const host_target = b.standardTargetOptions(.{});
    const host_optimize = b.standardOptimizeOption(.{});

    const model_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("firmware/model.zig"),
            .target = host_target,
            .optimize = host_optimize,
        }),
    });
    const run_model_tests = b.addRunArtifact(model_tests);

    const test_step = b.step("test", "Run firmware model tests on the host");
    test_step.dependOn(&run_model_tests.step);

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
    const csr_json_abs = b.root.joinString(b.allocator, csr_json_path) catch @panic("OOM");
    const maybe_contents: ?[]u8 = std.Io.Dir.cwd().readFileAlloc(
        b.graph.io,
        csr_json_abs,
        b.allocator,
        .limited(4 * 1024 * 1024),
    ) catch null;
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
    fw_exe.root_module.addObjectFile(b.path(crt0_path));
    fw_exe.setLinkerScript(b.path("firmware/linker.ld"));
    b.installArtifact(fw_exe);

    // ELF → flat binary
    // Zig's inbuilt objcopy adds zero padding, see: https://github.com/ziglang/zig/issues/25653
    const objcopy_prog = b.findProgram(.{
        .names = &.{ "riscv64-elf-objcopy", "riscv64-unknown-elf-objcopy", "riscv64-linux-gnu-objcopy" },
    }) orelse @panic("no riscv objcopy found in PATH");
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
