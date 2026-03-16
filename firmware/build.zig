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

    // crt0.o — pre-built from LiteX's crt0.S (required)
    //   riscv64-elf-gcc -c -march=rv32im_zicsr -mabi=ilp32 -nostdlib \
    //     $(python -c "import litex.soc.cores.cpu.vexriscv; import os; print(os.path.join(os.path.dirname(litex.soc.cores.cpu.vexriscv.__file__), 'crt0.S'))") \
    //     -o crt0.o
    const crt0_path = b.option([]const u8, "crt0", "Path to pre-built crt0.o") orelse {
        std.debug.print("error: -Dcrt0=<path> is required\n", .{});
        std.debug.print("build crt0.o from LiteX's crt0.S:\n", .{});
        std.debug.print("  riscv64-elf-gcc -c -march=rv32im_zicsr -mabi=ilp32 -nostdlib <vexriscv>/crt0.S -o crt0.o\n", .{});
        std.process.exit(1);
    };

    // --- Build firmware ELF ---
    const exe = b.addExecutable(.{
        .name = "firmware",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    exe.addObjectFile(.{ .cwd_relative = crt0_path });
    exe.setLinkerScript(b.path("linker.ld"));

    b.installArtifact(exe);

    // --- ELF → flat binary (GNU objcopy; Zig 0.15's built-in is buggy) ---
    const objcopy_prog = b.findProgram(
        &.{ "riscv64-elf-objcopy", "riscv64-unknown-elf-objcopy", "riscv32-elf-objcopy" },
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
