const std = @import("std");

const BUILD_DIR = "../build/sipeed_tang_nano_20k";

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{
        .default_target = .{
            .cpu_arch = .riscv32,
            .os_tag = .freestanding,
            .abi = .ilp32,
            .cpu_features_add = std.Target.riscv.featureSet(&.{
                .m, // integer muliply/divide
            }),
            .cpu_features_sub = std.Target.riscv.featureSet(&.{
                .f, .d, // no floats
            }),
        },
    });

    const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseSmall });

    const exe = b.addExecutable(.{
        .name = "firmware",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    exe.addObjectFile(b.path(BUILD_DIR ++ "/software/bios/crt0.o"));
    exe.setLinkerScript(b.path("linker.ld"));

    b.installArtifact(exe);

    // elf -> bin
    const bin = exe.addObjCopy(.{ .format = .bin });
    const install_bin = b.addInstallBinFile(bin.getOutput(), "firmware.bin");
    b.installArtifact(exe);
    b.getInstallStep().dependOn(&install_bin.step);

    const load_step = b.step("load", "Load firmware via picocom");
    const load = b.addSystemCommand(&.{
        "picocom",
        "/dev/ttyUSB1",
        "--kernel",
        b.getInstallPath(.bin, "firmware.bin"),
    });
    load_step.dependOn(b.getInstallStep());
    load_step.dependOn(&load.step);
}
