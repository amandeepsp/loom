/// Memory-mapped I/O and CSR access primitives.
///
/// Inspired by microzig's Mmio(PackedT) and Csr(addr, T) patterns.
/// See: https://github.com/ZigEmbeddedGroup/microzig

const std = @import("std");

/// Memory-mapped I/O register at a fixed address (32-bit, volatile).
pub fn Reg(comptime addr: usize) type {
    return struct {
        pub inline fn read() u32 {
            return @as(*const volatile u32, @ptrFromInt(addr)).*;
        }
        pub inline fn write(val: u32) void {
            @as(*volatile u32, @ptrFromInt(addr)).* = val;
        }
    };
}

/// RISC-V CSR access at a compile-time address via inline csrr/csrw.
pub fn Csr(comptime addr: u12) type {
    const ident = std.fmt.comptimePrint("{}", .{addr});
    return struct {
        pub inline fn read() u32 {
            return asm volatile ("csrr %[ret], " ++ ident
                : [ret] "=r" (-> u32),
            );
        }
        pub inline fn write(val: u32) void {
            asm volatile ("csrw " ++ ident ++ ", %[v]"
                :
                : [v] "r" (val),
            );
        }
    };
}
