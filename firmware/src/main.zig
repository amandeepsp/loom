const uart = @import("uart.zig");
const link = @import("link.zig");
const dispatch = @import("dispatch.zig");
const mmio = @import("mmio.zig");

/// VexRiscv CfuPlugin enable register (CSR 0xBC0, bit 31).
const cfu_ctrl = mmio.Csr(0xBC0);

/// Entry point — called from LiteX crt0.S after hardware init.
export fn main() void {
    cfu_ctrl.write(0x80000000);
    uart.init();
    uart.drain_rx();
    uart.write_bytes("[link] ready\n");

    while (true) {
        const header = link.recv_header() catch continue;
        dispatch.dispatch(header);
    }
}

/// Interrupt handler — called from crt0.S trap_entry.
export fn isr() void {}
