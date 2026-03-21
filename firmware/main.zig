const uart = @import("uart.zig");
const link = @import("link.zig");
const dispatch = @import("dispatch.zig");
const mmio = @import("mmio.zig");

/// VexRiscv CfuPlugin enable register (CSR 0xBC0, bit 31).
const cfu_ctrl = mmio.Csr(0xBC0);

/// RISC-V trap CSRs.
const mcause = mmio.Csr(0x342);
const mtval = mmio.Csr(0x343);
const mepc = mmio.Csr(0x341);

/// RISC-V mcause exception codes.
const CAUSE_ILLEGAL_INSTRUCTION = 2;
const CAUSE_LOAD_ACCESS_FAULT = 5;
const CAUSE_STORE_ACCESS_FAULT = 7;

/// Sequence ID of the request currently being processed.
/// Used by the trap handler to tag error responses.
var current_seq_id: u16 = 0;

/// Entry point — called from LiteX crt0.S after hardware init.
export fn main() void {
    cfu_ctrl.write(0x80000000);
    uart.init();
    uart.drain_rx();
    uart.write_bytes("[link] ready\n");

    while (true) {
        const header = link.recv_header() catch |err| {
            const code: link.StatusCode = switch (err) {
                error.BadMagic => .bad_magic,
                error.Timeout => .timeout,
            };
            link.send_error(0, code);
            continue;
        };
        current_seq_id = header.seq_id;
        dispatch.dispatch(header);
    }
}

/// Interrupt/trap handler — called from crt0.S trap_entry.
///
/// Mirrors the CFU-Playground pattern (base.c:trap_handler):
///   - Read mcause to identify the trap type
///   - For illegal instructions, report via protocol and advance mepc
///   - For other traps, report a generic trap fault
export fn isr() void {
    const cause = mcause.read();

    // Bit 31 set means it's an interrupt, not an exception.
    if (cause & 0x80000000 != 0) {
        // Async interrupt — nothing to handle for now.
        return;
    }

    const exception_code = cause & 0x7FFFFFFF;

    switch (exception_code) {
        CAUSE_ILLEGAL_INSTRUCTION => {
            // Send the faulting instruction word as the error payload
            // so the host can decode what went wrong.
            const faulting_insn = mtval.read();
            const insn_bytes = @as([4]u8, @bitCast(faulting_insn));
            link.send_response(current_seq_id, .illegal_instruction, &insn_bytes, 0);

            // Advance past the faulting instruction so we don't loop.
            // All our instructions are 32-bit (C extension disabled).
            mepc.write(mepc.read() + 4);
        },
        CAUSE_LOAD_ACCESS_FAULT, CAUSE_STORE_ACCESS_FAULT => {
            const fault_addr = mtval.read();
            const addr_bytes = @as([4]u8, @bitCast(fault_addr));
            link.send_response(current_seq_id, .trap_fault, &addr_bytes, 0);
            mepc.write(mepc.read() + 4);
        },
        else => {
            // Unknown exception — send cause code as payload.
            const cause_bytes = @as([4]u8, @bitCast(cause));
            link.send_response(current_seq_id, .trap_fault, &cause_bytes, 0);
            mepc.write(mepc.read() + 4);
        },
    }
}
