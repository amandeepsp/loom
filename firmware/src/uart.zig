/// UART driver for LiteX SoC.

const csr = @import("csr");
const mmio = @import("mmio.zig");

const rxtx = mmio.Reg(csr.uart_rxtx);
const txfull = mmio.Reg(csr.uart_txfull);
const rxempty = mmio.Reg(csr.uart_rxempty);
const ev_pending = mmio.Reg(csr.uart_ev_pending);
const ev_enable = mmio.Reg(csr.uart_ev_enable);

/// Clear pending events and enable TX/RX.
pub fn init() void {
    ev_enable.write(0);
    ev_pending.write(ev_pending.read());
    ev_enable.write(0x3); // UART_EV_TX | UART_EV_RX
}

pub fn write_byte(b: u8) void {
    while (txfull.read() != 0) {}
    rxtx.write(b);
}

pub fn read_byte_blocking() u8 {
    while (rxempty.read() != 0) {}
    const b: u8 = @truncate(rxtx.read());
    ev_pending.write(0x2); // ack UART_EV_RX
    return b;
}

pub fn write_bytes(buf: []const u8) void {
    for (buf) |b| write_byte(b);
}

pub fn read_bytes(buf: []u8) void {
    for (buf) |*b| b.* = read_byte_blocking();
}

/// Drain any stale bytes from the RX FIFO.
pub fn drain_rx() void {
    while (rxempty.read() == 0) {
        _ = rxtx.read();
        ev_pending.write(0x2);
    }
}
