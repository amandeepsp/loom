const csr = @import("csr");
const mmio = @import("mmio.zig");
const cpu_csr = @import("cpu_csr.zig");

/// LiteDRAMDMAReader's `done` CSR is asserted when the final read request has
/// been accepted, before all outstanding responses have necessarily reached
/// the CFU scratchpad bridge. Allow the bounded response/FIFO pipeline to
/// drain before swapping the scratchpad banks and starting compute.
pub inline fn drainFence() void {
    const start = cpu_csr.mcycle.read();
    while (cpu_csr.mcycle.read() -% start < 512) {}
}

fn DmaImpl(
    comptime base_addr: usize,
    comptime length_addr: usize,
    comptime enable_addr: usize,
    comptime done_addr: usize,
    comptime loop_addr: usize,
    comptime offset_addr: usize,
) type {
    return struct {
        const base_reg = mmio.Reg(base_addr);
        const length_reg = mmio.Reg(length_addr);
        const enable_reg = mmio.Reg(enable_addr);
        const done_reg = mmio.Reg(done_addr);
        const loop_reg = mmio.Reg(loop_addr);
        const offset_reg = mmio.Reg(offset_addr);

        pub inline fn configure(base_ptr: u32, len: u32) void {
            base_reg.write(base_ptr);
            length_reg.write(len);
        }

        pub inline fn setLoop(enabled: bool) void {
            loop_reg.write(@intFromBool(enabled));
        }

        pub inline fn start() void {
            enable_reg.write(1);
        }

        pub inline fn kick(base_ptr: u32, len: u32) void {
            stop();
            setLoop(false);
            configure(base_ptr, len);
            start();
        }

        pub inline fn stop() void {
            enable_reg.write(0);
        }

        pub inline fn isDone() bool {
            return done_reg.read() != 0;
        }

        pub inline fn wait() void {
            while (!isDone()) {}
        }

        pub inline fn offsetBytes() u32 {
            return offset_reg.read();
        }
    };
}

pub const Act = DmaImpl(
    csr.act_dma_base,
    csr.act_dma_length,
    csr.act_dma_enable,
    csr.act_dma_done,
    csr.act_dma_loop,
    csr.act_dma_offset,
);

pub const Wgt = DmaImpl(
    csr.wgt_dma_base,
    csr.wgt_dma_length,
    csr.wgt_dma_enable,
    csr.wgt_dma_done,
    csr.wgt_dma_loop,
    csr.wgt_dma_offset,
);
