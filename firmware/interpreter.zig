const std = @import("std");
const cpu_csr = @import("cpu_csr.zig");
const cfu = @import("cfu.zig");
const dma = @import("dma.zig");
const config = @import("fw_config");
const ir = @import("ir");
const link = @import("link.zig");
const memory = @import("memory.zig");
const uart = @import("uart.zig");

const max_tensors = 32;

// Sourced from build.zig so the firmware matches the current CFU build.
const array_rows = config.cfu_rows;
const array_cols = config.cfu_cols;
const tensor_depth = config.cfu_store_depth;

pub const ExecError = error{
    BadMagic,
    BadPayloadLen,
    BadAddress,
};

const PipeState = struct {
    compute_pending: bool = false,
};

const StreamContext = struct {
    remaining: usize,

    fn init(len: usize) StreamContext {
        return .{ .remaining = len };
    }

    fn read(self: *StreamContext, comptime T: type) ExecError!T {
        if (self.remaining < @sizeOf(T)) return error.BadPayloadLen;

        var value: T = undefined;
        uart.readBytes(std.mem.asBytes(&value));
        self.remaining -= @sizeOf(T);
        return value;
    }

    fn readOpcode(self: *StreamContext) ExecError!ir.InstructionType {
        if (self.remaining < 1) return error.BadPayloadLen;

        const raw = uart.readByte();
        self.remaining -= 1;
        return std.meta.intToEnum(ir.InstructionType, raw) catch error.BadPayloadLen;
    }

    fn readInstruction(self: *StreamContext, comptime T: type, opcode: ir.InstructionType) ExecError!T {
        if (self.remaining < @sizeOf(T) - 1) return error.BadPayloadLen;

        var value: T = undefined;
        const bytes = std.mem.asBytes(&value);
        bytes[0] = @intFromEnum(opcode);
        if (@sizeOf(T) > 1) {
            uart.readBytes(bytes[1..]);
        }
        self.remaining -= @sizeOf(T) - 1;
        return value;
    }

    fn drainRemaining(self: *StreamContext) void {
        if (self.remaining > 0) {
            link.drainPayload(self.remaining);
            self.remaining = 0;
        }
    }
};

pub fn execute(payload_len: u16, debug_buf: []u8) ExecError!u32 {
    var ctx = StreamContext.init(payload_len);
    errdefer ctx.drainRemaining();

    const header = try ctx.read(ir.ProgramHeader);
    if (header.magic != ir.program_magic) {
        debug_buf[0] = @intCast(payload_len);
        debug_buf[1] = @intCast(ctx.remaining);
        debug_buf[2] = @truncate(header.magic >> 24);
        debug_buf[3] = @intCast(header.version);
        return error.BadMagic;
    }
    if (header.version != ir.program_version) {
        debug_buf[0] = @intCast(payload_len);
        debug_buf[1] = @intCast(ctx.remaining);
        debug_buf[2] = @intCast(header.version);
        debug_buf[3] = @truncate(header.magic >> 24);
        return error.BadPayloadLen;
    }
    if (header.num_tensors > max_tensors) {
        debug_buf[0] = @intCast(payload_len);
        debug_buf[1] = @intCast(ctx.remaining);
        debug_buf[2] = header.num_tensors;
        debug_buf[3] = @intCast(header.version);
        return error.BadPayloadLen;
    }

    var descs: [max_tensors]ir.TensorDescriptor = undefined;
    for (descs[0..header.num_tensors]) |*desc| {
        desc.* = try ctx.read(ir.TensorDescriptor);
    }

    var state = PipeState{};
    errdefer if (state.compute_pending) cfu.computeWait();

    const cycle_start = cpu_csr.mcycle.read();
    var saw_done = false;
    var instr_idx: u8 = 0;

    for (0..header.num_instructions) |_| {
        const opcode = try ctx.readOpcode();
        debug_buf[0] = @intCast(payload_len);
        debug_buf[1] = @intCast(ctx.remaining);
        debug_buf[2] = @intCast(header.num_instructions);
        debug_buf[3] = instr_idx;
        debug_buf[4] = @intFromEnum(opcode);
        instr_idx += 1;

        switch (opcode) {
            .tile_load_act => try tileLoadAct(
                descs[0..header.num_tensors],
                try ctx.readInstruction(ir.TileLoadAct, opcode),
            ),
            .tile_load_wgt => try tileLoadWgt(
                descs[0..header.num_tensors],
                try ctx.readInstruction(ir.TileLoadWgt, opcode),
            ),
            .tile_mma => tileMma(&state, try ctx.readInstruction(ir.TileMma, opcode)),
            .tile_store => try tileStore(
                &state,
                descs[0..header.num_tensors],
                try ctx.readInstruction(ir.TileStore, opcode),
            ),
            .set_epilogue => try setEpilogue(
                descs[0..header.num_tensors],
                try ctx.readInstruction(ir.SetEpilogue, opcode),
            ),
            .done => {
                _ = try ctx.readInstruction(ir.Done, opcode);
                if (state.compute_pending) {
                    cfu.computeWait();
                    const act_dma = dma.Act.init();
                    const wgt_dma = dma.Wgt.init();
                    act_dma.stop();
                    wgt_dma.stop();
                    state.compute_pending = false;
                }
                saw_done = true;
                break;
            },
        }
    }

    if (!saw_done or ctx.remaining != 0) {
        debug_buf[0] = @intCast(payload_len);
        debug_buf[1] = @intCast(ctx.remaining);
        debug_buf[2] = @intFromBool(saw_done);
        debug_buf[3] = instr_idx;
        return error.BadPayloadLen;
    }
    return cpu_csr.mcycle.read() -% cycle_start;
}

fn tileLoadAct(descs: []const ir.TensorDescriptor, inst: ir.TileLoadAct) ExecError!void {
    const desc = try getTensor(descs, inst.tensor_id);
    if (desc.dtype != .signed8) return error.BadAddress;
    if (inst.m_offset >= desc.dim0) return error.BadAddress;
    if (inst.k_words == 0 or inst.k_words > tensor_depth) return error.BadAddress;

    const row_offset = try mulU32(inst.m_offset, desc.stride_row);
    var addr = try addU32(desc.base_addr, row_offset);
    addr = try addU32(addr, inst.k_offset);

    const len_bytes = try mulU32(inst.k_words, 4);
    if (!memory.rangeValid(addr, len_bytes)) return error.BadAddress;

    const act_dma = dma.Act.init();
    act_dma.kick(addr, len_bytes);
}

fn tileLoadWgt(descs: []const ir.TensorDescriptor, inst: ir.TileLoadWgt) ExecError!void {
    const desc = try getTensor(descs, inst.tensor_id);
    if (desc.dtype != .signed8) return error.BadAddress;
    if (inst.n_offset >= desc.dim1) return error.BadAddress;
    if (inst.k_words == 0 or inst.k_words > tensor_depth) return error.BadAddress;

    const row_offset = try mulU32(inst.k_offset, desc.stride_row);
    var addr = try addU32(desc.base_addr, row_offset);
    addr = try addU32(addr, inst.n_offset);

    const len_bytes = try mulU32(inst.k_words, 4);
    if (!memory.rangeValid(addr, len_bytes)) return error.BadAddress;

    const wgt_dma = dma.Wgt.init();
    wgt_dma.kick(addr, len_bytes);
}

fn tileMma(state: *PipeState, inst: ir.TileMma) void {
    if (state.compute_pending) {
        cfu.computeWait();
    }

    const act_dma = dma.Act.init();
    const wgt_dma = dma.Wgt.init();
    act_dma.wait();
    wgt_dma.wait();

    // TODO: Investigate if hardware supports this
    if (!inst.flags.last) {
        act_dma.setLoop(true);
        wgt_dma.setLoop(true);
    }

    cfu.computeStart(inst.flags.first, inst.flags.last, inst.k_count);
    state.compute_pending = true;
}

fn tileStore(state: *PipeState, descs: []const ir.TensorDescriptor, inst: ir.TileStore) ExecError!void {
    const desc = try getTensor(descs, inst.tensor_id);
    if (desc.dtype != .signed8) return error.BadAddress;
    if (inst.m_count == 0 or inst.n_count == 0) return error.BadAddress;
    if (inst.m_count > array_rows or inst.n_count > array_cols) return error.BadAddress;
    if (@as(u32, inst.m_offset) + inst.m_count > desc.dim0) return error.BadAddress;
    if (@as(u32, inst.n_offset) + inst.n_count > desc.dim1) return error.BadAddress;

    if (state.compute_pending) {
        cfu.computeWait();
        const act_dma = dma.Act.init();
        const wgt_dma = dma.Wgt.init();
        // TODO: Without this we were stuck, why?
        act_dma.stop();
        wgt_dma.stop();
        state.compute_pending = false;
    }

    for (0..inst.m_count) |row| {
        const row_index = @as(u32, inst.m_offset) + @as(u32, @intCast(row));
        const row_offset = try mulU32(row_index, desc.stride_row);
        var dst_addr = try addU32(desc.base_addr, row_offset);
        dst_addr = try addU32(dst_addr, inst.n_offset);
        if (!memory.rangeValid(dst_addr, inst.n_count)) return error.BadAddress;

        const dst: [*]i8 = @ptrFromInt(dst_addr);
        for (0..inst.n_count) |col| {
            const global_row: u32 = @as(u32, inst.m_offset) + @as(u32, @intCast(row));
            const global_col: u32 = @as(u32, inst.n_offset) + @as(u32, @intCast(col));
            const result_index: i32 = @intCast(global_row * @as(u32, array_cols) + global_col);
            dst[col] = @truncate(cfu.readResult(result_index));
        }
    }
}

fn setEpilogue(descs: []const ir.TensorDescriptor, inst: ir.SetEpilogue) ExecError!void {
    const bias_desc = try getTensor(descs, inst.bias_tid);
    const mult_desc = try getTensor(descs, inst.mult_tid);
    const shift_desc = try getTensor(descs, inst.shift_tid);

    if (bias_desc.dtype != .signedi32 or mult_desc.dtype != .signedi32 or shift_desc.dtype != .signedi32) {
        return error.BadAddress;
    }
    if (inst.n_count == 0 or inst.n_count > array_cols) return error.BadAddress;

    const bias_base = try addU32(bias_desc.base_addr, try mulU32(inst.n_offset, 4));
    const mult_base = try addU32(mult_desc.base_addr, try mulU32(inst.n_offset, 4));
    const shift_base = try addU32(shift_desc.base_addr, try mulU32(inst.n_offset, 4));
    const span_bytes = try mulU32(inst.n_count, 4);

    if (!memory.rangeValid(bias_base, span_bytes)) return error.BadAddress;
    if (!memory.rangeValid(mult_base, span_bytes)) return error.BadAddress;
    if (!memory.rangeValid(shift_base, span_bytes)) return error.BadAddress;

    for (0..array_rows) |row| {
        for (0..inst.n_count) |col| {
            const param_addr_offset = try mulU32(@as(u32, @intCast(col)), 4);
            const channel = @as(i32, @intCast(row * array_cols + col));
            cfu.writeEpiParam(.bias, channel, loadI32(try addU32(bias_base, param_addr_offset)));
            cfu.writeEpiParam(.multiplier, channel, loadI32(try addU32(mult_base, param_addr_offset)));
            cfu.writeEpiParam(.shift, channel, loadI32(try addU32(shift_base, param_addr_offset)));
        }
    }

    cfu.setOutputOffset(inst.output_offset);
    cfu.setActivationMin(inst.act_min);
    cfu.setActivationMax(inst.act_max);
}

fn getTensor(descs: []const ir.TensorDescriptor, tensor_id: u8) ExecError!ir.TensorDescriptor {
    if (tensor_id >= descs.len) return error.BadAddress;
    return descs[tensor_id];
}

fn addU32(lhs: anytype, rhs: anytype) ExecError!u32 {
    return std.math.add(u32, lhs, rhs) catch error.BadAddress;
}

fn mulU32(lhs: anytype, rhs: anytype) ExecError!u32 {
    return std.math.mul(u32, lhs, rhs) catch error.BadAddress;
}

fn loadI32(addr: u32) i32 {
    return @as(*const i32, @ptrFromInt(addr)).*;
}
