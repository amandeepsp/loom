const std = @import("std");
const cpu_csr = @import("cpu_csr.zig");
const cfu = @import("cfu.zig");
const dma = @import("dma.zig");
const config = @import("config");
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

pub fn execute(payload_len: u16, debug_buf: ?[]u8) ExecError!u32 {
    var ctx = StreamContext.init(payload_len);
    errdefer ctx.drainRemaining();

    const header = try ctx.read(ir.ProgramHeader);
    if (header.magic != ir.program_magic) {
        if (debug_buf) |buf| {
            buf[0] = @intCast(payload_len);
            buf[1] = @intCast(ctx.remaining);
            buf[2] = @truncate(header.magic >> 24);
            buf[3] = @intCast(header.version);
        }
        return error.BadMagic;
    }
    if (header.version != ir.program_version) {
        if (debug_buf) |buf| {
            buf[0] = @intCast(payload_len);
            buf[1] = @intCast(ctx.remaining);
            buf[2] = @intCast(header.version);
            buf[3] = @truncate(header.magic >> 24);
        }
        return error.BadPayloadLen;
    }
    if (header.num_tensors > max_tensors) {
        if (debug_buf) |buf| {
            buf[0] = @intCast(payload_len);
            buf[1] = @intCast(ctx.remaining);
            buf[2] = header.num_tensors;
            buf[3] = @intCast(header.version);
        }
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
    var instr_idx: u16 = 0;

    for (0..header.num_instructions) |_| {
        const opcode = try ctx.readOpcode();
        if (debug_buf) |buf| {
            buf[0] = @intCast(payload_len);
            buf[1] = @intCast(ctx.remaining);
            buf[2] = @intCast(header.num_instructions);
            buf[3] = @truncate(instr_idx);
            buf[4] = @intFromEnum(opcode);
        }
        instr_idx +%= 1;

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
                    dma.Act.stop();
                    dma.Wgt.stop();
                    state.compute_pending = false;
                }
                saw_done = true;
                break;
            },
        }
    }

    if (!saw_done or ctx.remaining != 0) {
        if (debug_buf) |buf| {
            buf[0] = @intCast(payload_len);
            buf[1] = @intCast(ctx.remaining);
            buf[2] = @intFromBool(saw_done);
            buf[3] = @truncate(instr_idx);
        }
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

    const len_bytes = try mulU32(inst.k_words, 8);
    if (!memory.rangeValid(addr, len_bytes)) return error.BadAddress;

    dma.Act.kick(addr, len_bytes);
}

fn tileLoadWgt(descs: []const ir.TensorDescriptor, inst: ir.TileLoadWgt) ExecError!void {
    const desc = try getTensor(descs, inst.tensor_id);
    if (desc.dtype != .signed8) return error.BadAddress;
    if (inst.n_offset >= desc.dim1) return error.BadAddress;
    if (inst.k_words == 0 or inst.k_words > tensor_depth) return error.BadAddress;

    const tile_size: u32 = @as(u32, array_cols);
    const n_tile = @as(u32, inst.n_offset) / tile_size;
    const n_tile_offset = try mulU32(n_tile, try mulU32(desc.dim0, tile_size));
    const row_offset = try mulU32(inst.k_offset, tile_size);
    var addr = try addU32(desc.base_addr, n_tile_offset);
    addr = try addU32(addr, row_offset);

    const len_bytes = try mulU32(inst.k_words, tile_size);
    if (!memory.rangeValid(addr, len_bytes)) return error.BadAddress;

    dma.Wgt.kick(addr, len_bytes);
}

fn tileMma(state: *PipeState, inst: ir.TileMma) void {
    if (state.compute_pending) {
        cfu.computeWait();
        state.compute_pending = false;
    }

    dma.Act.wait();
    dma.Wgt.wait();

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
        dma.Act.stop();
        dma.Wgt.stop();
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
            const result_index: i32 = @intCast(@as(u32, @intCast(row)) * @as(u32, array_cols) + @as(u32, @intCast(col)));
            const result = cfu.readResult(result_index);
            dst[col] = @truncate(result);
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

    // The hardware epilogue always processes all array_rows * array_cols channels
    // (sequencer EPILOGUE state walks the full psum grid), so we must write params
    // for every row even when tileStore only reads m_count of them.
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
