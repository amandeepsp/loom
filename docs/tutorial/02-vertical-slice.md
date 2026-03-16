# Part 2 — Vertical Slice: Host to Hardware and Back

> **Series:** [00-overview](00-overview.md) → [01-mac](01-mac.md) → **[02-vertical-slice](02-vertical-slice.md)** → [03-autonomous](03-autonomous.md) → [04-tinygrad](04-tinygrad.md) → [05-scaling](05-scaling.md)

This is the most important part of the tutorial. You will send a MAC
operation from the host over UART, have the firmware execute it on the
CFU hardware, and send the result back. It's the thinnest possible thread
through every layer of the system.

Everything else builds on this working.

---

## 2.1  Why a Vertical Slice First

Think about how you'd test a GPU for the first time. You wouldn't write
a full ML framework backend. You'd write a tiny kernel — "add two
numbers" — and verify the result comes back correct.

That's what this part is. One operation, end-to-end:

```
  Host Python          UART wire         Firmware (Zig)        CFU Hardware
  ┌──────────┐        ┌────────┐        ┌──────────────┐      ┌──────────┐
  │ Build    │───────►│ bytes  │───────►│ Decode       │─────►│ MAC      │
  │ request  │        │ on the │        │ OpType       │      │ 4 lanes  │
  │ packet   │        │ wire   │        │              │      │          │
  │          │◄───────│        │◄───────│ Send result  │◄─────│ result   │
  └──────────┘        └────────┘        └──────────────┘      └──────────┘
  Layer 4             Layer 2           Layer 1               Layer 0
```

If the echo doesn't work, you know the problem is UART. If the echo
works but the MAC result is wrong, the problem is in the decode or CFU
call. *Narrow the search space by testing each layer independently.*

---

## 2.2  Step 1: UART Echo

The simplest possible firmware. Read a byte, send it back.

### The UART Hardware

Your LiteX SoC has a UART peripheral with memory-mapped CSR registers.
Open `build/sipeed_tang_nano_20k/software/include/generated/csr.h` and
find the UART section.

**🤔 Exercise:** Identify these three registers:

| Register | Address | What it does |
|---|---|---|
| `uart_rxtx` | ? | Read: received byte. Write: send byte. |
| `uart_txfull` | ? | Read: 1 if TX buffer full, 0 if space available |
| `uart_rxempty` | ? | Read: 1 if RX buffer empty, 0 if byte available |

*How do you send a byte?*
1. Poll `uart_txfull` until it reads 0
2. Write the byte to `uart_rxtx`

*How do you receive a byte?*
1. Poll `uart_rxempty` until it reads 0
2. Read the byte from `uart_rxtx`

### Zig on Bare Metal

`firmware/src/link.zig` already imports the CSR headers:

```zig
const csr = @cImport({
    @cInclude("generated/csr.h");
});
```

This makes every C function from `csr.h` callable from Zig. So
`csr.uart_rxtx_write(byte)` sends a byte, and `csr.uart_rxtx_read()`
receives one.

**🤔 Exercise:** Write a `putByte` and `getByte` function in `link.zig`.
Each is 3–4 lines. Then write an echo loop in `main.zig`:

```
  export fn main() void {
      while (true) {
          // 1. receive a byte
          // 2. send it back
      }
  }
```

Build (`zig build`), upload (`zig build load`), test with picocom: type
a character, see it echoed back.

### Bare-Metal Gotchas

Things that will bite you if you're new to `freestanding` Zig:

**No stdout.** `std.debug.print` tries to write to stderr via a syscall.
There are no syscalls on bare metal. If you want debug output, write your
own `print` that loops over bytes and calls `putByte`.

**No allocator.** There is no heap. You have 8 KiB of SRAM total (stack +
`.data` + `.bss`). Use fixed-size buffers.

**Volatile MMIO.** The CSR functions from `csr.h` use `volatile` pointer
accesses, so the compiler won't optimize them away. If you ever write your
own MMIO, use `@as(*volatile u32, @ptrFromInt(addr)).*`.

**Memory clobber.** Look at `cfu.zig`'s inline asm — it uses
`.{ .memory = true }` as a clobber. The MAC instruction has a hidden
internal accumulator. Without the clobber, the compiler might reorder
or eliminate calls it considers redundant.

---

## 2.3  Step 2: Packet Framing

Once echo works, add structure. The link protocol needs to answer:

1. *Is this a valid packet?* → magic byte
2. *What operation?* → OpType
3. *How much data?* → length field
4. *What data?* → payload

**🤔 Exercise:** Sketch the packet format on paper before writing code:

```
  Request packet:
  ┌────────┬────────┬───────────────┬──────────────────┐
  │ 0xCF   │ OpType │ payload_len   │ payload bytes... │
  │ 1 byte │ 1 byte │ 2 bytes (LE)  │ N bytes          │
  └────────┴────────┴───────────────┴──────────────────┘

  Response packet:
  ┌────────┬────────┬───────────────┬──────────────────┐
  │ 0xFC   │ status │ payload_len   │ payload bytes... │
  │ 1 byte │ 1 byte │ 2 bytes (LE)  │ N bytes          │
  └────────┴────────┴───────────────┴──────────────────┘
```

**🤔 Design questions:**

- *Why magic bytes?* If the host sends garbage (wrong baud rate, partial
  packet from a crash), the firmware needs a way to re-synchronize. It
  scans for `0xCF` and discards everything else.

- *Why little-endian?* RISC-V is little-endian. Your host (x86) is also
  little-endian. No byte-swapping needed.

- *Do you need a checksum?* At 115200 baud over a short USB cable, bit
  errors are extremely rare. Skip it for now. Add it later if you have
  reliability problems.

- *What should happen on an unknown OpType?* Options: (a) ignore it (host
  hangs), (b) send an error response (robust), (c) reset. *Which did you
  pick? Why?*

### Firmware Implementation

In `link.zig`, you need functions like:

```
  recvPacket() → struct { optype, payload }
  sendResponse(status, payload)
```

**Don't implement these yet.** First, test packet framing in isolation.
Write a firmware that receives a request and echoes back a response with
the same payload. Test with a Python script on the host.

### Host-Side Test

Write a Python script that:
1. Opens the serial port at the correct baud rate
2. Sends a request packet (magic + optype + length + payload)
3. Waits for response magic byte
4. Reads the response

**🤔 Exercise:** What baud rate does your LiteX SoC use? Check the SoC
build configuration. Getting this wrong means garbage.

---

## 2.4  Step 3: MAC Execution

Wire it together. The firmware receives a `mac4` request and executes it.

### Request Payload

For a MAC operation, the host sends:

```
  ┌──────────┬───────────────┬────────────────┬────────────────┐
  │  offset  │ element_count │ input_data     │ weight_data    │
  │  1 byte  │ 2 bytes (LE)  │ N bytes        │ N bytes        │
  └──────────┴───────────────┴────────────────┴────────────────┘
```

### Firmware Processing

```
  1. Read offset, element_count
  2. For each group of 4 elements:
     a. Read 4 input bytes from UART → pack into u32
     b. Read 4 weight bytes from UART → pack into u32
     c. Call cfu.mac4(accumulator, packed_input, packed_weight)
  3. Send back the i32 accumulator as 4 bytes
```

**🤔 Exercise:** Trace a concrete example:

```
  input   = [1, 2, 3, 4, 5, 6, 7, 8]    (8 elements)
  weights = [1, 1, 1, 1, 1, 1, 1, 1]
  offset  = 128

  First chunk (elements 0–3):
    in0 = pack(1, 2, 3, 4)
    in1 = pack(1, 1, 1, 1)
    MAC = (1+128)×1 + (2+128)×1 + (3+128)×1 + (4+128)×1
        = 129 + 130 + 131 + 132
        = 522

  Second chunk (elements 4–7):
    in0 = pack(5, 6, 7, 8)
    in1 = pack(1, 1, 1, 1)
    MAC = (5+128)×1 + (6+128)×1 + (7+128)×1 + (8+128)×1
        = 133 + 134 + 135 + 136
        = 538

  Total accumulator = 522 + 538 = 1060
```

*How many `cfu_call` invocations?* 2.
*How many UART bytes in the request?* 1 (magic) + 1 (optype) + 2 (len) +
1 (offset) + 2 (count) + 8 (input) + 8 (weights) = 23 bytes.
*How many in the response?* 1 + 1 + 2 + 4 = 8 bytes.

**🤔 Exercise:** *Where does the data live during processing?* You're
reading bytes from UART and packing them 4 at a time. Do you need a buffer
for the whole input, or can you process in streaming fashion? (Hint: the
MAC accumulates — you only need 4 bytes at a time.)

### The Alternative: Batch Mode

Instead of streaming bytes from UART directly into the MAC, you could
receive all data first into a buffer, then process. The tradeoff:

```
  Streaming:                      Buffered:
  • No buffer needed              • Needs N + N bytes of SRAM
  • Can't retry on error          • Can verify checksum before processing
  • Simpler code                  • Can process data in any order
```

For an 8 KiB SRAM budget, streaming is safer. A 4096-element MAC would
need 8 KiB of buffer — your entire SRAM.

---

## 2.5  Step 4: Host-Side Test

Write a standalone Python script — no TinyGrad yet:

```
  1. Open serial port
  2. Build a mac4 request: magic + optype + length + offset + count + input + weights
  3. Send it
  4. Wait for response magic byte
  5. Read the 4-byte result
  6. Compare against a NumPy reference:
       expected = np.sum((input.astype(np.int16) + offset) * weights.astype(np.int16))
  7. Print PASS or FAIL
```

**🤔 Exercise:** Don't just test one case. Test:
- All zeros (should give 0)
- All ones with offset=128 (known result)
- Maximum values (input=127, weight=127, offset=128 → check for overflow)
- Offset other than 128

---

## 2.6  The Bottleneck You Just Discovered

Your vertical slice works. Congratulations. Now measure the round-trip time.

**🤔 Exercise:** For the 8-element example:

```
  Total bytes on the wire: 23 (request) + 8 (response) = 31 bytes
  At 115200 baud (8N1): 1 byte ≈ 87 µs
  Round-trip wire time: 31 × 87 µs ≈ 2.7 ms

  The actual MAC computation: 2 clock cycles × 37 ns/cycle = 74 ns

  You spent 2.7 ms transferring data for 74 ns of compute.
  Transfer / compute ratio: 36,000 : 1
```

This is the fundamental tension of selective lowering. The decision:

```
  Time_on_host  vs  Time_transfer + Time_compute

  NumPy on x86: 8 MACs ≈ 8 ns (pipelined SIMD)
  Over UART:    2,700,000 ns

  The host is ~340,000× faster for 8 elements.
```

**🤔 Exercise:** At what element count does UART offloading break even
against NumPy? Set up the equation:

```
  Host time ≈ N / 1e9  (seconds, ~1 GHz throughput)
  UART time ≈ (2N + 8) / 11500  (seconds, request + response bytes)

  Solve for N where UART_time < Host_time.
```

*The answer will surprise you.* UART is ~11.5 KB/s. NumPy on x86 does
billions of operations per second. **UART will never break even for raw
compute.** The value of the offload comes from:

1. **Freeing the host CPU** — it can do other work while the FPGA computes
2. **Power efficiency** — the FPGA uses milliwatts
3. **Learning** — you're building the architecture that matters at higher
   bandwidth (SPI, PCIe, integrated SoC)

This motivates:
- **Part 3:** Keep data on-chip (BSRAM stores) to avoid repeated transfers
- **Part 4:** Only lower operations where the compute-to-transfer ratio
  justifies it
- **Part 5:** Higher-bandwidth interfaces (PSRAM, DMA)

---

## 2.7  Adding a New Operation

Once `mac4` works, the pattern for adding any new operation is:

```
  1. hardware/   → New Instruction subclass, wire to funct3 slot N
  2. cfu.zig     → New inline function with funct3=N
  3. link.zig    → New OpType variant
  4. main.zig    → New branch in the dispatch switch
  5. Host Python → New request builder + reference implementation + test
```

**🤔 Exercise:** Walk through this for a hypothetical `relu` operation:
element-wise `max(x, 0)` on 4 packed INT8 bytes. What does each file
need? What's the packet format? How many bytes in, how many out?

---

## 2.8  Checkpoint

- [ ] UART echo firmware works (type → echo in picocom)
- [ ] Packet framing sends and receives correctly
- [ ] MAC over serial produces correct results (multiple test cases)
- [ ] Host-side Python test passes
- [ ] I have round-trip timing numbers
- [ ] I understand why UART bandwidth is the bottleneck
- [ ] I can explain the tradeoff: when does offloading help vs hurt?

---

**Previous:** [Part 1 — The MAC](01-mac.md)
**Next:** [Part 3 — Autonomous Compute: Taking the CPU Off the Hot Path](03-autonomous.md)
