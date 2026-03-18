# Unit 2: The Data Path — Drivers and Command Submission

> **Course:** [01-compute](01-compute.md) | **02-datapath** | [03-fusion](03-fusion.md) | [04-autonomy](04-autonomy.md)
>
> Learn how ML compilers and GPU hardware work, by building a tiny version on a Tang Nano 20K FPGA.

---

## The Real-World Problem

On a real GPU, launching a kernel costs ~5-10 microseconds *even before any compute happens*. That overhead is the driver stack:

```
  Userspace API (CUDA/Metal/Vulkan)
      │
      ▼
  Kernel-mode driver
      │
      ▼
  Command buffer (batch of GPU instructions)
      │
      ▼
  PCIe bus transfer
      │
      ▼
  GPU command processor / scheduler
      │
      ▼
  Shader execution on SMs / CUs
      │
      ▼
  DMA result back to host memory
```

Every layer adds latency. The command buffer is the central abstraction: the host-side driver packs multiple operations into a structured binary blob, sends it across the bus in one shot, and the device-side command processor decodes and executes it. The *format* of that buffer — magic numbers, operation codes, payload lengths — is the contract between driver and device.

This matters for ML because frameworks like PyTorch and JAX issue thousands of kernel launches per inference pass. If each launch costs 10 microseconds, 1000 launches cost 10 milliseconds of pure overhead — no compute, just setup. This is why CUDA Graphs, XLA fusion, and `torch.compile` exist: they amortise launch overhead by batching.

> **🔗 MLSys Connection:** CUDA's `cudaLaunchKernel()` writes a command into a stream's command buffer. The GPU's command processor reads these asynchronously. Your UART packet framing is the same abstraction: a structured binary message that tells the hardware what to do, sent over a serial link instead of PCIe.

---

## Your Stack: Driver (Host) → Bus → Command Processor (Device)

You are building the same layered architecture, just smaller:

```
  Host Python (your "userspace API")
      │
      ▼
  Serial protocol (your "command buffer format")
      │
      ▼
  UART wire at 115200 baud (your "PCIe bus")
      │
      ▼
  Firmware dispatch loop (your "command processor")
      │
      ▼
  CFU hardware execution (your "shader core")
      │
      ▼
  UART response back (your "DMA back to host")
```

Same pattern. Same concerns: framing, synchronisation, error handling, latency measurement.

The project already has a working implementation of this stack:

| Layer | GPU Equivalent | Your File |
|---|---|---|
| Host-side driver | CUDA runtime / kernel driver | `host/lib/protocol.py`, `host/client.py` |
| Wire protocol | Command buffer format | `host/lib/protocol.py` (format), `firmware/src/link.zig` (parser) |
| Device command processor | GPU command processor | `firmware/src/dispatch.zig` |
| Compute hardware | Shader/tensor core | `hardware/cfu.py` via `firmware/src/cfu.zig` |

---

## Building the Vertical Slice

The goal: send a single MAC4 instruction from the host, get the correct result back. This is your "Hello World" kernel launch.

### Step 1: UART Echo — Prove the Wire Works

Before sending structured commands, prove that bytes survive the round trip.

**Exercise 1.1:** Write a bare-metal firmware that echoes every byte it receives.

The LiteX SoC exposes UART as memory-mapped registers. You need three:

| Register | Purpose |
|---|---|
| `uart_rxtx` | Read: received byte. Write: send byte. |
| `uart_txfull` | 1 if TX buffer full, 0 if space available |
| `uart_rxempty` | 1 if RX buffer empty, 0 if byte available |

Your echo loop is 4 lines of Zig. Build it, upload it, test with picocom.

<details><summary>Hint 1</summary>

To receive a byte: spin on `rxempty` until it reads 0, then read from `rxtx`.
To send a byte: spin on `txfull` until it reads 0, then write to `rxtx`.

</details>

<details><summary>Hint 2</summary>

Look at `firmware/src/uart.zig` for the MMIO register addresses and the `read_byte_blocking()` / `write_byte()` pattern. The addresses come from LiteX's generated CSR map.

</details>

<details><summary>Solution</summary>See `firmware/src/uart.zig` for the complete UART driver, and `firmware/src/main.zig` for the main loop structure.</details>

> **🔗 MLSys Connection:** GPU bring-up follows the same pattern. Before running any kernels, engineers verify PCIe link integrity with loopback tests. If echo doesn't work, nothing else matters.

---

### Step 2: Packet Framing — Your Command Buffer Format

Raw bytes aren't enough. You need structure: *is this a valid packet? What operation? How much data?*

**Exercise 2.1:** Study the packet format, then implement send/receive on both sides.

**Request packet:**
```
┌────────┬────────┬───────────────┬───────────┬───────────┬──────────────────┐
│ 0xCF   │ OpType │ payload_len   │ seq_id    │ reserved  │ payload bytes... │
│ 1 byte │ 1 byte │ 2 bytes (LE)  │ 2 bytes   │ 2 bytes   │ N bytes          │
└────────┴────────┴───────────────┴───────────┴───────────┴──────────────────┘
```

**Response packet:**
```
┌────────┬────────┬───────────────┬───────────┬───────────┬──────────────────┐
│ 0xFC   │ status │ payload_len   │ seq_id    │ cycles_lo │ payload bytes... │
│ 1 byte │ 1 byte │ 2 bytes (LE)  │ 2 bytes   │ 2 bytes   │ N bytes          │
└────────┴────────┴───────────────┴───────────┴───────────┴──────────────────┘
```

**Design questions to answer before writing code:**

- Why magic bytes? (Think: what happens after a firmware crash or baud rate mismatch?)
- Why little-endian? (Think: what endianness is RISC-V? What about your x86 host?)
- Why a sequence ID? (Think: what if responses arrive out of order, or a response is lost?)
- What should happen on an unknown OpType?

<details><summary>Hint 1</summary>

The magic byte `0xCF` lets the firmware re-synchronise after garbage. It scans byte-by-byte until it sees `0xCF`, then reads the remaining 7 header bytes.

</details>

<details><summary>Hint 2</summary>

On the host side, `host/lib/protocol.py` has `make_request()` and `parse_response()`. On the firmware side, `firmware/src/link.zig` has `recv_header()` and `send_ok()`/`send_error()`. Study both.

</details>

<details><summary>Solution</summary>See `host/lib/protocol.py` for the host-side implementation and `firmware/src/link.zig` for the firmware-side parser.</details>

> **🔗 MLSys Connection:** GPU command buffers have the same anatomy: a header with opcode and size, followed by a payload of arguments and data pointers. NVIDIA's push buffer format, AMD's PM4 packets, and your UART packets all solve the same problem: structured communication between a host CPU and a compute device.

---

### Step 3: MAC Execution Over the Wire

Wire it together. The firmware receives a `mac4` request, calls the CFU hardware, and sends back the result.

**Exercise 3.1:** Trace the full data path for a concrete example.

The host sends two packed `i32` values (8 bytes of payload). The firmware unpacks them, issues a single `cfu.mac4_first(a, b)` instruction, and sends back the 4-byte result.

Note on the instruction encoding: `funct7` selects the instruction type (0 = accumulate, 1 = reset+compute), and `funct3` selects the instruction slot in the CFU dispatch. The CFU is wired via `.insn r CUSTOM_0, funct3, funct7, rd, rs1, rs2`.

```
Example:
  a = pack_bytes(1, 2, 3, 4)     → 0x04030201  (little-endian)
  b = pack_bytes(1, 1, 1, 1)     → 0x01010101

  MAC4 with INPUT_OFFSET=128:
    (1+128)*1 + (2+128)*1 + (3+128)*1 + (4+128)*1
    = 129 + 130 + 131 + 132
    = 522

  Host sends: [0xCF, 0x01, 0x08, 0x00, 0x01, 0x00, 0x00, 0x00,
               0x01, 0x02, 0x03, 0x04, 0x01, 0x01, 0x01, 0x01]
              (header: 8 bytes)    (payload: 8 bytes)

  Firmware receives, calls cfu.mac4_first(0x04030201, 0x01010101) → 522

  Host receives: [0xFC, 0x00, 0x04, 0x00, 0x01, 0x00, ...,
                  0x0A, 0x02, 0x00, 0x00]
                 (header)                    (522 as i32 LE)
```

**Exercise 3.2:** Look at `firmware/src/dispatch.zig`. Trace how `handle_mac4` reads the payload, calls the CFU, and constructs the response. Then look at `host/lib/protocol.py` and `host/client.py` to see how the host builds and parses packets.

<details><summary>Hint 1</summary>

The firmware reads exactly 8 bytes of payload, interprets them as two little-endian `i32` values, and passes them directly to the CFU instruction. No offset or element count — the simplest possible format.

</details>

<details><summary>Hint 2</summary>

The dispatch pattern is: `switch (header.op)` routes to a handler function. Each handler validates `payload_len`, reads the payload bytes from UART, does its work, and calls `link.send_ok()` or `link.send_error()`. Adding a new operation means adding a new enum variant and handler.

</details>

<details><summary>Solution</summary>See `firmware/src/dispatch.zig` for the firmware dispatch, `firmware/src/cfu.zig` for the CFU inline assembly, and `host/lib/protocol.py` for host-side packet construction.</details>

**Exercise 3.3:** Write test cases. Don't just test one input:

- All zeros (both operands zero — should give 0, but think about the offset)
- All ones with known offset (verify against hand calculation)
- Maximum values (input=127, weight=127 — does the accumulator overflow?)
- Negative values (INT8 range is -128 to 127)

<details><summary>Hint 1</summary>

Remember: the MAC4 instruction adds `INPUT_OFFSET = 128` to each input byte. So `input=0` becomes `0 + 128 = 128` before multiplication. "All zeros" doesn't give zero.

</details>

<details><summary>Solution</summary>Run `uv run python host/client.py --test -v` for the end-to-end test suite.</details>

---

## The Critical Measurement: Round-Trip Latency

Your vertical slice works. Now measure it.

**Exercise 4.1:** Calculate the transfer-to-compute ratio for a single MAC4 call.

```
Data on the wire:
  Request:  8 (header) + 8 (payload) = 16 bytes TX
  Response: 8 (header) + 4 (payload) = 12 bytes RX
  Total: 28 bytes

UART at 115200 baud, 8N1 encoding (10 bits per byte):
  1 byte ≈ 87 μs
  28 bytes ≈ 2.4 ms

Actual computation:
  1 CFU instruction = 1 clock cycle at 27 MHz ≈ 37 ns

Transfer-to-compute ratio: 2,400,000 ns / 37 ns ≈ 65,000 : 1
```

Now consider 8 elements (two MAC4 calls done in one request with streaming):

```
  Request:  8 (header) + 3 (offset+count) + 16 (data) = 27 bytes ≈ 2.3 ms
  Response: 8 (header) + 4 (result) = 12 bytes ≈ 1.0 ms
  Total wire time: ~3.3 ms

  Compute: 2 MAC instructions × 37 ns = 74 ns

  Ratio: ~36,000 : 1
```

For every nanosecond of useful compute, you spend 36,000 nanoseconds moving data.

**Exercise 4.2:** Compare against the host doing the same work:

```
  NumPy on x86: 8 MACs ≈ 8 ns (pipelined SIMD)
  Over UART:    3,300,000 ns

  The host is ~400,000× faster for 8 elements.
```

**Exercise 4.3:** At what element count does UART offloading break even against NumPy? Set up the equation and solve it.

<details><summary>Hint 1</summary>

UART throughput is ~11.5 KB/s. NumPy on modern x86 does billions of INT8 MACs per second. The crossover point may not exist — that's the point.

</details>

<details><summary>Hint 2</summary>

UART will *never* break even for raw compute throughput. The value of offloading comes from: (1) freeing the host CPU for other work, (2) power efficiency (FPGA uses milliwatts), (3) learning the architecture patterns that matter at higher bandwidth (SPI, PCIe, integrated SoC).

</details>

> **🔗 MLSys Connection:** This is exactly why GPU kernels do a LOT of work per launch. A CUDA kernel that computes a single multiply would be absurd — the launch overhead dominates. Instead, kernels process thousands or millions of elements. The arithmetic intensity (FLOPs per byte transferred) must exceed a threshold set by the hardware's compute-to-bandwidth ratio. This is the "roofline model" in action. Your UART link has a catastrophically low bandwidth ceiling, making the roofline obvious. On a real GPU with PCIe 4.0 (32 GB/s), the same principle applies but the threshold is higher — you still need high arithmetic intensity, just not as extreme.

---

## The Key Insight

This ratio — 36,000:1 for 8 elements — is the fundamental tension of accelerator design.

If your "kernel" does 1 MAC, the overhead dominates completely.
If your "kernel" does 10,000 MACs, the overhead is amortised to ~3.6:1.
If your "kernel" does 100,000 MACs *without returning to the host*, the overhead becomes negligible.

This motivates everything that comes next:

- **Unit 3 (Fusion):** Keep intermediate results on-chip instead of round-tripping them through UART. Fusing MAC + requantisation saves multiple round trips per output element.
- **Unit 4 (Autonomy):** Move the entire inner loop into hardware so the firmware triggers one "execute" command instead of thousands of individual MAC commands.
- **Bandwidth upgrades:** SPI (10 MB/s) is ~1000x faster than UART. PCIe (32 GB/s) is ~3,000,000x faster. The architecture patterns you learn here transfer directly.

> **🔗 MLSys Connection:** This is why `torch.compile` and XLA exist. Without compilation, PyTorch eagerly launches one kernel per operation — like sending one MAC at a time over UART. With compilation, the framework fuses operations and launches one big kernel — like batching thousands of MACs into a single command. The speedup isn't from faster compute; it's from amortising launch overhead.

---

## Adding New Operations

Once `mac4` works, the pattern for adding any new operation is:

```
1. hardware/cfu.py  → New Instruction subclass, wire to funct3 slot N
2. firmware/src/cfu.zig  → New inline function with funct3=N
3. firmware/src/link.zig → New OpType variant
4. firmware/src/dispatch.zig → New branch in the dispatch switch
5. host/lib/protocol.py → New request builder + opcode constant
6. host/client.py → New test case
```

**Exercise 5.1:** Walk through this for a hypothetical `relu` operation: element-wise `max(x, 0)` on 4 packed INT8 bytes. What does each file need? What's the packet format? How many bytes in, how many out?

<details><summary>Hint 1</summary>

ReLU on 4 packed INT8 values: input is one `u32` (4 bytes), output is one `u32` (4 bytes). For each byte lane, clamp negative values to 0. The hardware needs 4 comparators and 4 muxes.

</details>

<details><summary>Hint 2</summary>

Assign it `funct3=1` (the MAC is at `funct3=0`). The `funct7` field is free for sub-opcodes (e.g., `funct7=0` for ReLU, `funct7=1` for clamp to arbitrary range).

</details>

---

## Checkpoint

Before moving to Unit 3, verify:

- [ ] You understand the full data path: Python host -> serial protocol -> UART -> firmware dispatch -> CFU execute -> UART back
- [ ] You can send a MAC4 request from the host and get the correct result
- [ ] You have measured (or calculated) the round-trip latency
- [ ] You can explain why the transfer-to-compute ratio is ~36,000:1 for 8 elements
- [ ] You understand why this ratio motivates kernel fusion and autonomous execution
- [ ] You know the pattern for adding a new CFU operation (all 6 files)

---

## Suggested Readings

1. **GPU Driver Architecture:**
   - "Life of a Triangle" by Fabian Giesen — traces a draw call through the full NVIDIA driver stack. The command buffer concepts apply directly to compute kernels.
   - NVIDIA's CUDA Programming Guide, Chapter 3 (Programming Interface) — describes how `cudaLaunchKernel` works under the hood.

2. **Command Buffers and Submission:**
   - Vulkan specification, Chapter 6 (Command Buffers) — the most explicit documentation of how command buffers are structured and submitted. Your UART packets are a simplified version of Vulkan command buffers.

3. **Kernel Launch Overhead:**
   - "Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking" (Jia et al., 2018) — measures kernel launch latency at ~5-10 microseconds.
   - "Reducing CUDA Kernel Launch Overhead" (NVIDIA DevBlog) — discusses CUDA Graphs as a batching mechanism for amortising launch costs.

4. **The Roofline Model:**
   - "Roofline: An Insightful Visual Performance Model" (Williams, Waterman, Patterson, 2009) — the foundational paper on understanding compute vs. bandwidth bottlenecks. Your UART bandwidth ceiling makes this model trivially obvious.

5. **CFU-Playground:**
   - [CFU-Playground GitHub](https://github.com/google/CFU-Playground) — the project this accelerator is inspired by. Study `avg_pdti8` for the Tier 2 (SIMD + requantisation) pattern.

---

**Previous:** [Unit 1 — The MAC](01-compute.md)
**Next:** [Unit 3 — Kernel Fusion: Why Round Trips Kill You](03-fusion.md)
