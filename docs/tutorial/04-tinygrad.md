# Part 4 — Selective Lowering from TinyGrad

> **Series:** [00-overview](00-overview.md) → [01-mac](01-mac.md) → [02-vertical-slice](02-vertical-slice.md) → [03-autonomous](03-autonomous.md) → **[04-tinygrad](04-tinygrad.md)** → [05-scaling](05-scaling.md)

You have a working firmware that executes MAC operations on the FPGA
(Part 2), and a plan for autonomous compute (Part 3). Now: how does
TinyGrad decide *what* to offload?

This part connects the host-side ML framework to your hardware. The
firmware becomes an RPC server — TinyGrad calls it for specific operations
just like it would call CUDA for a GPU kernel.

---

## 4.1  The Lowering Decision

Not every operation benefits from offloading. The decision is simple math:

```
  Time_on_host  vs  Time_serialize + Time_transfer + Time_execute + Time_deserialize

  Lower to hardware ONLY when the right side is smaller.
```

For your system with a UART bottleneck:

| Operation | Lower? | Why |
|---|---|---|
| Large INT8 matmul | ✅ | Many MACs per byte transferred |
| Small matmul (<64 elements) | ❌ | Setup overhead > compute time |
| Element-wise add/mul | ❌ | 1 op per byte — UART kills you |
| Anything needing float | ❌ | Hardware only does INT8 |
| ReLU (max(x, 0)) | ❌ | Trivial on host CPU |
| Requantization | ✅ | Fused into the MAC pipeline (Part 3) |

**🤔 Exercise:** Calculate the crossover point for your system.

```
  UART bandwidth: ~11,500 bytes/sec
  Host throughput: ~1 billion MACs/sec (NumPy on modern x86)
  FPGA throughput: ~4 million MACs/sec (4 MACs/cycle × 27 MHz, Part 1)
                   ~108 million MACs/sec (autonomous, Part 3 target)

  For N INT8 MACs via UART:
    Transfer time:  (2N + overhead) / 11500 seconds
    Compute time:   N / 108e6 seconds (Part 3)
    Total offload:  (2N / 11500) + (N / 108e6)

  Host time:        N / 1e9 seconds

  Solve: 2N/11500 + N/108e6 < N/1e9
         2N/11500 < N/1e9  (transfer dominates)
         This is never true. UART is always slower.
```

*So why bother?* Because:
1. UART is a placeholder. SPI gives ~10 MB/s. PCIe gives GB/s.
2. With Part 3's BSRAM stores, weights load ONCE per layer (amortized).
3. The architecture you're learning IS the GPU architecture — just smaller.
4. For batch inference, activations stay on-device between layers.

**🤔 Key insight:** The real selective lowering question isn't "is UART
fast enough" — it's "which operations have enough compute density to
justify the transfer cost at your system's bandwidth?"

---

## 4.2  TinyGrad's Execution Pipeline

```
  User code:  c = a.dot(b)
                  │
  Tensor ops:     LazyBuffer (deferred execution)
                  │
  Schedule:       Fuse operations into ScheduleItems
                  │
  Linearize:      Convert to UOps IR
                  │
  Render:         UOps → target code (C, Metal, PTX, ...)
                  │
  Compile:        Source → binary (clang, nvcc, ...)
                  │
  Execute:        Run on hardware
```

**🤔 Exercise:** Read `tinygrad/runtime/ops_npy.py` (it's ~11 lines). It's
the simplest possible backend — it only stores data, it can't compute.
Answer:

1. `NpyAllocator` implements `_alloc`, `_as_buffer`, `_copyout`. *What
   does each do?*
2. `NpyDevice` passes `compilers=None, runtime=None`. *Can you run compute
   kernels on NPY?*
3. *Why is NPY useful despite not computing?* (Loading weights from disk.)

Then skim `tinygrad/runtime/ops_cpu.py` — a compute-capable backend that
emits C code. Find where the C renderer is selected, where clang is
invoked, and where the compiled `.so` is loaded and called. This is the
closest analog to what you'd build.

---

## 4.3  Where to Intercept

Selective lowering can intercept at different points:

```
  Interception points in TinyGrad:

  ┌────────────────┐
  │  Schedule level │ ← See fused operation groups
  │  (medium)       │   "This is a conv+bias+relu fusion"
  ├────────────────┤
  │  UOp level      │ ← Pattern-match op sequences
  │  (hard)         │   "If MULACC with int8 dtypes → offload"
  ├────────────────┤
  │  CUSTOM_FUNCTION│ ← Simplest
  │  (easy)         │   TinyGrad calls your Python function directly
  └────────────────┘
```

**Start with `CUSTOM_FUNCTION`.** It's the easiest path. TinyGrad lets you
register a custom callable for an operation. When encountered, it calls
your function instead of rendering to GPU/CPU code.

---

## 4.4  The CUSTOM_FUNCTION Approach

Your firmware is already an RPC server (from Part 2). `CUSTOM_FUNCTION`
is the client side:

```
  TinyGrad                    Your code                Firmware
  ┌──────────┐               ┌──────────────┐         ┌─────────┐
  │ Schedule │──► CUSTOM ──► │ Python fn:   │──UART──►│ Decode  │
  │ sees     │    FUNCTION   │  serialize   │         │ Execute │
  │ a matmul │               │  send        │◄─UART──│ Respond │
  │          │◄──────────────│  receive     │         │         │
  │ got      │   return      │  deserialize │         │         │
  │ result   │   tensor      └──────────────┘         └─────────┘
  └──────────┘
```

**🤔 Exercise:** Think about the Python function signature:

```python
def accel_matmul(a: Tensor, b: Tensor) -> Tensor:
    # 1. Validate: are a and b int8? Are shapes compatible?
    # 2. Serialize: pack a.numpy() and b.numpy() into link protocol
    # 3. Send: write request packet to serial port
    # 4. Receive: read response packet
    # 5. Deserialize: unpack result bytes into numpy array
    # 6. Return: Tensor from the result
    ...
```

*What if the tensors are too large for SRAM?* You need tiling — split the
matmul into tiles that fit, offload each tile, assemble the result.

*What if the tensors aren't int8?* Don't lower. Return a "not supported"
signal and let TinyGrad fall back to the host CPU.

---

## 4.5  Mapping UOps to Your Hardware

TinyGrad's IR uses UOps. The relevant ones for your hardware:

| UOp | What it does | Maps to your HW? |
|---|---|---|
| `MULACC` | Multiply-accumulate | **Yes** — `mac4` |
| `REDUCE_AXIS` (sum) | Sum over dimension | Via MAC accumulation |
| `WMMA` | Matrix multiply-accumulate | Future (systolic, Part 5) |
| `LOAD` | Read from buffer | Firmware receives data |
| `STORE` | Write to buffer | Firmware sends data |
| `ADD`, `MUL` | Element-wise | Not worth lowering |
| `MAX` | Element-wise max (ReLU) | Not worth lowering |
| `CAST` | Type conversion | Software |

**🤔 Exercise:** In TinyGrad, `a.dot(b)` where `a` and `b` are INT8
matrices becomes a fused schedule item containing `LOAD`s, nested `RANGE`
loops, `MUL`, `ADD` (accumulate), and a `STORE`. The optimizer may fuse
these into a `MULACC` or `REDUCE_AXIS`.

*Trace the UOps for a simple `Tensor.dot(a, b)` with
`DEBUG=4 NOOPT=1`.* What do you see? Which UOps correspond to the inner
loop you'd offload?

---

## 4.6  The Full Backend (Later)

For when `CUSTOM_FUNCTION` isn't enough, you build a full backend with
four components:

| Component | Base Class | Your Implementation |
|---|---|---|
| **Device** | `Compiled` | Wires the other three together |
| **Allocator** | `Allocator` | Host-side byte arrays (weights live here until sent to BSRAM) |
| **Renderer** | `Renderer` | Translates UOps → serialized op descriptors (not C code — you're sending packets, not compiling) |
| **Runtime** | callable | Sends packets, receives results |

You don't need a Compiler in the traditional sense — your firmware is
pre-compiled. The "compilation" step is packing UOps into link protocol
packets.

**🤔 Exercise:** Sketch the class interfaces. Don't implement — just
write signatures and one-line docstrings:

```python
class AccelAllocator(Allocator):
    def _alloc(self, size, options=None):
        """Allocate a host-side byte buffer."""
    def _copyin(self, dest, src: memoryview):
        """Copy data into a host-side buffer."""
    def _copyout(self, dest: memoryview, src):
        """Copy data out of a host-side buffer."""

class AccelDevice(Compiled):
    def __init__(self, device: str):
        """Set up the serial connection + allocator + runtime."""
```

See [tinygrad-backend.md](tinygrad-backend.md) for the full call chain
from Schedule through to execution.

---

## 4.7  INT8 Quantization: The Pragmatic Path

TinyGrad has `dtypes.int8` / `dtypes.char` but limited INT8 compute
support — most backends focus on float.

**🤔 Exercise:** Search the TinyGrad source for `dtypes.char`. Where is
it used? Does TinyGrad have built-in quantization? (As of 2026: minimal.)

The pragmatic path for a learning project:

1. **Quantize externally** — use PyTorch `torch.ao.quantization` or ONNX
   Runtime to produce INT8 weights + per-layer scale/zero-point
2. **Load as `dtypes.char` tensors** in TinyGrad
3. **Override the matmul lowering** for INT8 pairs → call your
   `CUSTOM_FUNCTION` instead of the default

*Or skip TinyGrad entirely for quantization* and write a standalone
Python export script that:
1. Loads a model
2. Quantizes it
3. For each layer, serializes weights + params into a binary blob
4. Calls your firmware for inference layer-by-layer
5. Collects results

This exercises the full path without fighting TinyGrad's type system.

---

## 4.8  The Incremental Path

Don't build the full backend at once:

```
  Step 1: Manual Python script
  ├── Serialize a matmul by hand
  ├── Send over UART
  ├── Compare result to NumPy
  └── This proves: protocol works, firmware works, math is correct

  Step 2: Python function wrapper
  ├── Wrap Step 1 in a function: accel_matmul(a, b) → result
  ├── Add input validation (shape, dtype)
  └── This proves: the interface is clean

  Step 3: CUSTOM_FUNCTION hook
  ├── Register accel_matmul with TinyGrad
  ├── Run: a.dot(b) where a, b are on "ACCEL" device
  └── This proves: TinyGrad integration works

  Step 4: Run a simple model
  ├── Single-layer INT8 conv via TinyGrad
  ├── Compare output to host-side reference
  └── This proves: real workloads work

  Step 5: Full backend (if needed)
  ├── AccelDevice, AccelAllocator, AccelRuntime
  ├── Multi-layer inference
  └── This proves: you have a real accelerator backend
```

Each step is independently testable. Each builds on the previous one.

---

## 4.9  Checkpoint

- [ ] I can calculate the lowering breakeven point for my system
- [ ] I understand TinyGrad's execution pipeline (at a high level)
- [ ] I have a working Python script that offloads a matmul via UART
- [ ] I can explain which UOps map to my hardware and which don't
- [ ] I know the difference between `CUSTOM_FUNCTION` and a full backend
- [ ] I have a quantization strategy (external, pragmatic)

---

**Previous:** [Part 3 — Autonomous Compute](03-autonomous.md)
**Next:** [Part 5 — Scaling Up](05-scaling.md)
