"""ACCEL device backend for tinygrad — Tang Nano 20K CFU accelerator.

This is a skeleton with TODO comments showing where the student fills in
logic. It is NOT a working implementation — it's a map of the territory.

Architecture mapping to CUDA equivalents:
    ACCELAllocator  ≈  cuMemAlloc / cuMemFree / cuMemcpyHtoD / cuMemcpyDtoH
    ACCELRenderer   ≈  nvcc (compile UOps → executable representation)
    ACCELProgram    ≈  cuLaunchKernel (send work to device, wait for result)
    ACCELDevice     ≈  cuDeviceGet + cuCtxCreate (wires everything together)

Key difference from CUDA: there is no "device memory" in the GPU sense.
BSRAM is too small (18 KiB) to hold all tensors. Data lives in host-side
numpy arrays and is transferred to BSRAM on each kernel launch. This is
more like a DMA accelerator than a GPU — the "device" has no persistent
memory between kernel calls.
"""

import numpy as np
from tinygrad.device import Compiled, Allocator
from tinygrad.renderer import Renderer


class ACCELAllocator(Allocator):
    """Manages "device memory" — which is really host-side numpy arrays.

    Why doesn't the allocator DMA to device?
    ─────────────────────────────────────────
    Unlike a GPU where cuMemAlloc reserves device DRAM, our BSRAM is:
    1. Only 18 KiB total (vs GB on a GPU)
    2. Statically partitioned (activation banks, filter stores, etc.)
    3. Loaded fresh on every kernel launch

    So _alloc just creates a host-side buffer. The actual BSRAM loading
    happens in ACCELProgram.__call__() as part of the kernel launch sequence.
    This is similar to how some embedded DMA engines work: memory is staged
    on the host, then burst-transferred at execution time.
    """

    def _alloc(self, size: int, options=None):
        """Allocate a host-side byte buffer.

        Returns a numpy array that acts as "device memory."
        """
        # TODO: return np.empty(size, dtype=np.uint8)
        raise NotImplementedError("Allocate host-side buffer")

    def _free(self, buf, options=None):
        """Free a host-side buffer.

        No-op for numpy arrays (garbage collected), but tinygrad expects
        the method to exist.
        """
        # TODO: pass (numpy handles deallocation)
        raise NotImplementedError

    def _copyin(self, dest, src: memoryview):
        """Copy data from host to "device" (host-side numpy array).

        In a GPU backend, this would be cuMemcpyHtoD.
        Here it's just a memcpy into our staging buffer.
        """
        # TODO: np.copyto(dest, np.frombuffer(src, dtype=np.uint8))
        raise NotImplementedError("Copy host → staging buffer")

    def _copyout(self, dest: memoryview, src):
        """Copy data from "device" back to host.

        In a GPU backend, this would be cuMemcpyDtoH.
        Here it's a memcpy from our staging buffer.
        """
        # TODO: dest[:] = src.tobytes()
        raise NotImplementedError("Copy staging buffer → host")


class ACCELRenderer(Renderer):
    """Translates tinygrad UOps into a compute descriptor for our hardware.

    In a GPU backend, the renderer converts UOps → PTX or CUDA C source code.
    Here, the "compiled output" is a Python dict describing the operation
    parameters, because our firmware is pre-compiled — we're sending config
    packets, not JIT-compiling code.

    The renderer's job is pattern matching:
    1. Look at the UOps graph
    2. Decide: can our hardware run this?
    3. If yes: emit a descriptor dict with dimensions, dtypes, etc.
    4. If no: raise NotImplementedError → tinygrad falls back to CPU

    What the hardware CAN run:
    - INT8 matrix multiply (pointwise / 1x1 conv)
    - With fused requantization (SRDHM + RDBPOT + clamp)

    What it CANNOT run (falls back to CPU):
    - Anything with float dtypes
    - Depthwise convolution (no channel reduction)
    - Element-wise ops (too little compute per byte transferred)
    - Pooling, softmax, reshape (no hardware support)
    """

    def render(self, name: str, uops) -> str:
        """Convert UOps into a serialized operation descriptor.

        Returns a string (or bytes) that ACCELProgram.__init__ will parse.
        For our backend, this is a JSON-like descriptor, not machine code.
        """
        # TODO: Pattern-match the UOps graph
        #
        # Look for the pattern:
        #   LOAD(activation, int8) → LOAD(weight, int8) → MUL → REDUCE_AXIS(sum)
        # This is a matmul / conv that our hardware can execute.
        #
        # Extract:
        #   - M, K, N dimensions
        #   - Activation buffer index
        #   - Weight buffer index
        #   - Output buffer index
        #   - Quantization parameters (if fused)
        #
        # Return a descriptor dict serialized as string:
        #   {"op": "matmul", "M": ..., "K": ..., "N": ...,
        #    "act_idx": 0, "wt_idx": 1, "out_idx": 2,
        #    "quant_mult": [...], "quant_shift": [...]}
        #
        # If the UOps don't match any supported pattern:
        #   raise NotImplementedError(f"ACCEL cannot run: {name}")

        raise NotImplementedError("UOp pattern matching not implemented")


class ACCELProgram:
    """Executes a compiled operation on the FPGA accelerator.

    In a GPU backend, this is cuLaunchKernel — it sends work to the device
    and waits for completion. Our "kernel launch" is:

    1. Send weights to BSRAM filter stores (via UART/SPI)
    2. Send activations to BSRAM activation banks
    3. Send quantization params to BSRAM param stores
    4. Write config registers (N, K, spatial_size)
    5. Assert START
    6. Wait for DONE
    7. Drain output FIFO

    The serial link (UART at 115200 baud) is the bottleneck. For a real
    accelerator you'd use SPI (10 MB/s) or a shared-memory bus.
    """

    def __init__(self, name: str, lib: str):
        """Parse the operation descriptor emitted by ACCELRenderer.

        Args:
            name: operation name (for debugging)
            lib: serialized descriptor string from renderer
        """
        # TODO: Parse the descriptor
        # self.name = name
        # self.desc = json.loads(lib)  # or whatever serialization format
        # self.M = self.desc["M"]
        # self.K = self.desc["K"]
        # self.N = self.desc["N"]
        raise NotImplementedError("Parse operation descriptor")

    def __call__(self, *bufs, global_size=None, local_size=None, vals=(), wait=False):
        """Execute the operation on the FPGA.

        Args:
            bufs: tuple of buffer references (activation, weights, output, ...)
            global_size: unused (not a GPU grid launch)
            local_size: unused
            vals: variable values (loop bounds, etc.)
            wait: whether to block until completion
        """
        # TODO: Implement the launch sequence
        #
        # Step 1: Compute tiling
        #   If M * K > activation BSRAM capacity (8 KiB) or
        #   N * K > filter store capacity (4 KiB):
        #     Split into tiles that fit
        #
        # Step 2: For each tile:
        #   a) Extract activation tile from bufs[0] (numpy array)
        #   b) Extract weight tile from bufs[1]
        #   c) Send weights to filter store BSRAM
        #      serial_link.send_packet(OP_LOAD_WEIGHTS, weight_bytes)
        #   d) Send activations to activation BSRAM
        #      serial_link.send_packet(OP_LOAD_ACTS, act_bytes)
        #   e) Send quantization params
        #      serial_link.send_packet(OP_LOAD_PARAMS, param_bytes)
        #   f) Send config + START
        #      serial_link.send_packet(OP_EXECUTE, config_bytes)
        #   g) Wait for DONE
        #      serial_link.recv_packet()  # blocks until firmware responds
        #   h) Drain output FIFO
        #      result_bytes = serial_link.recv_packet()
        #
        # Step 3: Assemble tile results into output buffer
        #   bufs[2][:] = assembled_result
        #
        raise NotImplementedError("FPGA kernel launch not implemented")


class ACCELDevice(Compiled):
    """Top-level device class that wires allocator, renderer, and runtime.

    Usage:
        # In tinygrad user code:
        from tinygrad import Tensor, Device
        Device.DEFAULT = "ACCEL"
        a = Tensor([1, 2, 3], dtype=dtypes.char)  # INT8
        b = Tensor([4, 5, 6], dtype=dtypes.char)
        c = a.dot(b)  # Dispatches to ACCELProgram if pattern matches
    """

    def __init__(self, device: str):
        # TODO: Initialize the serial connection to the FPGA
        # self.serial = SerialLink("/dev/ttyUSB1", baudrate=115200)

        # No compiler needed — firmware is pre-compiled on the FPGA.
        # The "compilation" step (ACCELRenderer) emits a descriptor dict,
        # not source code that needs a compiler toolchain.
        super().__init__(
            device,
            ACCELAllocator(),
            ACCELRenderer(),
            ACCELProgram,  # callable class, instantiated per-kernel
        )
