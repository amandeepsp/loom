"""MobileNet v2 0.25 INT8 inference runner.

Runs inference through tinygrad with ACCEL device for supported ops,
falling back to CPU for unsupported ops (depthwise conv, relu6, pooling).

Model: MobileNet v2 with width multiplier 0.25, input 96x96, INT8 quantized.
This is the smallest practical MobileNet variant — chosen because:
    - Total MACs: ~5.6M (manageable for 27 MHz FPGA)
    - Largest weight tensor: 96x32 = 3,072 bytes (fits in filter BSRAM)
    - Designed for TFLite INT8 quantization (good tooling support)

Layer types in MobileNet v2:
    - Pointwise conv (1x1): matrix multiply → HARDWARE
    - Depthwise conv (3x3): per-channel spatial filter → CPU
    - ReLU6: clamp(x, 0, 6) → CPU (trivial, not worth offloading)
    - Average pool: spatial reduction → CPU
    - Fully connected: small matmul → CPU (too small to amortize transfer)
    - Softmax: exp + normalize → CPU

Why some ops CANNOT be offloaded:
    Depthwise conv: each output channel depends on exactly ONE input channel.
    There's no reduction across channels — the K dimension is 1 (times the
    spatial filter size 3x3=9). Our MAC4 expects K to be a multiple of 4 for
    the SIMD lane packing. You'd need to pad to K=12 (waste 25% compute)
    and the spatial addressing pattern doesn't match our sequential BSRAM read.

    ReLU6 / element-wise ops: one operation per byte. Transfer cost via
    UART vastly exceeds compute cost on the host CPU.
"""

import numpy as np
# from tinygrad import Tensor, Device, dtypes


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_tflite_model(path: str) -> dict:
    """Load a TFLite model and extract layer-by-layer parameters.

    TODO: Use tflite_runtime or flatbuffers to parse the .tflite file.

    Returns a dict with:
        layers: list of layer descriptors, each containing:
            - op_type: "conv2d", "depthwise_conv2d", "average_pool2d", etc.
            - kernel_size: (H, W) spatial filter size
            - input_channels: K dimension
            - output_channels: N dimension
            - spatial_size: (H, W) of output activation
            - weights: np.ndarray of int8 filter weights
            - bias: np.ndarray of int32 per-channel bias
            - quant_multiplier: np.ndarray of int32 per-channel multipliers
            - quant_shift: np.ndarray of int32 per-channel shifts
            - input_zero_point: int8
            - output_zero_point: int8
    """
    # TODO: Parse the TFLite flatbuffer
    # import tflite_runtime.interpreter as tflite
    # interpreter = tflite.Interpreter(model_path=path)
    # interpreter.allocate_tensors()
    #
    # For each op in the graph:
    #   - Extract tensor data (weights, bias)
    #   - Extract quantization parameters from tensor metadata
    #   - Build layer descriptor
    #
    # The quantization multiplier and shift encode the output scale:
    #   output_scale = input_scale * weight_scale / output_scale
    # TFLite decomposes this into:
    #   output_scale ≈ multiplier * 2^(-shift)
    # where multiplier is a Q0.31 fixed-point number in [0.5, 1.0)
    raise NotImplementedError("TFLite model loading")


# ---------------------------------------------------------------------------
# Hardware capability check
# ---------------------------------------------------------------------------

def hardware_capable(layer: dict) -> bool:
    """Decide if a layer can run on the FPGA accelerator.

    A layer is hardware-capable if:
    1. It's a pointwise (1x1) convolution — these are pure matmuls
    2. Input channels (K) is a multiple of 4 (SIMD lane alignment)
    3. Weights fit in filter BSRAM (N * K <= 4096 bytes)
       - If not, we tile along N (output channels)
    4. Data type is INT8

    Returns True if the hardware can handle this layer (possibly with tiling).
    """
    # TODO: Implement the capability check
    #
    # if layer["op_type"] != "conv2d":
    #     return False  # Only standard conv, not depthwise
    # if layer["kernel_size"] != (1, 1):
    #     return False  # Only pointwise (1x1) — spatial convs need different addressing
    # if layer["input_channels"] % 4 != 0:
    #     return False  # SIMD lanes need aligned K dimension
    # return True
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Tiling strategy
# ---------------------------------------------------------------------------

# BSRAM budget constraints:
ACTIVATION_BSRAM_BYTES = 8192   # 4 banks x 2 KiB
FILTER_BSRAM_BYTES = 4096       # 2 stores x 2 KiB
PARAM_BSRAM_BYTES = 4096        # 2 stores x 2 KiB
OUTPUT_FIFO_BYTES = 2048        # 1 BSRAM


def compute_tiles(layer: dict) -> list:
    """Compute the tiling strategy for a layer that exceeds BSRAM capacity.

    Tiling dimensions (in priority order):
    1. Output channels (N): tile into groups that fit filter BSRAM
    2. Spatial: tile into row-groups that fit activation BSRAM
    3. Input channels (K): NOT tiled — hardware processes full K in one pass
       (using the first/last accumulator control pattern)

    Tiling math:

    Filter store constraint:
        N_tile * K <= FILTER_BSRAM_BYTES
        N_tile <= FILTER_BSRAM_BYTES / K

    Activation bank constraint:
        S_tile * K <= ACTIVATION_BSRAM_BYTES
        S_tile <= ACTIVATION_BSRAM_BYTES / K

    Example: 1x1 conv, K=16, N=32, spatial=48x48
        N_tile = 4096 / 16 = 256 → N=32 fits entirely, no N tiling needed
        S_tile = 8192 / 16 = 512 → 48x48=2304 needs ceil(2304/512)=5 tiles
        Total tiles: 1 (N) x 5 (spatial) = 5 tile launches

    Example: 1x1 conv, K=96, N=24, spatial=12x12
        N_tile = 4096 / 96 = 42 → N=24 fits, no N tiling
        S_tile = 8192 / 96 = 85 → 12x12=144 needs ceil(144/85)=2 tiles
        Total tiles: 1 x 2 = 2 tile launches

    Returns a list of tile descriptors, each containing:
        - n_start, n_end: output channel range for this tile
        - s_start, s_end: spatial position range for this tile
    """
    # TODO: Implement tiling computation
    #
    # K = layer["input_channels"]
    # N = layer["output_channels"]
    # S = layer["spatial_size"][0] * layer["spatial_size"][1]
    #
    # N_tile = min(N, FILTER_BSRAM_BYTES // K)
    # S_tile = min(S, ACTIVATION_BSRAM_BYTES // K)
    #
    # tiles = []
    # for n_start in range(0, N, N_tile):
    #     n_end = min(n_start + N_tile, N)
    #     for s_start in range(0, S, S_tile):
    #         s_end = min(s_start + S_tile, S)
    #         tiles.append({"n_start": n_start, "n_end": n_end,
    #                        "s_start": s_start, "s_end": s_end})
    # return tiles
    raise NotImplementedError


# ---------------------------------------------------------------------------
# CPU fallback for unsupported ops
# ---------------------------------------------------------------------------

def numpy_reference(layer: dict, input_data: np.ndarray) -> np.ndarray:
    """Execute a layer on the CPU using numpy.

    This handles all ops the hardware can't:
    - Depthwise conv: np.einsum or manual loop
    - ReLU6: np.clip(x, 0, 6) (but in quantized domain: clip to quantized 0 and 6)
    - Average pool: reshape + mean
    - Softmax: exp(x - max(x)) / sum(exp(x - max(x)))
    - Fully connected: np.dot (small enough that host is faster than transfer)
    """
    # TODO: Implement per-op-type numpy reference
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

def run_inference(model: dict, input_image: np.ndarray) -> np.ndarray:
    """Run full model inference, dispatching each layer to HW or CPU.

    The layer-by-layer execution is sequential — no pipelining between layers.
    This is the simplest correct approach. Pipelining (computing layer N+1
    while draining layer N's output) is a stretch optimization.

    For MobileNet v2 0.25 at 96x96:
        - ~18 pointwise conv layers → hardware (if capable)
        - ~17 depthwise conv layers → CPU
        - ~17 ReLU6 layers → CPU
        - 1 average pool → CPU
        - 1 fully connected → CPU
        - 1 softmax → CPU

    Expected timing (rough):
        Hardware layers: ~18 layers x ~2-5 ms each = ~36-90 ms compute
        UART transfer overhead: dominates at 115200 baud
        CPU layers: ~1-5 ms total (fast on modern host)
        Total: ~100-500 ms per inference (UART-bound)
    """
    # TODO: Implement the inference loop
    #
    # activation = preprocess(input_image)  # Quantize input to INT8
    #
    # for i, layer in enumerate(model["layers"]):
    #     if hardware_capable(layer):
    #         tiles = compute_tiles(layer)
    #         output_parts = []
    #         for tile in tiles:
    #             # Extract tile data
    #             act_tile = activation[tile["s_start"]:tile["s_end"]]
    #             wt_tile = layer["weights"][tile["n_start"]:tile["n_end"]]
    #
    #             # Send to FPGA and execute
    #             result = accel_execute(act_tile, wt_tile,
    #                                    layer["quant_multiplier"][tile["n_start"]:tile["n_end"]],
    #                                    layer["quant_shift"][tile["n_start"]:tile["n_end"]],
    #                                    layer["output_zero_point"])
    #             output_parts.append(result)
    #
    #         # Assemble tiles into full output
    #         activation = assemble_tiles(output_parts, tiles, layer)
    #     else:
    #         activation = numpy_reference(layer, activation)
    #
    #     print(f"Layer {i}: {layer['op_type']} "
    #           f"{'[HW]' if hardware_capable(layer) else '[CPU]'} "
    #           f"output shape: {activation.shape}")
    #
    # return activation
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(model_path: str, input_image: np.ndarray):
    """Compare our inference output against tflite_runtime's interpreter.

    This is the ground truth check. TFLite's interpreter uses the exact
    same quantization math (SRDHM, RDBPOT) that our hardware implements,
    so the outputs should match bit-for-bit for hardware-capable layers.

    Any discrepancy indicates a bug in:
    1. Our SRDHM/RDBPOT hardware implementation
    2. Our tiling logic (wrong data slice)
    3. Our activation/weight addressing (off-by-one in BSRAM)
    4. Our output assembly (tiles not stitched correctly)
    """
    # TODO: Implement validation
    #
    # import tflite_runtime.interpreter as tflite
    #
    # # Reference inference
    # interpreter = tflite.Interpreter(model_path=model_path)
    # interpreter.allocate_tensors()
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()
    # interpreter.set_tensor(input_details[0]["index"], input_image)
    # interpreter.invoke()
    # reference_output = interpreter.get_tensor(output_details[0]["index"])
    #
    # # Our inference
    # model = load_tflite_model(model_path)
    # our_output = run_inference(model, input_image)
    #
    # # Compare
    # if np.array_equal(reference_output, our_output):
    #     print("PASS: Outputs match bit-for-bit")
    # else:
    #     diff = np.abs(reference_output.astype(int) - our_output.astype(int))
    #     print(f"FAIL: Max diff = {diff.max()}, mean diff = {diff.mean():.3f}")
    #     print(f"      Mismatched elements: {np.count_nonzero(diff)} / {diff.size}")
    #
    # # Top-1 prediction check
    # ref_class = np.argmax(reference_output)
    # our_class = np.argmax(our_output)
    # print(f"Reference top-1: class {ref_class}")
    # print(f"Our top-1:       class {our_class}")
    # print(f"Match: {'YES' if ref_class == our_class else 'NO'}")
    raise NotImplementedError


if __name__ == "__main__":
    # TODO: Parse command line args for model path and input image
    #
    # model_path = "mobilenet_v2_0.25_96_integer_quant.tflite"
    # input_image = load_and_preprocess("test_image.jpg", size=(96, 96))
    # validate(model_path, input_image)
    print("MobileNet v2 0.25 INT8 inference runner")
    print("Usage: python mobilenet_runner.py <model.tflite> <image.jpg>")
    print("Not yet implemented — this is a skeleton for the course exercise.")
