"""Fresh int8 references and real dataset samples for loom model binaries.

References are computed from the source ONNX with ONNX Runtime at run time,
so they can never go stale relative to the model. The graph is cut just
before its final DequantizeLinear, which yields the same int8 tensor the
hardware writes, rather than a float that would need re-quantizing.

ORT and loom implement requantization rounding differently (ORT rescales in
float, loom uses TFLite-style SRDHM/RDBPOT), so expect small per-logit deltas.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx

REPO_ROOT = Path(__file__).resolve().parents[2]
MNIST_DIR = REPO_ROOT / "data/MNIST/raw"
CIFAR10_DIR = REPO_ROOT / "build/cifar-10-batches-py"

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
_CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)[:, None, None]
_CIFAR10_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)[:, None, None]


@dataclass(frozen=True)
class QuantBoundary:
    """Scale/zero-point of the model's int8 input and output tensors."""

    input_name: str
    input_shape: tuple[int, ...]
    input_scale: float
    input_zero_point: int
    output_int8_name: str


@dataclass(frozen=True)
class Sample:
    """One real dataset example, as float (ONNX input) and int8 (loom input)."""

    dataset: str
    index: int
    label: int
    label_name: str
    float_input: np.ndarray
    int8_input: np.ndarray


def quant_boundary(onnx_path: str | Path) -> QuantBoundary:
    """Find the input QuantizeLinear and the int8 input of the final DequantizeLinear."""

    model = onnx.load(str(onnx_path))
    graph = model.graph
    consts = {init.name: onnx.numpy_helper.to_array(init) for init in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant":
            consts[node.output[0]] = onnx.numpy_helper.to_array(node.attribute[0].t)

    graph_input = graph.input[0]
    dims = tuple(
        d.dim_value if d.dim_value > 0 else 1 for d in graph_input.type.tensor_type.shape.dim
    )

    quant = next((n for n in graph.node if n.op_type == "QuantizeLinear"), None)
    if quant is None:
        raise ValueError(f"{onnx_path}: no QuantizeLinear; expected a QOperator int8 model")
    output_name = graph.output[0].name
    dequant = next(
        (n for n in graph.node if n.op_type == "DequantizeLinear" and n.output[0] == output_name),
        None,
    )
    if dequant is None:
        raise ValueError(f"{onnx_path}: graph output is not produced by DequantizeLinear")

    return QuantBoundary(
        input_name=graph_input.name,
        input_shape=dims,
        input_scale=float(consts[quant.input[1]]),
        input_zero_point=int(consts[quant.input[2]]),
        output_int8_name=dequant.input[0],
    )


class OnnxReference:
    """Runs the source ONNX model and returns its int8 output for an int8 input."""

    def __init__(self, onnx_path: str | Path, extra_outputs: tuple[str, ...] = ()) -> None:
        import onnxruntime as ort

        self.boundary = quant_boundary(onnx_path)
        model = onnx.load(str(onnx_path))
        for name in (self.boundary.output_int8_name, *extra_outputs):
            model.graph.output.append(
                onnx.helper.make_tensor_value_info(name, onnx.TensorProto.INT8, None)
            )
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self._session = ort.InferenceSession(
            model.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )

    def quantize_input(self, float_input: np.ndarray) -> np.ndarray:
        b = self.boundary
        q = np.rint(float_input / b.input_scale) + b.input_zero_point
        return np.clip(q, -128, 127).astype(np.int8)

    def dequantize_input(self, int8_input: np.ndarray) -> np.ndarray:
        b = self.boundary
        x = (int8_input.astype(np.float32) - b.input_zero_point) * b.input_scale
        return x.reshape(b.input_shape).astype(np.float32)

    def run_int8(self, int8_input: np.ndarray) -> np.ndarray:
        """int8 input (any shape with the right element count) -> int8 output."""

        return self.run_int8_tensors(int8_input, ())[0]

    def run_int8_tensors(self, int8_input: np.ndarray, names: tuple[str, ...]) -> list[np.ndarray]:
        """Final int8 output followed by the requested int8 intermediates."""

        feed = {self.boundary.input_name: self.dequantize_input(int8_input)}
        outs = self._session.run([self.boundary.output_int8_name, *names], feed)
        return [np.asarray(o, dtype=np.int8) for o in outs]


def dataset_for_shape(shape: tuple[int, ...]) -> str:
    """Pick the dataset a model was built for from its input element count."""

    count = int(np.prod(shape))
    if count == 28 * 28:
        return "mnist"
    if count == 3 * 32 * 32:
        return "cifar10"
    raise ValueError(f"no dataset known for input shape {shape}")


def load_sample(dataset: str, index: int, reference: OnnxReference) -> Sample:
    """Load test-split example `index`, preprocessed to the model's float input."""

    if dataset == "mnist":
        images, labels = _mnist_test()
        # The quantized MNIST model was calibrated on raw [0, 1] pixels.
        x = images[index].astype(np.float32) / 255.0
        label = int(labels[index])
        name = str(label)
    elif dataset == "cifar10":
        images, labels = _cifar10_test()
        x = (images[index].astype(np.float32) / 255.0 - _CIFAR10_MEAN) / _CIFAR10_STD
        label = int(labels[index])
        name = CIFAR10_CLASSES[label]
    else:
        raise ValueError(f"unknown dataset {dataset!r}")

    x = x.reshape(reference.boundary.input_shape).astype(np.float32)
    return Sample(dataset, index, label, name, x, reference.quantize_input(x))


def _mnist_test() -> tuple[np.ndarray, np.ndarray]:
    def idx(path: Path, offset: int) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run `uv run python -m models.mnist` once")
        return np.frombuffer(path.read_bytes(), dtype=np.uint8, offset=offset)

    images = idx(MNIST_DIR / "t10k-images-idx3-ubyte", 16).reshape(-1, 28, 28)
    labels = idx(MNIST_DIR / "t10k-labels-idx1-ubyte", 8)
    return images, labels


def _cifar10_test() -> tuple[np.ndarray, np.ndarray]:
    path = CIFAR10_DIR / "test_batch"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; extract cifar-10-python.tar.gz into build/")
    with path.open("rb") as f:
        batch = pickle.load(f, encoding="bytes")
    images = np.asarray(batch[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
    return images, np.asarray(batch[b"labels"], dtype=np.int64)
