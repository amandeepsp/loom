"""Compiler numerics vs. ONNX on real data, plus runtime/manifest plumbing.

These run offline (no board, no sim). The first test would have caught the
TOSA-path requantization shift bug: the compiler emitted TOSA's raw shift
(which includes the multiplier's 31 fractional bits) where loom's epilogue
expects only the residual shift, collapsing every MNIST logit to the output
zero point.
"""

from __future__ import annotations

import numpy as np
import onnx
import pytest

from compiler import compile_onnx_model
from compiler.backend.reference import CIFAR10_DIR, MNIST_DIR, OnnxReference, load_sample
from compiler.backend.runtime import ModelBinaryArtifact, ModelBinaryRuntime
from shared.reference import cpu_requantize
from tests.mnist_model_binary import (
    DEFAULT_ONNX,
    compiled_layer_params,
    mnist_digit,
    mnist_reference,
    onnx_hidden_and_output,
    qgemm_output_zero_points,
)

needs_mnist = pytest.mark.skipif(
    not (MNIST_DIR / "t10k-images-idx3-ubyte").exists(), reason="MNIST test set not downloaded"
)


@needs_mnist
def test_mnist_compiled_requantization_matches_onnx():
    compiled = compile_onnx_model(DEFAULT_ONNX)
    (b0, m0, s0), (b1, m1, s1) = compiled_layer_params(compiled)
    zp0, zp1 = qgemm_output_zero_points()
    weights = {i.name: onnx.numpy_helper.to_array(i) for i in onnx.load(str(DEFAULT_ONNX)).graph.initializer}
    w0 = weights["net.1.weight_quantized"].astype(np.int32)
    w1 = weights["net.3.weight_quantized"].astype(np.int32)

    reference = mnist_reference()
    correct = 0
    for index in range(10):
        x, label = mnist_digit(reference, index)
        ref_h, ref_out = onnx_hidden_and_output(reference, x)

        h = cpu_requantize(x.astype(np.int32) @ w0.T, b0, m0, s0, zp0, -128, 127)
        out = cpu_requantize(h.astype(np.int32) @ w1.T, b1, m1, s1, zp1, -128, 127)

        assert np.abs(h.astype(np.int32) - ref_h).max() <= 1, f"image {index}: hidden layer"
        assert np.abs(out.astype(np.int32) - ref_out).max() <= 1, f"image {index}: logits"
        correct += int(out.argmax()) == label
    assert correct >= 9


def test_requantization_shift_is_residual():
    """The epilogue SRDHM consumes 31 bits; stored shifts must be small residuals."""
    for _bias, _mult, shift in compiled_layer_params(compile_onnx_model(DEFAULT_ONNX)):
        assert shift.min() >= 0 and shift.max() < 31


def test_manifest_records_source_onnx():
    compiled = compile_onnx_model(DEFAULT_ONNX)
    artifact = ModelBinaryArtifact.from_bytes(compiled.to_bytes())
    source = artifact.manifest["source"]
    assert source["onnx"].endswith("mnist_int8.onnx")
    assert len(source["onnx_sha256"]) == 64


@pytest.mark.parametrize(
    "dataset,onnx_path,root",
    [
        ("mnist", "models/out/mnist_int8.onnx", MNIST_DIR),
        ("cifar10", "models/out/mobilenet_v1_025_int8.onnx", CIFAR10_DIR),
    ],
)
def test_onnx_reference_classifies_real_samples(dataset, onnx_path, root):
    if not root.exists():
        pytest.skip(f"{dataset} test set not available")
    reference = OnnxReference(onnx_path)
    hits = 0
    for index in range(20):
        sample = load_sample(dataset, index, reference)
        assert sample.int8_input.dtype == np.int8
        hits += int(reference.run_int8(sample.int8_input).argmax()) == sample.label
    assert hits >= 15, f"{dataset}: ONNX reference only {hits}/20 correct — preprocessing drift?"


class _RecordingTransport:
    def __init__(self, output_size: int) -> None:
        self.writes: list[tuple[int, int]] = []
        self._output_size = output_size

    def write_mem(self, addr: int, data: bytes) -> None:
        self.writes.append((addr, len(data)))

    def read_mem(self, addr: int, length: int) -> bytes:
        return bytes(length)

    def exec_model(self, addr: int, length: int) -> int:
        return 1


def test_runtime_uploads_model_blob_once():
    artifact = ModelBinaryArtifact.from_bytes(compile_onnx_model(DEFAULT_ONNX).to_bytes())
    transport = _RecordingTransport(80)
    runtime = ModelBinaryRuntime(artifact, transport)
    x = np.zeros((1, 784), dtype=np.int8)
    runtime.run_quantized(x)
    runtime.run_quantized(x)
    blob_writes = [w for w in transport.writes if w == (runtime.model_addr, len(artifact.blob))]
    assert len(blob_writes) == 1
    assert len(transport.writes) == 3  # blob once + input twice


class _ResidentTransport(_RecordingTransport):
    """Device that already holds the blob: checksum returns its CRC."""

    def __init__(self, blob: bytes) -> None:
        super().__init__(80)
        import zlib

        self._crc = zlib.crc32(blob)
        self.checksum_calls = 0

    def checksum(self, addr: int, length: int) -> int:
        self.checksum_calls += 1
        return self._crc


def test_runtime_skips_upload_when_model_resident():
    artifact = ModelBinaryArtifact.from_bytes(compile_onnx_model(DEFAULT_ONNX).to_bytes())
    transport = _ResidentTransport(artifact.blob)
    runtime = ModelBinaryRuntime(artifact, transport)
    runtime.run_quantized(np.zeros((1, 784), dtype=np.int8))
    assert transport.checksum_calls == 1
    assert runtime.uploaded_bytes == 0
    assert all(length != len(artifact.blob) for _addr, length in transport.writes)


def test_runtime_uploads_when_resident_model_differs():
    artifact = ModelBinaryArtifact.from_bytes(compile_onnx_model(DEFAULT_ONNX).to_bytes())
    transport = _ResidentTransport(b"some other model")
    runtime = ModelBinaryRuntime(artifact, transport)
    runtime.run_quantized(np.zeros((1, 784), dtype=np.int8))
    assert runtime.uploaded_bytes == len(artifact.blob)
