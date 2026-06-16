import struct

import numpy as np
import pytest

import compiler.backend.run_model as run_model_cli
from compiler.backend import (
    ModelBinaryArtifact,
    ModelBinaryRuntime,
    ModelRuntime,
    compile_onnx_model,
    model_binary_manifest,
)
from compiler.backend.model import SECTION_CONSTANTS, SECTION_RISCV, SECTION_SCHEDULE


class FakeTransport:
    def __init__(self, output_addr: int, output_data: bytes):
        self.output_addr = output_addr
        self.output_data = output_data
        self.writes: list[tuple[int, bytes]] = []
        self.exec_calls: list[tuple[int, int]] = []
        self.reads: list[tuple[int, int]] = []

    def write_mem(self, addr: int, data: bytes) -> None:
        self.writes.append((addr, bytes(data)))

    def read_mem(self, addr: int, length: int) -> bytes:
        self.reads.append((addr, length))
        assert addr == self.output_addr
        assert length == len(self.output_data)
        return self.output_data

    def exec_model(self, addr: int, length: int) -> int:
        self.exec_calls.append((addr, length))
        return 1234

    def close(self) -> None:
        pass


def test_model_runtime_uploads_raw_input_model_and_reads_unpadded_output():
    compiled = compile_onnx_model("models/out/mnist_int8.onnx")
    first = compiled.kernels[0]
    final = compiled.kernels[-1]
    contract = compiled.manifest()["runtime_contract"]
    input_buf = next(buf for buf in compiled.buffers if buf.name == first.input_buffer)
    output_buf = next(buf for buf in compiled.buffers if buf.name == final.output_buffer)

    assert contract["boundary"] == "quantized-int8"
    assert contract["input"]["buffer"] == input_buf.name
    assert contract["input"]["shape"] == [first.m, first.k]
    assert contract["output"]["buffer"] == output_buf.name
    assert contract["output"]["shape"] == [final.m, final.n]

    padded_output = np.arange(final.padded_m * final.n, dtype=np.int8).reshape(final.padded_m, final.n)
    transport = FakeTransport(output_buf.addr, padded_output.tobytes())
    runtime = ModelRuntime(compiled, transport)

    matrix = np.arange(first.m * first.k, dtype=np.int16).reshape(first.m, first.k).astype(np.int8)
    result = runtime.run_quantized(matrix)

    assert transport.writes[0] == (input_buf.addr, matrix.tobytes())
    assert transport.writes[1] == (runtime.model_addr, runtime.model_blob)
    assert transport.exec_calls == [(runtime.model_addr, len(runtime.model_blob))]
    assert transport.reads == [(output_buf.addr, output_buf.size)]
    assert result.cycles == 1234
    np.testing.assert_array_equal(result.output, padded_output[: final.m, :])


def test_model_binary_artifact_runtime_uses_embedded_manifest_only(tmp_path):
    compiled = compile_onnx_model("models/out/mnist_int8.onnx")
    blob = compiled.to_bytes(include_debug_sections=False)
    artifact_path = tmp_path / "mnist.lmb"
    artifact_path.write_bytes(blob)

    artifact = ModelBinaryArtifact.from_file(artifact_path)
    contract = artifact.manifest["runtime_contract"]
    input_buf = artifact.buffer(contract["input"]["buffer"])
    output_buf = artifact.buffer(contract["output"]["buffer"])
    output_shape = tuple(contract["output"]["shape"])
    padded_shape = tuple(contract["output"]["padded_shape"])

    padded_output = np.arange(output_buf.size, dtype=np.int8).reshape(padded_shape)
    transport = FakeTransport(output_buf.addr, padded_output.tobytes())
    runtime = ModelBinaryRuntime(artifact, transport)
    matrix = np.arange(input_buf.size, dtype=np.int16).reshape(contract["input"]["shape"]).astype(np.int8)

    result = runtime.run_quantized(matrix)

    assert artifact.blob == blob
    assert artifact.model_addr == compiled.recommended_model_addr()
    assert transport.writes[0] == (input_buf.addr, matrix.tobytes())
    assert transport.writes[1] == (artifact.model_addr, blob)
    assert transport.exec_calls == [(artifact.model_addr, len(blob))]
    assert transport.reads == [(output_buf.addr, output_buf.size)]
    assert result.output.shape == output_shape
    assert result.model_addr == artifact.model_addr
    np.testing.assert_array_equal(result.output, padded_output[: output_shape[0], : output_shape[1]])


def test_model_binary_manifest_parser_rejects_bad_blob():
    with pytest.raises(ValueError, match="bad model binary magic"):
        model_binary_manifest(b"BAD!" + bytes(32))


def test_model_binary_artifact_rejects_missing_executable_sections():
    blob = bytearray(compile_onnx_model("models/out/mnist_int8.onnx").to_bytes())
    entry = _section_entry_offset(blob, SECTION_CONSTANTS)
    struct.pack_into("<H", blob, entry, 99)

    with pytest.raises(ValueError, match="missing required section 2"):
        ModelBinaryArtifact.from_bytes(blob)


def test_model_binary_artifact_rejects_bad_riscv_record():
    blob = bytearray(compile_onnx_model("models/out/mnist_int8.onnx").to_bytes())
    _typ, _flags, riscv_off, _size = _section_entry(blob, SECTION_RISCV)
    first_record = riscv_off + 4
    struct.pack_into("<I", blob, first_record, 2)

    with pytest.raises(ValueError, match="invalid code length 2"):
        ModelBinaryArtifact.from_bytes(blob)


def test_model_binary_artifact_rejects_bad_schedule_reference():
    blob = bytearray(compile_onnx_model("models/out/mnist_int8.onnx").to_bytes())
    _typ, _flags, schedule_off, _size = _section_entry(blob, SECTION_SCHEDULE)
    first_command_function_index = schedule_off + 4 + 4
    struct.pack_into("<I", blob, first_command_function_index, 99)

    with pytest.raises(ValueError, match="references missing RISC-V function 99"):
        ModelBinaryArtifact.from_bytes(blob)


def test_run_model_cli_executes_precompiled_artifact(monkeypatch, tmp_path, capsys):
    compiled = compile_onnx_model("models/out/mnist_int8.onnx")
    artifact = ModelBinaryArtifact.from_bytes(compiled.to_bytes(include_debug_sections=False))
    contract = artifact.manifest["runtime_contract"]
    input_buf = artifact.buffer(contract["input"]["buffer"])
    output_buf = artifact.buffer(contract["output"]["buffer"])
    padded_output = np.arange(output_buf.size, dtype=np.int8)
    fake_transport = FakeTransport(output_buf.addr, padded_output.tobytes())

    model_path = tmp_path / "model.lmb"
    input_path = tmp_path / "input.bin"
    output_path = tmp_path / "output.bin"
    expected_path = tmp_path / "expected.bin"
    model_path.write_bytes(artifact.blob)
    input_path.write_bytes(bytes(input_buf.size))
    expected_path.write_bytes(padded_output[:10].tobytes())

    def fake_tcp_transport(endpoint: str, timeout_s: float):
        assert endpoint == "tcp://127.0.0.1:21450"
        assert timeout_s == 12
        return fake_transport

    monkeypatch.setattr(run_model_cli, "TcpTransport", fake_tcp_transport)
    monkeypatch.setattr(
        "sys.argv",
        [
            "loom-run-model",
            str(model_path),
            str(input_path),
            "--port",
            "tcp://127.0.0.1:21450",
            "--timeout",
            "12",
            "-o",
            str(output_path),
            "--expect-output",
            str(expected_path),
            "--tolerance",
            "0",
            "--check-prediction",
        ],
    )

    assert run_model_cli.main() == 0

    assert fake_transport.writes[0] == (input_buf.addr, bytes(input_buf.size))
    assert fake_transport.writes[1] == (artifact.model_addr, artifact.blob)
    assert fake_transport.exec_calls == [(artifact.model_addr, len(artifact.blob))]
    assert output_path.read_bytes() == padded_output[:10].tobytes()
    out = capsys.readouterr().out
    assert "model: addr=0x400459c0" in out
    assert "sim output:" in out
    assert "output max delta: 0" in out
    assert "prediction: expected=9 actual=9" in out


def test_model_runtime_places_blob_after_planned_buffers():
    compiled = compile_onnx_model("models/out/mnist_int8.onnx")
    blob = compiled.to_bytes(include_debug_sections=False)
    addr = ModelRuntime.default_model_addr(compiled, len(blob))
    for buf in compiled.buffers:
        assert addr >= buf.addr + buf.size or addr + len(blob) <= buf.addr


def test_model_runtime_rejects_overlapping_model_address():
    compiled = compile_onnx_model("models/out/mnist_int8.onnx")
    output_buf = next(buf for buf in compiled.buffers if buf.name == compiled.kernels[-1].output_buffer)
    transport = FakeTransport(output_buf.addr, bytes(output_buf.size))
    with pytest.raises(ValueError, match="overlaps planned buffer"):
        ModelRuntime(compiled, transport, model_addr=compiled.buffers[0].addr)


def test_model_runtime_requires_quantized_input_shape_and_dtype():
    compiled = compile_onnx_model("models/out/mnist_int8.onnx")
    output_buf = next(buf for buf in compiled.buffers if buf.name == compiled.kernels[-1].output_buffer)
    runtime = ModelRuntime(compiled, FakeTransport(output_buf.addr, bytes(output_buf.size)))

    with pytest.raises(TypeError, match="dtype int8"):
        runtime.run_quantized(np.zeros((1, 784), dtype=np.float32))
    with pytest.raises(ValueError, match="input shape"):
        runtime.run_quantized(np.zeros((2, 784), dtype=np.int8))


def _section_entry_offset(blob: bytes | bytearray, section_type: int) -> int:
    _magic, _version, _flags, section_count = struct.unpack_from("<4sHHI", blob)
    cursor = 12
    for _ in range(section_count):
        typ, _flags, _off, _size = struct.unpack_from("<HHII", blob, cursor)
        if typ == section_type:
            return cursor
        cursor += 12
    raise AssertionError(f"section {section_type} not found")


def _section_entry(blob: bytes | bytearray, section_type: int) -> tuple[int, int, int, int]:
    return struct.unpack_from("<HHII", blob, _section_entry_offset(blob, section_type))
