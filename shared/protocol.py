"""Shared wire protocol implementation for Python transports.

Mirrors shared/protocol.zig — single source of truth for framing,
header packing, and error handling.
"""

from __future__ import annotations

import os
import socket as _socket
import struct
from dataclasses import dataclass

MAGIC_REQ = 0xCF
MAGIC_RESP = 0xFC

OP_PING = 0x00
OP_READ = 0x10
OP_WRITE = 0x11
OP_EXEC_MODEL = 0x13
OP_PERF = 0x14
OP_CHECKSUM = 0x15
OP_PROFILE = 0x16

STATUS_OK = 0x00
WRITE_CHUNK_MAX = 1024


@dataclass(frozen=True)
class Response:
    """Parsed response from the loom."""

    status: int
    payload: bytes
    cycles_lo: int


@dataclass(frozen=True)
class PerfCounters:
    seq_busy_cycles: int
    dma_active_cycles: int
    overlap_cycles: int


class ProtocolError(RuntimeError):
    """Raised when the wire protocol encounters an error."""


class FramingProtocol:
    """Handles request/response framing over a byte stream (socket or serial).

    Usage:
        proto = FramingProtocol()
        proto.write_request(stream, OP_WRITE, payload)
        resp = proto.read_response(stream)
    """

    def write_request(
        self,
        stream,
        op: int,
        payload: bytes,
        seq_id: int = 0,
    ) -> None:
        """Send a framed request."""
        if len(payload) > 0xFFFF:
            raise ProtocolError("payload too large")
        header = struct.pack("<BBHHH", MAGIC_REQ, op, len(payload), seq_id, 0)
        write = stream.sendall if hasattr(stream, "sendall") else stream.write
        write(header)
        write(payload)

    def read_response(self, stream) -> Response:
        """Read a framed response, skipping any non-magic bytes (firmware banner)."""
        # Scan for response magic byte
        while True:
            magic = self._read_exact(stream, 1)
            if magic[0] == MAGIC_RESP:
                break

        rest = self._read_exact(stream, 7)
        status, payload_len, _seq_id, cycles_lo = struct.unpack("<BHHH", rest)
        payload = self._read_exact(stream, payload_len)
        return Response(status, payload, cycles_lo)

    @staticmethod
    def _read_exact(stream, length: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < length:
            chunk = stream.recv(length - len(chunks)) if hasattr(stream, "recv") else stream.read(length - len(chunks))
            if not chunk:
                raise ProtocolError("connection closed while reading response")
            chunks.extend(chunk)
        return bytes(chunks)


READ_CHUNK_MAX = 8192

# Must match the SoC build (`--uart-baudrate`); override with LOOM_BAUD.
DEFAULT_BAUD = int(os.environ.get("LOOM_BAUD", "115200"))


class BaseTransport:
    """Common transport logic: chunked mem ops, model exec, checksum over FramingProtocol."""

    def __init__(self) -> None:
        self._proto = FramingProtocol()
        self._seq_id = 0

    def _next_seq(self) -> int:
        sid = self._seq_id
        self._seq_id = (self._seq_id + 1) & 0xFFFF
        return sid

    def _request(self, op: int, payload: bytes, *, expected_len: int | None = None) -> bytes:
        raise NotImplementedError

    def write_mem(self, addr: int, data: bytes) -> None:
        chunk_max = min(WRITE_CHUNK_MAX, 0xFFFF - 4)
        for offset in range(0, len(data), chunk_max):
            chunk = data[offset : offset + chunk_max]
            self._request(OP_WRITE, struct.pack("<I", addr + offset) + chunk, expected_len=0)

    def read_mem(self, addr: int, length: int) -> bytes:
        out = bytearray()
        # Long device->host bursts drop bytes on the Tang Nano's USB-UART
        # bridge at 1 Mbaud (64 KiB reads fail, 16 KiB pass); keep reads small.
        chunk_max = READ_CHUNK_MAX
        for offset in range(0, length, chunk_max):
            chunk_len = min(chunk_max, length - offset)
            payload = struct.pack("<II", addr + offset, chunk_len)
            out.extend(self._request(OP_READ, payload, expected_len=chunk_len))
        return bytes(out)

    def profile(self) -> list[int]:
        """Per-step cycles of the last exec_model: [constants, step0, step1, ...]."""
        data = self._request(OP_PROFILE, b"")
        (count,) = struct.unpack_from("<I", data)
        return list(struct.unpack_from(f"<{count}I", data, 4))

    def checksum(self, addr: int, length: int) -> int:
        """CRC-32 of device memory (matches zlib.crc32)."""
        data = self._request(OP_CHECKSUM, struct.pack("<II", addr, length), expected_len=4)
        return struct.unpack("<I", data)[0]

    def exec_model(self, addr: int, length: int) -> int:
        data = self._request(OP_EXEC_MODEL, struct.pack("<II", addr, length), expected_len=4)
        return struct.unpack("<I", data)[0]

    def perf_counters(self, *, reset: bool = False) -> PerfCounters:
        data = self._request(OP_PERF, struct.pack("<I", int(reset)), expected_len=3 * 4)
        return PerfCounters(*struct.unpack("<3I", data))

    def ping(self) -> None:
        self._request(OP_PING, b"", expected_len=0)


class TcpTransport(BaseTransport):
    """Transport over a TCP socket (e.g., Verilator simulation)."""

    def __init__(self, endpoint: str, timeout_s: float = 300):
        super().__init__()
        host, _, port = endpoint.removeprefix("tcp://").rpartition(":")
        self._sock = _socket.create_connection((host, int(port)), timeout=timeout_s)
        self._sock.settimeout(timeout_s)

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def _request(self, op: int, payload: bytes, *, expected_len: int | None = None) -> bytes:
        sid = self._next_seq()
        self._proto.write_request(self._sock, op, payload, sid)
        resp = self._proto.read_response(self._sock)
        if resp.status != STATUS_OK:
            raise ProtocolError(f"device error status=0x{resp.status:02x} debug={list(resp.payload)}")
        if expected_len is not None and len(resp.payload) != expected_len:
            raise ProtocolError(f"bad response length: expected {expected_len}, got {len(resp.payload)}")
        return resp.payload


class SerialTransport(BaseTransport):
    """Transport over a physical serial port using pyserial."""

    def __init__(self, port: str, baud_rate: int = DEFAULT_BAUD, timeout_s: float = 120):
        super().__init__()
        import serial

        self._port = serial.Serial(port, baudrate=baud_rate, timeout=timeout_s)

    def close(self) -> None:
        if self._port is not None:
            self._port.close()
            self._port = None

    def _request(self, op: int, payload: bytes, *, expected_len: int | None = None) -> bytes:
        sid = self._next_seq()
        self._port.reset_input_buffer()  # drop stale bytes from an aborted response
        self._proto.write_request(self._port, op, payload, sid)
        resp = self._proto.read_response(self._port)
        if resp.status != STATUS_OK:
            raise ProtocolError(f"device error status=0x{resp.status:02x} debug={list(resp.payload)}")
        if expected_len is not None and len(resp.payload) != expected_len:
            raise ProtocolError(f"bad response length: expected {expected_len}, got {len(resp.payload)}")
        return resp.payload
