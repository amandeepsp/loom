"""Shared protocol definitions for CFU link."""

import struct

MAGIC_REQ = 0xCF
MAGIC_RESP = 0xFC


def make_mac4_request(a: int, b: int, seq_id: int = 1) -> bytes:
    payload = struct.pack("<ii", a, b)
    header = struct.pack("<BBHHH", MAGIC_REQ, 0x01, len(payload), seq_id, 0)
    return header + payload


def parse_response(data: bytes) -> dict:
    if len(data) < 8:
        raise ValueError(f"Response too short: {len(data)} bytes")
    magic, status, payload_len, seq_id, cycles = struct.unpack("<BBHHH", data[:8])
    if magic != MAGIC_RESP:
        raise ValueError(f"Bad response magic: 0x{magic:02x} in {data[:8].hex()}")
    payload = data[8 : 8 + payload_len]
    result = struct.unpack("<i", payload)[0] if payload_len >= 4 else None
    return {"status": status, "seq_id": seq_id, "cycles": cycles, "result": result}
