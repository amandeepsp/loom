"""Shared protocol definitions for CFU link."""

import struct

MAGIC_REQ = 0xCF
MAGIC_RESP = 0xFC
OP_PING = 0x00
OP_MAC4 = 0x01


def make_request(opcode: int, payload: bytes = b"", seq_id: int = 1) -> bytes:
    header = struct.pack("<BBHHH", MAGIC_REQ, opcode, len(payload), seq_id, 0)
    return header + payload


def make_ping_request(seq_id: int = 1) -> bytes:
    return make_request(OP_PING, seq_id=seq_id)


def make_mac4_request(a: int, b: int, seq_id: int = 1) -> bytes:
    payload = struct.pack("<ii", a, b)
    return make_request(OP_MAC4, payload, seq_id)


def parse_response(data: bytes) -> dict:
    if len(data) < 8:
        raise ValueError(f"Response too short: {len(data)} bytes")
    magic, status, payload_len, seq_id, cycles = struct.unpack("<BBHHH", data[:8])
    if magic != MAGIC_RESP:
        raise ValueError(f"Bad response magic: 0x{magic:02x} in {data[:8].hex()}")
    payload = data[8 : 8 + payload_len]
    result = struct.unpack("<i", payload)[0] if payload_len >= 4 else None
    return {
        "status": status,
        "seq_id": seq_id,
        "cycles": cycles,
        "payload": payload,
        "result": result,
    }
