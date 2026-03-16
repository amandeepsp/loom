#!/usr/bin/env python3
"""Host-side client for the CFU link protocol over litex_sim.

Spawns litex_sim --non-interactive and communicates binary data
over stdin/stdout pipes.

Usage:
    uv run python host/client.py --test
    uv run python host/client.py --test -v
"""

import argparse
import os
import select
import struct
import subprocess
import sys
import time
import threading

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


class SimLink:
    """Communicate with firmware in litex_sim via stdin/stdout pipes."""

    def __init__(self, cfu="top.v", firmware="firmware/zig-out/bin/firmware.bin",
                 ram_size=0x800000, verbose=False):
        self.verbose = verbose

        cmd = [
            sys.executable, "-m", "litex.tools.litex_sim",
            "--cpu-type", "vexriscv",
            "--cpu-variant", "full+cfu",
            "--cpu-cfu", cfu,
            "--integrated-main-ram-size", str(ram_size),
            "--ram-init", firmware,
            "--non-interactive",
        ]
        if verbose:
            print(f"[host] launching sim...")

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={**os.environ},
        )

    def wait_for_ready(self, timeout=120) -> str:
        """Read stdout until '[link] ready\\n' appears."""
        deadline = time.time() + timeout
        buf = b""
        while time.time() < deadline:
            ch = self.proc.stdout.read(1)
            if not ch:
                raise RuntimeError("Sim process ended unexpectedly")
            buf += ch
            if buf.endswith(b"[link] ready\n"):
                text = buf.decode("utf-8", errors="replace")
                if self.verbose:
                    print(text, end="")
                return text
        raise TimeoutError(f"Timed out. Got: {buf[-200:]!r}")

    def mac4(self, a: int, b: int, seq_id: int = 1) -> int:
        req = make_mac4_request(a, b, seq_id)
        if self.verbose:
            print(f"[host] >> {req.hex()}")
        self.proc.stdin.write(req)
        self.proc.stdin.flush()

        # Read 8-byte response header
        resp_hdr = self._read_exact(8)
        if self.verbose:
            print(f"[host] << hdr: {resp_hdr.hex()}")

        magic, status, payload_len, r_seq_id, cycles = struct.unpack("<BBHHH", resp_hdr)

        # Skip non-response bytes (firmware debug output mixed in)
        while magic != MAGIC_RESP:
            # Shift by 1 byte and read another
            resp_hdr = resp_hdr[1:] + self._read_exact(1)
            magic, status, payload_len, r_seq_id, cycles = struct.unpack("<BBHHH", resp_hdr)

        resp_payload = self._read_exact(payload_len) if payload_len > 0 else b""
        if self.verbose:
            print(f"[host] << status={status} payload_len={payload_len} payload={resp_payload.hex()}")
        resp = parse_response(resp_hdr + resp_payload)

        if resp["status"] != 0:
            raise RuntimeError(f"Firmware error: status=0x{resp['status']:02x}")
        return resp["result"]

    def _read_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.proc.stdout.read(n - len(buf))
            if not chunk:
                raise RuntimeError(f"EOF after {len(buf)}/{n} bytes")
            buf += chunk
        return buf

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def run_tests(link: SimLink):
    def check(a, b, expected, desc=""):
        result = link.mac4(a, b)
        ok = result == expected
        tag = "PASS" if ok else f"FAIL (got 0x{result:08x}, want 0x{expected:08x})"
        print(f"  mac4(0x{a:08x}, 0x{b:08x}) = 0x{result:08x}  {tag}  {desc}")
        return ok

    print("\n== Host-side link tests ==\n")
    passed = total = 0

    for a, b, exp, desc in [
        (0x00000000, 0x00000000, 0x00000000, "zeros"),
        (0x00000000, 0x01010101, 0x00000200, "offset only"),
        (0x01010101, 0x01010101, 0x00000204, "1s * 1s"),
        (0x01020304, 0x04030201, 0x00000514, "mixed"),
    ]:
        total += 1
        if check(a, b, exp, desc):
            passed += 1

    print(f"\nResults: {passed}/{total}", "ALL PASSED" if passed == total else "SOME FAILED")
    return passed == total


def main():
    parser = argparse.ArgumentParser(description="CFU host client")
    parser.add_argument("--test", action="store_true", help="Run test suite")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--cfu", default="top.v")
    parser.add_argument("--firmware", default="firmware/zig-out/bin/firmware.bin")
    args = parser.parse_args()

    if args.test:
        sim = SimLink(cfu=args.cfu, firmware=args.firmware, verbose=args.verbose)
        try:
            sim.wait_for_ready()
            success = run_tests(sim)
            sys.exit(0 if success else 1)
        finally:
            sim.close()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
