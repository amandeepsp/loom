"""SimLink - communicates with firmware in litex_sim via stdin/stdout pipes."""

import os
import struct
import subprocess
import sys
import time

from .protocol import MAGIC_RESP, make_mac4_request, parse_response


class SimLink:
    def __init__(
        self,
        cfu="top.v",
        firmware="firmware/zig-out/bin/firmware.bin",
        ram_size=0x800000,
        verbose=False,
    ):
        self.verbose = verbose

        cmd = [
            sys.executable,
            "-m",
            "litex.tools.litex_sim",
            "--cpu-type",
            "vexriscv",
            "--cpu-variant",
            "full+cfu",
            "--cpu-cfu",
            cfu,
            "--integrated-main-ram-size",
            str(ram_size),
            "--ram-init",
            firmware,
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
            magic, status, payload_len, r_seq_id, cycles = struct.unpack(
                "<BBHHH", resp_hdr
            )

        resp_payload = self._read_exact(payload_len) if payload_len > 0 else b""
        if self.verbose:
            print(
                f"[host] << status={status} payload_len={payload_len} payload={resp_payload.hex()}"
            )
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
