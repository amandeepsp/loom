"""SerialLink - communicates with firmware on a real FPGA over a serial port."""

import struct
import time

import crcmod
import serial

from .protocol import MAGIC_RESP, make_mac4_request, parse_response

_crc16 = crcmod.predefined.mkCrcFun("xmodem")


class SerialLink:
    """Communicate with firmware on a real FPGA over a serial port.

    Uploads the firmware binary via the LiteX SFL boot protocol (same as
    litex_term --kernel), then enters the link protocol loop.
    """

    def __init__(
        self,
        port: str,
        firmware: str = "firmware/zig-out/bin/firmware.bin",
        kernel_address: int = 0x40000000,
        baudrate: int = 115200,
        verbose: bool = False,
        no_upload: bool = False,
    ):
        self.verbose = verbose
        self.firmware = firmware
        self.kernel_address = kernel_address
        self.no_upload = no_upload
        self.ser = serial.Serial(port, baudrate, timeout=1)
        if verbose:
            print(f"[host] opened {port} @ {baudrate}")

    def wait_for_ready(self, timeout=60) -> str:
        """Wait for SFL prompt, upload firmware, then wait for '[link] ready'."""
        deadline = time.time() + timeout
        buf = b""

        sfl_prompt_req = b"F7:    boot from serial\n"
        sfl_prompt_ack = b"\x06"
        sfl_magic_req = b"sL5DdSMmkekro\n"
        sfl_magic_ack = b"z6IHG7cYDID6o\n"

        if self.no_upload:
            print("[host] skipping upload, waiting for '[link] ready'...")
        else:
            print(
                "[host] waiting for BIOS SFL boot request... (press board reset button)"
            )
        prompted = False
        while time.time() < deadline:
            ch = self.ser.read(1)
            if not ch:
                continue
            buf += ch
            if self.verbose and ch == b"\n":
                lines = buf.decode("utf-8", errors="replace").rstrip().rsplit("\n", 1)
                print(f"[bios] {lines[-1]}")
            if not self.no_upload:
                if not prompted and buf.endswith(sfl_prompt_req):
                    if self.verbose:
                        print("[host] got SFL prompt, sending ACK...")
                    self.ser.write(sfl_prompt_ack)
                    prompted = True
                if buf.endswith(sfl_magic_req):
                    if self.verbose:
                        print("[host] got SFL magic, sending ACK...")
                    self.ser.write(sfl_magic_ack)
                    self.ser.flush()
                    time.sleep(0.5)
                    self.ser.reset_input_buffer()
                    self._sfl_upload()
                    break
            if self.no_upload and "[link] ready" in buf.decode(
                "utf-8", errors="replace"
            ):
                break
        else:
            if not self.no_upload:
                text = buf.decode("utf-8", errors="replace")
                raise TimeoutError(
                    f"Timed out waiting for SFL magic. Got:\n{text[-500:]}"
                )

        # Phase 2: wait for firmware ready message
        buf = b""
        while time.time() < deadline:
            ch = self.ser.read(1)
            if not ch:
                continue
            buf += ch
            text = buf.decode("utf-8", errors="replace")
            if "[link] ready" in text:
                if self.verbose:
                    print(text, end="")
                print("[host] ready (serial)")
                self.ser.timeout = 10
                return ""
        raise TimeoutError(
            f"Firmware uploaded but no ready message. Got: {buf[-200:]!r}"
        )

    def _sfl_upload(self):
        """Upload firmware binary using LiteX SFL protocol."""
        with open(self.firmware, "rb") as f:
            data = f.read()

        sfl_cmd_load = b"\x01"
        sfl_cmd_jump = b"\x02"
        sfl_ack_success = b"K"
        chunk_size = 251  # sfl_payload_length(255) - 4 bytes for address

        address = self.kernel_address
        offset = 0
        total = len(data)

        while offset < total:
            chunk = data[offset : offset + chunk_size]
            payload = address.to_bytes(4, "big") + chunk

            crc = _crc16(sfl_cmd_load + payload)
            frame = (
                bytes([len(payload)]) + crc.to_bytes(2, "big") + sfl_cmd_load + payload
            )
            self.ser.write(frame)

            reply = self.ser.read(1)
            if reply != sfl_ack_success:
                raise RuntimeError(
                    f"SFL upload failed at offset {offset}: reply={reply!r}"
                )

            address += len(chunk)
            offset += len(chunk)

            if self.verbose and offset % 4096 == 0:
                print(
                    f"\r[host] uploaded {offset}/{total} bytes ({100 * offset // total}%)",
                    end="",
                )

        if self.verbose:
            print(f"\r[host] uploaded {total}/{total} bytes (100%)")

        # Send jump command to boot the firmware
        payload = self.kernel_address.to_bytes(4, "big")
        crc = _crc16(sfl_cmd_jump + payload)
        frame = bytes([len(payload)]) + crc.to_bytes(2, "big") + sfl_cmd_jump + payload
        self.ser.write(frame)
        reply = self.ser.read(1)
        if self.verbose:
            print(f"[host] jump command sent, reply={reply!r}")

    def mac4(self, a: int, b: int, seq_id: int = 1) -> int:
        req = make_mac4_request(a, b, seq_id)
        if self.verbose:
            print(f"[host] >> {req.hex()}")
        self.ser.write(req)
        self.ser.flush()

        resp_hdr = self._read_exact(8)
        if self.verbose:
            print(f"[host] << hdr: {resp_hdr.hex()}")

        magic, status, payload_len, r_seq_id, cycles = struct.unpack("<BBHHH", resp_hdr)

        while magic != MAGIC_RESP:
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
            chunk = self.ser.read(n - len(buf))
            if not chunk:
                raise RuntimeError(f"Serial read timeout after {len(buf)}/{n} bytes")
            buf += chunk
        return buf

    def close(self):
        self.ser.close()
