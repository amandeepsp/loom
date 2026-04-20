#!/usr/bin/env -S uv run python

import argparse
import binascii
import pathlib
import subprocess
import sys
import time

import serial


SFL_MAGIC_REQ = b"sL5DdSMmkekro\n"
SFL_MAGIC_ACK = b"z6IHG7cYDID6o\n"

SFL_CMD_LOAD = 0x01
SFL_CMD_JUMP = 0x02

SFL_ACK_SUCCESS = b"K"
SFL_ACK_CRCERROR = b"C"
SFL_ACK_UNKNOWN = b"U"
SFL_ACK_ERROR = b"E"

LINK_READY_MARKER = b"[link] ready"


def crc16(data: bytes) -> int:
    return binascii.crc_hqx(data, 0)


def encode_frame(cmd: int, payload: bytes) -> bytes:
    body = bytes([cmd]) + payload
    return bytes([len(payload)]) + crc16(body).to_bytes(2, "big") + body


def wait_for_magic(port: serial.Serial, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    window = bytearray()

    while time.monotonic() < deadline:
        try:
            chunk = port.read(1)
        except serial.SerialException:
            time.sleep(0.05)
            continue
        if not chunk:
            continue

        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()

        window += chunk
        if len(window) > len(SFL_MAGIC_REQ):
            del window[:-len(SFL_MAGIC_REQ)]
        if bytes(window) == SFL_MAGIC_REQ:
            return

    raise TimeoutError("timed out waiting for LiteX serial boot magic")


def read_ack(port: serial.Serial, timeout: float) -> bytes:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            ack = port.read(1)
        except serial.SerialException:
            time.sleep(0.05)
            continue
        if ack:
            return ack
    raise TimeoutError("timed out waiting for device ACK")


def open_port(port_name: str, speed: int, timeout: float = 5.0) -> serial.Serial:
    deadline = time.monotonic() + timeout
    while True:
        try:
            return serial.Serial(port_name, speed, timeout=0.05, write_timeout=1.0)
        except serial.SerialException:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.1)


def upload_image(port: serial.Serial, image: bytes, address: int, chunk_size: int, ack_timeout: float) -> None:
    position = 0
    total = len(image)

    print(f"[upload-once] Uploading 0x{address:08x} ({total} bytes) in {chunk_size}-byte chunks.")

    while position < total:
        chunk = image[position : position + chunk_size]
        frame = encode_frame(SFL_CMD_LOAD, address.to_bytes(4, "big") + chunk)
        port.write(frame)
        ack = read_ack(port, ack_timeout)
        if ack != SFL_ACK_SUCCESS:
            raise RuntimeError(
                f"unexpected ACK at offset {position}: got {ack!r}, expected {SFL_ACK_SUCCESS!r}"
            )

        position += len(chunk)
        address += len(chunk)
        percent = (100 * position) // total
        print(f"[upload-once] {position}/{total} bytes ({percent}%)", end="\r", flush=True)

    print()


def jump_to_image(port: serial.Serial, address: int, ack_timeout: float) -> None:
    port.write(encode_frame(SFL_CMD_JUMP, address.to_bytes(4, "big")))
    ack = read_ack(port, ack_timeout)
    if ack != SFL_ACK_SUCCESS:
        raise RuntimeError(f"unexpected jump ACK: got {ack!r}, expected {SFL_ACK_SUCCESS!r}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload a LiteX kernel over serial boot and exit without holding the port open.",
    )
    parser.add_argument("port", help="Serial port, for example /dev/ttyUSB1.")
    parser.add_argument("kernel", help="Kernel image to upload.")
    parser.add_argument("--speed", type=int, default=115200, help="Serial baudrate.")
    parser.add_argument("--base", type=lambda x: int(x, 0), default=0x40000000, help="Kernel load address.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=60,
        help="Image data bytes per SFL load frame (address bytes are added separately).",
    )
    parser.add_argument("--boot-timeout", type=float, default=8.0, help="Time to wait for serial boot magic.")
    parser.add_argument("--ack-timeout", type=float, default=1.0, help="Time to wait for each device ACK.")
    parser.add_argument(
        "--post-boot-timeout",
        type=float,
        default=10.0,
        help="Max time to read post-jump UART output (and to wait for [link] ready when enabled).",
    )
    parser.add_argument(
        "--wait-link-ready",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After jump, keep reading until firmware prints [link] ready or post-boot-timeout (default: on).",
    )
    parser.add_argument("--reset-command", help="Command to run after opening the port, before waiting for boot magic.")
    args = parser.parse_args()

    # LiteX SFL length covers the 4-byte load address plus image bytes.
    if args.chunk_size <= 0 or args.chunk_size > 247:
        print("[upload-once] chunk-size must be in the range 1..247.", file=sys.stderr)
        return 2

    image = pathlib.Path(args.kernel).read_bytes()

    if args.reset_command:
        print(f"[upload-once] Running reset command: {args.reset_command}")
        subprocess.run(args.reset_command, shell=True, check=True)
        time.sleep(0.5)

    with open_port(args.port, args.speed) as port:
        port.reset_input_buffer()
        port.reset_output_buffer()

        try:
            wait_for_magic(port, args.boot_timeout)
            port.write(SFL_MAGIC_ACK)
            upload_image(port, image, args.base, args.chunk_size, args.ack_timeout)
            jump_to_image(port, args.base, args.ack_timeout)
            print("[upload-once] Jumped to uploaded image.")
        except Exception as exc:
            print(f"[upload-once] {exc}", file=sys.stderr)
            return 1

        deadline = time.monotonic() + args.post_boot_timeout
        seen = bytearray()
        while time.monotonic() < deadline:
            chunk = port.read(256)
            if chunk:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                if args.wait_link_ready:
                    seen.extend(chunk)
                    if LINK_READY_MARKER in seen:
                        break
            else:
                time.sleep(0.01)
        else:
            if args.wait_link_ready and LINK_READY_MARKER not in seen:
                print(
                    "[upload-once] warning: post-boot window ended before [link] ready; "
                    "host driver will retry ping.",
                    file=sys.stderr,
                )

        port.reset_input_buffer()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
