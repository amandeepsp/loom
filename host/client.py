#!/usr/bin/env python3
"""Host-side client helpers for the CFU link protocol.

Supports two transports:
  - SimLink:    spawns litex_sim and talks over stdin/stdout pipes
  - SerialLink: talks to a real FPGA over a serial port

Usage:
    uv run python host/client.py --test                          # sim
    uv run python host/client.py --test --serial /dev/ttyUSB1   # real FPGA (flashes)
    uv run python host/client.py --test --serial /dev/ttyUSB1 --no-upload
    uv run python host/numpy_sim.py                              # library-style NumPy example
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING




from lib.seriallink import SerialLink
from lib.simlink import SimLink

DEFAULT_BAUD = 115200
DEFAULT_CFU = "top.v"
DEFAULT_FIRMWARE = "firmware/zig-out/bin/firmware.bin"
INPUT_OFFSET = 128


def create_link(
    *,
    serial: str | None = None,
    baud: int = DEFAULT_BAUD,
    cfu: str = DEFAULT_CFU,
    firmware: str = DEFAULT_FIRMWARE,
    verbose: bool = False,
    no_upload: bool = False,
):
    if serial:
        return SerialLink(
            serial,
            firmware=firmware,
            baudrate=baud,
            verbose=verbose,
            no_upload=no_upload,
        )
    return SimLink(cfu=cfu, firmware=firmware, verbose=verbose)


@contextmanager
def open_link(**kwargs):
    link = create_link(**kwargs)
    try:
        link.wait_for_ready()
        yield link
    finally:
        link.close()


def _pack_u8x4(chunk) -> int:
    packed = 0
    for lane, value in enumerate(chunk):
        packed |= (int(value) & 0xFF) << (8 * lane)
    return packed


def _as_u8_vector(values) -> "np.ndarray":
    import numpy as np

    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError("mac4_numpy expects integer array-like inputs")
    flat = array.reshape(-1)
    if flat.size == 0:
        return np.zeros(0, dtype=np.uint8)
    if np.any(flat < 0) or np.any(flat > 0xFF):
        raise ValueError("mac4_numpy expects values in the range [0, 255]")
    return flat.astype(np.uint8, copy=False)


def iter_mac4_chunks(lhs, rhs):
    import numpy as np

    lhs_array = _as_u8_vector(lhs)
    rhs_array = _as_u8_vector(rhs)
    if lhs_array.shape != rhs_array.shape:
        raise ValueError(
            f"mac4_numpy expects matching shapes, got {lhs_array.shape} and {rhs_array.shape}"
        )

    if lhs_array.size == 0:
        return

    pad = (-lhs_array.size) % 4
    if pad:
        lhs_array = np.pad(lhs_array, (0, pad), constant_values=0)
        rhs_array = np.pad(rhs_array, (0, pad), constant_values=0)

    for offset in range(0, lhs_array.size, 4):
        yield (
            _pack_u8x4(lhs_array[offset : offset + 4]),
            _pack_u8x4(rhs_array[offset : offset + 4]),
        )


def mac4_numpy(link, lhs, rhs) -> int:
    total = 0
    for packed_lhs, packed_rhs in iter_mac4_chunks(lhs, rhs):
        total += link.mac4(packed_lhs, packed_rhs)
    return total


def mac4_numpy_reference(lhs, rhs, input_offset: int = INPUT_OFFSET) -> int:
    lhs_array = _as_u8_vector(lhs).astype("int32", copy=False)
    rhs_array = _as_u8_vector(rhs).astype("int32", copy=False)
    if lhs_array.shape != rhs_array.shape:
        raise ValueError(
            f"mac4_numpy_reference expects matching shapes, got {lhs_array.shape} and {rhs_array.shape}"
        )
    return int(((lhs_array + input_offset) * rhs_array).sum(dtype="int64"))


def run_tests(link):
    if hasattr(link, "ping"):
        ok = link.ping()
        print(f"\n== Host-side link tests ==\n\n  ping() = {'PASS' if ok else 'FAIL'}")
        if not ok:
            return False
    else:
        print("\n== Host-side link tests ==\n")

    def check(a, b, expected, desc=""):
        result = link.mac4(a, b)
        ok = result == expected
        tag = "PASS" if ok else f"FAIL (got 0x{result:08x}, want 0x{expected:08x})"
        print(f"  mac4(0x{a:08x}, 0x{b:08x}) = 0x{result:08x}  {tag}  {desc}")
        return ok

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

    print(
        f"\nResults: {passed}/{total}",
        "ALL PASSED" if passed == total else "SOME FAILED",
    )
    return passed == total


def parse_args():
    parser = argparse.ArgumentParser(description="CFU host client")
    parser.add_argument("--test", action="store_true", help="Run test suite")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--serial", metavar="PORT", help="Serial port for real FPGA (e.g. /dev/ttyUSB1)"
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip firmware upload (assume already running)",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=DEFAULT_BAUD,
        help=f"Serial baud rate (default: {DEFAULT_BAUD})",
    )
    parser.add_argument("--cfu", default=DEFAULT_CFU)
    parser.add_argument("--firmware", default=DEFAULT_FIRMWARE)
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.test:
        return 0

    with open_link(
        serial=args.serial,
        baud=args.baud,
        cfu=args.cfu,
        firmware=args.firmware,
        verbose=args.verbose,
        no_upload=args.no_upload,
    ) as link:
        return 0 if run_tests(link) else 1


if __name__ == "__main__":
    sys.exit(main())
