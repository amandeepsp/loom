#!/usr/bin/env python3
"""Host-side client for the CFU link protocol.

Supports two transports:
  - SimLink:    spawns litex_sim and talks over stdin/stdout pipes
  - SerialLink: talks to a real FPGA over a serial port

Usage:
    uv run python host/client.py --test                        # sim
    uv run python host/client.py --test --serial /dev/ttyUSB1  # real FPGA (flashes)
    uv run python host/client.py --test --serial /dev/ttyUSB1 --no-upload  # already running
"""

import argparse
import sys

from lib.seriallink import SerialLink
from lib.simlink import SimLink


def run_tests(link):
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

    print(
        f"\nResults: {passed}/{total}",
        "ALL PASSED" if passed == total else "SOME FAILED",
    )
    return passed == total


def main():
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
        "--baud", type=int, default=115200, help="Serial baud rate (default: 115200)"
    )
    parser.add_argument("--cfu", default="top.v")
    parser.add_argument("--firmware", default="firmware/zig-out/bin/firmware.bin")
    args = parser.parse_args()

    if args.test:
        if args.serial:
            link = SerialLink(
                args.serial,
                firmware=args.firmware,
                baudrate=args.baud,
                verbose=args.verbose,
                no_upload=args.no_upload,
            )
        else:
            link = SimLink(cfu=args.cfu, firmware=args.firmware, verbose=args.verbose)
        try:
            link.wait_for_ready()
            success = run_tests(link)
            sys.exit(0 if success else 1)
        finally:
            link.close()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
