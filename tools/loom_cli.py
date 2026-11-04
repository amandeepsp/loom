"""`loom` — one entry point for talking to the board.

    loom board                      what is the board doing right now?
    loom compile model.onnx -o m.lmb
    loom run m.lmb                  first test image, checked against a fresh ONNX reference
    loom run m.lmb --image 7 --count 20
    loom run m.lmb --input raw_i8.bin

`--port` defaults to $LOOM_PORT (or /dev/ttyUSB1); a `tcp://host:port`
endpoint targets the Verilator sim. Baud rate comes from $LOOM_BAUD.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np

from shared.board import clock_hz

DEFAULT_PORT = os.environ.get("LOOM_PORT", "/dev/ttyUSB1")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="loom", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_board = sub.add_parser("board", help="Report board state (firmware / BIOS / silent).")
    _add_link_args(p_board)
    p_board.set_defaults(fn=cmd_board)

    p_compile = sub.add_parser("compile", help="Compile ONNX to a .lmb (loom-compile-model).",
                               add_help=False)
    p_compile.set_defaults(fn=None)

    p_run = sub.add_parser("run", help="Run a .lmb on real test images and check it.")
    p_run.add_argument("model", help=".lmb model binary")
    src = p_run.add_mutually_exclusive_group()
    src.add_argument("--image", type=int, default=0, help="first test-split index (default 0)")
    src.add_argument("--input", help="raw int8 input file instead of a dataset image")
    p_run.add_argument("--count", type=int, default=1, help="number of consecutive images")
    p_run.add_argument("--onnx", help="source ONNX for the reference (default: from manifest)")
    p_run.add_argument("--tolerance", type=int, default=None,
                       help="fail if any logit differs from ONNX by more than this")
    p_run.add_argument("-o", "--output", help="write the last raw int8 output here")
    p_run.add_argument("--profile", action="store_true",
                       help="print per-step cycle breakdown of the last run")
    _add_link_args(p_run)
    p_run.set_defaults(fn=cmd_run)

    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["compile"]:
        from compiler.backend.__main__ import main as compile_main

        sys.argv = ["loom compile", *argv[1:]]
        return compile_main()

    args = parser.parse_args(argv)
    return args.fn(args)


def _add_link_args(p: argparse.ArgumentParser) -> None:
    from shared.protocol import DEFAULT_BAUD

    p.add_argument("--port", default=DEFAULT_PORT, help="serial device or tcp://host:port")
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    p.add_argument("--timeout", type=float, default=1800)


def _open(args):
    from shared.protocol import SerialTransport, TcpTransport

    if args.port.startswith("tcp://"):
        return TcpTransport(args.port, timeout_s=args.timeout)
    return SerialTransport(args.port, baud_rate=args.baud, timeout_s=args.timeout)


# --- board ------------------------------------------------------------------


def probe_board(port: str, baud: int) -> str:
    """Return 'firmware', 'bios', 'silent', or 'missing'."""

    import serial

    from shared.protocol import ProtocolError, SerialTransport

    if not Path(port).exists():
        return "missing"
    transport = SerialTransport(port, baud_rate=baud, timeout_s=1.0)
    try:
        transport.ping()
        return "firmware"
    except ProtocolError:
        pass
    finally:
        transport.close()
    with serial.Serial(port, baud, timeout=0.5) as raw:
        raw.reset_input_buffer()
        raw.write(b"\r")
        time.sleep(0.3)
        if b"litex" in raw.read(256):
            return "bios"
    return "silent"


def cmd_board(args) -> int:
    if args.port.startswith("tcp://"):
        _open(args).ping()
        print(f"{args.port}: firmware responding")
        return 0
    state = probe_board(args.port, args.baud)
    hints = {
        "firmware": "firmware responding — ready for `loom run`",
        "bios": "LiteX BIOS prompt, no firmware loaded — run `just up`",
        "silent": f"no response at {args.baud} baud — wrong LOOM_BAUD, or run `just reset`",
        "missing": "device not found — is the board plugged in? (check LOOM_PORT)",
    }
    print(f"{args.port}: {hints[state]}")
    return 0 if state == "firmware" else 1


# --- run --------------------------------------------------------------------


def cmd_run(args) -> int:
    from compiler.backend.reference import OnnxReference, dataset_for_shape, load_sample
    from compiler.backend.runtime import ModelBinaryArtifact, ModelBinaryRuntime

    artifact = ModelBinaryArtifact.from_file(args.model)
    contract = artifact.manifest["runtime_contract"]
    input_shape = tuple(contract["input"]["shape"])

    reference = _reference_for(artifact, args.onnx)
    if args.input:
        raw = np.frombuffer(Path(args.input).read_bytes(), dtype=np.int8)
        cases = [(Path(args.input).name, None, raw)]
    else:
        if reference is None:
            print("error: dataset images need the source ONNX (pass --onnx)", file=sys.stderr)
            return 2
        dataset = dataset_for_shape(reference.boundary.input_shape)
        cases = []
        for index in range(args.image, args.image + args.count):
            s = load_sample(dataset, index, reference)
            cases.append((f"{dataset}[{index}]", (s.label, s.label_name), s.int8_input))

    transport = _open(args)
    runtime = ModelBinaryRuntime(artifact, transport)
    failures = agree = correct = 0
    try:
        for i, (name, label, x) in enumerate(cases):
            t0 = time.monotonic()
            result = runtime.run_quantized(x.reshape(input_shape))
            wall = time.monotonic() - t0
            out = result.output.flatten()
            pred = int(out.argmax())

            parts = [f"{name:14}", f"pred={pred}"]
            if label is not None:
                ok = pred == label[0]
                correct += ok
                parts.append(f"label={label[0]} ({label[1]}) {'✓' if ok else '✗'}")
            if reference is not None:
                ref = reference.run_int8(x).flatten()
                delta = int(np.abs(out.astype(np.int32) - ref.astype(np.int32)).max())
                same = pred == int(ref.argmax())
                agree += same
                parts.append(f"onnx_pred={int(ref.argmax())} max|Δ|={delta}")
                if args.tolerance is not None and delta > args.tolerance:
                    failures += 1
                    parts.append("FAIL")
            parts.append(f"{result.cycles / clock_hz() * 1e3:.1f} ms on-chip")
            if i == 0:
                parts.append(f"(wall {wall:.1f}s, uploaded {runtime.uploaded_bytes / 1024:.0f} KiB)")
            print("  ".join(parts))
            if args.output:
                Path(args.output).write_bytes(out.astype(np.int8).tobytes())
        if args.profile:
            print_profile(artifact.manifest, transport.profile())
    finally:
        transport.close()

    n = len(cases)
    if n > 1:
        summary = [f"{n} images"]
        if cases[0][1] is not None:
            summary.append(f"accuracy {correct}/{n}")
        if reference is not None:
            summary.append(f"agrees with ONNX on {agree}/{n}")
        print("  ".join(summary))
    return 1 if failures else 0


def profile_rows(manifest: dict, cycles: list[int]) -> list[tuple[str, int]]:
    """Name each firmware profile entry: constants load, then schedule steps."""
    names = ["(load constants)"] + [f["name"] for f in manifest.get("riscv_functions", [])]
    return [(names[i] if i < len(names) else f"step{i}", c) for i, c in enumerate(cycles)]


def print_profile(manifest: dict, cycles: list[int], top: int = 25) -> None:
    rows = profile_rows(manifest, cycles)
    total = sum(c for _, c in rows) or 1
    print(f"\nprofile: {total:,} cycles ({total / clock_hz() * 1e3:.1f} ms) over {len(rows)} steps")
    for name, c in sorted(rows, key=lambda r: -r[1])[:top]:
        print(f"  {c:>12,}  {100 * c / total:5.1f}%  {name}")
    by_kind: dict[str, int] = {}
    for name, c in rows:
        kind = "pack" if "pack" in name else "load constants" if name.startswith("(") else "gemm"
        by_kind[kind] = by_kind.get(kind, 0) + c
    print("  by kind: " + ", ".join(f"{k} {100 * v / total:.1f}%" for k, v in by_kind.items()))


def _reference_for(artifact, onnx_override: str | None):
    from compiler.backend.reference import OnnxReference

    if onnx_override:
        return OnnxReference(onnx_override)
    source = artifact.manifest.get("source")
    if not source:
        print("note: .lmb has no source ONNX recorded (recompile, or pass --onnx); "
              "skipping reference check", file=sys.stderr)
        return None
    path = Path(source["onnx"])
    if not path.exists():
        print(f"note: source ONNX {path} not found; skipping reference check", file=sys.stderr)
        return None
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != source.get("onnx_sha256"):
        print(f"warning: {path} changed since this .lmb was compiled — recompile it",
              file=sys.stderr)
    return OnnxReference(path)


if __name__ == "__main__":
    raise SystemExit(main())
