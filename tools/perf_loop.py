"""Hill-climbing harness: measure one experiment on the board and log it.

    python -m tools.perf_loop --label baseline            # records the golden
    python -m tools.perf_loop --label "hoist epilogue params"

Each run compiles the model from ONNX with the *current* compiler, syncs it
to the board (chunked CRC diff, so only changed bytes cross the UART), runs
a fixed set of test images, and appends a JSON line to the experiment log.

Correctness gate (all must hold, else the experiment is rejected):

1. Per-layer exactness (MobileNet): after the last image, every GEMM layer's
   output is recomputed on the host from the board's own packed input with
   the plan's exact epilogue; the total number of wrong elements must stay
   within --wrong-budget. (The baseline is not zero: a known hardware datapath
   bug corrupts a handful of elements, e.g. ~8/8192 in mobilenet_conv_6.)
2. End to end: argmax agrees with ONNX Runtime on every image and the max
   logit delta vs ONNX stays within --delta-budget.

A golden snapshot of outputs is still recorded and compared for information,
but the hardware bug makes outputs slightly non-deterministic, so it is not
part of the gate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG = REPO_ROOT / "analytics/perf_log.jsonl"
GOLDEN_DIR = REPO_ROOT / "build/perf"
MODELS = {
    "mobilenet": REPO_ROOT / "models/out/mobilenet_v1_025_int8.onnx",
    "mnist": REPO_ROOT / "models/out/mnist_int8.onnx",
}


def git_rev() -> str:
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                           cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    return rev + ("+dirty" if dirty else "")


def main() -> int:
    from compiler import compile_onnx_model
    from compiler.backend.reference import OnnxReference, dataset_for_shape, load_sample
    from compiler.backend.runtime import ModelBinaryArtifact, ModelBinaryRuntime
    from shared.protocol import DEFAULT_BAUD, SerialTransport
    from tools.loom_cli import DEFAULT_PORT, profile_rows

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", required=True, help="what this experiment changes")
    p.add_argument("--model", choices=sorted(MODELS), default="mobilenet")
    p.add_argument("--images", type=int, default=3)
    p.add_argument("--port", default=DEFAULT_PORT)
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    p.add_argument("--csr-json", default=str(REPO_ROOT / "build/sipeed_tang_nano_20k/csr.json"))
    p.add_argument("--reset-golden", action="store_true")
    p.add_argument("--wrong-budget", type=int, default=32,
                   help="max wrong GEMM output elements summed over layers")
    p.add_argument("--delta-budget", type=int, default=24,
                   help="max |logit - ONNX logit| over images")
    p.add_argument("--no-log", action="store_true")
    args = p.parse_args()

    onnx_path = MODELS[args.model]
    t_compile = time.monotonic()
    compiled = compile_onnx_model(onnx_path, csr_json=args.csr_json)
    artifact = ModelBinaryArtifact.from_bytes(compiled.to_bytes(include_debug_sections=False))
    t_compile = time.monotonic() - t_compile

    reference = OnnxReference(onnx_path)
    dataset = dataset_for_shape(reference.boundary.input_shape)
    shape = tuple(artifact.manifest["runtime_contract"]["input"]["shape"])
    samples = [load_sample(dataset, i, reference) for i in range(args.images)]

    transport = SerialTransport(args.port, baud_rate=args.baud, timeout_s=1800)
    runtime = ModelBinaryRuntime(artifact, transport)
    outputs, cycles = [], []
    t_sync = time.monotonic()
    try:
        for s in samples:
            r = runtime.run_quantized(s.int8_input.reshape(shape))
            outputs.append(r.output.astype(np.int8).copy())
            cycles.append(r.cycles)
        profile = transport.profile()
        layer_wrong = (
            layer_errors(artifact, onnx_path, transport) if args.model == "mobilenet" else {}
        )
    finally:
        transport.close()
    t_sync = time.monotonic() - t_sync
    outputs = np.stack(outputs)

    golden_path = GOLDEN_DIR / f"golden_{args.model}_{args.images}.npy"
    if args.reset_golden or not golden_path.exists():
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(golden_path, outputs)
        exact, golden_note = True, "recorded new golden"
    else:
        golden = np.load(golden_path)
        exact = golden.shape == outputs.shape and bool((golden == outputs).all())
        golden_note = "bit-identical to golden" if exact else (
            f"MISMATCH vs golden: max|Δ|={int(np.abs(golden.astype(int) - outputs).max())}")

    onnx_outs = [reference.run_int8(s.int8_input).reshape(o.shape) for o, s in zip(outputs, samples)]
    onnx_agree = sum(int(o.argmax()) == int(r.argmax()) for o, r in zip(outputs, onnx_outs))
    delta = max(int(np.abs(o.astype(int) - r.astype(int)).max()) for o, r in zip(outputs, onnx_outs))
    wrong_total = sum(layer_wrong.values())
    reasons = []
    if onnx_agree != len(samples):
        reasons.append(f"ONNX argmax disagrees on {len(samples) - onnx_agree} image(s)")
    if delta > args.delta_budget:
        reasons.append(f"logit delta {delta} > {args.delta_budget}")
    if wrong_total > args.wrong_budget:
        reasons.append(f"{wrong_total} wrong GEMM elements > {args.wrong_budget}")
    passed = not reasons
    correct = sum(int(o.argmax()) == s.label for o, s in zip(outputs, samples))

    rows = profile_rows(artifact.manifest, profile)
    by_kind: dict[str, int] = {}
    for name, c in rows:
        kind = "load_constants" if name.startswith("(") else "pack" if "pack" in name else "gemm"
        by_kind[kind] = by_kind.get(kind, 0) + c
    # Steady state = last image (one-time work like constant loading is
    # paid by the first); first-image cycles are logged alongside.
    mean_cycles = int(cycles[-1])

    previous = []
    if LOG.exists():
        previous = [json.loads(line) for line in LOG.read_text().splitlines() if line.strip()]
    base = next((e for e in previous if e["model"] == args.model and e.get("baseline")), None)
    best = min((e["cycles"] for e in previous
                if e["model"] == args.model and e.get("passed", e.get("exact"))), default=None)

    entry = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "label": args.label,
        "rev": git_rev(),
        "model": args.model,
        "images": args.images,
        "cycles": mean_cycles,
        "ms": round(mean_cycles / 48e3, 2),
        "first_cycles": int(cycles[0]),
        "passed": passed,
        "reject_reasons": reasons,
        "golden_identical": exact,
        "layer_wrong": {k: v for k, v in layer_wrong.items() if v},
        "onnx_max_delta": delta,
        "onnx_agree": f"{onnx_agree}/{len(samples)}",
        "accuracy": f"{correct}/{len(samples)}",
        "lmb_bytes": len(artifact.blob),
        "uploaded_bytes": runtime.uploaded_bytes,
        "by_kind": by_kind,
        "top_steps": sorted(rows, key=lambda r: -r[1])[:8],
        "baseline": base is None,
    }
    if base:
        entry["speedup_vs_baseline"] = round(base["cycles"] / mean_cycles, 3)

    total = sum(by_kind.values()) or 1
    print(f"[{args.model}] {args.label}")
    print(f"  cycles {mean_cycles:,} ({entry['ms']} ms)"
          + (f"  speedup {entry['speedup_vs_baseline']}x vs baseline" if base else "  (baseline)")
          + (f"  best so far {best:,}" if best else ""))
    if cycles[0] != cycles[-1]:
        print(f"  first image {cycles[0]:,} cycles (includes one-time work)")
    print(("  PASS" if passed else "  REJECT: " + "; ".join(reasons))
          + f"  (wrong GEMM elems {wrong_total}"
          + (f": {entry['layer_wrong']}" if entry["layer_wrong"] else "") + f", ONNX max|Δ| {delta})")
    print(f"  {golden_note}; ONNX agree {entry['onnx_agree']}; accuracy {entry['accuracy']}")
    print("  " + ", ".join(f"{k} {100 * v / total:.1f}%" for k, v in by_kind.items()))
    for name, c in entry["top_steps"][:5]:
        print(f"    {c:>12,}  {name}")
    print(f"  compile {t_compile:.1f}s, board {t_sync:.1f}s, uploaded {runtime.uploaded_bytes // 1024} KiB")

    if not args.no_log:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    return 0 if passed else 1


def layer_errors(artifact, onnx_path: Path, transport) -> dict[str, int]:
    """Wrong output elements per GEMM layer vs an exact host recomputation.

    Uses the board's packed input for each layer, so it checks the GEMM kernels
    (DMA, array, epilogue, stores) in isolation from the CPU pack code.
    """
    from compiler.frontend.qlinear_mobilenet import parse_qlinear_mobilenet
    from shared.reference import cpu_requantize

    plan = parse_qlinear_mobilenet(onnx_path)
    buffers = {b["name"]: b for b in artifact.manifest["buffers"]}
    errors: dict[str, int] = {}
    for layer, k in zip(plan.layers, artifact.manifest["kernels"]):
        tile, m, kk, n, pm = k["tile"], k["m"], k["k"], k["n"], k["padded_m"]
        pin = buffers[k["packed_input_buffer"]]
        out = buffers[k["output_buffer"]]
        packed = np.frombuffer(transport.read_mem(pin["addr"], pin["size"]), np.int8)
        actual = np.frombuffer(transport.read_mem(out["addr"], out["size"]), np.int8).reshape(pm, n)
        a = packed.reshape(pm // tile, kk, tile).transpose(0, 2, 1).reshape(pm, kk)
        acc = a.astype(np.int64) @ layer.weight.astype(np.int64)
        expected = cpu_requantize(acc, layer.bias, layer.multiplier, layer.shift,
                                  layer.output_zero_point, layer.activation_min,
                                  layer.activation_max)
        errors[k["name"]] = int((actual[:m] != expected[:m]).sum())
    return errors


if __name__ == "__main__":
    sys.exit(main())
