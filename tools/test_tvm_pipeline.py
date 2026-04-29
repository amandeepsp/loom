#!/usr/bin/env -S uv run python
"""End-to-end TVM pipeline test on Verilator sim.

ONNX → Relax → VM execution on accel.
Compare against onnxruntime CPU reference.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from compiler import (
    lower_pipeline,
    AccelRuntime,
    RuntimeConfig,
    TcpTransport,
    register_runtime_functions,
)

log = logging.getLogger("tvm_pipeline")


def run_cpu_reference(onnx_path: Path, input_arr: np.ndarray) -> np.ndarray:
    """Run ONNX model via onnxruntime for CPU reference."""
    sess = ort.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name
    return sess.run(None, {input_name: input_arr})[0]


def build_accel_vm(lowered_mod: tvm.IRModule, runtime: AccelRuntime) -> relax.VirtualMachine:
    """Build a TVM VirtualMachine for the accel-targeted Relax module."""
    registered = register_runtime_functions(lowered_mod, runtime=runtime)
    log.info("Registered %d accel symbols", len(registered))
    ex = relax.build(lowered_mod, target="llvm")
    return relax.VirtualMachine(ex, tvm.cpu())


def main() -> int:
    parser = argparse.ArgumentParser(description="TVM pipeline e2e test on Verilator sim")
    parser.add_argument("--tcp", default="tcp://127.0.0.1:21450")
    parser.add_argument("--onnx", default=str(REPO_ROOT / "models/out/mnist_int8.onnx"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--driver-timeout", type=float, default=1800.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        log.error("ONNX model not found: %s", onnx_path)
        log.error("Run: uv run python -m models.mnist")
        return 1

    # --- Random input ---
    rng = np.random.default_rng(args.seed)
    input_img = rng.integers(-128, 128, size=(1, 784), dtype=np.int8)
    vm_input = np.zeros((1, 1, 28, 28), dtype=np.float32)
    vm_input[0, 0, :, :] = input_img.astype(np.float32).reshape(28, 28)

    # --- CPU reference ---
    cpu_out = run_cpu_reference(onnx_path, vm_input)
    cpu_pred = int(cpu_out.argmax())
    log.info("CPU ref:  pred=%d logits=%s", cpu_pred, cpu_out.flatten().tolist())

    # --- TVM compilation ---
    model_proto = onnx.load(str(onnx_path))
    shape_dict = {"input": [1, 1, 28, 28]}
    mod = from_onnx(model_proto, shape_dict=shape_dict, keep_params_in_input=False)
    lowered = lower_pipeline(mod)

    # --- Accel execution ---
    transport = TcpTransport(args.tcp, timeout_s=args.driver_timeout)
    runtime = AccelRuntime(transport, RuntimeConfig())
    vm = build_accel_vm(lowered, runtime)

    log.info("Running VM inference...")
    t0 = time.monotonic()
    vm_out = vm["main"](vm_input)
    sim_out = vm_out.numpy()
    t1 = time.monotonic()
    transport.close()

    sim_pred = int(sim_out.argmax())
    log.info("Accel:    pred=%d (%.1fs) logits=%s", sim_pred, t1 - t0, sim_out.flatten().tolist())

    # --- Compare ---
    delta = np.abs(cpu_out.flatten() - sim_out.flatten())
    log.info("max|Δ|=%.6f MAE=%.6f", float(delta.max()), float(delta.mean()))

    if cpu_pred == sim_pred:
        log.info("PASS")
        return 0
    log.error("FAIL: prediction mismatch")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
