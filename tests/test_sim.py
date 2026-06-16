"""Verilator simulation tests: layer-by-layer GEMM and full model-binary pipeline."""

import logging
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from compiler import TcpTransport
from compiler.quant_utils import quantize_multiplier_less_than_one
from shared.reference import cpu_requantize
from shared.sim_harness import run_gemm_on_sim
from tests.mnist_model_binary import run_mnist_model_binary

log = logging.getLogger(__name__)

# ── MNIST model QDQ constants (from debug pipeline) ──────────────────────
_L0_IS  = 0.003921568859368563
_L0_IZP = -128
_L0_WS  = 0.0026108562014997005
_L0_OS  = 0.025416772812604904
_L0_OZP = -128

_L1_IS  = _L0_OS
_L1_IZP = -128
_L1_WS  = 0.0017632287926971912
_L1_OS  = 0.040668897330760956
_L1_OZP = 21


def _epilogue_params(
    input_scale: float, input_zp: int, weight_scale: float,
    output_scale: float,
    weight: np.ndarray, bias_raw: np.ndarray, n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    combined = (input_scale * weight_scale) / output_scale
    mult, shft = quantize_multiplier_less_than_one(combined)
    mult_i, shft_i = int(mult), int(shft)

    sum_w = weight.sum(axis=1).astype(np.int32)
    bias = bias_raw.astype(np.int32) - input_zp * sum_w

    return bias, np.full(n, mult_i, dtype=np.int32), np.full(n, shft_i, dtype=np.int32)


# ═══════════════════════════════════════════════════════════════════════════
# Layer-by-layer GEMM test
# ═══════════════════════════════════════════════════════════════════════════

def test_sim_gemm(sim_port):
    """Per-layer GEMM with proper requantization params, both layers."""
    import onnx

    onnx_path = REPO_ROOT / "models/out/mnist_int8.onnx"
    model = onnx.load(str(onnx_path))
    weights = {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}
    w0 = weights["net.1.weight_quantized"]   # [256, 784]
    b0 = weights["net.1.bias_quantized"]     # [256]
    w1 = weights["net.3.weight_quantized"]   # [10, 256]
    b1 = weights["net.3.bias_quantized"]     # [10]

    n0, n1 = 256, 10

    l0_bias, l0_mult, l0_shift = _epilogue_params(
        _L0_IS, _L0_IZP, _L0_WS, _L0_OS, w0, b0, n0)
    l1_bias, l1_mult, l1_shift = _epilogue_params(
        _L1_IS, _L1_IZP, _L1_WS, _L1_OS, w1, b1, n1)

    rng = np.random.default_rng(42)
    input_img = rng.integers(-128, 128, size=(1, 784), dtype=np.int8)

    # CPU reference (bit-accurate)
    cpu_acc0 = input_img.astype(np.int32) @ w0.T.astype(np.int32)
    cpu_h = cpu_requantize(cpu_acc0, l0_bias, l0_mult, l0_shift, _L0_OZP, -128, 127)
    cpu_acc1 = cpu_h.astype(np.int32) @ w1.T.astype(np.int32)
    cpu_out = cpu_requantize(cpu_acc1, l1_bias, l1_mult, l1_shift, _L1_OZP, -128, 127)

    # Sim inference
    transport = TcpTransport(sim_port, timeout_s=1800)
    try:
        sim_h = run_gemm_on_sim(
            transport, input_img, w0.T.astype(np.int8),
            bias=l0_bias, multiplier=l0_mult, shift=l0_shift,
            output_offset=_L0_OZP, activation_min=-128, activation_max=127,
        )
        sim_out = run_gemm_on_sim(
            transport, sim_h, w1.T.astype(np.int8),
            bias=l1_bias, multiplier=l1_mult, shift=l1_shift,
            output_offset=_L1_OZP, activation_min=-128, activation_max=127,
        )
    finally:
        transport.close()

    delta_h = np.abs(cpu_h.astype(np.int32) - sim_h.astype(np.int32))
    delta_out = np.abs(cpu_out.astype(np.int32) - sim_out.astype(np.int32))

    assert delta_h.max() <= 2, f"layer 0 max|Δ|={delta_h.max()}"
    assert delta_out.max() <= 1, f"layer 1 max|Δ|={delta_out.max()}"
    assert int(sim_out.argmax()) == int(cpu_out.argmax())


def test_sim_model_binary_exec_model(sim_port):
    """Full compiled model binary through firmware exec_model on Verilator sim."""
    transport = TcpTransport(sim_port, timeout_s=1800)
    try:
        result = run_mnist_model_binary(
            transport,
            onnx_path=REPO_ROOT / "models/out/mnist_int8.onnx",
            csr_json=REPO_ROOT / "build/sim/csr.json",
        )
    finally:
        transport.close()

    assert result.hidden_max_delta <= 2, (
        f"model binary hidden max|delta|={result.hidden_max_delta}\n"
        f"hidden actual[:16]={result.loom_hidden[:, :16].tolist()}\n"
        f"hidden expected[:16]={result.cpu_hidden[:, :16].tolist()}"
    )
    assert result.output_max_delta <= 2, (
        f"model binary max|delta|={result.output_max_delta}\n"
        f"actual={result.loom_output.tolist()}\n"
        f"expected={result.cpu_output.tolist()}"
    )
    assert result.loom_pred == result.cpu_pred


# ═══════════════════════════════════════════════════════════════════════════
# Full ONNX to model-binary pipeline test
# ═══════════════════════════════════════════════════════════════════════════

def test_sim_pipeline(sim_port):
    """Alias for the model-binary pipeline kept for the historical test name."""
    test_sim_model_binary_exec_model(sim_port)
