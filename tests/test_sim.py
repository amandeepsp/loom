"""Verilator simulation tests: layer-by-layer GEMM and full model-binary pipeline."""

import logging
import sys
from pathlib import Path

import numpy as np
import onnx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from compiler import TcpTransport, compile_onnx_model
from shared.sim_harness import run_gemm_on_sim
from tests.mnist_model_binary import (
    DEFAULT_ONNX,
    compiled_layer_params,
    mnist_digit,
    mnist_reference,
    onnx_hidden_and_output,
    qgemm_output_zero_points,
    run_mnist_model_binary,
)

log = logging.getLogger(__name__)
SIM_CSR = REPO_ROOT / "build/sim/csr.json"


# ═══════════════════════════════════════════════════════════════════════════
# Layer-by-layer GEMM test
# ═══════════════════════════════════════════════════════════════════════════

def test_sim_gemm(sim_port):
    """Each MNIST layer as a standalone GEMM, using the compiler's epilogue params."""
    model = onnx.load(str(DEFAULT_ONNX))
    weights = {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}
    w0 = weights["net.1.weight_quantized"]   # [256, 784]
    w1 = weights["net.3.weight_quantized"]   # [10, 256]

    (b0, m0, s0), (b1, m1, s1) = compiled_layer_params(
        compile_onnx_model(DEFAULT_ONNX, csr_json=SIM_CSR)
    )
    zp0, zp1 = qgemm_output_zero_points(DEFAULT_ONNX)

    reference = mnist_reference()
    input_img, _label = mnist_digit(reference, 0)
    ref_h, ref_out = onnx_hidden_and_output(reference, input_img)

    transport = TcpTransport(sim_port, timeout_s=1800)
    try:
        sim_h = run_gemm_on_sim(
            transport, input_img, w0.T.astype(np.int8),
            bias=b0, multiplier=m0, shift=s0,
            output_offset=zp0, activation_min=-128, activation_max=127,
        )
        sim_out = run_gemm_on_sim(
            transport, sim_h, w1.T.astype(np.int8),
            bias=b1, multiplier=m1, shift=s1,
            output_offset=zp1, activation_min=-128, activation_max=127,
        )
    finally:
        transport.close()

    delta_h = np.abs(ref_h.astype(np.int32) - sim_h.astype(np.int32))
    delta_out = np.abs(ref_out.astype(np.int32) - sim_out.astype(np.int32))

    assert delta_h.max() <= 1, f"layer 0 max|Δ|={delta_h.max()}"
    assert delta_out.max() <= 1, f"layer 1 max|Δ|={delta_out.max()}"
    assert int(sim_out.argmax()) == int(ref_out.argmax())


def test_sim_model_binary_exec_model(sim_port):
    """Full compiled model binary through firmware exec_model on Verilator sim."""
    transport = TcpTransport(sim_port, timeout_s=1800)
    try:
        result = run_mnist_model_binary(transport, onnx_path=DEFAULT_ONNX, csr_json=SIM_CSR)
    finally:
        transport.close()

    assert result.hidden_max_delta <= 1, (
        f"model binary hidden max|delta|={result.hidden_max_delta}\n"
        f"hidden actual[:16]={result.loom_hidden[:, :16].tolist()}\n"
        f"hidden expected[:16]={result.cpu_hidden[:, :16].tolist()}"
    )
    assert result.output_max_delta <= 1, (
        f"model binary max|delta|={result.output_max_delta}\n"
        f"actual={result.loom_output.tolist()}\n"
        f"expected={result.cpu_output.tolist()}"
    )
    assert result.loom_pred == result.cpu_pred == result.label
