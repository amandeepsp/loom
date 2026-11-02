"""Collect source-backed theoretical-versus-achieved showcase analytics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from compiler.backend.model import (
    SECTION_CONSTANTS,
    SECTION_MANIFEST,
    SECTION_RISCV,
    SECTION_SCHEDULE,
    model_binary_sections,
)
from compiler.backend.runtime import ModelBinaryArtifact
from compiler.frontend.qlinear_mobilenet import parse_qlinear_mobilenet
from shared.protocol import SerialTransport


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = REPO_ROOT / "build/sipeed_tang_nano_20k/gateware/sipeed_tang_nano_20k.report"
DEFAULT_MODEL = REPO_ROOT / "models/out/mobilenet_v1_025_int8.onnx"
DEFAULT_LMB = REPO_ROOT / "build/mobilenet_v1_025.lmb"
DEFAULT_EVAL = REPO_ROOT / "analytics/measurements/mobilenet_cifar200.json"
DEFAULT_SWEEP = REPO_ROOT / "analytics/measurements/gemm_sweep.json"
TILE = 8
DEFAULT_CLOCK_HZ = 48_000_000
SDRAM_INTERFACE_BYTES_PER_CYCLE = 4

BENCHMARK_CASES = (
    (8, 8, 8),
    (8, 64, 8),
    (8, 256, 8),
    (8, 512, 8),
    (16, 64, 16),
    (32, 128, 32),
    (64, 256, 64),
    (1024, 27, 8),
    (1024, 72, 8),
    (256, 288, 32),
    (16, 1152, 128),
    (4, 2304, 256),
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _ceil_tile(value: int, tile: int = TILE) -> int:
    return math.ceil(value / tile) * tile


def _source() -> dict[str, Any]:
    return {
        "id": "showcase_sql",
        "label": "loom showcase metrics",
        "path": "analytics/queries/showcase.sql",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "description": "Reads the reviewed JSON datasets emitted by tools.showcase_metrics.",
            "filters": [
                "MobileNet-v1 0.25 INT8",
                "8x8 int8 CFU",
                "48 MHz board clock",
                "CIFAR-10 stratified sample where accuracy is shown",
            ],
            "metric_definitions": [
                "Peak GOPS = CFU rows * CFU columns * clock_hz * 2 / 1e9.",
                "Peak utilization = issued operations / (peak operations per cycle * measured cycles).",
                "Array-active GOPS use sequencer-busy cycles; accelerator-window GOPS use the DMA/sequencer union.",
                "Streamed-KIR mcycle measurements include UART instruction delivery.",
                "Arithmetic intensity uses two operations per MAC.",
            ],
            "tables_used": [
                "build/sipeed_tang_nano_20k/gateware/sipeed_tang_nano_20k.report",
                "models/out/mobilenet_v1_025_int8.onnx",
                "build/mobilenet_v1_025.lmb",
                "analytics/measurements/mobilenet_cifar200.json",
                "analytics/measurements/gemm_sweep.json",
            ],
        },
    }


def _layer_rows(plan: Any, artifact: ModelBinaryArtifact) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, layer in enumerate(plan.layers):
        padded_m = _ceil_tile(layer.m)
        padded_n = _ceil_tile(layer.n)
        dense_macs = layer.m * layer.k * layer.n
        issued_macs = padded_m * layer.k * padded_n
        semantic_macs = (
            layer.m * int(np.count_nonzero(layer.weight))
            if layer.kind == "depthwise_conv2d"
            else dense_macs
        )
        if layer.packing is None:
            original_input_bytes = layer.m * layer.k
        else:
            original_input_bytes = (
                layer.packing.input_h * layer.packing.input_w * layer.packing.channels
            )
        minimum_weight_bytes = (
            int(np.count_nonzero(layer.weight))
            if layer.kind == "depthwise_conv2d"
            else layer.k * layer.n
        )
        minimum_bytes = original_input_bytes + minimum_weight_bytes + layer.m * layer.n + 12 * layer.n
        packed_input = artifact.buffer(f"gemm{index}.input_packed").size
        planned_bytes = (
            packed_input
            + artifact.buffer(f"gemm{index}.weights").size
            + artifact.buffer(f"gemm{index}.output").size
            + artifact.buffer(f"gemm{index}.bias").size
            + artifact.buffer(f"gemm{index}.multiplier").size
            + artifact.buffer(f"gemm{index}.shift").size
        )
        rows.append(
            {
                "layer": index,
                "name": layer.name,
                "kind": layer.kind,
                "shape": f"{layer.m}x{layer.k}x{layer.n}",
                "m": layer.m,
                "k": layer.k,
                "n": layer.n,
                "dense_macs": dense_macs,
                "issued_macs": issued_macs,
                "semantic_macs": semantic_macs,
                "padding_efficiency_pct": 100.0 * dense_macs / issued_macs,
                "semantic_efficiency_pct": 100.0 * semantic_macs / issued_macs,
                "minimum_bytes": minimum_bytes,
                "planned_working_set_bytes": planned_bytes,
                "packing_amplification": packed_input / original_input_bytes,
                "ideal_ai_ops_per_byte": 2.0 * semantic_macs / minimum_bytes,
                "planned_ai_ops_per_byte": 2.0 * semantic_macs / planned_bytes,
            }
        )
    return rows


def _model_sections(lmb_path: Path) -> list[dict[str, Any]]:
    names = {
        SECTION_MANIFEST: "Manifest",
        SECTION_CONSTANTS: "Constants",
        SECTION_RISCV: "RISC-V code",
        SECTION_SCHEDULE: "Schedule",
    }
    sections = model_binary_sections(lmb_path.read_bytes())
    return [
        {"section": names.get(kind, f"Section {kind}"), "bytes": len(payload)}
        for kind, payload in sorted(sections.items())
    ]


def _sweep_rows(
    sweep: dict[str, Any] | None,
    *,
    clock_hz: int,
    peak_gops: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not sweep:
        return [], []
    by_shape: dict[tuple[int, int, int], dict[str, Any]] = {}
    for item in sweep.get("results", []):
        key = (int(item["m"]), int(item["k"]), int(item["n"]))
        by_shape.setdefault(key, {})[str(item["variant"])] = item
    roofline: list[dict[str, Any]] = []
    overlap: list[dict[str, Any]] = []
    bandwidth = SDRAM_INTERFACE_BYTES_PER_CYCLE * clock_hz
    for (m, k, n), variants in sorted(by_shape.items()):
        if "lmb" in variants:
            # Schema 2: one compiled-.lmb measurement; no KIR variant pair.
            variants = {"pipelined": variants["lmb"], "non-pipelined": variants["lmb"]}
        if "pipelined" not in variants or "non-pipelined" not in variants:
            continue
        operations = 2 * m * k * n
        minimum_bytes = m * k + k * n + m * n + 12 * n
        ai = operations / minimum_bytes
        theoretical_gops = min(peak_gops, bandwidth * ai / 1e9)
        pipe_cycles = int(variants["pipelined"]["cycles"])
        nonpipe_cycles = int(variants["non-pipelined"]["cycles"])
        streamed_kir_gops = operations * clock_hz / pipe_cycles / 1e9
        nonpipe_streamed_kir_gops = operations * clock_hz / nonpipe_cycles / 1e9
        pipe_counters = variants["pipelined"].get("counters", {})
        seq_busy = int(pipe_counters.get("seq_busy_cycles", 0))
        dma_active = int(pipe_counters.get("dma_active_cycles", 0))
        overlap_cycles = int(pipe_counters.get("overlap_cycles", 0))
        accelerator_cycles = seq_busy + dma_active - overlap_cycles
        array_active_gops = operations * clock_hz / seq_busy / 1e9 if seq_busy else None
        accelerator_window_gops = (
            operations * clock_hz / accelerator_cycles / 1e9 if accelerator_cycles else None
        )
        measured_ai = None
        label = f"{m}x{k}x{n}"
        roofline.append(
            {
                "shape": label,
                "m": m,
                "k": k,
                "n": n,
                "arithmetic_intensity": ai,
                "compute_peak_gops": peak_gops,
                "theoretical_gops": theoretical_gops,
                "array_active_gops": array_active_gops,
                "accelerator_window_gops": accelerator_window_gops,
                "streamed_kir_gops": streamed_kir_gops,
                "nonpipe_streamed_kir_gops": nonpipe_streamed_kir_gops,
                "array_active_pct_of_compute_peak": (
                    100.0 * array_active_gops / peak_gops
                    if array_active_gops is not None
                    else None
                ),
                "measured_ai_ops_per_byte": measured_ai,
            }
        )
        overlap.append(
            {
                "shape": label,
                "m": m,
                "k": k,
                "n": n,
                "pipelined_cycles": pipe_cycles,
                "non_pipelined_cycles": nonpipe_cycles,
                "overlap_speedup": nonpipe_cycles / pipe_cycles,
                "cycles_saved": nonpipe_cycles - pipe_cycles,
                "seq_busy_cycles": seq_busy,
                "dma_active_cycles": dma_active,
                "measured_overlap_cycles": overlap_cycles,
                "exposed_compute_cycles": max(0, seq_busy - overlap_cycles),
                "exposed_dma_cycles": max(0, dma_active - overlap_cycles),
                "overlap_efficiency_pct": (
                    100.0 * overlap_cycles / min(seq_busy, dma_active)
                    if min(seq_busy, dma_active) > 0
                    else None
                ),
                "measured_traffic_bytes": None,
            }
        )
    roofline.sort(key=lambda row: row["arithmetic_intensity"])
    overlap.sort(key=lambda row: row["overlap_speedup"], reverse=True)
    return roofline, overlap


def collect(
    *,
    report_path: Path,
    model_path: Path,
    lmb_path: Path,
    evaluation_path: Path,
    sweep_path: Path,
    output_dir: Path,
) -> Path:
    for path in (report_path, model_path, lmb_path, evaluation_path):
        if not path.is_file():
            raise FileNotFoundError(f"required showcase source is missing: {path}")
    report = _read_json(report_path)
    evaluation = _read_json(evaluation_path)
    sweep = _read_json(sweep_path) if sweep_path.is_file() else None
    artifact = ModelBinaryArtifact.from_file(lmb_path)
    plan = parse_qlinear_mobilenet(model_path)

    clock_hz = int(evaluation.get("clock_hz", DEFAULT_CLOCK_HZ))
    peak_macs_per_cycle = TILE * TILE
    peak_gops = 2.0 * peak_macs_per_cycle * clock_hz / 1e9
    mean_cycles = int(evaluation["cycles"]["mean"])
    seconds = mean_cycles / clock_hz
    layer_rows = _layer_rows(plan, artifact)
    dense_macs = sum(int(row["dense_macs"]) for row in layer_rows)
    issued_macs = sum(int(row["issued_macs"]) for row in layer_rows)
    semantic_macs = sum(int(row["semantic_macs"]) for row in layer_rows)
    lower_bound_cycles = math.ceil(issued_macs / peak_macs_per_cycle)
    sample_count = int(evaluation["sample"]["count"])
    profile = evaluation.get("hardware_profile")
    model_cycle_breakdown: list[dict[str, Any]] = []
    profile_metrics: dict[str, float | int | None] = {
        "array_active_gops": None,
        "accelerator_window_gops": None,
        "accelerator_duty_pct": None,
        "overlap_efficiency_pct": None,
    }
    if profile:
        profile_total = int(profile["total_cycles"])
        profile_seq = int(profile["seq_busy_cycles"])
        profile_dma = int(profile["dma_active_cycles"])
        profile_overlap = int(profile["overlap_cycles"])
        profile_union = profile_seq + profile_dma - profile_overlap
        profile_metrics = {
            "array_active_gops": 2.0 * issued_macs * clock_hz / profile_seq / 1e9,
            "accelerator_window_gops": 2.0 * issued_macs * clock_hz / profile_union / 1e9,
            "accelerator_duty_pct": 100.0 * profile_union / profile_total,
            "overlap_efficiency_pct": 100.0 * profile_overlap / min(profile_seq, profile_dma),
        }
        model_cycle_breakdown = [
            {"phase": "Outside CFU/DMA", "cycles": profile_total - profile_union},
            {"phase": "DMA only", "cycles": profile_dma - profile_overlap},
            {"phase": "Sequencer only", "cycles": profile_seq - profile_overlap},
            {"phase": "DMA + sequencer overlap", "cycles": profile_overlap},
        ]

    hardware_summary = [
        {
            "theoretical_peak_gops": peak_gops,
            "achieved_issued_gops": 2.0 * issued_macs / seconds / 1e9,
            "peak_utilization": issued_macs / (mean_cycles * peak_macs_per_cycle),
            "inference_ms": seconds * 1000.0,
            "exactness": evaluation["agreement"]["hardware_compiler_exact_logits"] / sample_count,
            "peak_lower_bound_cycles": lower_bound_cycles,
            "measured_cycles": mean_cycles,
            **profile_metrics,
        }
    ]
    throughput = [
        {"measure": "Theoretical peak", "gops": peak_gops, "basis": "architecture"},
        *(
            [
                {"measure": "Array active", "gops": profile_metrics["array_active_gops"], "basis": "measured counters"},
                {"measure": "CFU + DMA window", "gops": profile_metrics["accelerator_window_gops"], "basis": "measured counters"},
            ]
            if profile
            else []
        ),
        {"measure": "End-to-end issued", "gops": 2.0 * issued_macs / seconds / 1e9, "basis": "measured mcycle"},
        {"measure": "End-to-end semantic", "gops": 2.0 * semantic_macs / seconds / 1e9, "basis": "measured mcycle"},
    ]
    efficiency = [
        {"stage": "Ideal array capacity", "efficiency_pct": 100.0},
        *(
            [
                {"stage": "Array active", "efficiency_pct": 100.0 * profile_metrics["array_active_gops"] / peak_gops},
                {"stage": "CFU + DMA window", "efficiency_pct": 100.0 * profile_metrics["accelerator_window_gops"] / peak_gops},
            ]
            if profile
            else []
        ),
        {"stage": "End-to-end peak", "efficiency_pct": 100.0 * issued_macs / (mean_cycles * peak_macs_per_cycle)},
        {"stage": "End-to-end semantic", "efficiency_pct": 100.0 * semantic_macs / (mean_cycles * peak_macs_per_cycle)},
    ]
    resources = []
    for resource in ("LUT4", "DFF", "BSRAM", "MULT9X9"):
        item = report["utilization"][resource]
        resources.append(
            {
                "resource": resource,
                "used": int(item["used"]),
                "available": int(item["available"]),
                "utilization_pct": 100.0 * item["used"] / item["available"],
            }
        )

    accuracy = evaluation["accuracy"]
    accuracy_rows = [
        {"backend": "FP32", "accuracy_pct": 100.0 * accuracy["fp32_correct"] / sample_count},
        {"backend": "ORT INT8", "accuracy_pct": 100.0 * accuracy["ort_int8_correct"] / sample_count},
        {"backend": "Compiler INT8", "accuracy_pct": 100.0 * accuracy["compiler_int8_correct"] / sample_count},
        {"backend": "Hardware INT8", "accuracy_pct": 100.0 * accuracy["hardware_int8_correct"] / sample_count},
    ]
    manifest = artifact.manifest
    software_summary = [
        {
            "fp32_accuracy": accuracy["fp32_correct"] / sample_count,
            "ort_accuracy": accuracy["ort_int8_correct"] / sample_count,
            "compiler_hardware_accuracy": accuracy["hardware_int8_correct"] / sample_count,
            "ort_prediction_agreement": evaluation["agreement"]["hardware_ort_prediction"] / sample_count,
            "model_binary_mib": len(artifact.blob) / (1024 * 1024),
            "kernels": len(manifest["kernels"]),
            "riscv_functions": len(manifest["riscv_functions"]),
        }
    ]
    kinds: dict[str, int] = {}
    for layer in plan.layers:
        kinds[layer.kind] = kinds.get(layer.kind, 0) + 1
    operator_mix = [
        {"operator": kind.replace("_", " ").title(), "count": count}
        for kind, count in sorted(kinds.items(), key=lambda item: item[1], reverse=True)
    ]
    roofline, overlap = _sweep_rows(sweep, clock_hz=clock_hz, peak_gops=peak_gops)
    has_hw_counters = bool(
        sweep
        and sweep.get("results")
        and all("counters" in result for result in sweep["results"])
    )
    instrumentation = [
        {"metric": "Total inference cycles", "status": "Measured", "source": "mcycle"},
        {"metric": "Streamed-KIR GEMM runtime", "status": "Measured", "source": "mcycle including UART instruction delivery" if sweep else "run benchmark command"},
        {"metric": "DMA transaction-active cycles", "status": "Measured" if has_hw_counters else "Not captured", "source": "compiled-model profile and CFU counters" if profile else "GEMM sweep CFU counters"},
        {"metric": "DMA / sequencer overlap cycles", "status": "Measured" if has_hw_counters else "Not captured", "source": "compiled-model concurrent counters" if profile else "GEMM sweep concurrent counters"},
        {"metric": "DMA traffic bytes", "status": "Derived", "source": "emitted tile schedule and tensor shapes"},
        {"metric": "Sequencer phase cycles", "status": "Derived", "source": "emitted k_count/first/last schedule and fixed RTL state costs"},
        {"metric": "Per-layer achieved cycles", "status": "Not instrumented", "source": "requires model profiler buffer"},
        {"metric": "Power and energy", "status": "Not instrumented", "source": "requires external power measurement"},
    ]

    datasets = {
        "hardware_summary": hardware_summary,
        "throughput_comparison": throughput,
        "efficiency_ladder": efficiency,
        "resource_utilization": resources,
        "roofline": roofline,
        "gemm_overlap": overlap,
        "software_summary": software_summary,
        "accuracy_comparison": accuracy_rows,
        "operator_mix": operator_mix,
        "layer_metrics": layer_rows,
        "model_sections": _model_sections(lmb_path),
        "model_cycle_breakdown": model_cycle_breakdown,
        "instrumentation_status": instrumentation,
    }
    generated_at = _now()
    for dataset_id, rows in datasets.items():
        _write_json(output_dir / "datasets" / f"{dataset_id}.json", rows)
    _write_json(output_dir / "metrics.json", {"generated_at": generated_at, "datasets": datasets})
    artifact_path = output_dir / "artifact.json"
    _write_json(
        artifact_path,
        _dashboard_artifact(
            datasets=datasets,
            generated_at=generated_at,
            has_sweep=bool(roofline),
            has_hw_counters=has_hw_counters,
        ),
    )
    return artifact_path


def _dashboard_artifact(
    *,
    datasets: dict[str, list[dict[str, Any]]],
    generated_at: str,
    has_sweep: bool,
    has_hw_counters: bool,
) -> dict[str, Any]:
    source = _source()
    cards = [
        {"id": "peak", "dataset": "hardware_summary", "sourceId": "showcase_sql", "description": "8x8 int8 array at 48 MHz; multiply and add count separately.", "metrics": [{"label": "Theoretical peak", "field": "theoretical_peak_gops", "format": "number"}]},
        {"id": "array_active", "dataset": "hardware_summary", "sourceId": "showcase_sql", "description": "Issued operations divided by measured sequencer-busy cycles.", "metrics": [{"label": "Array-active GOPS", "field": "array_active_gops", "format": "number"}]},
        {"id": "accelerator_window", "dataset": "hardware_summary", "sourceId": "showcase_sql", "description": "Issued operations divided by the measured union of CFU and DMA cycles.", "metrics": [{"label": "CFU + DMA GOPS", "field": "accelerator_window_gops", "format": "number"}]},
        {"id": "achieved", "dataset": "hardware_summary", "sourceId": "showcase_sql", "description": "Issued padded operations divided by complete compiled-model runtime.", "metrics": [{"label": "End-to-end GOPS", "field": "achieved_issued_gops", "format": "number"}]},
        {"id": "overlap_efficiency", "dataset": "hardware_summary", "sourceId": "showcase_sql", "description": "Concurrent DMA/sequencer cycles divided by the shorter active interval.", "metrics": [{"label": "DMA/compute overlap, %", "field": "overlap_efficiency_pct", "format": "number"}]},
        {"id": "utilization", "dataset": "hardware_summary", "sourceId": "showcase_sql", "description": "Issued operations as a share of all array slots during end-to-end execution.", "metrics": [{"label": "Peak utilization", "field": "peak_utilization", "format": "percent"}]},
        {"id": "latency", "dataset": "hardware_summary", "sourceId": "showcase_sql", "description": "Mean over 200 stratified CIFAR-10 images.", "metrics": [{"label": "Inference latency, ms", "field": "inference_ms", "format": "number"}]},
        {"id": "exactness", "dataset": "hardware_summary", "sourceId": "showcase_sql", "description": "Hardware/compiler exact logits over the reviewed sample.", "metrics": [{"label": "Hardware exactness", "field": "exactness", "format": "percent"}]},
        {"id": "fp32_accuracy", "dataset": "software_summary", "sourceId": "showcase_sql", "description": "Ground-truth accuracy on 200 stratified CIFAR-10 images.", "metrics": [{"label": "FP32 accuracy", "field": "fp32_accuracy", "format": "percent"}]},
        {"id": "ort_accuracy", "dataset": "software_summary", "sourceId": "showcase_sql", "description": "ORT QOperator INT8 accuracy on the same images.", "metrics": [{"label": "ORT INT8 accuracy", "field": "ort_accuracy", "format": "percent"}]},
        {"id": "compiler_accuracy", "dataset": "software_summary", "sourceId": "showcase_sql", "description": "Compiler and hardware predictions are bit-identical.", "metrics": [{"label": "Compiler/HW accuracy", "field": "compiler_hardware_accuracy", "format": "percent"}]},
        {"id": "agreement", "dataset": "software_summary", "sourceId": "showcase_sql", "description": "Prediction agreement between compiler/hardware and ORT INT8.", "metrics": [{"label": "ORT prediction agreement", "field": "ort_prediction_agreement", "format": "percent"}]},
        {"id": "binary", "dataset": "software_summary", "sourceId": "showcase_sql", "description": "Deployable model binary including constants, code, and schedule.", "metrics": [{"label": "Model binary, MiB", "field": "model_binary_mib", "format": "number"}]},
    ]
    charts: list[dict[str, Any]] = [
        {"id": "throughput", "title": "Where throughput is lost", "subtitle": "Same MobileNet workload measured over the array, CFU+DMA window, and complete inference", "intent": "comparison", "question": "Is the array slow, or is it mostly idle?", "rationale": "Separates array-active throughput from accelerator-window and complete-model throughput.", "type": "horizontalBar", "dataset": "throughput_comparison", "sourceId": "showcase_sql", "encodings": {"x": {"field": "measure", "type": "nominal"}, "y": {"field": "gops", "type": "quantitative", "label": "GOPS"}}, "valueFormat": "number", "unit": "GOPS", "layout": "full"},
        {"id": "efficiency", "title": "Compute-efficiency ladder", "subtitle": "100% ideal capacity to useful and end-to-end work", "intent": "comparison", "question": "Where is theoretical array capacity lost?", "rationale": "Separates padding and structural-zero efficiency from total system duty cycle.", "type": "horizontalBar", "dataset": "efficiency_ladder", "sourceId": "showcase_sql", "encodings": {"x": {"field": "stage", "type": "ordinal"}, "y": {"field": "efficiency_pct", "type": "quantitative", "label": "Efficiency", "unit": "%"}}, "valueFormat": "number", "unit": "%", "layout": "half"},
        {"id": "resources", "title": "FPGA resource utilization", "subtitle": "Post-route Tang Nano 20K build", "intent": "comparison", "question": "Which device resources constrain the design?", "rationale": "Compares utilization on a common percentage scale.", "type": "bar", "dataset": "resource_utilization", "sourceId": "showcase_sql", "encodings": {"x": {"field": "resource", "type": "nominal"}, "y": {"field": "utilization_pct", "type": "quantitative", "label": "Utilization", "unit": "%"}}, "valueFormat": "number", "unit": "%", "layout": "half"},
        {"id": "model_cycles", "title": "Compiled-model cycle breakdown", "subtitle": "Disjoint phases from one 48.99M-cycle MobileNet inference", "intent": "composition", "question": "Where does complete inference time go?", "rationale": "Separates sequencer, DMA, their overlap, and time outside both hardware engines.", "type": "horizontalBar", "dataset": "model_cycle_breakdown", "sourceId": "showcase_sql", "encodings": {"x": {"field": "phase", "type": "nominal"}, "y": {"field": "cycles", "type": "quantitative", "label": "Cycles"}}, "valueFormat": "compact", "unit": "cycles", "layout": "full"},
        {"id": "accuracy", "title": "Accuracy by execution backend", "subtitle": "200 CIFAR-10 test images, stratified 20 per class", "intent": "comparison", "question": "Where does accuracy change from FP32 to deployed hardware?", "rationale": "Uses one denominator and sample across every backend.", "type": "bar", "dataset": "accuracy_comparison", "sourceId": "showcase_sql", "encodings": {"x": {"field": "backend", "type": "nominal"}, "y": {"field": "accuracy_pct", "type": "quantitative", "label": "Accuracy", "unit": "%"}}, "valueFormat": "number", "unit": "%", "layout": "half"},
        {"id": "operators", "title": "Compiled operator mix", "subtitle": "29 CFU-backed MobileNet kernels", "intent": "composition", "question": "What work does the compiler lower to the accelerator?", "rationale": "Shows the kernel composition without implying runtime share.", "type": "horizontalBar", "dataset": "operator_mix", "sourceId": "showcase_sql", "encodings": {"x": {"field": "operator", "type": "nominal"}, "y": {"field": "count", "type": "quantitative"}}, "valueFormat": "number", "layout": "half"},
        {"id": "sections", "title": "Deployable model-binary sections", "subtitle": "Constants, generated RV32I, schedule, and manifest", "intent": "composition", "question": "What occupies the deployable model binary?", "rationale": "Compares exact section sizes from the current .lmb.", "type": "horizontalBar", "dataset": "model_sections", "sourceId": "showcase_sql", "encodings": {"x": {"field": "section", "type": "nominal"}, "y": {"field": "bytes", "type": "quantitative", "label": "Bytes"}}, "valueFormat": "compact", "unit": "bytes", "layout": "half"},
    ]
    if has_sweep:
        charts[3:3] = [
            {"id": "roofline", "title": "GEMM roofline with measurement boundaries", "subtitle": "Array-active excludes DMA; CFU+DMA compares with the combined memory/compute roof; streamed KIR includes UART", "intent": "trend", "question": "How does arithmetic intensity translate into accelerator throughput?", "rationale": "Keeps compute-only, accelerator-window, and serial instruction-delivery rates distinct.", "type": "line", "dataset": "roofline", "sourceId": "showcase_sql", "encodings": {"x": {"field": "arithmetic_intensity", "type": "quantitative", "label": "Arithmetic intensity", "unit": "operations/byte"}, "y": {"fields": ["compute_peak_gops", "theoretical_gops", "array_active_gops", "accelerator_window_gops", "streamed_kir_gops"], "type": "quantitative", "label": "Throughput", "unit": "GOPS"}}, "valueFormat": "number", "unit": "GOPS", "layout": "full"},
        ]
    tables = [
        {"id": "layers", "title": "Layer efficiency and arithmetic intensity", "subtitle": "Static compiler plan; achieved per-layer cycles require profiling counters", "dataset": "layer_metrics", "sourceId": "showcase_sql", "defaultSort": {"field": "issued_macs", "direction": "desc"}, "density": "dense", "layout": "full", "columns": [{"field": "layer", "label": "Layer", "format": "number"}, {"field": "kind", "label": "Operator"}, {"field": "shape", "label": "MxKxN"}, {"field": "issued_macs", "label": "Issued MACs", "format": "compact"}, {"field": "semantic_efficiency_pct", "label": "Semantic / issued, %", "format": "number"}, {"field": "ideal_ai_ops_per_byte", "label": "Ideal ops/byte", "format": "number"}, {"field": "planned_ai_ops_per_byte", "label": "Planned ops/byte", "format": "number"}]},
        {"id": "instrumentation", "title": "Measurement coverage", "subtitle": "Unavailable counters are disclosed rather than modeled as achieved values", "dataset": "instrumentation_status", "sourceId": "showcase_sql", "defaultSort": {"field": "status", "direction": "asc"}, "density": "spacious", "layout": "full", "columns": [{"field": "metric", "label": "Metric"}, {"field": "status", "label": "Status"}, {"field": "source", "label": "Source or next instrumentation"}]},
    ]
    blocks: list[dict[str, Any]] = [
        {"id": "hardware_heading", "type": "markdown", "body": "## Hardware\nArchitectural ceilings, achieved board performance, utilization, and overlap."},
        {"id": "hardware_cards", "type": "metric-strip", "cardIds": ["peak", "array_active", "accelerator_window", "achieved", "overlap_efficiency", "utilization", "latency", "exactness"]},
        {"id": "throughput_block", "type": "chart", "chartId": "throughput"},
        {"id": "efficiency_block", "type": "chart", "chartId": "efficiency"},
        {"id": "resources_block", "type": "chart", "chartId": "resources"},
        {"id": "model_cycles_block", "type": "chart", "chartId": "model_cycles"},
    ]
    if has_sweep:
        blocks.extend([
            {"id": "roofline_block", "type": "chart", "chartId": "roofline"},
        ])
    blocks.extend([
        {"id": "software_heading", "type": "markdown", "body": "## Software and compiler\nAccuracy preservation, operator coverage, code generation, and model layout."},
        {"id": "software_cards", "type": "metric-strip", "cardIds": ["fp32_accuracy", "ort_accuracy", "compiler_accuracy", "agreement", "binary"]},
        {"id": "accuracy_block", "type": "chart", "chartId": "accuracy"},
        {"id": "operators_block", "type": "chart", "chartId": "operators"},
        {"id": "sections_block", "type": "chart", "chartId": "sections"},
        {"id": "layers_block", "type": "table", "tableId": "layers"},
        {"id": "coverage_heading", "type": "markdown", "body": "## Measurement coverage\nMeasured, derived, and not-yet-instrumented fields are kept distinct."},
        {"id": "instrumentation_block", "type": "table", "tableId": "instrumentation"},
    ])
    return {
        "surface": "dashboard",
        "manifest": {
            "version": 1,
            "surface": "dashboard",
            "title": "loom: theoretical vs achieved",
            "description": "Hardware and compiler performance characterization for MobileNet-v1 0.25 on Tang Nano 20K.",
            "generatedAt": generated_at,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": [{"id": source["id"], "label": source["label"], "path": source["path"]}],
            "blocks": blocks,
        },
        "snapshot": {"version": 1, "generatedAt": generated_at, "status": "ready", "datasets": datasets},
        "sources": [source],
    }


def benchmark(*, port: str, output: Path, timeout_s: float) -> None:
    """Measure on-chip cycles + CFU counters for each GEMM shape via a compiled .lmb."""
    from tests.test_gemm import run_gemm

    results: list[dict[str, Any]] = []
    transport = SerialTransport(port, timeout_s=timeout_s)
    try:
        for index, (m, k, n) in enumerate(BENCHMARK_CASES, start=1):
            r = run_gemm(transport, m, k, n)
            print(f"[{index}/{len(BENCHMARK_CASES)}] M={m} K={k} N={n} cycles={r.cycles} "
                  f"max|Δ|={r.max_delta}", flush=True)
            if r.max_delta > 1:
                raise RuntimeError(f"GEMM {(m, k, n)} failed verification: max|Δ|={r.max_delta}")
            results.append(
                {
                    "m": m,
                    "k": k,
                    "n": n,
                    "variant": "lmb",
                    "cycles": r.cycles,
                    "verified": True,
                    "verification_tolerance_lsb": 1,
                    "counters": r.counters.__dict__ if r.counters else {},
                }
            )
    finally:
        transport.close()
    _write_json(
        output,
        {
            "schema_version": 2,
            "captured_at": _now(),
            "board": "Sipeed Tang Nano 20K",
            "clock_hz": DEFAULT_CLOCK_HZ,
            "measurement": "compiled .lmb via exec_model; cycles are on-chip mcycle",
            "results": results,
        },
    )
    print(f"wrote {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    collect_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    collect_parser.add_argument("--lmb", type=Path, default=DEFAULT_LMB)
    collect_parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVAL)
    collect_parser.add_argument("--sweep", type=Path, default=DEFAULT_SWEEP)
    collect_parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "build/showcase")
    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--port", default="/dev/ttyUSB1")
    benchmark_parser.add_argument("--output", type=Path, default=DEFAULT_SWEEP)
    benchmark_parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.command == "benchmark":
        benchmark(port=args.port, output=args.output, timeout_s=args.timeout)
    else:
        artifact_path = collect(
            report_path=args.report,
            model_path=args.model,
            lmb_path=args.lmb,
            evaluation_path=args.evaluation,
            sweep_path=args.sweep,
            output_dir=args.output_dir,
        )
        print(f"wrote {artifact_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
