#!/usr/bin/env -S uv run python
"""Create the compiler-supported INT8 QOperator MobileNet-v1 0.25 model.

The checked-in FP32 model uses a 32x32 input and ten output classes. Static
calibration uses a deterministic subset of the CIFAR-10 training split with the
same normalization used to train the supplied classifier.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from torchvision import datasets, transforms


class Cifar10CalibrationDataReader(CalibrationDataReader):
    """Representative, reproducible calibration inputs from CIFAR-10 train."""

    def __init__(self, root: Path, samples: int = 512, seed: int = 0) -> None:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616),
                ),
            ]
        )
        dataset = datasets.CIFAR10(str(root), train=True, download=False, transform=transform)
        if not 0 < samples <= len(dataset):
            raise ValueError(f"samples must be in [1, {len(dataset)}], got {samples}")
        indices = np.random.default_rng(seed).permutation(len(dataset))[:samples]
        self._inputs = [
            {"input": dataset[int(index)][0].numpy()[None].astype(np.float32, copy=False)}
            for index in indices
        ]
        self.rewind()

    def get_next(self):
        return next(self._iterator, None)

    def rewind(self) -> None:
        self._iterator = iter(self._inputs)


def quantize_mobilenet(
    fp32_path: Path,
    output_path: Path,
    *,
    data_root: Path,
    samples: int,
    seed: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        str(fp32_path),
        str(output_path),
        Cifar10CalibrationDataReader(data_root, samples=samples, seed=seed),
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["Conv", "Gemm"],
        per_channel=True,
        # Prevent a near-constant activation channel from producing a tiny
        # input_scale * weight_scale that overflows the int32 bias encoding.
        extra_options={"MinimumRealRange": 1.0e-4},
    )
    model = onnx.load(str(output_path))
    counts: dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    if counts.get("QLinearConv") != 27 or counts.get("QGemm") != 1:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"unexpected quantized MobileNet operator set: {counts}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantize MobileNet-v1 0.25 for loom")
    parser.add_argument(
        "--fp32",
        type=Path,
        default=Path("models/out/mobilenet_v1_025_fp32.onnx"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/out/mobilenet_v1_025_int8.onnx"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("build"),
        help="directory containing the extracted cifar-10-batches-py dataset",
    )
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    quantize_mobilenet(
        args.fp32,
        args.output,
        data_root=args.data_root,
        samples=args.samples,
        seed=args.seed,
    )
    logging.info("INT8 MobileNet saved to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
