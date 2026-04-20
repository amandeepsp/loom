#!/usr/bin/env -S uv run python
"""
Minimal MNIST: train and statically-quantized int8 ONNX export.

Usage:
    uv run python -m models.mnist [--epochs 5] [--out-dir models/out]
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantFormat, quantize_static
from onnxruntime.quantization.calibrate import CalibrationDataReader
from onnxruntime.quantization.shape_inference import quant_pre_process
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)


class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MNISTCalibrationDataReader(CalibrationDataReader):
    def __init__(self, data_dir: str, num_samples: int = 512):
        self.num_samples = num_samples
        self._count = 0
        self._transform = transforms.Compose([transforms.ToTensor()])
        self._ds = datasets.MNIST(data_dir, train=True, download=True, transform=self._transform)
        self._loader = torch.utils.data.DataLoader(self._ds, batch_size=1, shuffle=True)
        self._data_iter = iter(self._loader)

    def get_next(self):
        if self._count >= self.num_samples:
            return None
        images, _ = next(self._data_iter)
        self._count += 1
        return {"input": images.numpy().astype(np.float32)}


def train(model, epochs, batch_size, lr, data_dir):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            criterion(model(images), labels).backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                correct += (model(images).argmax(1) == labels).sum().item()
                total += labels.size(0)
        logger.info(f"Epoch {epoch + 1}/{epochs} — Test acc: {100 * correct / total:.1f}%")


def _export_fp32_onnx(model, fp32_path: Path) -> None:
    model.eval()
    dummy = torch.randn(1, 1, 28, 28)
    torch.onnx.export(
        model,
        dummy,
        str(fp32_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    logger.info(f"FP32 ONNX saved to {fp32_path}")


def export_onnx_static_quant(model, out_dir: Path, data_dir: str) -> Path:
    """Export statically-quantized int8 ONNX using ONNX Runtime static quantization.

    The exported ONNX has fixed scale/zero_point constants baked in as ONNX
    Constant nodes (via QDQ format), making them available at compile time for
    hardware epilogue parameter derivation.
    """
    fp32_path = out_dir / "mnist_fp32.onnx"
    int8_path = out_dir / "mnist_int8.onnx"

    _export_fp32_onnx(model, fp32_path)

    prep_path = out_dir / "mnist_fp32_prep.onnx"
    quant_pre_process(str(fp32_path), str(prep_path))

    logger.info("Running static quantization calibration...")
    calibration_reader = MNISTCalibrationDataReader(data_dir)
    quantize_static(
        str(prep_path),
        str(int8_path),
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=["Gemm", "Relu"],
    )
    prep_path.unlink()
    logger.info(f"Statically-quantized ONNX saved to {int8_path}")

    sess = ort.InferenceSession(str(int8_path))
    test_input = np.random.randn(10, 1, 28, 28).astype(np.float32)
    out = sess.run(None, {"input": test_input})
    logger.info(f"Static int8 model inference OK, output shape: {out[0].shape}")

    return int8_path


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train MNIST + ONNX int8 export")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--out-dir", type=str, default="models/out")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = MNISTNet()
    train(model, args.epochs, args.batch_size, args.lr, args.data_dir)
    export_onnx_static_quant(model, out_dir, args.data_dir)


if __name__ == "__main__":
    main()
