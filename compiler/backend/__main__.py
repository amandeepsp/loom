"""CLI for compiling ONNX models into loom model binaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model import compile_onnx_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile ONNX into a loom model binary.")
    parser.add_argument("onnx")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--memory-base", type=lambda x: int(x, 0), default=0x40010100)
    parser.add_argument(
        "--csr-json",
        default="build/sipeed_tang_nano_20k/csr.json",
        help="LiteX csr.json whose DMA register addresses are encoded in the RV32I kernels.",
    )
    parser.add_argument("--tile", type=int, default=8, help="CFU tile dimension.")
    parser.add_argument("--cfu-word-bits", type=int, default=64)
    parser.add_argument("--cfu-store-depth-words", type=int, default=512)
    parser.add_argument("--manifest", action="store_true")
    parser.add_argument(
        "--debug-sections",
        action="store_true",
        help="Include diagnostic ASM/xDSL IR sections in the output binary.",
    )
    args = parser.parse_args()

    compiled = compile_onnx_model(
        args.onnx,
        batch_size=args.batch_size,
        memory_base=args.memory_base,
        tile=args.tile,
        cfu_word_bits=args.cfu_word_bits,
        cfu_store_depth_words=args.cfu_store_depth_words,
        csr_json=args.csr_json,
    )
    Path(args.output).write_bytes(compiled.to_bytes(include_debug_sections=args.debug_sections))
    if args.manifest:
        print(json.dumps(compiled.manifest(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
