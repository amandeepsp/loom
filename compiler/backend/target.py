"""Target metadata for loom GEMM lowering."""

from __future__ import annotations

from dataclasses import dataclass

from compiler.backend.rewrites import compute_k_tile


@dataclass(frozen=True)
class DmaRegs:
    base: int
    length: int
    enable: int
    done: int
    loop: int
    offset: int


@dataclass(frozen=True)
class CsrMap:
    act_dma: DmaRegs
    wgt_dma: DmaRegs

    @staticmethod
    def from_json_dict(data: dict) -> "CsrMap":
        regs = data["csr_registers"]

        def reg(name: str) -> int:
            return int(regs[name]["addr"])

        return CsrMap(
            act_dma=DmaRegs(
                reg("act_dma_base"),
                reg("act_dma_length"),
                reg("act_dma_enable"),
                reg("act_dma_done"),
                reg("act_dma_loop"),
                reg("act_dma_offset"),
            ),
            wgt_dma=DmaRegs(
                reg("wgt_dma_base"),
                reg("wgt_dma_length"),
                reg("wgt_dma_enable"),
                reg("wgt_dma_done"),
                reg("wgt_dma_loop"),
                reg("wgt_dma_offset"),
            ),
        )


@dataclass(frozen=True)
class GemmTargetSpec:
    m: int
    k: int
    n: int
    tile: int
    input_addr: int
    weight_addr: int
    output_addr: int
    bias_addr: int
    mult_addr: int
    shift_addr: int
    output_offset: int
    activation_min: int = -128
    activation_max: int = 127
    cfu_word_bits: int = 64
    cfu_store_depth_words: int = 512

    @property
    def k_tile(self) -> int:
        return compute_k_tile(
            cfu_word_bits=self.cfu_word_bits,
            cfu_store_depth_words=self.cfu_store_depth_words,
        )
