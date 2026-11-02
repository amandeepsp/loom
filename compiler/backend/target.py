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
    # Status: done *and* every beat delivered to the scratchpad. Present on
    # bitstreams with the `drained` CSR; when set, no delay fence is needed.
    drained: int | None = None


@dataclass(frozen=True)
class CsrMap:
    act_dma: DmaRegs
    wgt_dma: DmaRegs

    @staticmethod
    def from_json_dict(data: dict) -> "CsrMap":
        regs = data["csr_registers"]

        def reg(name: str) -> int:
            return int(regs[name]["addr"])

        def opt(name: str) -> int | None:
            return int(regs[name]["addr"]) if name in regs else None

        return CsrMap(
            act_dma=DmaRegs(
                reg("act_dma_base"),
                reg("act_dma_length"),
                reg("act_dma_enable"),
                reg("act_dma_done"),
                reg("act_dma_loop"),
                reg("act_dma_offset"),
                opt("act_dma_drained"),
            ),
            wgt_dma=DmaRegs(
                reg("wgt_dma_base"),
                reg("wgt_dma_length"),
                reg("wgt_dma_enable"),
                reg("wgt_dma_done"),
                reg("wgt_dma_loop"),
                reg("wgt_dma_offset"),
                opt("wgt_dma_drained"),
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
    # Block-diagonal GEMMs: the packed input has `input_k_stride` K rows per
    # M tile (default: k), and N tile n reads rows starting at n * input_n_k_offset.
    input_k_stride: int = 0
    input_n_k_offset: int = 0

    @property
    def a_k_stride(self) -> int:
        return self.input_k_stride or self.k

    @property
    def k_tile(self) -> int:
        return compute_k_tile(
            cfu_word_bits=self.cfu_word_bits,
            cfu_store_depth_words=self.cfu_store_depth_words,
        )
