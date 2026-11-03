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
    # Optional CFU features (Batch A bitstreams only; see cfu_features()).
    packed_read: bool = False
    keep_wgt: bool = False
    keep_act: bool = False
    # Fused store: write results straight into the next layer's packed tile
    # layout ([m/tile][n][tile]) through a 384-byte rebased requant LUT,
    # replacing the next layer's separate pack step.
    packed_output_addr: int = 0
    store_lut_addr: int = 0
    # Channel-major (CHW) store: element (m, c) goes to output_addr + c *
    # output_channel_stride + m, feeding a depthwise consumer's dwpack kernel.
    output_channel_stride: int = 0
    # `result_lut` CFU: the epilogue maps results through a 256-entry LUT
    # (cfu_lut_addr: 256 bytes indexed by the result's unsigned byte, uploaded
    # once per GEMM; 0 disables it) and READ_RESULT funct7=2 reads columns.
    result_lut: bool = False
    cfu_lut_addr: int = 0
    # `col_params` CFU: epilogue params are read per column, so only the
    # row-0 slots (channel index = column) need uploading.
    col_params: bool = False

    @property
    def cpu_store_lut(self) -> bool:
        """The CPU applies store_lut_addr itself (no CFU result LUT)."""
        return bool(self.store_lut_addr) and not (self.result_lut and self.cfu_lut_addr)

    @property
    def a_k_stride(self) -> int:
        return self.input_k_stride or self.k

    @property
    def k_tile(self) -> int:
        return compute_k_tile(
            cfu_word_bits=self.cfu_word_bits,
            cfu_store_depth_words=self.cfu_store_depth_words,
        )


def cfu_features(csr_json=None) -> frozenset[str]:
    """CFU features the target bitstream supports.

    Comma-separated names: `packed_read` (READ_RESULT funct7=1), `keep_wgt`
    (compute flag bit 2), `keep_act` (compute flag bit 3). Taken from
    $LOOM_CFU_FEATURES if set, else from a `loom_cfu_features` file next to
    the bitstream's csr.json (written by `just soc`). Absent means none:
    older bitstreams ignore these and would silently compute garbage.
    """
    import os
    from pathlib import Path

    raw = os.environ.get("LOOM_CFU_FEATURES")
    if raw is None and csr_json is not None:
        marker = Path(csr_json).parent / "loom_cfu_features"
        raw = marker.read_text() if marker.exists() else ""
    return frozenset(f.strip() for f in (raw or "").replace("\n", ",").split(",") if f.strip())
