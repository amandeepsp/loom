"""Small labeled RV32 word builder used by loom code encoders."""

from __future__ import annotations

from . import riscv


class RV32Builder:
    def __init__(self) -> None:
        self.words: list[int] = []
        self.labels: dict[str, int] = {}
        self.fixups: list[tuple[int, str, str, int, int]] = []
        self._label_counter = 0

    def emit(self, word: int) -> None:
        self.words.append(word & 0xFFFFFFFF)

    def emit_many(self, words: list[int]) -> None:
        for word in words:
            self.emit(word)

    def li(self, rd: int, value: int) -> None:
        self.emit_many(riscv.li(rd, value))

    def sw_abs(self, addr: int, value_reg: int, tmp: int = riscv.T0) -> None:
        self.li(tmp, addr)
        self.emit(riscv.sw(value_reg, tmp, 0))

    def sw_reg_abs(self, addr: int, value_reg: int, tmp: int = riscv.T0) -> None:
        self.sw_abs(addr, value_reg, tmp)

    def sw_imm_abs(
        self,
        addr: int,
        value: int,
        tmp_addr: int = riscv.T0,
        tmp_value: int = riscv.T1,
    ) -> None:
        self.li(tmp_value, value)
        self.sw_abs(addr, tmp_value, tmp_addr)

    def wait_done(self, done_addr: int) -> None:
        self.li(riscv.T0, done_addr)
        loop_word_index = len(self.words)
        self.emit(riscv.lw(riscv.T1, riscv.T0, 0))
        byte_offset = (loop_word_index - len(self.words)) * 4
        self.emit(riscv.beq(riscv.T1, riscv.ZERO, byte_offset))

    def new_label(self, prefix: str) -> str:
        label = f"{prefix}_{self._label_counter}"
        self._label_counter += 1
        return label

    def label(self, name: str) -> None:
        if name in self.labels:
            raise ValueError(f"duplicate label: {name}")
        self.labels[name] = len(self.words)

    def branch_label(self, kind: str, rs1: int, rs2: int, label: str) -> None:
        self.fixups.append((len(self.words), label, kind, rs1, rs2))
        self.emit(0)

    def j_label(self, label: str) -> None:
        self.branch_label("beq", riscv.ZERO, riscv.ZERO, label)

    def _resolve(self) -> list[int]:
        words = list(self.words)
        for pos, label, kind, rs1, rs2 in self.fixups:
            if label not in self.labels:
                raise ValueError(f"unknown label: {label}")
            offset = (self.labels[label] - pos) * 4
            if kind == "beq":
                words[pos] = riscv.beq(rs1, rs2, offset)
            elif kind == "bne":
                words[pos] = riscv.bne(rs1, rs2, offset)
            elif kind == "blt":
                words[pos] = riscv.blt(rs1, rs2, offset)
            else:
                raise ValueError(f"unknown branch kind: {kind}")
        return words

    def bytes(self) -> bytes:
        return b"".join(word.to_bytes(4, "little") for word in self._resolve())
