"""Small RV32I/CUSTOM0 instruction encoding helpers."""

from __future__ import annotations

from dataclasses import dataclass

CUSTOM0_OPCODE = 0x0B

ZERO = 0
RA = 1
SP = 2
T0 = 5
T1 = 6
T2 = 7
S0 = 8
S1 = 9
A0 = 10
A1 = 11
A2 = 12
A3 = 13
A4 = 14
A5 = 15
A6 = 16
A7 = 17
S2 = 18
S3 = 19
S4 = 20
T3 = 28
T4 = 29
T5 = 30
T6 = 31


@dataclass(frozen=True)
class RTypeInstruction:
    opcode: int
    funct3: int
    funct7: int
    rd: int
    rs1: int
    rs2: int

    def encode(self) -> int:
        return encode_r_type(
            opcode=self.opcode,
            funct3=self.funct3,
            funct7=self.funct7,
            rd=self.rd,
            rs1=self.rs1,
            rs2=self.rs2,
        )

    def to_bytes(self) -> bytes:
        return self.encode().to_bytes(4, "little")


def encode_r_type(*, opcode: int, funct3: int, funct7: int, rd: int, rs1: int, rs2: int) -> int:
    """Encode a RISC-V R-type instruction word."""

    _check_range("opcode", opcode, 7)
    _check_range("funct3", funct3, 3)
    _check_range("funct7", funct7, 7)
    _check_range("rd", rd, 5)
    _check_range("rs1", rs1, 5)
    _check_range("rs2", rs2, 5)
    return (
        (funct7 << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (funct3 << 12)
        | (rd << 7)
        | opcode
    )


def encode_i_type(*, opcode: int, funct3: int, rd: int, rs1: int, imm: int) -> int:
    imm = _signed_imm("imm", imm, 12) & 0xFFF
    _check_range("opcode", opcode, 7)
    _check_range("funct3", funct3, 3)
    _check_range("rd", rd, 5)
    _check_range("rs1", rs1, 5)
    return (imm << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def encode_s_type(*, opcode: int, funct3: int, rs1: int, rs2: int, imm: int) -> int:
    imm = _signed_imm("imm", imm, 12) & 0xFFF
    _check_range("opcode", opcode, 7)
    _check_range("funct3", funct3, 3)
    _check_range("rs1", rs1, 5)
    _check_range("rs2", rs2, 5)
    return (
        ((imm >> 5) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (funct3 << 12)
        | ((imm & 0x1F) << 7)
        | opcode
    )


def encode_b_type(*, opcode: int, funct3: int, rs1: int, rs2: int, offset: int) -> int:
    if offset % 2 != 0:
        raise ValueError("branch offset must be 2-byte aligned")
    imm = _signed_imm("offset", offset, 13) & 0x1FFF
    _check_range("opcode", opcode, 7)
    _check_range("funct3", funct3, 3)
    _check_range("rs1", rs1, 5)
    _check_range("rs2", rs2, 5)
    return (
        ((imm >> 12) << 31)
        | (((imm >> 5) & 0x3F) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (funct3 << 12)
        | (((imm >> 1) & 0xF) << 8)
        | (((imm >> 11) & 0x1) << 7)
        | opcode
    )


def lui(rd: int, imm20: int) -> int:
    _check_range("rd", rd, 5)
    _check_range("imm20", imm20, 20)
    return (imm20 << 12) | (rd << 7) | 0x37


def addi(rd: int, rs1: int, imm: int) -> int:
    return encode_i_type(opcode=0x13, funct3=0, rd=rd, rs1=rs1, imm=imm)


def lw(rd: int, rs1: int, imm: int = 0) -> int:
    return encode_i_type(opcode=0x03, funct3=2, rd=rd, rs1=rs1, imm=imm)


def lbu(rd: int, rs1: int, imm: int = 0) -> int:
    return encode_i_type(opcode=0x03, funct3=4, rd=rd, rs1=rs1, imm=imm)


def sw(rs2: int, rs1: int, imm: int = 0) -> int:
    return encode_s_type(opcode=0x23, funct3=2, rs1=rs1, rs2=rs2, imm=imm)


def sb(rs2: int, rs1: int, imm: int = 0) -> int:
    return encode_s_type(opcode=0x23, funct3=0, rs1=rs1, rs2=rs2, imm=imm)


def beq(rs1: int, rs2: int, offset: int) -> int:
    return encode_b_type(opcode=0x63, funct3=0, rs1=rs1, rs2=rs2, offset=offset)


def bne(rs1: int, rs2: int, offset: int) -> int:
    return encode_b_type(opcode=0x63, funct3=1, rs1=rs1, rs2=rs2, offset=offset)


def blt(rs1: int, rs2: int, offset: int) -> int:
    return encode_b_type(opcode=0x63, funct3=4, rs1=rs1, rs2=rs2, offset=offset)


def add(rd: int, rs1: int, rs2: int) -> int:
    return encode_r_type(opcode=0x33, funct3=0, funct7=0, rd=rd, rs1=rs1, rs2=rs2)


def xor(rd: int, rs1: int, rs2: int) -> int:
    return encode_r_type(opcode=0x33, funct3=4, funct7=0, rd=rd, rs1=rs1, rs2=rs2)


def or_(rd: int, rs1: int, rs2: int) -> int:
    return encode_r_type(opcode=0x33, funct3=6, funct7=0, rd=rd, rs1=rs1, rs2=rs2)


def andi(rd: int, rs1: int, imm: int) -> int:
    return encode_i_type(opcode=0x13, funct3=7, rd=rd, rs1=rs1, imm=imm)


def slli(rd: int, rs1: int, shamt: int) -> int:
    if not 0 <= shamt < 32:
        raise ValueError(f"shamt={shamt} does not fit in RV32 shift immediate")
    return encode_i_type(opcode=0x13, funct3=1, rd=rd, rs1=rs1, imm=shamt)


def srli(rd: int, rs1: int, shamt: int) -> int:
    if not 0 <= shamt < 32:
        raise ValueError(f"shamt={shamt} does not fit in RV32 shift immediate")
    return encode_i_type(opcode=0x13, funct3=5, rd=rd, rs1=rs1, imm=shamt)


def sltu(rd: int, rs1: int, rs2: int) -> int:
    return encode_r_type(opcode=0x33, funct3=3, funct7=0, rd=rd, rs1=rs1, rs2=rs2)


def sltiu(rd: int, rs1: int, imm: int) -> int:
    return encode_i_type(opcode=0x13, funct3=3, rd=rd, rs1=rs1, imm=imm)


def ret() -> int:
    return encode_i_type(opcode=0x67, funct3=0, rd=ZERO, rs1=RA, imm=0)


def li(rd: int, value: int) -> list[int]:
    """Expand a 32-bit load-immediate pseudo-instruction."""

    value &= 0xFFFFFFFF
    signed = value if value < (1 << 31) else value - (1 << 32)
    if -2048 <= signed <= 2047:
        return [addi(rd, ZERO, signed)]

    low = ((signed + 0x800) & 0xFFF) - 0x800
    high = (signed - low) >> 12
    high &= 0xFFFFF
    code = [lui(rd, high)]
    if low:
        code.append(addi(rd, rd, low))
    return code


def custom0(funct3: int, funct7: int, *, rd: int = 0, rs1: int = 0, rs2: int = 0) -> RTypeInstruction:
    return RTypeInstruction(CUSTOM0_OPCODE, funct3, funct7, rd, rs1, rs2)


def compute_start(*, rd: int = 0, flags_reg: int, k_count_reg: int) -> RTypeInstruction:
    return custom0(0, 0, rd=rd, rs1=flags_reg, rs2=k_count_reg)


def compute_wait(*, rd: int = 0) -> RTypeInstruction:
    return custom0(1, 0, rd=rd, rs1=0, rs2=0)


def write_epilogue_param(field: int, *, channel_reg: int, value_reg: int, rd: int = 0) -> RTypeInstruction:
    return custom0(2, field, rd=rd, rs1=channel_reg, rs2=value_reg)


def write_config(register: int, *, value_reg: int, rd: int = 0) -> RTypeInstruction:
    return custom0(3, register, rd=rd, rs1=0, rs2=value_reg)


def read_result(*, rd: int, index_reg: int) -> RTypeInstruction:
    return custom0(4, 0, rd=rd, rs1=index_reg, rs2=0)


def _check_range(name: str, value: int, bits: int) -> None:
    if not 0 <= value < (1 << bits):
        raise ValueError(f"{name}={value} does not fit in {bits} bits")


def _signed_imm(name: str, value: int, bits: int) -> int:
    lo = -(1 << (bits - 1))
    hi = (1 << (bits - 1)) - 1
    if not lo <= value <= hi:
        raise ValueError(f"{name}={value} does not fit in signed {bits} bits")
    return value
