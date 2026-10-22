"""RV32IM encode/decode. The decoder fills every field, so stages never re-read raw bits."""

from __future__ import annotations

from dataclasses import dataclass, field

XLEN = 32
MASK32 = 0xFFFFFFFF

# --------------------------------------------------------------------------
# Registers
# --------------------------------------------------------------------------

ABI_NAMES = [
    "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2",
    "s0", "s1", "a0", "a1", "a2", "a3", "a4", "a5",
    "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7",
    "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6",
]

REG_ALIASES: dict[str, int] = {}
for _i, _n in enumerate(ABI_NAMES):
    REG_ALIASES[_n] = _i
    REG_ALIASES["x%d" % _i] = _i
REG_ALIASES["fp"] = 8  # s0 doubles as the frame pointer


def reg_number(token: str) -> int:
    key = token.strip().lower()
    if key not in REG_ALIASES:
        raise KeyError("unknown register %r" % token)
    return REG_ALIASES[key]


def reg_name(index: int) -> str:
    return "x%d/%s" % (index, ABI_NAMES[index])


# --------------------------------------------------------------------------
# Bit helpers
# --------------------------------------------------------------------------

def sign_extend(value: int, width: int) -> int:
    """Low ``width`` bits of ``value`` as two's complement."""
    sign = 1 << (width - 1)
    return (value & ((1 << width) - 1) ^ sign) - sign


def to_signed(value: int) -> int:
    return sign_extend(value & MASK32, 32)


def to_unsigned(value: int) -> int:
    return value & MASK32


def bits(word: int, hi: int, lo: int) -> int:
    return (word >> lo) & ((1 << (hi - lo + 1)) - 1)


# --------------------------------------------------------------------------
# Opcode / funct tables
# --------------------------------------------------------------------------

OP_LUI = 0b0110111
OP_AUIPC = 0b0010111
OP_JAL = 0b1101111
OP_JALR = 0b1100111
OP_BRANCH = 0b1100011
OP_LOAD = 0b0000011
OP_STORE = 0b0100011
OP_IMM = 0b0010011
OP_REG = 0b0110011
OP_MISC_MEM = 0b0001111
OP_SYSTEM = 0b1110011
OP_AMO = 0b0101111

BRANCH_F3 = {"beq": 0b000, "bne": 0b001, "blt": 0b100, "bge": 0b101,
             "bltu": 0b110, "bgeu": 0b111}
LOAD_F3 = {"lb": 0b000, "lh": 0b001, "lw": 0b010, "lbu": 0b100, "lhu": 0b101}
STORE_F3 = {"sb": 0b000, "sh": 0b001, "sw": 0b010}
IMM_F3 = {"addi": 0b000, "slli": 0b001, "slti": 0b010, "sltiu": 0b011,
          "xori": 0b100, "srli": 0b101, "srai": 0b101, "ori": 0b110,
          "andi": 0b111}
IMM_PLAIN_F3 = {"addi": 0b000, "slti": 0b010, "sltiu": 0b011,
                "xori": 0b100, "ori": 0b110, "andi": 0b111}

# name -> (funct3, funct7)
REG_OPS = {
    "add": (0b000, 0b0000000), "sub": (0b000, 0b0100000),
    "sll": (0b001, 0b0000000), "slt": (0b010, 0b0000000),
    "sltu": (0b011, 0b0000000), "xor": (0b100, 0b0000000),
    "srl": (0b101, 0b0000000), "sra": (0b101, 0b0100000),
    "or": (0b110, 0b0000000), "and": (0b111, 0b0000000),
    # RV32M
    "mul": (0b000, 0b0000001), "mulh": (0b001, 0b0000001),
    "mulhsu": (0b010, 0b0000001), "mulhu": (0b011, 0b0000001),
    "div": (0b100, 0b0000001), "divu": (0b101, 0b0000001),
    "rem": (0b110, 0b0000001), "remu": (0b111, 0b0000001),
}
MULDIV = {"mul", "mulh", "mulhsu", "mulhu", "div", "divu", "rem", "remu"}

# RV32A: funct5 (bits 31:27) with funct3 = 010 for word-width atomics.
AMO_F5 = {
    "amoadd.w": 0b00000, "amoswap.w": 0b00001, "lr.w": 0b00010,
    "sc.w": 0b00011, "amoxor.w": 0b00100, "amoor.w": 0b01000,
    "amoand.w": 0b01100, "amomin.w": 0b10000, "amomax.w": 0b10100,
    "amominu.w": 0b11000, "amomaxu.w": 0b11100,
}
AMO_OPS = set(AMO_F5) - {"lr.w", "sc.w"}

CSR_F3 = {"csrrw": 0b001, "csrrs": 0b010, "csrrc": 0b011,
          "csrrwi": 0b101, "csrrsi": 0b110, "csrrci": 0b111}

MEM_WIDTH = {"lb": 1, "lbu": 1, "lh": 2, "lhu": 2, "lw": 4,
             "sb": 1, "sh": 2, "sw": 4}

# Instructions that end a basic block by redirecting the PC.
CONTROL_FLOW = set(BRANCH_F3) | {"jal", "jalr"}

NOP_WORD = 0x00000013  # addi x0, x0, 0


@dataclass
class Instr:
    raw: int = 0
    pc: int = 0
    name: str = "unknown"
    rd: int = 0
    rs1: int = 0
    rs2: int = 0
    imm: int = 0
    csr: int = 0
    # Control signals derived once, at decode.
    uses_rs1: bool = False
    uses_rs2: bool = False
    writes_rd: bool = False
    is_load: bool = False
    is_store: bool = False
    is_branch: bool = False
    is_jump: bool = False
    is_system: bool = False
    is_csr: bool = False
    is_fence: bool = False
    is_muldiv: bool = False
    is_atomic: bool = False
    amo_op: str = ""
    mem_width: int = 0
    illegal: bool = False
    # Filled in by the pipeline as the instruction flows through.
    seq: int = -1
    core: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def reads(self) -> set:
        out = set()
        if self.uses_rs1 and self.rs1 != 0:
            out.add(self.rs1)
        if self.uses_rs2 and self.rs2 != 0:
            out.add(self.rs2)
        return out

    @property
    def writes(self) -> set:
        return {self.rd} if self.writes_rd and self.rd != 0 else set()

    def text(self) -> str:
        n, rd, rs1, rs2 = self.name, self.rd, self.rs1, self.rs2
        a = ABI_NAMES
        if n in ("nop", "ecall", "ebreak", "mret", "sret", "wfi",
                 "fence", "fence.i", "illegal"):
            return n
        if n in ("lui", "auipc"):
            return "%s %s, 0x%x" % (n, a[rd], (self.imm >> 12) & 0xFFFFF)
        if n == "jal":
            return "jal %s, %+d" % (a[rd], self.imm)
        if n == "jalr":
            return "jalr %s, %+d(%s)" % (a[rd], self.imm, a[rs1])
        if n in BRANCH_F3:
            return "%s %s, %s, %+d" % (n, a[rs1], a[rs2], self.imm)
        if n in LOAD_F3:
            return "%s %s, %+d(%s)" % (n, a[rd], self.imm, a[rs1])
        if n in STORE_F3:
            return "%s %s, %+d(%s)" % (n, a[rs2], self.imm, a[rs1])
        if n in ("slli", "srli", "srai"):
            return "%s %s, %s, %d" % (n, a[rd], a[rs1], self.imm & 31)
        if n in IMM_PLAIN_F3:
            return "%s %s, %s, %d" % (n, a[rd], a[rs1], self.imm)
        if n in REG_OPS:
            return "%s %s, %s, %s" % (n, a[rd], a[rs1], a[rs2])
        if n == "lr.w":
            return "lr.w %s, (%s)" % (a[rd], a[rs1])
        if n in AMO_F5:
            return "%s %s, %s, (%s)" % (n, a[rd], a[rs2], a[rs1])
        if n in ("csrrw", "csrrs", "csrrc"):
            return "%s %s, %s, %s" % (n, a[rd], csr_name(self.csr), a[rs1])
        if n in ("csrrwi", "csrrsi", "csrrci"):
            return "%s %s, %s, %d" % (n, a[rd], csr_name(self.csr), self.imm)
        return n

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<%s @0x%08x>" % (self.text(), self.pc)


def csr_name(addr: int) -> str:
    from .csr import CSR_NAMES
    return CSR_NAMES.get(addr, "0x%03x" % addr)


# --------------------------------------------------------------------------
# Decoder
# --------------------------------------------------------------------------

def decode(word: int, pc: int = 0) -> Instr:
    word &= MASK32
    ins = Instr(raw=word, pc=pc)
    opcode = bits(word, 6, 0)
    rd = bits(word, 11, 7)
    funct3 = bits(word, 14, 12)
    rs1 = bits(word, 19, 15)
    rs2 = bits(word, 24, 20)
    funct7 = bits(word, 31, 25)
    ins.rd, ins.rs1, ins.rs2 = rd, rs1, rs2

    if opcode == OP_LUI:
        ins.name, ins.writes_rd = "lui", True
        ins.imm = sign_extend(word & 0xFFFFF000, 32)
    elif opcode == OP_AUIPC:
        ins.name, ins.writes_rd = "auipc", True
        ins.imm = sign_extend(word & 0xFFFFF000, 32)
    elif opcode == OP_JAL:
        ins.name, ins.writes_rd, ins.is_jump = "jal", True, True
        imm = ((bits(word, 31, 31) << 20) | (bits(word, 19, 12) << 12) |
               (bits(word, 20, 20) << 11) | (bits(word, 30, 21) << 1))
        ins.imm = sign_extend(imm, 21)
    elif opcode == OP_JALR and funct3 == 0:
        ins.name, ins.writes_rd, ins.is_jump = "jalr", True, True
        ins.uses_rs1 = True
        ins.imm = sign_extend(bits(word, 31, 20), 12)
    elif opcode == OP_BRANCH:
        name = _lookup(BRANCH_F3, funct3)
        if name is None:
            return _illegal(ins)
        ins.name, ins.is_branch = name, True
        ins.uses_rs1 = ins.uses_rs2 = True
        imm = ((bits(word, 31, 31) << 12) | (bits(word, 7, 7) << 11) |
               (bits(word, 30, 25) << 5) | (bits(word, 11, 8) << 1))
        ins.imm = sign_extend(imm, 13)
    elif opcode == OP_LOAD:
        name = _lookup(LOAD_F3, funct3)
        if name is None:
            return _illegal(ins)
        ins.name, ins.is_load = name, True
        ins.writes_rd = ins.uses_rs1 = True
        ins.mem_width = MEM_WIDTH[name]
        ins.imm = sign_extend(bits(word, 31, 20), 12)
    elif opcode == OP_STORE:
        name = _lookup(STORE_F3, funct3)
        if name is None:
            return _illegal(ins)
        ins.name, ins.is_store = name, True
        ins.uses_rs1 = ins.uses_rs2 = True
        ins.mem_width = MEM_WIDTH[name]
        ins.imm = sign_extend((bits(word, 31, 25) << 5) | bits(word, 11, 7), 12)
    elif opcode == OP_IMM:
        ins.uses_rs1, ins.writes_rd = True, True
        if funct3 == 0b001:
            if funct7 != 0:
                return _illegal(ins)
            ins.name, ins.imm = "slli", rs2
        elif funct3 == 0b101:
            if funct7 not in (0, 0b0100000):
                return _illegal(ins)
            ins.name = "srai" if funct7 == 0b0100000 else "srli"
            ins.imm = rs2
        else:
            name = _lookup(IMM_PLAIN_F3, funct3)
            if name is None:
                return _illegal(ins)
            ins.name = name
            ins.imm = sign_extend(bits(word, 31, 20), 12)
        if ins.name == "addi" and rd == 0 and rs1 == 0 and ins.imm == 0:
            ins.name, ins.writes_rd, ins.uses_rs1 = "nop", False, False
    elif opcode == OP_REG:
        name = None
        for key, (f3, f7) in REG_OPS.items():
            if f3 == funct3 and f7 == funct7:
                name = key
                break
        if name is None:
            return _illegal(ins)
        ins.name, ins.writes_rd = name, True
        ins.uses_rs1 = ins.uses_rs2 = True
        ins.is_muldiv = name in MULDIV
    elif opcode == OP_AMO:
        funct5 = bits(word, 31, 27)
        name = None
        for key, f5 in AMO_F5.items():
            if f5 == funct5:
                name = key
                break
        if name is None or funct3 != 0b010:
            return _illegal(ins)
        ins.name = name
        ins.is_atomic = True
        ins.amo_op = name
        ins.mem_width = 4
        ins.uses_rs1 = True
        ins.writes_rd = True
        ins.uses_rs2 = name != "lr.w"
    elif opcode == OP_MISC_MEM:
        ins.name = "fence.i" if funct3 == 1 else "fence"
        ins.is_fence = True
    elif opcode == OP_SYSTEM:
        if funct3 == 0:
            imm12 = bits(word, 31, 20)
            ins.name = {0x000: "ecall", 0x001: "ebreak", 0x102: "sret",
                        0x302: "mret", 0x105: "wfi"}.get(imm12)
            if ins.name is None:
                return _illegal(ins)
            ins.is_system = True
        else:
            name = _lookup(CSR_F3, funct3)
            if name is None:
                return _illegal(ins)
            ins.name, ins.is_csr, ins.is_system = name, True, True
            ins.writes_rd = True
            ins.csr = bits(word, 31, 20)
            if name.endswith("i"):
                ins.imm = rs1  # zero-extended 5-bit immediate
            else:
                ins.uses_rs1 = True
    else:
        return _illegal(ins)
    return ins


def _lookup(table, value):
    for key, val in table.items():
        if val == value:
            return key
    return None


def _illegal(ins: Instr) -> Instr:
    ins.illegal = True
    ins.name = "illegal"
    ins.uses_rs1 = ins.uses_rs2 = ins.writes_rd = False
    return ins


# --------------------------------------------------------------------------
# Encoder (used by the assembler and by the round-trip tests)
# --------------------------------------------------------------------------

def encode(name: str, rd: int = 0, rs1: int = 0, rs2: int = 0,
           imm: int = 0, csr: int = 0) -> int:
    name = name.lower()
    if name == "nop":
        return NOP_WORD
    if name == "lui":
        return ((imm & 0xFFFFF000) | (rd << 7) | OP_LUI) & MASK32
    if name == "auipc":
        return ((imm & 0xFFFFF000) | (rd << 7) | OP_AUIPC) & MASK32
    if name == "jal":
        i = imm & 0x1FFFFF
        word = ((((i >> 20) & 1) << 31) | (((i >> 1) & 0x3FF) << 21) |
                (((i >> 11) & 1) << 20) | (((i >> 12) & 0xFF) << 12))
        return (word | (rd << 7) | OP_JAL) & MASK32
    if name == "jalr":
        return (((imm & 0xFFF) << 20) | (rs1 << 15) | (rd << 7) | OP_JALR) & MASK32
    if name in BRANCH_F3:
        i = imm & 0x1FFF
        word = ((((i >> 12) & 1) << 31) | (((i >> 5) & 0x3F) << 25) |
                (((i >> 1) & 0xF) << 8) | (((i >> 11) & 1) << 7))
        return (word | (rs2 << 20) | (rs1 << 15) |
                (BRANCH_F3[name] << 12) | OP_BRANCH) & MASK32
    if name in LOAD_F3:
        return (((imm & 0xFFF) << 20) | (rs1 << 15) | (LOAD_F3[name] << 12) |
                (rd << 7) | OP_LOAD) & MASK32
    if name in STORE_F3:
        i = imm & 0xFFF
        return (((i >> 5) << 25) | (rs2 << 20) | (rs1 << 15) |
                (STORE_F3[name] << 12) | ((i & 0x1F) << 7) | OP_STORE) & MASK32
    if name in ("slli", "srli", "srai"):
        f7 = 0b0100000 if name == "srai" else 0
        f3 = 0b001 if name == "slli" else 0b101
        return ((f7 << 25) | ((imm & 31) << 20) | (rs1 << 15) | (f3 << 12) |
                (rd << 7) | OP_IMM) & MASK32
    if name in IMM_PLAIN_F3:
        return (((imm & 0xFFF) << 20) | (rs1 << 15) | (IMM_PLAIN_F3[name] << 12) |
                (rd << 7) | OP_IMM) & MASK32
    if name in REG_OPS:
        f3, f7 = REG_OPS[name]
        return ((f7 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) |
                (rd << 7) | OP_REG) & MASK32
    if name in AMO_F5:
        return ((AMO_F5[name] << 27) | (rs2 << 20) | (rs1 << 15) |
                (0b010 << 12) | (rd << 7) | OP_AMO) & MASK32
    if name in ("fence", "fence.i"):
        f3 = 1 if name == "fence.i" else 0
        pred_succ = 0x0FF if name == "fence" else 0
        return ((pred_succ << 20) | (f3 << 12) | OP_MISC_MEM) & MASK32
    if name in ("ecall", "ebreak", "mret", "sret", "wfi"):
        imm12 = {"ecall": 0x000, "ebreak": 0x001, "sret": 0x102,
                 "mret": 0x302, "wfi": 0x105}[name]
        return ((imm12 << 20) | OP_SYSTEM) & MASK32
    if name in CSR_F3:
        src = (imm & 31) if name.endswith("i") else rs1
        return (((csr & 0xFFF) << 20) | (src << 15) | (CSR_F3[name] << 12) |
                (rd << 7) | OP_SYSTEM) & MASK32
    raise ValueError("cannot encode %r" % name)
