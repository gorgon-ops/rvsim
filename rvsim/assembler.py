"""A two-pass RV32IM assembler, with a source map so every view can show the original line."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .csr import CSR_ADDRS
from .isa import (ABI_NAMES, AMO_F5, BRANCH_F3, IMM_PLAIN_F3, LOAD_F3, MASK32,
                  REG_OPS, STORE_F3, decode, encode, reg_number)

TEXT_BASE = 0x00001000
DATA_BASE = 0x00002000


class AsmError(Exception):
    def __init__(self, line_no: int, message: str, text: str = ""):
        super().__init__("line %d: %s%s" % (line_no, message,
                                            ("\n    " + text) if text else ""))
        self.line_no = line_no


@dataclass
class SourceLine:
    addr: int
    line_no: int
    text: str
    word: int = 0


@dataclass
class Program:
    text: bytearray = field(default_factory=bytearray)
    data: bytearray = field(default_factory=bytearray)
    text_base: int = TEXT_BASE
    data_base: int = DATA_BASE
    symbols: dict = field(default_factory=dict)
    source: dict = field(default_factory=dict)   # addr -> SourceLine
    listing: list = field(default_factory=list)  # SourceLine in address order

    @property
    def entry(self) -> int:
        return self.symbols.get("_start", self.text_base)

    def disassembly(self) -> list:
        out = []
        for sl in self.listing:
            ins = decode(sl.word, sl.addr)
            out.append((sl.addr, sl.word, ins.text(), sl.text, sl.line_no))
        return out


# --------------------------------------------------------------------------
# Tokenising helpers
# --------------------------------------------------------------------------

_LABEL_RE = re.compile(r"^([A-Za-z_.$][A-Za-z0-9_.$]*):")
_NUM_RE = re.compile(r"^[-+]?(0[xX][0-9a-fA-F]+|0[bB][01]+|\d+)$")


def _strip_comment(line: str) -> str:
    out = []
    in_str = False
    escape = False
    for i, ch in enumerate(line):
        if in_str:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            continue
        if ch == "#" or ch == ";":
            break
        if ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
            break
        out.append(ch)
    return "".join(out).strip()


def _split_operands(rest: str) -> list:
    parts, buf, depth, in_str, escape = [], [], 0, False, False
    for ch in rest:
        if in_str:
            buf.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            buf.append(ch)
        elif ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


_MEM_RE = re.compile(r"^(.*?)\(\s*([A-Za-z0-9_$]+)\s*\)$")


def _parse_mem(operand: str):
    """``off(reg)`` -> (offset_expr, reg), or None."""
    m = _MEM_RE.match(operand.strip())
    if not m:
        return None
    off = m.group(1).strip() or "0"
    try:
        reg = reg_number(m.group(2))
    except KeyError:
        return None
    return off, reg


def _unescape(text: str) -> bytes:
    body = text.strip()
    if not (body.startswith('"') and body.endswith('"')):
        raise ValueError("expected a quoted string, got %r" % text)
    return body[1:-1].encode().decode("unicode_escape").encode("latin-1")


# --------------------------------------------------------------------------
# Expression evaluation
# --------------------------------------------------------------------------

_HI_RE = re.compile(r"%hi\((.*)\)$", re.I)
_LO_RE = re.compile(r"%lo\((.*)\)$", re.I)


def _atom(token: str, symbols: dict, line_no: int) -> int:
    token = token.strip()
    if not token:
        return 0
    if _NUM_RE.match(token):
        neg = token.startswith("-")
        body = token.lstrip("+-")
        if body.lower().startswith("0x"):
            val = int(body, 16)
        elif body.lower().startswith("0b"):
            val = int(body, 2)
        else:
            val = int(body, 10)
        return -val if neg else val
    if len(token) == 3 and token[0] == "'" and token[2] == "'":
        return ord(token[1])
    if token in symbols:
        return symbols[token]
    if token in CSR_ADDRS:
        return CSR_ADDRS[token]
    raise AsmError(line_no, "undefined symbol %r" % token)


def _eval(expr: str, symbols: dict, line_no: int) -> int:
    expr = expr.strip()
    m = _HI_RE.match(expr)
    if m:
        value = _eval(m.group(1), symbols, line_no) & MASK32
        # %hi must compensate for the sign extension of the paired %lo.
        return ((value + 0x800) >> 12) & 0xFFFFF
    m = _LO_RE.match(expr)
    if m:
        value = _eval(m.group(1), symbols, line_no)
        low = value & 0xFFF
        return low - 0x1000 if low & 0x800 else low
    tokens = re.split(r"([+\-])", expr)
    total = _atom(tokens[0], symbols, line_no)
    i = 1
    while i + 1 < len(tokens):
        op, operand = tokens[i], tokens[i + 1]
        value = _atom(operand, symbols, line_no)
        total = total + value if op == "+" else total - value
        i += 2
    return total


def _split_hi_lo(value: int):
    """Split a 32-bit constant into the lui/addi pair that rebuilds it."""
    value &= MASK32
    hi = (value + 0x800) >> 12 & 0xFFFFF
    lo = value & 0xFFF
    if lo & 0x800:
        lo -= 0x1000
    return hi, lo


# --------------------------------------------------------------------------
# Pseudo-instruction expansion
# --------------------------------------------------------------------------

PSEUDO_2REG = {
    "mv": ("addi", 0), "not": ("xori", -1), "neg": ("sub", None),
    "seqz": ("sltiu", 1), "snez": ("sltu", None), "sltz": ("slt", None),
    "sgtz": ("slt", None),
}
BRANCH_ZERO = {"beqz": "beq", "bnez": "bne", "blez": "bge", "bgez": "bge",
               "bltz": "blt", "bgtz": "blt"}
BRANCH_SWAP = {"bgt": "blt", "ble": "bge", "bgtu": "bltu", "bleu": "bgeu"}
CSR_PSEUDO = {"csrr": "csrrs", "csrw": "csrrw", "csrs": "csrrs",
              "csrc": "csrrc", "csrwi": "csrrwi", "csrsi": "csrrsi",
              "csrci": "csrrci"}


def _li_words(rd: int, value: int) -> list:
    value = value & MASK32
    signed = value - (1 << 32) if value & 0x80000000 else value
    if -2048 <= signed < 2048:
        return [("addi", dict(rd=rd, rs1=0, imm=signed))]
    hi, lo = _split_hi_lo(value)
    seq = [("lui", dict(rd=rd, imm=hi << 12))]
    if lo:
        seq.append(("addi", dict(rd=rd, rs1=rd, imm=lo)))
    return seq


def _pseudo_size(mnemonic: str, operands: list, symbols: dict,
                 line_no: int) -> int:
    """Instruction count of a pseudo-op, needed by pass 1."""
    if mnemonic == "li":
        try:
            value = _eval(operands[1], symbols, line_no)
        except AsmError:
            return 2  # forward reference: assume the worst case
        return len(_li_words(0, value))
    if mnemonic in ("la", "call", "tail"):
        return 2
    return 1


# --------------------------------------------------------------------------
# Assembler
# --------------------------------------------------------------------------

def assemble(source: str, text_base: int = TEXT_BASE,
             data_base: int = DATA_BASE) -> Program:
    prog = Program(text_base=text_base, data_base=data_base)
    symbols: dict = {}
    lines = source.splitlines()

    # ---- pre-pass: .equ constants ---------------------------------------
    for line_no, raw in enumerate(lines, 1):
        body = _strip_comment(raw)
        m = _LABEL_RE.match(body)
        if m:
            body = body[m.end():].strip()
        if not body:
            continue
        parts = body.split(None, 1)
        if parts[0].lower() in (".equ", ".set", ".define"):
            ops = _split_operands(parts[1] if len(parts) > 1 else "")
            if len(ops) != 2:
                raise AsmError(line_no, ".equ needs NAME, value")
            symbols[ops[0].strip()] = _eval(ops[1], symbols, line_no)

    # ---- pass 1: lay out sections and collect labels ---------------------
    parsed = []  # (section, addr, line_no, mnemonic, operands, text)
    text_pc, data_pc = text_base, data_base
    section = "text"
    for line_no, raw in enumerate(lines, 1):
        body = _strip_comment(raw)
        while True:
            m = _LABEL_RE.match(body)
            if not m:
                break
            label = m.group(1)
            if label in symbols and label not in ("_start",):
                raise AsmError(line_no, "duplicate label %r" % label)
            symbols[label] = text_pc if section == "text" else data_pc
            body = body[m.end():].strip()
        if not body:
            continue
        parts = body.split(None, 1)
        mnemonic = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""
        operands = _split_operands(rest)

        if mnemonic in (".equ", ".set", ".define", ".globl", ".global",
                        ".type", ".size", ".file", ".option"):
            continue
        if mnemonic in (".text", ".data", ".bss", ".rodata", ".section"):
            if mnemonic == ".section":
                target = operands[0].strip().lstrip(".") if operands else "text"
                section = "data" if target in ("data", "rodata", "bss") else "text"
            else:
                section = "data" if mnemonic in (".data", ".bss", ".rodata") else "text"
            continue
        if mnemonic == ".org":
            addr = _eval(operands[0], symbols, line_no)
            if section == "text":
                text_pc = text_base + addr if addr < text_base else addr
            else:
                data_pc = data_base + addr if addr < data_base else addr
            continue

        if section == "data" or mnemonic in (".word", ".half", ".byte",
                                             ".string", ".asciz", ".ascii",
                                             ".space", ".zero", ".align",
                                             ".balign"):
            size = _data_size(mnemonic, operands, symbols, line_no,
                              data_pc if section == "data" else text_pc)
            parsed.append((section, data_pc if section == "data" else text_pc,
                           line_no, mnemonic, operands, body))
            if section == "data":
                data_pc += size
            else:
                text_pc += size
            continue

        count = _pseudo_size(mnemonic, operands, symbols, line_no)
        parsed.append(("text", text_pc, line_no, mnemonic, operands, body))
        text_pc += 4 * count

    # ---- pass 2: emit ----------------------------------------------------
    text_end, data_end = text_base, data_base
    text_words: dict = {}
    data_bytes: dict = {}
    for section, addr, line_no, mnemonic, operands, body in parsed:
        if mnemonic in (".word", ".half", ".byte", ".string", ".asciz",
                        ".ascii", ".space", ".zero", ".align", ".balign"):
            chunk = _emit_data(mnemonic, operands, symbols, line_no, addr)
            for off, byte in enumerate(chunk):
                if section == "data":
                    data_bytes[addr + off] = byte
                else:
                    data_bytes[addr + off] = byte
            end = addr + len(chunk)
            if section == "data":
                data_end = max(data_end, end)
            else:
                text_end = max(text_end, end)
            continue
        words = _emit_instruction(mnemonic, operands, symbols, line_no, addr, body)
        for i, word in enumerate(words):
            pc = addr + 4 * i
            text_words[pc] = word
            sl = SourceLine(addr=pc, line_no=line_no, text=body, word=word)
            prog.source[pc] = sl
            prog.listing.append(sl)
        text_end = max(text_end, addr + 4 * len(words))

    prog.symbols = symbols
    prog.text = bytearray(max(0, text_end - text_base))
    for pc, word in text_words.items():
        off = pc - text_base
        prog.text[off:off + 4] = word.to_bytes(4, "little")
    for addr, byte in data_bytes.items():
        if addr >= data_base:
            data_end = max(data_end, addr + 1)
    prog.data = bytearray(max(0, data_end - data_base))
    for addr, byte in data_bytes.items():
        if addr >= data_base:
            prog.data[addr - data_base] = byte
        else:
            off = addr - text_base
            if 0 <= off < len(prog.text):
                prog.text[off] = byte
    prog.listing.sort(key=lambda sl: sl.addr)
    return prog


def _data_size(mnemonic: str, operands: list, symbols: dict, line_no: int,
               addr: int) -> int:
    if mnemonic == ".word":
        return 4 * len(operands)
    if mnemonic == ".half":
        return 2 * len(operands)
    if mnemonic == ".byte":
        return len(operands)
    if mnemonic in (".string", ".asciz"):
        return sum(len(_unescape(o)) + 1 for o in operands)
    if mnemonic == ".ascii":
        return sum(len(_unescape(o)) for o in operands)
    if mnemonic in (".space", ".zero"):
        return _eval(operands[0], symbols, line_no)
    if mnemonic in (".align", ".balign"):
        n = _eval(operands[0], symbols, line_no)
        boundary = (1 << n) if mnemonic == ".align" else n
        if boundary <= 0:
            return 0
        return (-addr) % boundary
    raise AsmError(line_no, "unknown directive %s" % mnemonic)


def _emit_data(mnemonic: str, operands: list, symbols: dict, line_no: int,
               addr: int) -> bytes:
    try:
        if mnemonic == ".word":
            return b"".join((_eval(o, symbols, line_no) & MASK32).to_bytes(4, "little")
                            for o in operands)
        if mnemonic == ".half":
            return b"".join((_eval(o, symbols, line_no) & 0xFFFF).to_bytes(2, "little")
                            for o in operands)
        if mnemonic == ".byte":
            return bytes(_eval(o, symbols, line_no) & 0xFF for o in operands)
        if mnemonic in (".string", ".asciz"):
            return b"".join(_unescape(o) + b"\x00" for o in operands)
        if mnemonic == ".ascii":
            return b"".join(_unescape(o) for o in operands)
        if mnemonic in (".space", ".zero"):
            return bytes(_eval(operands[0], symbols, line_no))
        if mnemonic in (".align", ".balign"):
            return bytes(_data_size(mnemonic, operands, symbols, line_no, addr))
    except AsmError:
        raise
    except Exception as exc:  # pragma: no cover - malformed input
        raise AsmError(line_no, str(exc))
    raise AsmError(line_no, "unknown directive %s" % mnemonic)


def _reg(token: str, line_no: int) -> int:
    try:
        return reg_number(token)
    except KeyError:
        raise AsmError(line_no, "expected a register, got %r" % token)


def _emit_instruction(mnemonic: str, operands: list, symbols: dict,
                      line_no: int, addr: int, body: str) -> list:
    try:
        return _emit_instruction_inner(mnemonic, operands, symbols, line_no, addr)
    except AsmError:
        raise
    except (IndexError, KeyError, ValueError) as exc:
        raise AsmError(line_no, "%s: %s" % (mnemonic, exc), body)


def _emit_instruction_inner(mnemonic: str, operands: list, symbols: dict,
                            line_no: int, addr: int) -> list:
    ev = lambda e: _eval(e, symbols, line_no)          # noqa: E731
    rg = lambda t: _reg(t, line_no)                    # noqa: E731

    # ---- pseudo-instructions --------------------------------------------
    if mnemonic == "nop":
        return [encode("nop")]
    if mnemonic == "li":
        rd = rg(operands[0])
        return [encode(n, **kw) for n, kw in _li_words(rd, ev(operands[1]))]
    if mnemonic == "la":
        rd = rg(operands[0])
        target = ev(operands[1]) & MASK32
        delta = (target - addr) & MASK32
        hi, lo = _split_hi_lo(delta)
        return [encode("auipc", rd=rd, imm=hi << 12),
                encode("addi", rd=rd, rs1=rd, imm=lo)]
    if mnemonic in ("call", "tail"):
        link = 1 if mnemonic == "call" else 0
        scratch = 1 if mnemonic == "call" else 6  # ra / t1
        target = ev(operands[0]) & MASK32
        delta = (target - addr) & MASK32
        hi, lo = _split_hi_lo(delta)
        return [encode("auipc", rd=scratch, imm=hi << 12),
                encode("jalr", rd=link, rs1=scratch, imm=lo)]
    if mnemonic in PSEUDO_2REG:
        rd, rs = rg(operands[0]), rg(operands[1])
        if mnemonic == "mv":
            return [encode("addi", rd=rd, rs1=rs, imm=0)]
        if mnemonic == "not":
            return [encode("xori", rd=rd, rs1=rs, imm=-1)]
        if mnemonic == "neg":
            return [encode("sub", rd=rd, rs1=0, rs2=rs)]
        if mnemonic == "seqz":
            return [encode("sltiu", rd=rd, rs1=rs, imm=1)]
        if mnemonic == "snez":
            return [encode("sltu", rd=rd, rs1=0, rs2=rs)]
        if mnemonic == "sltz":
            return [encode("slt", rd=rd, rs1=rs, rs2=0)]
        if mnemonic == "sgtz":
            return [encode("slt", rd=rd, rs1=0, rs2=rs)]
    if mnemonic in BRANCH_ZERO:
        rs = rg(operands[0])
        offset = ev(operands[1]) - addr
        real = BRANCH_ZERO[mnemonic]
        if mnemonic in ("blez", "bgtz"):  # compare against zero on the left
            return [encode(real, rs1=0, rs2=rs, imm=offset)]
        return [encode(real, rs1=rs, rs2=0, imm=offset)]
    if mnemonic in BRANCH_SWAP:
        rs1, rs2 = rg(operands[0]), rg(operands[1])
        offset = ev(operands[2]) - addr
        return [encode(BRANCH_SWAP[mnemonic], rs1=rs2, rs2=rs1, imm=offset)]
    if mnemonic == "j":
        return [encode("jal", rd=0, imm=ev(operands[0]) - addr)]
    if mnemonic == "jr":
        return [encode("jalr", rd=0, rs1=rg(operands[0]), imm=0)]
    if mnemonic == "ret":
        return [encode("jalr", rd=0, rs1=1, imm=0)]
    if mnemonic in CSR_PSEUDO:
        real = CSR_PSEUDO[mnemonic]
        if mnemonic == "csrr":
            return [encode(real, rd=rg(operands[0]), rs1=0,
                           csr=ev(operands[1]))]
        csr = ev(operands[0])
        if real.endswith("i"):
            return [encode(real, rd=0, csr=csr, imm=ev(operands[1]))]
        return [encode(real, rd=0, csr=csr, rs1=rg(operands[1]))]

    # ---- atomics (the .aq / .rl ordering suffixes are accepted and
    # ignored: this model has a single in-order memory port per core) ----
    base = mnemonic
    for suffix in (".aqrl", ".aq", ".rl"):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break
    if base in AMO_F5:
        rd = rg(operands[0])
        if base == "lr.w":
            mem = _parse_mem(operands[1])
            rs1 = mem[1] if mem else rg(operands[1])
            return [encode(base, rd=rd, rs1=rs1, rs2=0)]
        rs2 = rg(operands[1])
        mem = _parse_mem(operands[2])
        rs1 = mem[1] if mem else rg(operands[2])
        return [encode(base, rd=rd, rs1=rs1, rs2=rs2)]

    # ---- real instructions ----------------------------------------------
    if mnemonic in ("ecall", "ebreak", "mret", "sret", "wfi", "fence",
                    "fence.i"):
        return [encode(mnemonic)]
    if mnemonic in ("lui", "auipc"):
        rd = rg(operands[0])
        value = ev(operands[1])
        # A bare number is the 20-bit immediate, as in GNU as.
        imm = value << 12 if -0x80000 <= value < 0x100000 else value
        return [encode(mnemonic, rd=rd, imm=imm)]
    if mnemonic == "jal":
        if len(operands) == 1:
            return [encode("jal", rd=1, imm=ev(operands[0]) - addr)]
        return [encode("jal", rd=rg(operands[0]), imm=ev(operands[1]) - addr)]
    if mnemonic == "jalr":
        if len(operands) == 1:
            mem = _parse_mem(operands[0])
            if mem:
                return [encode("jalr", rd=1, rs1=mem[1], imm=ev(mem[0]))]
            return [encode("jalr", rd=1, rs1=rg(operands[0]), imm=0)]
        if len(operands) == 2:
            mem = _parse_mem(operands[1])
            if mem:
                return [encode("jalr", rd=rg(operands[0]), rs1=mem[1],
                               imm=ev(mem[0]))]
            return [encode("jalr", rd=rg(operands[0]), rs1=rg(operands[1]),
                           imm=0)]
        return [encode("jalr", rd=rg(operands[0]), rs1=rg(operands[1]),
                       imm=ev(operands[2]))]
    if mnemonic in BRANCH_F3:
        rs1, rs2 = rg(operands[0]), rg(operands[1])
        return [encode(mnemonic, rs1=rs1, rs2=rs2, imm=ev(operands[2]) - addr)]
    if mnemonic in LOAD_F3:
        rd = rg(operands[0])
        mem = _parse_mem(operands[1])
        if mem:
            return [encode(mnemonic, rd=rd, rs1=mem[1], imm=ev(mem[0]))]
        # lw rd, symbol  ->  auipc + load
        target = ev(operands[1]) & MASK32
        delta = (target - addr) & MASK32
        hi, lo = _split_hi_lo(delta)
        return [encode("auipc", rd=rd, imm=hi << 12),
                encode(mnemonic, rd=rd, rs1=rd, imm=lo)]
    if mnemonic in STORE_F3:
        rs2 = rg(operands[0])
        mem = _parse_mem(operands[1])
        if mem:
            return [encode(mnemonic, rs2=rs2, rs1=mem[1], imm=ev(mem[0]))]
        if len(operands) < 3:
            raise AsmError(line_no,
                           "%s to a symbol needs a scratch register" % mnemonic)
        tmp = rg(operands[2])
        target = ev(operands[1]) & MASK32
        delta = (target - addr) & MASK32
        hi, lo = _split_hi_lo(delta)
        return [encode("auipc", rd=tmp, imm=hi << 12),
                encode(mnemonic, rs1=tmp, rs2=rs2, imm=lo)]
    if mnemonic in ("slli", "srli", "srai"):
        return [encode(mnemonic, rd=rg(operands[0]), rs1=rg(operands[1]),
                       imm=ev(operands[2]) & 31)]
    if mnemonic in IMM_PLAIN_F3:
        return [encode(mnemonic, rd=rg(operands[0]), rs1=rg(operands[1]),
                       imm=ev(operands[2]))]
    if mnemonic in REG_OPS:
        return [encode(mnemonic, rd=rg(operands[0]), rs1=rg(operands[1]),
                       rs2=rg(operands[2]))]
    if mnemonic in ("csrrw", "csrrs", "csrrc"):
        return [encode(mnemonic, rd=rg(operands[0]), csr=ev(operands[1]),
                       rs1=rg(operands[2]))]
    if mnemonic in ("csrrwi", "csrrsi", "csrrci"):
        return [encode(mnemonic, rd=rg(operands[0]), csr=ev(operands[1]),
                       imm=ev(operands[2]))]
    raise AsmError(line_no, "unknown instruction %r" % mnemonic)


def assemble_file(path: str, **kwargs) -> Program:
    with open(path, "r", encoding="utf-8") as fh:
        return assemble(fh.read(), **kwargs)
