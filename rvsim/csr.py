"""M-mode CSRs and trap plumbing: mtvec handlers, CLINT timer, IPIs, synchronous
exceptions.
"""

from __future__ import annotations

MASK32 = 0xFFFFFFFF

# --------------------------------------------------------------------------
# CSR addresses
# --------------------------------------------------------------------------

MSTATUS = 0x300
MISA = 0x301
MEDELEG = 0x302
MIDELEG = 0x303
MIE = 0x304
MTVEC = 0x305
MCOUNTEREN = 0x306
MSCRATCH = 0x340
MEPC = 0x341
MCAUSE = 0x342
MTVAL = 0x343
MIP = 0x344
MCYCLE = 0xB00
MINSTRET = 0xB02
MCYCLEH = 0xB80
MINSTRETH = 0xB82
CYCLE = 0xC00
TIME = 0xC01
INSTRET = 0xC02
MVENDORID = 0xF11
MARCHID = 0xF12
MIMPID = 0xF13
MHARTID = 0xF14

CSR_NAMES = {
    MSTATUS: "mstatus", MISA: "misa", MEDELEG: "medeleg", MIDELEG: "mideleg",
    MIE: "mie", MTVEC: "mtvec", MCOUNTEREN: "mcounteren", MSCRATCH: "mscratch",
    MEPC: "mepc", MCAUSE: "mcause", MTVAL: "mtval", MIP: "mip",
    MCYCLE: "mcycle", MINSTRET: "minstret", MCYCLEH: "mcycleh",
    MINSTRETH: "minstreth", CYCLE: "cycle", TIME: "time", INSTRET: "instret",
    MVENDORID: "mvendorid", MARCHID: "marchid", MIMPID: "mimpid",
    MHARTID: "mhartid",
}
CSR_ADDRS = {v: k for k, v in CSR_NAMES.items()}

READ_ONLY = {MVENDORID, MARCHID, MIMPID, MHARTID, MISA,
             CYCLE, TIME, INSTRET}

# mstatus fields
MSTATUS_MIE = 1 << 3
MSTATUS_MPIE = 1 << 7
MSTATUS_MPP = 0b11 << 11

# mie / mip bits
MIP_MSIP = 1 << 3   # inter-processor interrupt
MIP_MTIP = 1 << 7
MIP_MEIP = 1 << 11

INTERRUPT_BITS = [
    (MIP_MEIP, 11, "external"),
    (MIP_MSIP, 3, "software"),
    (MIP_MTIP, 7, "timer"),
]

# --------------------------------------------------------------------------
# Trap causes
# --------------------------------------------------------------------------

CAUSE_INSTR_MISALIGNED = 0
CAUSE_INSTR_ACCESS_FAULT = 1
CAUSE_ILLEGAL_INSTRUCTION = 2
CAUSE_BREAKPOINT = 3
CAUSE_LOAD_MISALIGNED = 4
CAUSE_LOAD_ACCESS_FAULT = 5
CAUSE_STORE_MISALIGNED = 6
CAUSE_STORE_ACCESS_FAULT = 7
CAUSE_ECALL_M = 11

EXCEPTION_NAMES = {
    CAUSE_INSTR_MISALIGNED: "instruction address misaligned",
    CAUSE_INSTR_ACCESS_FAULT: "instruction access fault",
    CAUSE_ILLEGAL_INSTRUCTION: "illegal instruction",
    CAUSE_BREAKPOINT: "breakpoint",
    CAUSE_LOAD_MISALIGNED: "load address misaligned",
    CAUSE_LOAD_ACCESS_FAULT: "load access fault",
    CAUSE_STORE_MISALIGNED: "store address misaligned",
    CAUSE_STORE_ACCESS_FAULT: "store access fault",
    CAUSE_ECALL_M: "environment call from M-mode",
}
INTERRUPT_NAMES = {3: "machine software interrupt",
                   7: "machine timer interrupt",
                   11: "machine external interrupt"}


def cause_name(cause: int) -> str:
    if cause & 0x80000000:
        return INTERRUPT_NAMES.get(cause & 0x7FFFFFFF, "interrupt %d" % (cause & 0x7FFFFFFF))
    return EXCEPTION_NAMES.get(cause, "exception %d" % cause)


class Trap(Exception):
    def __init__(self, cause: int, tval: int = 0):
        super().__init__(cause_name(cause))
        self.cause = cause
        self.tval = tval & MASK32


class CSRFile:
    def __init__(self, hart_id: int = 0):
        self.hart_id = hart_id
        self.regs = {addr: 0 for addr in CSR_NAMES}
        # RV32IM: I and M bits, MXL=1 (32-bit).
        self.regs[MISA] = (1 << 30) | (1 << 8) | (1 << 12)
        self.regs[MHARTID] = hart_id
        self.regs[MVENDORID] = 0
        self.regs[MARCHID] = 0x5256  # "RV"
        self.regs[MIMPID] = 1
        self.regs[MSTATUS] = MSTATUS_MPP  # previous privilege = M
        self.wfi = False

    # -- raw access used by the machine itself (bypasses read-only rules) ---

    def get(self, addr: int) -> int:
        return self.regs.get(addr, 0) & MASK32

    def set(self, addr: int, value: int) -> None:
        self.regs[addr] = value & MASK32

    # -- architectural access performed by csrr*/csrw* instructions --------

    def read(self, addr: int) -> int:
        if addr not in self.regs:
            raise Trap(CAUSE_ILLEGAL_INSTRUCTION, addr)
        if addr == CYCLE:
            return self.get(MCYCLE)
        if addr == INSTRET:
            return self.get(MINSTRET)
        if addr == TIME:
            return self.get(TIME)
        return self.get(addr)

    def write(self, addr: int, value: int) -> None:
        if addr not in self.regs:
            raise Trap(CAUSE_ILLEGAL_INSTRUCTION, addr)
        if addr in READ_ONLY:
            raise Trap(CAUSE_ILLEGAL_INSTRUCTION, addr)
        if addr == MTVEC:
            value &= ~0b10  # only direct (0) and vectored (1) modes exist
        if addr == MIP:
            # Software may only touch MSIP; the timer and external bits belong to the
            # CLINT/PLIC.
            value = (self.get(MIP) & ~MIP_MSIP) | (value & MIP_MSIP)
        self.set(addr, value)

    # -- interrupt arbitration --------------------------------------------

    @property
    def global_ie(self) -> bool:
        return bool(self.get(MSTATUS) & MSTATUS_MIE)

    def pending_interrupt(self) -> int | None:
        """Highest-priority enabled pending cause. A WFI-parked hart wakes even with MIE clear."""
        ready = self.get(MIP) & self.get(MIE)
        if not ready:
            return None
        if not self.global_ie and not self.wfi:
            return None
        for bit, code, _label in INTERRUPT_BITS:
            if ready & bit:
                return 0x80000000 | code
        return None

    def raise_line(self, bit: int, active: bool) -> None:
        mip = self.get(MIP)
        self.set(MIP, (mip | bit) if active else (mip & ~bit))

    # -- trap entry / return ----------------------------------------------

    def enter_trap(self, cause: int, epc: int, tval: int) -> int:
        status = self.get(MSTATUS)
        mpie = MSTATUS_MPIE if (status & MSTATUS_MIE) else 0
        status = (status & ~(MSTATUS_MIE | MSTATUS_MPIE)) | mpie | MSTATUS_MPP
        self.set(MSTATUS, status)
        self.set(MEPC, epc & ~1)
        self.set(MCAUSE, cause)
        self.set(MTVAL, tval)
        self.wfi = False
        tvec = self.get(MTVEC)
        base, mode = tvec & ~0b11, tvec & 0b11
        if mode == 1 and (cause & 0x80000000):
            return (base + 4 * (cause & 0x7FFFFFFF)) & MASK32
        return base

    def exit_trap(self) -> int:
        """``mret``: restore MIE from MPIE and return to ``mepc``."""
        status = self.get(MSTATUS)
        mie = MSTATUS_MIE if (status & MSTATUS_MPIE) else 0
        status = (status & ~(MSTATUS_MIE | MSTATUS_MPIE)) | mie | MSTATUS_MPIE
        self.set(MSTATUS, status)
        return self.get(MEPC)

    def snapshot(self) -> dict:
        keep = (MSTATUS, MIE, MIP, MTVEC, MEPC, MCAUSE, MTVAL, MSCRATCH,
                MCYCLE, MINSTRET, MHARTID)
        return {CSR_NAMES[a]: self.get(a) for a in keep}
