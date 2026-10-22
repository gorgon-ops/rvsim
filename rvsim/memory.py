"""Sparse physical memory and devices. Above :data:`MMIO_BASE` bypasses the caches."""

from __future__ import annotations

MASK32 = 0xFFFFFFFF
PAGE_BITS = 12
PAGE_SIZE = 1 << PAGE_BITS

# ---- device map ----------------------------------------------------------
MMIO_BASE = 0x02000000
CLINT_MSIP = 0x02000000        # 4 bytes per hart; write 1 to post an IPI
CLINT_MTIMECMP = 0x02004000    # 8 bytes per hart
CLINT_MTIME = 0x0200BFF8       # 8 bytes, free-running
UART_TX = 0x10000000
HALT = 0x10000004              # word write halts the hart with that exit code
MMIO_END = 0x10001000


def is_mmio(addr: int) -> bool:
    return MMIO_BASE <= addr < MMIO_END


class PhysicalMemory:
    def __init__(self):
        self.pages: dict = {}

    # -- raw byte access ---------------------------------------------------

    def _page(self, addr: int) -> bytearray:
        index = addr >> PAGE_BITS
        page = self.pages.get(index)
        if page is None:
            page = bytearray(PAGE_SIZE)
            self.pages[index] = page
        return page

    def load_bytes(self, addr: int, payload: bytes) -> None:
        for i, byte in enumerate(payload):
            a = (addr + i) & MASK32
            self._page(a)[a & (PAGE_SIZE - 1)] = byte

    def read_block(self, addr: int, size: int) -> bytearray:
        out = bytearray(size)
        for i in range(size):
            a = (addr + i) & MASK32
            page = self.pages.get(a >> PAGE_BITS)
            out[i] = page[a & (PAGE_SIZE - 1)] if page else 0
        return out

    def write_block(self, addr: int, payload: bytes) -> None:
        self.load_bytes(addr, payload)

    # -- word helpers ------------------------------------------------------

    def read(self, addr: int, width: int = 4) -> int:
        return int.from_bytes(self.read_block(addr, width), "little")

    def write(self, addr: int, value: int, width: int = 4) -> None:
        self.load_bytes(addr, (value & ((1 << (8 * width)) - 1)).to_bytes(width, "little"))

    def nonzero_words(self, base: int, count: int):
        out = []
        for i in range(count):
            addr = base + 4 * i
            value = self.read(addr, 4)
            if value:
                out.append((addr, value))
        return out


class Devices:
    def __init__(self, num_harts: int = 1):
        self.num_harts = num_harts
        self.mtime = 0
        self.mtimecmp = [0xFFFFFFFFFFFFFFFF] * num_harts
        self.msip = [0] * num_harts
        self.console = []
        self.halted: dict = {}
        self.events = []

    def tick(self) -> None:
        self.mtime += 1

    def timer_pending(self, hart: int) -> bool:
        return self.mtime >= self.mtimecmp[hart]

    def software_pending(self, hart: int) -> bool:
        return bool(self.msip[hart])

    def read(self, addr: int, width: int, hart: int) -> int:
        if CLINT_MSIP <= addr < CLINT_MSIP + 4 * self.num_harts:
            return self.msip[(addr - CLINT_MSIP) // 4]
        if CLINT_MTIMECMP <= addr < CLINT_MTIMECMP + 8 * self.num_harts:
            index = (addr - CLINT_MTIMECMP) // 8
            high = (addr - CLINT_MTIMECMP) % 8 >= 4
            value = self.mtimecmp[index]
            return (value >> 32) & MASK32 if high else value & MASK32
        if CLINT_MTIME <= addr < CLINT_MTIME + 8:
            high = (addr - CLINT_MTIME) >= 4
            return (self.mtime >> 32) & MASK32 if high else self.mtime & MASK32
        return 0

    def write(self, addr: int, value: int, width: int, hart: int) -> None:
        if CLINT_MSIP <= addr < CLINT_MSIP + 4 * self.num_harts:
            target = (addr - CLINT_MSIP) // 4
            self.msip[target] = value & 1
            self.events.append(("ipi", hart, target, value & 1))
            return
        if CLINT_MTIMECMP <= addr < CLINT_MTIMECMP + 8 * self.num_harts:
            index = (addr - CLINT_MTIMECMP) // 8
            high = (addr - CLINT_MTIMECMP) % 8 >= 4
            current = self.mtimecmp[index]
            if high:
                self.mtimecmp[index] = (current & MASK32) | ((value & MASK32) << 32)
            else:
                self.mtimecmp[index] = (current & ~MASK32) | (value & MASK32)
            self.events.append(("mtimecmp", hart, index, self.mtimecmp[index]))
            return
        if addr == UART_TX:
            self.console.append(chr(value & 0xFF))
            return
        if addr == HALT:
            self.halted[hart] = value & MASK32
            return
