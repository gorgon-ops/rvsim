"""Set-associative caches with MESI line states, used for L1I, L1D and the inclusive L2."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

MASK32 = 0xFFFFFFFF

INVALID = "I"
SHARED = "S"
EXCLUSIVE = "E"
MODIFIED = "M"

STATE_LABELS = {INVALID: "Invalid", SHARED: "Shared",
                EXCLUSIVE: "Exclusive", MODIFIED: "Modified"}


@dataclass
class CacheConfig:
    name: str = "L1D"
    size: int = 1024            # bytes
    block_size: int = 32        # bytes per line
    assoc: int = 2              # ways per set
    latency: int = 1            # cycles to look up
    replacement: str = "lru"    # lru | fifo | random
    write_back: bool = True
    write_allocate: bool = True

    @property
    def num_sets(self) -> int:
        sets = self.size // (self.block_size * self.assoc)
        if sets < 1:
            raise ValueError("%s: size too small for %d-way x %d B lines"
                             % (self.name, self.assoc, self.block_size))
        return sets

    def describe(self) -> str:
        return ("%s: %dB, %d sets x %d ways x %dB, %s, %s, %d-cycle"
                % (self.name, self.size, self.num_sets, self.assoc,
                   self.block_size, self.replacement.upper(),
                   "write-back" if self.write_back else "write-through",
                   self.latency))


@dataclass
class CacheLine:
    tag: int = 0
    state: str = INVALID
    data: bytearray = field(default_factory=bytearray)
    dirty: bool = False
    stamp: int = 0          # last touch (LRU) or fill time (FIFO)
    sharers: set = field(default_factory=set)  # L2 only: cores holding a copy

    @property
    def valid(self) -> bool:
        return self.state != INVALID


@dataclass
class CacheStats:
    reads: int = 0
    writes: int = 0
    read_hits: int = 0
    write_hits: int = 0
    misses: int = 0
    evictions: int = 0
    writebacks: int = 0
    invalidations: int = 0
    upgrades: int = 0

    @property
    def accesses(self) -> int:
        return self.reads + self.writes

    @property
    def hits(self) -> int:
        return self.read_hits + self.write_hits

    @property
    def hit_rate(self) -> float:
        return self.hits / self.accesses if self.accesses else 0.0

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["accesses"] = self.accesses
        d["hits"] = self.hits
        d["hit_rate"] = round(self.hit_rate, 4)
        return d


class Cache:
    def __init__(self, config: CacheConfig, seed: int = 1):
        self.cfg = config
        self.sets = [[CacheLine(data=bytearray(config.block_size))
                      for _ in range(config.assoc)]
                     for _ in range(config.num_sets)]
        self.stats = CacheStats()
        self.rng = random.Random(seed)
        self.clock = 0

    # -- address breakdown -------------------------------------------------

    @property
    def offset_bits(self) -> int:
        return self.cfg.block_size.bit_length() - 1

    @property
    def index_bits(self) -> int:
        return self.cfg.num_sets.bit_length() - 1

    def block_addr(self, addr: int) -> int:
        return addr & ~(self.cfg.block_size - 1) & MASK32

    def split(self, addr: int):
        offset = addr & (self.cfg.block_size - 1)
        index = (addr >> self.offset_bits) % self.cfg.num_sets
        tag = addr >> (self.offset_bits + self.index_bits)
        return tag, index, offset

    # -- lookup ------------------------------------------------------------

    def find(self, addr: int):
        tag, index, _ = self.split(addr)
        for line in self.sets[index]:
            if line.valid and line.tag == tag:
                return line
        return None

    def touch(self, line: CacheLine) -> None:
        self.clock += 1
        if self.cfg.replacement == "lru":
            line.stamp = self.clock

    def victim(self, index: int) -> CacheLine:
        ways = self.sets[index]
        for line in ways:
            if not line.valid:
                return line
        if self.cfg.replacement == "random":
            return self.rng.choice(ways)
        return min(ways, key=lambda ln: ln.stamp)  # lru and fifo both use stamp

    def install(self, addr: int, data: bytes, state: str):
        tag, index, _ = self.split(addr)
        line = self.find(addr)
        evicted = None
        if line is None:
            line = self.victim(index)
            if line.valid:
                self.stats.evictions += 1
                evicted = (self.line_addr(line, index), bytes(line.data), line.dirty)
                if line.dirty:
                    self.stats.writebacks += 1
            line.tag = tag
            line.dirty = False
            line.sharers = set()
        line.data = bytearray(data)
        line.state = state
        self.clock += 1
        line.stamp = self.clock
        return evicted

    def line_addr(self, line: CacheLine, index: int) -> int:
        return ((line.tag << (self.offset_bits + self.index_bits)) |
                (index << self.offset_bits)) & MASK32

    def invalidate(self, addr: int):
        line = self.find(addr)
        if line is None:
            return None
        self.stats.invalidations += 1
        payload = bytes(line.data) if line.dirty else None
        line.state = INVALID
        line.dirty = False
        line.sharers = set()
        return payload

    def downgrade(self, addr: int):
        """M/E -> S for another core's read. Returns dirty data."""
        line = self.find(addr)
        if line is None:
            return None
        payload = bytes(line.data) if line.dirty else None
        if line.state in (MODIFIED, EXCLUSIVE):
            line.state = SHARED
            line.dirty = False
        return payload

    # -- data access on a resident line ------------------------------------

    def read_bytes(self, addr: int, width: int) -> bytes:
        line = self.find(addr)
        if line is None:
            raise RuntimeError("%s: read of 0x%08x with no resident line"
                               % (self.cfg.name, addr))
        offset = addr & (self.cfg.block_size - 1)
        self.touch(line)
        return bytes(line.data[offset:offset + width])

    def write_bytes(self, addr: int, payload: bytes) -> None:
        line = self.find(addr)
        if line is None:
            raise RuntimeError("%s: write to 0x%08x with no resident line"
                               % (self.cfg.name, addr))
        offset = addr & (self.cfg.block_size - 1)
        line.data[offset:offset + len(payload)] = payload
        line.dirty = True
        line.state = MODIFIED
        self.touch(line)

    # -- inspection for the trace / UI -------------------------------------

    def snapshot(self) -> list:
        out = []
        for index, ways in enumerate(self.sets):
            for way, line in enumerate(ways):
                out.append({
                    "set": index,
                    "way": way,
                    "state": line.state,
                    "tag": line.tag,
                    "addr": self.line_addr(line, index) if line.valid else None,
                    "dirty": line.dirty,
                    "sharers": sorted(line.sharers),
                })
        return out

    def describe(self) -> str:
        return self.cfg.describe()
