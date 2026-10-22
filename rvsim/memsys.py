"""Private L1I/L1D per core, shared L2, DRAM, and a MESI snooping bus.

Misses are cycle-stepped through a small state machine, emitting an event at
each transition, so the visualiser can show a block travelling the hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .cache import (Cache, CacheConfig, EXCLUSIVE, INVALID, MODIFIED, SHARED)
from .memory import Devices, PhysicalMemory, is_mmio

MASK32 = 0xFFFFFFFF

# Request walk states.
S_L1 = "l1"
S_ARBITRATE = "arbitrate"
S_SNOOP = "snoop"
S_TRANSFER = "transfer"      # cache-to-cache flush from another core
S_L2 = "l2"
S_DRAM = "dram"
S_FILL = "fill"
S_MMIO = "mmio"
S_DONE = "done"

STATE_LABELS = {
    S_L1: "L1 lookup", S_ARBITRATE: "bus arbitration", S_SNOOP: "snooping",
    S_TRANSFER: "cache-to-cache transfer", S_L2: "L2 lookup",
    S_DRAM: "DRAM access", S_FILL: "refill", S_MMIO: "device access",
    S_DONE: "complete",
}


@dataclass
class MemoryConfig:
    l1i: CacheConfig = field(default_factory=lambda: CacheConfig(
        name="L1I", size=1024, block_size=32, assoc=2, latency=1))
    l1d: CacheConfig = field(default_factory=lambda: CacheConfig(
        name="L1D", size=1024, block_size=32, assoc=2, latency=1))
    l2: CacheConfig = field(default_factory=lambda: CacheConfig(
        name="L2", size=8192, block_size=32, assoc=4, latency=8))
    dram_latency: int = 40
    bus_latency: int = 2
    mmio_latency: int = 2

    def describe(self) -> list:
        return [self.l1i.describe(), self.l1d.describe(), self.l2.describe(),
                "DRAM: %d-cycle, bus: %d-cycle, MMIO: %d-cycle"
                % (self.dram_latency, self.bus_latency, self.mmio_latency)]


@dataclass
class MemRequest:
    rid: int
    core: int
    kind: str
    addr: int
    width: int = 4
    store_data: bytes = b""
    state: str = S_L1
    timer: int = 0
    value: int = 0
    done: bool = False
    hit: bool = True      # was it satisfied by the L1?
    level: str = "L1"     # deepest level reached, for stats
    bus_op: str = ""      # BusRd | BusRdX | BusUpgr
    issued_cycle: int = 0
    finished_cycle: int = 0
    supplied: bytes | None = None
    shared_elsewhere: bool = False
    amo_op: str = ""
    amo_operand: int = 0
    sc_failed: bool = False

    @property
    def is_store(self) -> bool:
        return self.kind == "store"

    @property
    def needs_write(self) -> bool:
        return self.kind in ("store", "sc", "amo")

    @property
    def latency(self) -> int:
        return max(1, self.finished_cycle - self.issued_cycle)


class MemorySystem:
    def __init__(self, num_cores: int, config: MemoryConfig | None = None,
                 devices: Devices | None = None, emit=None, seed: int = 1):
        self.cfg = config or MemoryConfig()
        self.num_cores = num_cores
        self.dram = PhysicalMemory()
        self.devices = devices or Devices(num_cores)
        self.emit = emit or (lambda *a, **k: None)
        self.l1i = [Cache(_clone(self.cfg.l1i, i), seed + i) for i in range(num_cores)]
        self.l1d = [Cache(_clone(self.cfg.l1d, i), seed + 100 + i) for i in range(num_cores)]
        self.l2 = Cache(self.cfg.l2, seed + 200)
        self.pending: list = []
        self.bus_owner = None      # rid currently holding the bus
        self.bus_busy_until = -1
        self.next_rid = 1
        self.cycle = 0
        self.bus_traffic = 0
        self.bus_bytes = 0
        self.coherence_log = []
        self.last_grant_core = -1
        # LR/SC reservation per core: the reserved block address, or None.
        self.reservations = [None] * num_cores

    # ------------------------------------------------------------------
    # Issue / completion
    # ------------------------------------------------------------------

    def issue(self, core: int, kind: str, addr: int, width: int = 4,
              store_data: bytes = b"", amo_op: str = "",
              amo_operand: int = 0) -> MemRequest:
        req = MemRequest(rid=self.next_rid, core=core, kind=kind,
                         addr=addr & MASK32, width=width,
                         store_data=bytes(store_data),
                         amo_op=amo_op, amo_operand=amo_operand,
                         issued_cycle=self.cycle)
        self.next_rid += 1
        if is_mmio(req.addr):
            req.state = S_MMIO
            req.timer = max(0, self.cfg.mmio_latency - 1)
            req.level = "MMIO"
            self.emit("mem", req=req.rid, core=core, kind=kind,
                      addr=req.addr, frm="CORE%d" % core, to="MMIO",
                      note="uncached device access")
        else:
            cache = self._l1(req)
            req.state = S_L1
            # The issue cycle counts as the first lookup cycle, so a 1-cycle hit does
            # not stall.
            req.timer = max(0, cache.cfg.latency - 1)
            self.emit("mem", req=req.rid, core=core, kind=kind, addr=req.addr,
                      frm="CORE%d" % core, to=cache.cfg.name,
                      note="%s 0x%08x" % (kind, req.addr))
        self.pending.append(req)
        self._run(req)
        if req.done:
            self.pending.remove(req)
        return req

    def _l1(self, req: MemRequest) -> Cache:
        return self.l1i[req.core] if req.kind == "fetch" else self.l1d[req.core]

    # ------------------------------------------------------------------
    # Per-cycle advance
    # ------------------------------------------------------------------

    def step(self, cycle: int) -> None:
        self.cycle = cycle
        for req in list(self.pending):
            if req.done:
                continue
            if req.timer > 0:
                req.timer -= 1
            if req.timer == 0:
                self._run(req)
        self.pending = [r for r in self.pending if not r.done]

    def _run(self, req: MemRequest) -> None:
        guard = 0
        while not req.done and req.timer == 0:
            self._advance(req)
            guard += 1
            if guard > 32:  # pragma: no cover - would mean a cycle in the FSM
                raise RuntimeError("memory request %d is looping" % req.rid)

    def _advance(self, req: MemRequest) -> None:
        handler = {
            S_L1: self._do_l1, S_ARBITRATE: self._do_arbitrate,
            S_SNOOP: self._do_snoop, S_TRANSFER: self._do_transfer,
            S_L2: self._do_l2, S_DRAM: self._do_dram, S_FILL: self._do_fill,
            S_MMIO: self._do_mmio,
        }[req.state]
        handler(req)

    # -- L1 -------------------------------------------------------------

    def _do_l1(self, req: MemRequest) -> None:
        cache = self._l1(req)
        stats = cache.stats
        block = cache.block_addr(req.addr)
        line = cache.find(req.addr)

        # A store-conditional with no reservation fails without touching the bus:
        # the point of LR/SC.
        if req.kind == "sc" and self.reservations[req.core] != block:
            stats.writes += 1
            req.value = 1
            req.sc_failed = True
            req.hit = True
            req.state = S_DONE
            req.done = True
            req.finished_cycle = self.cycle
            self.emit("mem", req=req.rid, core=req.core, addr=req.addr,
                      frm=cache.cfg.name, to="CORE%d" % req.core, kind="sc",
                      note="sc.w FAILED: reservation lost", complete=True,
                      hit=True)
            return

        if req.needs_write:
            stats.writes += 1
        else:
            stats.reads += 1

        if line is not None and line.valid:
            if not req.needs_write:
                stats.read_hits += 1
                self._complete(req, cache, hit=True)
                return
            if line.state in (MODIFIED, EXCLUSIVE):
                stats.write_hits += 1
                self._complete(req, cache, hit=True)
                return
            # Shared line: we have the data, not the permission.
            stats.upgrades += 1
            req.bus_op = "BusUpgr"
        else:
            stats.misses += 1
            req.bus_op = "BusRdX" if req.needs_write else "BusRd"
        req.hit = False
        req.state = S_ARBITRATE
        req.timer = 0
        self.emit("mem", req=req.rid, core=req.core, kind=req.kind,
                  addr=req.addr, frm=cache.cfg.name, to="BUS",
                  note="%s miss -> %s" % (cache.cfg.name, req.bus_op),
                  miss=True)

    # -- bus ------------------------------------------------------------

    def _do_arbitrate(self, req: MemRequest) -> None:
        if self.bus_owner is not None and self.bus_owner != req.rid:
            req.timer = 1  # bus is busy; retry next cycle
            return
        if self.bus_owner is None:
            waiting = [r for r in self.pending
                       if r.state == S_ARBITRATE and not r.done]
            # Round-robin, so no core starves.
            waiting.sort(key=lambda r: ((r.core - self.last_grant_core - 1)
                                        % self.num_cores, r.rid))
            if waiting and waiting[0].rid != req.rid:
                req.timer = 1
                return
            self.bus_owner = req.rid
            self.last_grant_core = req.core
            self.bus_traffic += 1
        req.state = S_SNOOP
        req.timer = self.cfg.bus_latency
        self.emit("bus", req=req.rid, core=req.core, addr=req.addr,
                  op=req.bus_op, frm="BUS", to="SNOOP",
                  note="bus granted to core %d (%s)" % (req.core, req.bus_op))

    def _do_snoop(self, req: MemRequest) -> None:
        block = self.l1d[req.core].block_addr(req.addr)

        # Racy upgrade: another core took our Shared line while we waited for the
        # bus, so this becomes a full read-for-ownership.
        own = self._l1(req).find(block)
        if req.bus_op == "BusUpgr" and (own is None or not own.valid):
            req.bus_op = "BusRdX"
            self.emit("coherence", req=req.rid, core=req.core, addr=block,
                      op="BusRdX",
                      note="upgrade raced with an invalidate; "
                           "reissued as BusRdX")

        supplier = None
        shared_elsewhere = False
        for other in range(self.num_cores):
            if other == req.core:
                continue
            cache = self.l1d[other]
            line = cache.find(block)
            if line is None or not line.valid:
                continue
            if req.bus_op == "BusRd":
                shared_elsewhere = True
                if line.state in (MODIFIED, EXCLUSIVE):
                    dirty = cache.downgrade(block)
                    self._note_coherence(req, other, line.state, SHARED,
                                         "snoop hit: %s -> S" % line.state)
                    if dirty is not None:
                        supplier = (other, dirty)
                        self._push_to_l2(block, dirty)
                else:
                    self._note_coherence(req, other, SHARED, SHARED,
                                         "snoop hit: already S")
            else:  # BusRdX / BusUpgr: everyone else drops the line
                dirty = cache.invalidate(block)
                self._note_coherence(req, other, line.state, INVALID,
                                     "snoop invalidate: %s -> I" % line.state)
                if self.reservations[other] == block:
                    self.reservations[other] = None
                    self.emit("coherence", req=req.rid, core=req.core,
                              target=other, addr=block, op=req.bus_op,
                              note="core %d loses its LR reservation" % other)
                if dirty is not None:
                    supplier = (other, dirty)
                    self._push_to_l2(block, dirty)
        req.hit = False
        if req.bus_op == "BusUpgr":
            # We already hold the data; only permission was missing.
            req.level = "COHERENCE"
            req.state = S_FILL
            req.timer = 0
            req.shared_elsewhere = False
            self.emit("bus", req=req.rid, core=req.core, addr=req.addr,
                      op=req.bus_op, frm="SNOOP", to="L1D",
                      note="upgrade granted, S -> M")
            return
        req.shared_elsewhere = shared_elsewhere
        if supplier is not None:
            req.state = S_TRANSFER
            req.timer = self.cfg.bus_latency
            req.level = "L1-to-L1"
            self.emit("bus", req=req.rid, core=req.core, addr=req.addr,
                      op=req.bus_op, frm="CORE%d.L1D" % supplier[0],
                      to="CORE%d.L1D" % req.core,
                      note="dirty copy forwarded from core %d" % supplier[0],
                      supplier=supplier[0])
            req.supplied = supplier[1]
            return
        req.state = S_L2
        req.timer = self.cfg.l2.latency
        self.emit("mem", req=req.rid, core=req.core, addr=req.addr,
                  frm="BUS", to="L2", note="L2 lookup", kind=req.kind)

    def _do_transfer(self, req: MemRequest) -> None:
        req.state = S_FILL
        req.timer = 0
        self.bus_bytes += self.cfg.l1d.block_size

    # -- L2 / DRAM ------------------------------------------------------

    def _do_l2(self, req: MemRequest) -> None:
        block = self.l2.block_addr(req.addr)
        line = self.l2.find(block)
        if req.is_store:
            self.l2.stats.writes += 1
        else:
            self.l2.stats.reads += 1
        if line is not None and line.valid:
            if req.is_store:
                self.l2.stats.write_hits += 1
            else:
                self.l2.stats.read_hits += 1
            req.level = "L2"
            req.supplied = bytes(line.data)
            line.sharers.add(req.core)
            self.l2.touch(line)
            req.state = S_FILL
            req.timer = 0
            self.emit("mem", req=req.rid, core=req.core, addr=req.addr,
                      frm="L2", to="CORE%d.L1D" % req.core, kind=req.kind,
                      note="L2 hit", hit=True)
            return
        self.l2.stats.misses += 1
        req.level = "DRAM"
        req.state = S_DRAM
        req.timer = self.cfg.dram_latency
        self.emit("mem", req=req.rid, core=req.core, addr=req.addr,
                  frm="L2", to="DRAM", kind=req.kind,
                  note="L2 miss -> DRAM (%d cycles)" % self.cfg.dram_latency,
                  miss=True)

    def _do_dram(self, req: MemRequest) -> None:
        block = self.l2.block_addr(req.addr)
        data = bytes(self.dram.read_block(block, self.cfg.l2.block_size))
        evicted = self.l2.install(block, data, SHARED)
        if evicted is not None:
            self._writeback(req, "L2", evicted)
        line = self.l2.find(block)
        if line is not None:
            line.sharers.add(req.core)
        req.supplied = data
        req.state = S_FILL
        req.timer = 0
        self.bus_bytes += self.cfg.l2.block_size
        self.emit("mem", req=req.rid, core=req.core, addr=req.addr,
                  frm="DRAM", to="L2", kind=req.kind,
                  note="block 0x%08x returned" % block)

    def _push_to_l2(self, block: int, data: bytes) -> None:
        """Park a dirty block in the shared L2, still marked dirty."""
        evicted = self.l2.install(block, data, MODIFIED)
        line = self.l2.find(block)
        if line is not None:
            line.dirty = True
        if evicted is not None and evicted[2]:
            self.dram.write_block(evicted[0], evicted[1])

    def _writeback(self, req: MemRequest, level: str, evicted) -> None:
        """Dirty data leaving a level lands in the next one down, not DRAM."""
        addr, data, dirty = evicted
        if not dirty:
            self.emit("mem", req=req.rid, core=req.core, addr=addr,
                      frm=level, to="-", kind="evict",
                      note="clean eviction, no writeback needed",
                      writeback=False)
            return
        self.bus_bytes += len(data)
        if level == "L2":
            self.dram.write_block(addr, data)
            destination = "DRAM"
        else:
            self._push_to_l2(addr, data)
            destination = "L2"
        self.emit("mem", req=req.rid, core=req.core, addr=addr,
                  frm=level, to=destination, kind="writeback",
                  note="dirty line 0x%08x written back to %s"
                       % (addr, destination),
                  writeback=True)

    # -- refill ---------------------------------------------------------

    def _do_fill(self, req: MemRequest) -> None:
        cache = self._l1(req)
        block = cache.block_addr(req.addr)
        if req.bus_op == "BusUpgr":
            line = cache.find(block)
            if line is not None:
                line.state = MODIFIED
        else:
            data = req.supplied
            if data is None:
                data = bytes(self.dram.read_block(block, cache.cfg.block_size))
            if req.kind == "fetch":
                state = SHARED
            elif req.is_store:
                state = MODIFIED
            elif req.shared_elsewhere:
                state = SHARED
            else:
                state = EXCLUSIVE
            evicted = cache.install(block, data, state)
            if evicted is not None:
                self._writeback(req, cache.cfg.name, evicted)
            if req.kind != "fetch":
                l2line = self.l2.find(block)
                if l2line is not None:
                    if req.is_store:
                        l2line.sharers = {req.core}
                    else:
                        l2line.sharers.add(req.core)
            self.emit("mem", req=req.rid, core=req.core, addr=req.addr,
                      frm=cache.cfg.name, to="CORE%d" % req.core,
                      kind=req.kind,
                      note="line filled in state %s" % state, fill=state)
        self._release_bus(req)
        self._complete(req, cache, hit=False)

    def _release_bus(self, req: MemRequest) -> None:
        if self.bus_owner == req.rid:
            self.bus_owner = None

    # -- MMIO -----------------------------------------------------------

    def _do_mmio(self, req: MemRequest) -> None:
        if req.is_store:
            value = int.from_bytes(req.store_data, "little")
            self.devices.write(req.addr, value, req.width, req.core)
            self.emit("mem", req=req.rid, core=req.core, addr=req.addr,
                      frm="CORE%d" % req.core, to="MMIO", kind="store",
                      note="device write 0x%08x" % value)
        else:
            req.value = self.devices.read(req.addr, req.width, req.core)
            self.emit("mem", req=req.rid, core=req.core, addr=req.addr,
                      frm="MMIO", to="CORE%d" % req.core, kind="load",
                      note="device read 0x%08x" % req.value)
        req.hit = True
        req.state = S_DONE
        req.done = True
        req.finished_cycle = self.cycle

    # -- completion -----------------------------------------------------

    def _complete(self, req: MemRequest, cache: Cache, hit: bool) -> None:
        if hit:
            req.hit = True
            req.level = cache.cfg.name
        block = cache.block_addr(req.addr)

        if req.kind == "store":
            self._do_write(req, cache, req.store_data)
        elif req.kind == "lr":
            req.value = int.from_bytes(cache.read_bytes(req.addr, req.width),
                                       "little")
            self.reservations[req.core] = block
            self.emit("coherence", req=req.rid, core=req.core, addr=block,
                      op="LR",
                      note="core %d reserves 0x%08x" % (req.core, block))
        elif req.kind == "sc":
            if self.reservations[req.core] == block:
                self._do_write(req, cache, req.store_data)
                req.value = 0
            else:
                req.value = 1
                req.sc_failed = True
            self.reservations[req.core] = None
        elif req.kind == "amo":
            old = int.from_bytes(cache.read_bytes(req.addr, req.width),
                                 "little")
            new = amo_apply(req.amo_op, old, req.amo_operand)
            self._do_write(req, cache, new.to_bytes(4, "little"))
            req.value = old
            self.emit("memdata", req=req.rid, core=req.core, addr=req.addr,
                      kind="amo", value=new,
                      note="%s: [0x%08x] 0x%08x -> 0x%08x"
                           % (req.amo_op, req.addr, old, new))
        else:
            req.value = int.from_bytes(cache.read_bytes(req.addr, req.width),
                                       "little")
        req.state = S_DONE
        req.done = True
        req.finished_cycle = self.cycle
        self.emit("mem", req=req.rid, core=req.core, addr=req.addr,
                  frm=cache.cfg.name, to="CORE%d" % req.core, kind=req.kind,
                  note="%s complete (%s, %d cycles)"
                       % (req.kind, "hit" if req.hit else req.level,
                          req.latency),
                  complete=True, hit=req.hit)

    # ------------------------------------------------------------------

    def _do_write(self, req: MemRequest, cache: Cache, payload: bytes) -> None:
        """L2 and DRAM stay stale until eviction or a snoop: that is what makes coherence
        load-bearing.
        """
        cache.write_bytes(req.addr, payload)
        if not self.cfg.l1d.write_back:
            block = cache.block_addr(req.addr)
            l2line = self.l2.find(block)
            if l2line is not None:
                off = req.addr & (self.cfg.l2.block_size - 1)
                l2line.data[off:off + len(payload)] = payload
            else:
                self.dram.write_block(req.addr, payload)

    def flush_all(self) -> None:
        for cache in self.l1d:
            for index, ways in enumerate(cache.sets):
                for line in ways:
                    if line.valid and line.dirty:
                        self._push_to_l2(cache.line_addr(line, index),
                                         bytes(line.data))
                        line.dirty = False
        for index, ways in enumerate(self.l2.sets):
            for line in ways:
                if line.valid and line.dirty:
                    self.dram.write_block(self.l2.line_addr(line, index),
                                          bytes(line.data))
                    line.dirty = False

    def _note_coherence(self, req: MemRequest, core: int, old: str, new: str,
                        note: str) -> None:
        self.coherence_log.append({
            "cycle": self.cycle, "addr": req.addr, "requester": req.core,
            "target": core, "op": req.bus_op, "from": old, "to": new,
        })
        self.emit("coherence", req=req.rid, core=req.core, target=core,
                  addr=req.addr, op=req.bus_op, old=old, new=new, note=note)

    def stats(self) -> dict:
        return {
            "l1i": [c.stats.as_dict() for c in self.l1i],
            "l1d": [c.stats.as_dict() for c in self.l1d],
            "l2": self.l2.stats.as_dict(),
            "bus_transactions": self.bus_traffic,
            "bus_bytes": self.bus_bytes,
            "coherence_events": len(self.coherence_log),
        }


SIGN_BIT = 1 << 31


def _signed(value: int) -> int:
    return value - (1 << 32) if value & SIGN_BIT else value


def amo_apply(op: str, old: int, operand: int) -> int:
    so, sv = _signed(old), _signed(operand)
    table = {
        "amoswap.w": lambda: operand,
        "amoadd.w": lambda: old + operand,
        "amoxor.w": lambda: old ^ operand,
        "amoor.w": lambda: old | operand,
        "amoand.w": lambda: old & operand,
        "amomin.w": lambda: old if so < sv else operand,
        "amomax.w": lambda: old if so > sv else operand,
        "amominu.w": lambda: min(old, operand),
        "amomaxu.w": lambda: max(old, operand),
    }
    if op not in table:
        raise ValueError("unknown atomic %r" % op)
    return table[op]() & MASK32


def _clone(cfg: CacheConfig, core: int) -> CacheConfig:
    return CacheConfig(name="%s%d" % (cfg.name, core), size=cfg.size,
                       block_size=cfg.block_size, assoc=cfg.assoc,
                       latency=cfg.latency, replacement=cfg.replacement,
                       write_back=cfg.write_back,
                       write_allocate=cfg.write_allocate)
