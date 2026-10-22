"""Structured execution trace, replayed by both the terminal and web views.

Stored as per-cycle deltas, with instructions referenced by sequence number into
one ``instrs`` table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

# Event kinds, in roughly the order they happen within a cycle.
KINDS = ["fetch", "predict", "regread", "forward", "alu", "branch", "mem",
         "bus", "coherence", "memdata", "csr", "regwrite", "retire", "trap",
         "flush", "exec", "device"]

STAGES = ["IF", "ID", "EX", "MEM", "WB"]

# How each kind is drawn as a packet in the datapath view: (source, destination).
FLOW_HINTS = {
    "fetch": ("PC", "L1I"),
    "regread": ("REGS", "EX"),
    "forward": ("EX", "EX"),
    "alu": ("EX", "EX"),
    "memdata": ("L1D", "REGS"),
    "regwrite": ("WB", "REGS"),
    "csr": ("WB", "CSR"),
}


@dataclass
class CycleRecord:
    cycle: int
    cores: list = field(default_factory=list)
    events: list = field(default_factory=list)
    cache_updates: list = field(default_factory=list)
    bus: dict = field(default_factory=dict)
    devices: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        out = {"c": self.cycle, "cores": self.cores}
        if self.events:
            out["ev"] = self.events
        if self.cache_updates:
            out["ca"] = self.cache_updates
        if self.bus:
            out["b"] = self.bus
        if self.devices:
            out["d"] = self.devices
        return out


class Tracer:
    def __init__(self, enabled: bool = True, max_cycles: int = 200000):
        self.enabled = enabled
        self.max_cycles = max_cycles
        self.records: list = []
        self._events: list = []
        self._prev_regs: dict = {}
        self._prev_csrs: dict = {}
        self._prev_cache: dict = {}
        self._prev_devices: dict = {}
        self.instrs: dict = {}     # "core:seq" -> [pc, disassembly]
        self.meta: dict = {}
        self.stats: dict = {}

    # -- event sink -----------------------------------------------------

    def emit(self, etype: str, **fields) -> None:
        """A caller's ``kind`` field is domain data (load/store/fetch) and is kept."""
        if not self.enabled:
            return
        event = {"k": etype}
        event.update({k: v for k, v in fields.items() if v is not None})
        self._events.append(event)

    # -- per-cycle capture ----------------------------------------------

    def capture(self, cycle: int, system) -> CycleRecord | None:
        if not self.enabled or len(self.records) >= self.max_cycles:
            self._events = []
            return None
        rec = CycleRecord(cycle=cycle)
        for core in system.cores:
            rec.cores.append(self._core_record(core))
        rec.cache_updates = self._cache_delta(system.memsys)
        rec.bus = self._bus_state(system.memsys)
        rec.devices = self._device_delta(system.devices)
        rec.events = self._events
        self._events = []
        self.records.append(rec)
        return rec

    def _core_record(self, core) -> dict:
        snap = core.snapshot()
        entry = {"pc": snap["pc"]}

        # Five sequence numbers, -1 empty; disassembly lives once in instrs.
        stages = []
        for stage in STAGES:
            slot = snap["stages"].get(stage)
            if not slot:
                stages.append(-1)
                continue
            seq = slot["seq"]
            stages.append(seq)
            if slot["name"] == "fetch":
                entry["fp"] = slot["pc"]
            key = "%d:%d" % (core.id, seq)
            # First seen as an anonymous fetch; replaced once it decodes.
            known = self.instrs.get(key)
            if known is None or known[2] == "fetch":
                self.instrs[key] = [slot["pc"], slot["text"], slot["name"],
                                    slot["cat"], slot["rd"], slot["rs1"],
                                    slot["rs2"]]
        entry["st"] = stages

        if snap["stall"]:
            entry["sr"] = snap["stall"]
        if snap["wfi"]:
            entry["wfi"] = 1
        if snap["halted"]:
            entry["halt"] = 1

        previous = self._prev_regs.get(core.id)
        if previous is None:
            delta = {str(i): v for i, v in enumerate(snap["regs"]) if v}
        else:
            delta = {str(i): v for i, v in enumerate(snap["regs"])
                     if v != previous[i]}
        if delta:
            entry["r"] = delta
        self._prev_regs[core.id] = list(snap["regs"])

        previous_csr = self._prev_csrs.get(core.id, {})
        csr_delta = {k: v for k, v in snap["csrs"].items()
                     if previous_csr.get(k) != v and k not in ("mcycle",)}
        if csr_delta:
            entry["cs"] = csr_delta
        self._prev_csrs[core.id] = dict(snap["csrs"])

        if core.retired_this_cycle is not None:
            entry["ret"] = core.retired_this_cycle.seq
        return entry

    def _bus_state(self, mem) -> dict:
        if mem.bus_owner is None and not mem.pending:
            return {}
        out = {}
        if mem.bus_owner is not None:
            out["owner"] = mem.bus_owner
        if mem.pending:
            out["p"] = [[r.rid, r.core, r.state, r.addr, r.kind, r.bus_op]
                        for r in mem.pending]
        return out

    def _device_delta(self, devices) -> dict:
        current = {"mtimecmp": list(devices.mtimecmp), "msip": list(devices.msip)}
        if current == self._prev_devices:
            return {}
        self._prev_devices = {"mtimecmp": list(current["mtimecmp"]),
                              "msip": list(current["msip"])}
        return current

    def _cache_delta(self, memsys) -> list:
        updates = []
        caches = [("L1I", i, c) for i, c in enumerate(memsys.l1i)]
        caches += [("L1D", i, c) for i, c in enumerate(memsys.l1d)]
        caches += [("L2", None, memsys.l2)]
        for name, core, cache in caches:
            key = (name, core)
            previous = self._prev_cache.get(key, {})
            current = {}
            for entry in cache.snapshot():
                slot = (entry["set"], entry["way"])
                value = (entry["state"], entry["addr"], entry["dirty"],
                         tuple(entry["sharers"]))
                current[slot] = value
                if previous.get(slot) != value:
                    updates.append({
                        "cache": name, "core": core, "set": entry["set"],
                        "way": entry["way"], "state": entry["state"],
                        "addr": entry["addr"], "dirty": entry["dirty"],
                        "sharers": entry["sharers"],
                    })
            self._prev_cache[key] = current
        return updates

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "meta": self.meta,
            "instrs": self.instrs,
            "cycles": [r.to_json() for r in self.records],
            "stats": self.stats,
        }

    def write(self, path: str, compact: bool = True) -> str:
        payload = self.to_dict()
        with open(path, "w", encoding="utf-8") as fh:
            if compact:
                json.dump(payload, fh, separators=(",", ":"))
            else:
                json.dump(payload, fh, indent=1)
        return path
