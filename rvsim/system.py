"""The chip. Each cycle: devices, interrupt lines, memory requests, then cores, so a
core sees results available at the start of the cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import csr as csrmod
from .assembler import Program
from .core import Core, CoreConfig
from .memory import Devices
from .memsys import MemorySystem, MemoryConfig
from .trace import Tracer

MASK32 = 0xFFFFFFFF


@dataclass
class SystemConfig:
    num_cores: int = 1
    core: CoreConfig = field(default_factory=CoreConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    max_cycles: int = 200000
    timer_period: int = 0      # 0 = program sets mtimecmp itself
    trace: bool = True
    trace_limit: int = 200000
    seed: int = 1

    def describe(self) -> list:
        out = ["cores: %d" % self.num_cores]
        out += self.core.describe()
        out += self.memory.describe()
        return out


class HaltReason:
    HALT_WRITE = "halt device written"
    MAX_CYCLES = "cycle limit reached"
    IDLE = "all cores parked with no interrupt source"


class System:
    def __init__(self, program: Program, config: SystemConfig | None = None):
        self.cfg = config or SystemConfig()
        self.program = program
        self.tracer = Tracer(self.cfg.trace, self.cfg.trace_limit)
        self.devices = Devices(self.cfg.num_cores)
        self.memsys = MemorySystem(self.cfg.num_cores, self.cfg.memory,
                                   self.devices, emit=self.tracer.emit,
                                   seed=self.cfg.seed)
        self._load(program)
        self.cores = [Core(i, self.memsys, self.cfg.core,
                           emit=self.tracer.emit, entry=program.entry)
                      for i in range(self.cfg.num_cores)]
        if self.cfg.timer_period:
            for i in range(self.cfg.num_cores):
                self.devices.mtimecmp[i] = self.cfg.timer_period
        self.cycle = 0
        self.halt_reason = None
        self.tracer.meta = self._meta()

    def _load(self, program: Program) -> None:
        self.memsys.dram.load_bytes(program.text_base, bytes(program.text))
        self.memsys.dram.load_bytes(program.data_base, bytes(program.data))

    def _meta(self) -> dict:
        return {
            "config": self.cfg.describe(),
            "num_cores": self.cfg.num_cores,
            "entry": self.program.entry,
            "text_base": self.program.text_base,
            "data_base": self.program.data_base,
            "symbols": {k: v for k, v in sorted(self.program.symbols.items(),
                                                key=lambda kv: kv[1])},
            "listing": [
                {"addr": a, "word": w, "asm": d, "src": s, "line": ln}
                for a, w, d, s, ln in self.program.disassembly()
            ],
            "caches": {
                "l1i": _cache_meta(self.memsys.l1i[0]),
                "l1d": _cache_meta(self.memsys.l1d[0]),
                "l2": _cache_meta(self.memsys.l2),
                "dram_latency": self.cfg.memory.dram_latency,
                "bus_latency": self.cfg.memory.bus_latency,
            },
            "branch": {
                "scheme": self.cfg.core.branch_scheme,
                "btb_entries": self.cfg.core.btb_entries,
            },
        }

    @property
    def running(self) -> bool:
        return not all(core.halted for core in self.cores)

    def step(self) -> None:
        self.cycle += 1
        self.devices.tick()
        for core in self.cores:
            core.csr.set(csrmod.TIME, self.devices.mtime & MASK32)
            core.csr.raise_line(csrmod.MIP_MTIP,
                                self.devices.timer_pending(core.id))
            core.csr.raise_line(csrmod.MIP_MSIP,
                                self.devices.software_pending(core.id))
        self.memsys.step(self.cycle)
        for core in self.cores:
            core.tick(self.cycle)
        for hart, code in list(self.devices.halted.items()):
            if hart < len(self.cores) and not self.cores[hart].halted:
                self.cores[hart].halted = True
                self.cores[hart].exit_code = code
                self.tracer.emit("device", core=hart, kind="halt", value=code,
                                 note="core %d halted with code %d" % (hart, code))
        self.tracer.capture(self.cycle, self)

    def run(self, max_cycles: int | None = None) -> str:
        limit = max_cycles or self.cfg.max_cycles
        while self.running and self.cycle < limit:
            self.step()
            if self._deadlocked():
                self.halt_reason = HaltReason.IDLE
                for core in self.cores:
                    core.halted = True
                break
        if self.halt_reason is None:
            self.halt_reason = (HaltReason.HALT_WRITE if not self.running
                                else HaltReason.MAX_CYCLES)
        # So a memory dump reflects what the program wrote.
        self.memsys.flush_all()
        self.tracer.stats = self.stats()
        self.tracer.meta["halt_reason"] = self.halt_reason
        self.tracer.meta["cycles"] = self.cycle
        return self.halt_reason

    def _deadlocked(self) -> bool:
        """Every core parked in WFI with nothing that can ever wake it."""
        if not self.running:
            return False  # halted normally, not deadlocked
        for core in self.cores:
            if core.halted:
                continue
            if not core.csr.wfi:
                return False
            if core.csr.pending_interrupt() is not None:
                return False
            # A timer that is armed and enabled will eventually wake it.
            timer_armed = (self.devices.mtimecmp[core.id] != 0xFFFFFFFFFFFFFFFF
                           and core.csr.get(csrmod.MIE) & csrmod.MIP_MTIP)
            if timer_armed:
                return False
            if any(self.devices.msip):
                return False
        return True

    def stats(self) -> dict:
        total_retired = sum(c.stats.retired for c in self.cores)
        return {
            "cycles": self.cycle,
            "halt_reason": self.halt_reason,
            "aggregate": {
                "retired": total_retired,
                "ipc": round(total_retired / self.cycle, 4) if self.cycle else 0,
            },
            "cores": [dict(c.stats.as_dict(), id=c.id,
                           branch=c.bpu.stats(),
                           exit_code=c.exit_code) for c in self.cores],
            "memory": self.memsys.stats(),
            "console": "".join(self.devices.console),
        }

    def register_dump(self, core: int = 0) -> list:
        from .isa import ABI_NAMES
        regs = self.cores[core].regs
        return [(i, ABI_NAMES[i], regs[i]) for i in range(32)]


def _cache_meta(cache) -> dict:
    cfg = cache.cfg
    return {"name": cfg.name.rstrip("0123456789") or cfg.name,
            "size": cfg.size, "block": cfg.block_size, "assoc": cfg.assoc,
            "sets": cfg.num_sets, "latency": cfg.latency,
            "replacement": cfg.replacement}
