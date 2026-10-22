"""python -m rvsim run       programs/sort.s
python -m rvsim pipeline  programs/hazards.s
python -m rvsim disasm    programs/branches.s
python -m rvsim trace     programs/spinlock.s --cores 4 -o trace.json
python -m rvsim compare   programs/branches.s
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .assembler import assemble_file, AsmError
from .cache import CacheConfig
from .core import CoreConfig
from .csr import cause_name
from .isa import ABI_NAMES
from .memsys import MemoryConfig
from .system import System, SystemConfig

# ---- terminal colours (disabled when piped or when NO_COLOR is set) -------

_TTY = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(code: str, text: str) -> str:
    return "\033[%sm%s\033[0m" % (code, text) if _TTY else text


def bold(t):    return _c("1", t)
def dim(t):     return _c("2", t)
def red(t):     return _c("31", t)
def green(t):   return _c("32", t)
def yellow(t):  return _c("33", t)
def blue(t):    return _c("34", t)
def magenta(t): return _c("35", t)
def cyan(t):    return _c("36", t)


STAGE_COLOUR = {"IF": cyan, "ID": blue, "EX": magenta, "MEM": yellow,
                "WB": green}


# --------------------------------------------------------------------------
# Configuration from flags
# --------------------------------------------------------------------------

def add_machine_flags(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("machine")
    g.add_argument("--cores", type=int, default=1, help="number of harts")
    g.add_argument("--bp", default="gshare",
                   choices=["gshare", "bimodal", "taken", "not-taken", "btfn"],
                   help="branch direction predictor")
    g.add_argument("--bp-entries", type=int, default=64)
    g.add_argument("--btb-entries", type=int, default=32)
    g.add_argument("--no-forwarding", action="store_true",
                   help="disable EX/MEM and MEM/WB forwarding")
    g.add_argument("--mul-latency", type=int, default=3)
    g.add_argument("--div-latency", type=int, default=12)
    c = parser.add_argument_group("memory hierarchy")
    c.add_argument("--l1i-size", type=int, default=1024)
    c.add_argument("--l1d-size", type=int, default=1024)
    c.add_argument("--l1d-assoc", type=int, default=2)
    c.add_argument("--l1i-assoc", type=int, default=2)
    c.add_argument("--block", type=int, default=32, help="cache line bytes")
    c.add_argument("--l2-size", type=int, default=8192)
    c.add_argument("--l2-assoc", type=int, default=4)
    c.add_argument("--l2-latency", type=int, default=8)
    c.add_argument("--dram-latency", type=int, default=40)
    c.add_argument("--bus-latency", type=int, default=2)
    c.add_argument("--replacement", default="lru",
                   choices=["lru", "fifo", "random"])
    r = parser.add_argument_group("run")
    r.add_argument("--timer", type=int, default=0,
                   help="preload mtimecmp so a timer interrupt fires")
    r.add_argument("--max-cycles", type=int, default=200000)


def build_config(args, trace: bool = True) -> SystemConfig:
    core = CoreConfig(
        forwarding=not args.no_forwarding,
        branch_scheme=args.bp,
        bpred_entries=args.bp_entries,
        btb_entries=args.btb_entries,
        mul_latency=args.mul_latency,
        div_latency=args.div_latency,
    )
    memory = MemoryConfig(
        l1i=CacheConfig(name="L1I", size=args.l1i_size, block_size=args.block,
                        assoc=args.l1i_assoc, latency=1,
                        replacement=args.replacement),
        l1d=CacheConfig(name="L1D", size=args.l1d_size, block_size=args.block,
                        assoc=args.l1d_assoc, latency=1,
                        replacement=args.replacement),
        l2=CacheConfig(name="L2", size=args.l2_size, block_size=args.block,
                       assoc=args.l2_assoc, latency=args.l2_latency,
                       replacement=args.replacement),
        dram_latency=args.dram_latency,
        bus_latency=args.bus_latency,
    )
    return SystemConfig(num_cores=args.cores, core=core, memory=memory,
                        max_cycles=args.max_cycles, timer_period=args.timer,
                        trace=trace)


def load(args, trace: bool = True) -> System:
    try:
        program = assemble_file(args.source)
    except AsmError as exc:
        print(red("assembly error: ") + str(exc), file=sys.stderr)
        raise SystemExit(2)
    return System(program, build_config(args, trace))


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

def cmd_run(args) -> int:
    system = load(args, trace=False)
    print(bold("machine"))
    for line in system.cfg.describe():
        print("  " + line)
    print()
    reason = system.run()
    stats = system.stats()

    print(bold("result"))
    print("  halted: %s after %s cycles"
          % (reason, bold(str(stats["cycles"]))))
    if stats["console"]:
        print("  console: %r" % stats["console"])
    print()

    for core in stats["cores"]:
        print(bold("core %d" % core["id"]))
        print("  retired %d  IPC %s  CPI %s"
              % (core["retired"], green("%.3f" % core["ipc"]),
                 "%.3f" % core["cpi"]))
        print("  stalls: %d total = %d load-use, %d I-cache, %d D-cache, "
              "%d execute, %d serialize"
              % (core["stall_cycles"], core["load_use_stalls"],
                 core["icache_stalls"], core["dcache_stalls"],
                 core["exec_stalls"], core["serialize_stalls"]))
        b = core["branch"]
        print("  branches: %d, mispredicts %s (%s accuracy, %s scheme)"
              % (b["predictions"], red(str(b["mispredictions"])),
                 yellow("%.1f%%" % (100 * b["accuracy"])), b["scheme"]))
        print("  jumps:    %d, mispredicts %d (%.1f%% accuracy), "
              "BTB %d/%d hit, RAS %d used / %d correct"
              % (b["jumps"], b["jump_mispredictions"],
                 100 * b["jump_accuracy"], b["btb_hits"],
                 b["btb_hits"] + b["btb_misses"], b["ras_used"],
                 b["ras_correct"]))
        print("  forwards: %d, flushes: %d, traps: %d (%d interrupts)"
              % (core["forwards"], core["flushes"], core["traps"],
                 core["interrupts"]))
        print()

    mem = stats["memory"]
    print(bold("memory hierarchy"))
    for i in range(system.cfg.num_cores):
        for label, data in (("L1I", mem["l1i"][i]), ("L1D", mem["l1d"][i])):
            print("  core%d %s: %d accesses, %s hits, %d misses (%s), "
                  "%d evictions, %d writebacks"
                  % (i, label, data["accesses"], data["hits"], data["misses"],
                     _rate(data["hit_rate"]), data["evictions"],
                     data["writebacks"]))
    l2 = mem["l2"]
    print("  shared L2: %d accesses, %d hits, %d misses (%s), %d writebacks"
          % (l2["accesses"], l2["hits"], l2["misses"], _rate(l2["hit_rate"]),
             l2["writebacks"]))
    print("  bus: %d transactions, %d bytes moved, %d coherence events"
          % (mem["bus_transactions"], mem["bus_bytes"],
             mem["coherence_events"]))
    print()

    if system.memsys.coherence_log and not args.quiet:
        print(bold("coherence log (first 20)"))
        for entry in system.memsys.coherence_log[:20]:
            print("  c%-6d core%d %-7s on 0x%08x -> core%d %s to %s"
                  % (entry["cycle"], entry["requester"], entry["op"],
                     entry["addr"], entry["target"],
                     magenta(entry["from"]), magenta(entry["to"])))
        print()

    if not args.quiet:
        for i in range(system.cfg.num_cores):
            print_registers(system, i)

    if args.dump_memory:
        print_memory(system)
    return 0


def _rate(value: float) -> str:
    text = "%.1f%%" % (100 * value)
    return green(text) if value > 0.9 else (yellow(text) if value > 0.6
                                            else red(text))


def print_registers(system: System, core: int) -> None:
    print(bold("core %d registers" % core))
    regs = system.cores[core].regs
    for row in range(8):
        cells = []
        for col in range(4):
            i = row + 8 * col
            value = regs[i]
            label = "%-4s" % ABI_NAMES[i]
            text = "%s=%08x" % (label, value)
            cells.append(text if value else dim(text))
        print("  " + "  ".join(cells))
    csrs = system.cores[core].csr.snapshot()
    # mcause means nothing until something has trapped.
    cause = ("  mcause=%08x (%s)" % (csrs["mcause"], cause_name(csrs["mcause"]))
             if system.cores[core].stats.traps else "  no traps taken")
    print("  " + dim("pc=%08x  mstatus=%08x  mepc=%08x%s"
                     % (system.cores[core].pc, csrs["mstatus"], csrs["mepc"],
                        cause)))
    print()


def print_memory(system: System, words: int = 48) -> None:
    print(bold("data memory"))
    base = system.program.data_base
    reverse = {v: k for k, v in system.program.symbols.items()}
    for i in range(words):
        addr = base + 4 * i
        value = system.memsys.dram.read(addr, 4)
        label = reverse.get(addr, "")
        if value or label:
            print("  0x%08x  %08x  %11d  %s"
                  % (addr, value, _signed(value), cyan(label)))
    print()


def _signed(value: int) -> int:
    return value - (1 << 32) if value & 0x80000000 else value


# --------------------------------------------------------------------------
# pipeline -- the cycle x instruction diagram
# --------------------------------------------------------------------------

def cmd_pipeline(args) -> int:
    system = load(args, trace=True)
    system.run()
    core = args.core
    records = system.tracer.records
    first, last = args.start, min(args.start + args.cycles, len(records))
    window = records[first:last]
    if not window:
        print("no cycles in that range (ran %d)" % len(records))
        return 1

    # seq -> {cycle: stage}
    from .trace import STAGES as STAGE_NAMES
    timeline = {}
    stalls = {}
    for rec in window:
        view = rec.cores[core]
        stalls[rec.cycle] = view.get("sr", "")
        for index, seq in enumerate(view["st"]):
            if seq >= 0:
                timeline.setdefault(seq, {})[rec.cycle] = STAGE_NAMES[index]

    seqs = sorted(timeline)[:args.rows]
    labels = {}
    for key, value in system.tracer.instrs.items():
        core_id, seq = key.split(":")
        if int(core_id) == core:
            labels[int(seq)] = (value[0], value[1])

    width = max((len(labels.get(s, (0, ""))[1]) for s in seqs), default=10)
    width = min(max(width, 10), 26)
    header = " " * (width + 14)
    cycles = [rec.cycle for rec in window]
    header += "".join("%-4d" % c if c % 5 == 0 else "    " for c in cycles)
    print(bold("pipeline, core %d  (cycles %d-%d)"
               % (core, cycles[0], cycles[-1])))
    print(dim(header))

    for seq in seqs:
        pc, text = labels.get(seq, (0, "?"))
        line = "%s %s " % (dim("%3d" % seq), dim("%08x" % pc))
        line += "%-*s " % (width, text)
        for cycle in cycles:
            stage = timeline[seq].get(cycle)
            if stage:
                line += STAGE_COLOUR[stage]("%-4s" % stage)
            else:
                line += "    "
        print(line)

    print()
    reasons = [(c, r) for c, r in sorted(stalls.items()) if r]
    if reasons:
        print(bold("stalls in this window"))
        shown = 0
        previous = None
        for cycle, reason in reasons:
            if reason != previous:
                print("  c%-6d %s" % (cycle, yellow(reason)))
                previous = reason
                shown += 1
            if shown >= 25:
                print("  " + dim("..."))
                break
        print()
    stats = system.stats()["cores"][core]
    print("retired %d in %d cycles, IPC %.3f"
          % (stats["retired"], system.cycle, stats["ipc"]))
    return 0


# --------------------------------------------------------------------------
# disasm / trace / compare
# --------------------------------------------------------------------------

def cmd_disasm(args) -> int:
    try:
        program = assemble_file(args.source)
    except AsmError as exc:
        print(red("assembly error: ") + str(exc), file=sys.stderr)
        return 2
    reverse = {}
    for name, addr in program.symbols.items():
        reverse.setdefault(addr, []).append(name)
    print(bold("%-10s %-10s %-28s %s" % ("addr", "word", "disassembly", "source")))
    for addr, word, text, src, line_no in program.disassembly():
        for label in reverse.get(addr, []):
            print(cyan("%08x <%s>:" % (addr, label)))
        print("%08x   %08x   %-28s %s"
              % (addr, word, text, dim("%3d| %s" % (line_no, src))))
    if program.data:
        print()
        print(bold("data at 0x%08x (%d bytes)"
                   % (program.data_base, len(program.data))))
    return 0


def cmd_trace(args) -> int:
    system = load(args, trace=True)
    if getattr(args, "limit", 0):
        system.tracer.max_cycles = args.limit
    reason = system.run()
    path = args.output
    system.tracer.write(path, compact=not args.pretty)
    size = os.path.getsize(path)
    print("%s: %d cycles, %d events, %.1f KB"
          % (path, len(system.tracer.records),
             sum(len(r.events) for r in system.tracer.records), size / 1024))
    print("halted: %s" % reason)
    return 0


def cmd_compare(args) -> int:
    variants = [
        ("baseline", {}),
        ("no forwarding", {"no_forwarding": True}),
        ("static not-taken", {"bp": "not-taken"}),
        ("bimodal", {"bp": "bimodal"}),
        ("direct-mapped L1D", {"l1d_assoc": 1}),
        ("4-way L1D", {"l1d_assoc": 4}),
        ("tiny L1D (256B)", {"l1d_size": 256}),
        ("64-byte lines", {"block": 64}),
        ("slow DRAM (120)", {"dram_latency": 120}),
    ]
    rows = []
    for label, overrides in variants:
        local = argparse.Namespace(**vars(args))
        for key, value in overrides.items():
            setattr(local, key, value)
        system = load(local, trace=False)
        system.run()
        stats = system.stats()
        core = stats["cores"][0]
        mem = stats["memory"]
        rows.append((label, stats["cycles"], core["ipc"],
                     core["branch"]["accuracy"], mem["l1d"][0]["hit_rate"],
                     core["stall_cycles"]))

    baseline = rows[0][1]
    print(bold("%-20s %9s %7s %8s %9s %8s %8s"
               % ("configuration", "cycles", "IPC", "bp acc", "L1D hit",
                  "stalls", "vs base")))
    for label, cycles, ipc, acc, hit, stalls in rows:
        delta = cycles / baseline if baseline else 1.0
        marker = green("%.2fx" % delta) if delta <= 1.02 else red("%.2fx" % delta)
        print("%-20s %9d %7.3f %7.1f%% %8.1f%% %8d %8s"
              % (label, cycles, ipc, 100 * acc, 100 * hit, stalls, marker))
    return 0


# --------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="rvsim",
        description="RV32IMA pipeline, cache and coherence simulator")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="run a program and report statistics")
    p.add_argument("source")
    p.add_argument("--dump-memory", action="store_true")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="statistics only, no register dump")
    add_machine_flags(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("pipeline", help="cycle-by-cycle pipeline diagram")
    p.add_argument("source")
    p.add_argument("--core", type=int, default=0)
    p.add_argument("--start", type=int, default=0, help="first cycle shown")
    p.add_argument("--cycles", type=int, default=40, help="cycles to show")
    p.add_argument("--rows", type=int, default=30, help="instructions to show")
    add_machine_flags(p)
    p.set_defaults(func=cmd_pipeline)

    p = sub.add_parser("disasm", help="assemble and disassemble")
    p.add_argument("source")
    p.set_defaults(func=cmd_disasm)

    p = sub.add_parser("trace", help="write a JSON execution trace")
    p.add_argument("source")
    p.add_argument("-o", "--output", default="trace.json")
    p.add_argument("--pretty", action="store_true")
    p.add_argument("--limit", type=int, default=0,
                   help="stop recording after this many cycles")
    add_machine_flags(p)
    p.set_defaults(func=cmd_trace)

    p = sub.add_parser("compare",
                       help="run one program under several configurations")
    p.add_argument("source")
    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument("--dump-memory", action="store_true")
    add_machine_flags(p)
    p.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
