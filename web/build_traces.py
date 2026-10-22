"""Build the trace bundles the web visualiser loads: python web/build_traces.py

Windowed to a few thousand cycles at most, since the page plays in slow motion.
"""

from __future__ import annotations

import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rvsim.assembler import assemble_file
from rvsim.cache import CacheConfig
from rvsim.core import CoreConfig
from rvsim.memsys import MemoryConfig
from rvsim.system import System, SystemConfig

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "traces")

# name, source, blurb, cores, cycle window, config overrides
DEMOS = [
    ("tour", "tour.s",
     "A short program written to be watched. Every idea worth understanding "
     "shows up once, in order.",
     1, 400, {}),
    ("hazards", "hazards.s",
     "A dependency chain, a load-use bubble and a multi-cycle divide.",
     1, 240, {}),
    ("branches", "branches.s",
     "Loops the predictor finds easy, hard and impossible, plus a call in a "
     "loop for the return-address stack.",
     1, 1000, {}),
    ("cache", "cache.s",
     "Sequential, reuse, conflict and strided access patterns, so you can "
     "watch lines fill, hit and thrash.",
     1, 1400, {"l1d_assoc": 1}),
    ("sort", "sort.s",
     "Insertion sort over 16 words: real data-dependent branches.",
     1, 1600, {}),
    ("interrupt", "interrupt.s",
     "A timer interrupt arrives mid-loop. The pipeline drains, the handler "
     "runs, and mret puts everything back.",
     1, 1600, {}),
    ("spinlock", "spinlock.s",
     "Four cores contending for one lock line, handed between caches one "
     "owner at a time.",
     4, 1400, {}),
]


def build_one(name, source, blurb, cores, limit, overrides):
    program = assemble_file(os.path.join(ROOT, "programs", source))
    core = CoreConfig()
    memory = MemoryConfig()
    if "l1d_assoc" in overrides:
        memory.l1d = CacheConfig(name="L1D", size=1024, block_size=32,
                                 assoc=overrides["l1d_assoc"], latency=1)
    config = SystemConfig(num_cores=cores, core=core, memory=memory,
                          max_cycles=200000, trace=True, trace_limit=limit)
    system = System(program, config)
    system.run()

    payload = system.tracer.to_dict()
    payload["meta"]["name"] = name
    payload["meta"]["blurb"] = blurb
    payload["meta"]["source"] = io.open(
        os.path.join(ROOT, "programs", source), encoding="utf-8").read()
    payload["meta"]["windowed"] = len(payload["cycles"]) < system.cycle
    payload["meta"]["total_cycles"] = system.cycle

    body = json.dumps(payload, separators=(",", ":"))
    path = os.path.join(OUT, name + ".js")
    with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("window.__rvtrace(%s,%s);\n" % (json.dumps(name), body))
    size = os.path.getsize(path)
    print("  %-10s %5d cycles (of %5d)  %6.1f KB"
          % (name, len(payload["cycles"]), system.cycle, size / 1024))
    return payload


INLINE_DEFAULT = "tour"
START = "<!-- INLINE-TRACE-START -->"
END = "<!-- INLINE-TRACE-END -->"


def inline_default(payload):
    """Embedded in index.html, so the page animates with no extra request."""
    path = os.path.join(HERE, "index.html")
    page = io.open(path, encoding="utf-8").read()
    if START not in page or END not in page:
        print("  (index.html has no inline markers; skipping)")
        return
    head, rest = page.split(START, 1)
    tail = rest.split(END, 1)[1]
    body = json.dumps(payload, separators=(",", ":"))
    block = "%s\n<script>window.__rvtrace(%s,%s);</script>\n%s" % (
        START, json.dumps(INLINE_DEFAULT), body, END)
    out = head + block + tail
    io.open(path, "w", encoding="utf-8", newline="\n").write(out)
    print("  inlined %s into index.html (page is now %.1f KB)"
          % (INLINE_DEFAULT, len(out) / 1024))


def main():
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    print("building traces into %s" % OUT)
    index = []
    inline_payload = None
    for entry in DEMOS:
        payload = build_one(*entry)
        if entry[0] == INLINE_DEFAULT:
            inline_payload = payload
        index.append({
            "name": entry[0],
            "title": entry[0].replace("_", " "),
            "blurb": entry[2],
            "cores": entry[3],
            "cycles": len(payload["cycles"]),
            "total": payload["meta"]["total_cycles"],
        })
    with io.open(os.path.join(OUT, "index.json"), "w", encoding="utf-8",
                 newline="\n") as fh:
        json.dump(index, fh, indent=1)
    if inline_payload is not None:
        inline_default(inline_payload)
    print("done")


if __name__ == "__main__":
    main()
