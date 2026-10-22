# rvsim — an RV32IMA processor simulator

A cycle-accurate model of a small RISC-V machine, written to be *watched*
rather than just measured: a five-stage in-order pipeline with forwarding and
hazard interlocks, branch prediction, a coherent multi-level cache hierarchy,
several cores, and precise traps and interrupts.

Programs are real RISC-V assembly. They are assembled by the included
assembler, executed instruction by instruction, and every architectural
result is real — the register values, the memory contents and the cycle
counts all come from the same machine.

**Python 3.9+, standard library only.** No dependencies, no build step.

```bash
python -m rvsim run      programs/sort.s
python -m rvsim pipeline programs/hazards.s --start 165 --cycles 18
python -m rvsim compare  programs/cache.s
python -m unittest discover -s tests
```

---

## What is modelled

### The core (`rvsim/core.py`)

| | |
|---|---|
| Pipeline | IF · ID · EX · MEM · WB, in order |
| Register file | x0–x31, x0 hardwired to zero |
| Forwarding | EX/MEM and MEM/WB paths into both ALU inputs |
| Interlocks | load-use bubble with forwarding; full RAW interlock without |
| Branch prediction | bimodal, gshare, or static, plus a BTB and a return-address stack |
| Multi-cycle units | `mul` holds EX for 3 cycles, `div`/`rem` for 12 |
| Traps | precise: an exception rides the pipe and only commits in WB |
| Interrupts | taken at an instruction boundary; the pipeline drains first |
| `wfi` | parks the core until an interrupt arrives |

Branch resolution happens in EX, so a mispredict squashes the two younger
instructions and refetches. Unconditional jumps and returns are predicted
from the BTB entry's *kind* rather than from the direction predictor — a
detail that is easy to get wrong and costs a great deal when you do.

### The memory system (`rvsim/memsys.py`, `rvsim/cache.py`)

Private L1I and L1D per core, a shared inclusive L2, and DRAM, connected by a
snooping bus. Every cache level is the same configurable class: size,
line size, associativity, replacement policy (LRU / FIFO / random), and
lookup latency.

Misses are **cycle-stepped, not resolved instantly**. A request walks a state
machine — L1 lookup, bus arbitration, snoop, cache-to-cache transfer or L2,
DRAM, refill — and each transition takes real cycles and emits a trace event.
That is what makes the visualiser able to show a block physically travelling
through the hierarchy.

Coherence is **MESI**, maintained by snooping:

- a read miss issues `BusRd`; another core holding the line Modified flushes
  it and drops to Shared, supplying the data core-to-core
- a write to a Shared line issues `BusUpgr` — permission only, no data
- a write miss issues `BusRdX` and invalidates every other copy
- dirty data leaving a level lands in the next level down, **not** in DRAM,
  so the protocol is load-bearing rather than decorative

`tests/test_rvsim.py` checks the central invariant on every cycle of a real
multicore run: no line is ever Modified in two caches at once.

### Atomics

RV32A: `lr.w` / `sc.w` with per-core reservations that a remote write kills,
plus `amoswap` / `amoadd` / `amoand` / `amoor` / `amoxor` / `amomin` /
`amomax` (and the unsigned variants). This is what makes the multicore demo
mean something: four cores take a real lock and the shared counter comes out
exactly right.

### Traps and interrupts (`rvsim/csr.py`)

Machine mode, with `mstatus`, `mie`, `mip`, `mtvec` (direct and vectored),
`mepc`, `mcause`, `mtval`, `mscratch`, `mcycle`, `minstret`, `mhartid`.
Exceptions: illegal instruction, misaligned load/store, breakpoint, and
environment call. Interrupts: machine timer and machine software (IPI),
driven by a CLINT at the usual addresses.

---

## Programs

| file | what it shows |
|---|---|
| `programs/tour.s` | a short program written to be watched: one of each idea, in order |
| `programs/hazards.s` | a RAW chain, the load-use bubble forwarding cannot remove, multi-cycle divide |
| `programs/branches.s` | loops that are easy, hard and impossible to predict; a call in a loop for the RAS |
| `programs/cache.s` | sequential, reuse, conflict and strided access patterns |
| `programs/sort.s` | insertion sort over 16 words — a real algorithm with verifiable output |
| `programs/interrupt.s` | a timer interrupt mid-loop, an `ecall`, and `wfi` |
| `programs/spinlock.s` | four harts contending for one lock line |

The assembler supports labels, `.text` / `.data`, `.word` / `.half` /
`.byte` / `.string` / `.space` / `.align` / `.equ`, and the usual
pseudo-instructions (`li`, `la`, `mv`, `call`, `ret`, `j`, `beqz`, `csrr`,
`csrw`, …), with `%hi` / `%lo`.

---

## The cycle-by-cycle view

`python -m rvsim pipeline` prints the classic instruction × cycle diagram.
A well-predicted loop, running at one instruction per cycle:

```
 67 00001008 addi s0, s0, 1   MEM WB
 68 0000100c addi t0, t0, -1  EX  MEM WB
 69 00001010 bne t0, zero, -8 ID  EX  MEM WB
 70 00001008 addi s0, s0, 1   IF  ID  EX  MEM WB
 71 0000100c addi t0, t0, -1      IF  ID  EX  MEM WB
 72 00001010 bne t0, zero, -8         IF  ID  EX  MEM WB
```

And the same view where a multiply and a divide hold EX, with everything
behind them backing up into ID and IF:

```
 13 00001034 mul s1, t5, t6      ID  EX  EX  EX  MEM WB
 14 00001038 div s2, s1, t6      IF  ID  ID  ID  EX  EX  EX  EX  EX  EX  EX  EX  EX  EX  EX  EX  MEM WB
 15 0000103c rem s3, s1, t6          IF  IF  IF  ID  ID  ID  ID  ID  ID  ID  ID  ID  ID  ID  ID  EX  EX
 16 00001040 sw s1, +8(s0)                       IF  IF  IF  IF  IF  IF  IF  IF  IF  IF  IF  IF  IF  IF
```

---

## Comparing configurations

`python -m rvsim compare` runs one program under several machines. On
`programs/cache.s`, whose inner loop touches three addresses that collide in
the same set:

```
configuration           cycles     IPC   bp acc   L1D hit   stalls  vs base
baseline                  9356   0.235    91.6%     41.8%     7087    1.00x
no forwarding            10672   0.206    91.6%     41.8%     8403    1.14x
static not-taken         10024   0.220     1.4%     41.8%     7087    1.07x
bimodal                   9314   0.236    97.3%     41.8%     7087    1.00x
direct-mapped L1D         9276   0.237    91.6%     42.8%     7007    0.99x
4-way L1D                 4876   0.452    91.6%     93.3%     2607    0.52x
64-byte lines             8407   0.262    91.6%     43.7%     6138    0.90x
slow DRAM (120)          13116   0.168    91.6%     41.8%    10847    1.40x
```

Two things worth noticing. Associativity is not a free win — going from
2-way to direct-mapped changes almost nothing here, because halving the ways
doubles the number of sets and the collisions move rather than disappear;
it is the step to 4-way, which finally holds all three lines at once, that
halves the runtime. And gshare loses to plain bimodal on this program: the
loops are short and regular, so the global history only adds aliasing.

---

## The animation

`web/index.html` puts the animation on the chip's own wiring diagram. Along
the top, a card for each instruction travels left to right through the five
steps, changing colour as it goes; underneath, the values move along the
wires they actually travel.

Each clock tick is broken into its separate moments, told in order, so it
reads as a story rather than a flash of simultaneous activity:

```
step 1 · fetch    The CPU asks memory for the instruction at 0x1010.
step 1 · fetch    Reading these bytes out of memory.
                  the 4 bytes at 0x1010 are 00c42023  ->  sw a2, +0(s0)
step 1 · fetch    Found it in the instruction cache straight away - no waiting.
step 2 · decode   The decoder works out that this is a store to memory,
                  and collects a2 and s0.
step 3 · execute  Shortcut: s0 was only just calculated and has not been saved
                  yet, so the value 8192 is handed straight back to the maths
                  unit instead of making it wait.
step 4 · memory   Writing 13 into memory at 0x2000.
step 5 · save     There is no answer to save this tick.
clock edge        The clock ticks. Everything that finished its step moves
                  one place right.
```

Empty steps get their own beat rather than being skipped, because a bubble
is a real thing worth seeing. Long uneventful stalls play at speed with a
counter showing what the wait costs, so a 50-tick memory access reads as
expensive without being fifty seconds of dead air.

It is driven by the same trace the terminal uses, so the two can never
disagree with each other.

```bash
python web/build_traces.py     # regenerate the bundled traces
```

Live: **https://claude.ai/artifact/NSAPMdJpNQeVEHcJCHT3hz**

---

## Layout

```
rvsim/
  isa.py         RV32IMA encode / decode, register naming
  assembler.py   two-pass assembler: labels, directives, pseudo-instructions
  csr.py         machine-mode CSRs, trap causes, interrupt arbitration
  core.py        the five-stage pipeline
  bpred.py       bimodal / gshare / static, BTB, return-address stack
  cache.py       set-associative cache with MESI line states
  memsys.py      cache hierarchy, snooping bus, coherence, atomics
  memory.py      DRAM and memory-mapped devices (CLINT, UART, halt)
  system.py      the chip: cores + memory + devices + clock
  trace.py       structured per-cycle trace
  cli.py         run / pipeline / disasm / trace / compare
programs/        demo programs in RISC-V assembly
tests/           51 tests, standard-library unittest
web/             the visualiser and its trace bundles
legacy/          the prototypes this grew out of
```

## Simplifications

Worth stating plainly, since the point of the project is that the rest is
honest:

- One outstanding memory request per core: a miss blocks the pipeline
  rather than allowing hit-under-miss.
- The bus carries one transaction at a time, granted round-robin; there is
  no split-transaction or pipelined bus.
- M-mode only. No virtual memory, no PMP, no floating point.
- `.aq` / `.rl` ordering suffixes are accepted and ignored — with a single
  in-order memory port per core there is nothing for them to reorder.
- L2 is shared and inclusive but not itself snooped; coherence is enforced
  between the L1 data caches.
