"""Test suite for the RV32IM simulator. Standard library only:

    python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rvsim.assembler import assemble, AsmError
from rvsim.cache import Cache, CacheConfig, MODIFIED, SHARED, INVALID
from rvsim.core import CoreConfig
from rvsim.csr import (CAUSE_ECALL_M, CAUSE_ILLEGAL_INSTRUCTION,
                       CAUSE_LOAD_MISALIGNED, MCAUSE, MEPC)
from rvsim.isa import ABI_NAMES, decode, encode, reg_number
from rvsim.memsys import MemoryConfig
from rvsim.system import System, SystemConfig

HALT = 0x10000004

PROLOGUE = """
    .equ HALT_ADDR, 0x10000004
"""

EPILOGUE = """
    .text
_halt:
    li   t6, HALT_ADDR
    sw   x0, 0(t6)
    j    _halt
"""


def run(source, cores=1, cycles=20000, **core_kwargs):
    """Assemble and run, returning the finished System."""
    mem = core_kwargs.pop("memory", None) or MemoryConfig()
    timer = core_kwargs.pop("timer_period", 0)
    program = assemble(PROLOGUE + source + EPILOGUE)
    cfg = SystemConfig(num_cores=cores, core=CoreConfig(**core_kwargs),
                       memory=mem, max_cycles=cycles, timer_period=timer)
    system = System(program, cfg)
    system.run()
    return system


def reg(system, name, core=0):
    return system.cores[core].regs[reg_number(name)]


def sreg(system, name, core=0):
    value = reg(system, name, core)
    return value - (1 << 32) if value & 0x80000000 else value


# --------------------------------------------------------------------------


class TestEncoding(unittest.TestCase):
    def test_round_trip_every_format(self):
        cases = [
            ("addi", dict(rd=1, rs1=2, imm=-2048)),
            ("addi", dict(rd=31, rs1=0, imm=2047)),
            ("andi", dict(rd=3, rs1=4, imm=-1)),
            ("slli", dict(rd=5, rs1=6, imm=31)),
            ("srai", dict(rd=5, rs1=6, imm=17)),
            ("add", dict(rd=7, rs1=8, rs2=9)),
            ("sub", dict(rd=7, rs1=8, rs2=9)),
            ("sra", dict(rd=7, rs1=8, rs2=9)),
            ("mul", dict(rd=10, rs1=11, rs2=12)),
            ("divu", dict(rd=10, rs1=11, rs2=12)),
            ("remu", dict(rd=10, rs1=11, rs2=12)),
            ("lw", dict(rd=13, rs1=14, imm=-4)),
            ("lbu", dict(rd=13, rs1=14, imm=2047)),
            ("sw", dict(rs1=15, rs2=16, imm=-2048)),
            ("sh", dict(rs1=15, rs2=16, imm=1000)),
            ("beq", dict(rs1=17, rs2=18, imm=-4096)),
            ("bgeu", dict(rs1=17, rs2=18, imm=4094)),
            ("jal", dict(rd=19, imm=-1048576)),
            ("jal", dict(rd=19, imm=1048574)),
            ("jalr", dict(rd=20, rs1=21, imm=-1)),
            ("lui", dict(rd=22, imm=0xABCDE000)),
            ("auipc", dict(rd=22, imm=0x12345000)),
            ("csrrw", dict(rd=23, rs1=24, csr=0x305)),
            ("csrrsi", dict(rd=23, imm=17, csr=0x344)),
        ]
        for name, kwargs in cases:
            with self.subTest(name=name, **kwargs):
                word = encode(name, **kwargs)
                ins = decode(word)
                self.assertEqual(ins.name, name)
                for field, expected in kwargs.items():
                    if field == "imm" and name in ("lui", "auipc"):
                        self.assertEqual(ins.imm & 0xFFFFF000,
                                         expected & 0xFFFFF000)
                    else:
                        self.assertEqual(getattr(ins, field), expected,
                                         "%s.%s" % (name, field))

    def test_illegal_word_decodes_as_illegal(self):
        self.assertTrue(decode(0xFFFFFFFF).illegal)
        self.assertTrue(decode(0x00000000).illegal)

    def test_nop_is_recognised(self):
        ins = decode(0x00000013)
        self.assertEqual(ins.name, "nop")
        self.assertFalse(ins.writes_rd)


class TestAssembler(unittest.TestCase):
    def test_labels_and_pseudo_ops(self):
        prog = assemble("""
            .text
        _start:
            li   a0, 1
            li   a1, 0x deadbeef
        """.replace("0x dead", "0xdead"))
        words = [sl.word for sl in prog.listing]
        self.assertEqual(len(words), 3)  # li small = 1, li large = 2
        self.assertEqual(prog.symbols["_start"], prog.text_base)

    def test_li_materialises_exact_values(self):
        for value in (0, 1, -1, 2047, -2048, 2048, 0x7FFFFFFF, -0x80000000,
                      0xDEADBEEF, 0x800, 0xFFFFF800):
            with self.subTest(value=value):
                system = run("_start:\n li a0, %d\n" % (value if value < 2**31
                                                        else value - 2**32))
                self.assertEqual(reg(system, "a0"), value & 0xFFFFFFFF)

    def test_data_directives(self):
        prog = assemble("""
            .data
        nums:
            .word 1, -1, 0x100
        text:
            .string "ab"
        pad:
            .byte 7
        """)
        self.assertEqual(bytes(prog.data[:12]),
                         (1).to_bytes(4, "little") +
                         (0xFFFFFFFF).to_bytes(4, "little") +
                         (0x100).to_bytes(4, "little"))
        self.assertEqual(bytes(prog.data[12:15]), b"ab\x00")
        self.assertEqual(prog.data[15], 7)

    def test_unknown_instruction_reports_line(self):
        with self.assertRaises(AsmError) as ctx:
            assemble("_start:\n  frobnicate a0, a1\n")
        self.assertIn("line 2", str(ctx.exception))

    def test_la_computes_pc_relative_address(self):
        system = run("""
        _start:
            la   a0, target
            j    after
        target:
            nop
        after:
            nop
        """)
        self.assertEqual(reg(system, "a0"),
                         system.program.symbols["target"])


class TestArithmetic(unittest.TestCase):
    def test_add_sub_and_immediates(self):
        system = run("""
        _start:
            li   t0, 100
            li   t1, 42
            add  a0, t0, t1
            sub  a1, t0, t1
            addi a2, t0, -150
            and  a3, t0, t1
            or   a4, t0, t1
            xor  a5, t0, t1
        """)
        self.assertEqual(reg(system, "a0"), 142)
        self.assertEqual(reg(system, "a1"), 58)
        self.assertEqual(sreg(system, "a2"), -50)
        self.assertEqual(reg(system, "a3"), 100 & 42)
        self.assertEqual(reg(system, "a4"), 100 | 42)
        self.assertEqual(reg(system, "a5"), 100 ^ 42)

    def test_shifts_including_arithmetic_right(self):
        system = run("""
        _start:
            li   t0, -16
            srai a0, t0, 2
            srli a1, t0, 28
            slli a2, t0, 1
            li   t1, 3
            sra  a3, t0, t1
        """)
        self.assertEqual(sreg(system, "a0"), -4)
        self.assertEqual(reg(system, "a1"), 0xF)
        self.assertEqual(reg(system, "a2"), (-32) & 0xFFFFFFFF)
        self.assertEqual(sreg(system, "a3"), -2)

    def test_set_less_than_signed_and_unsigned(self):
        system = run("""
        _start:
            li   t0, -1
            li   t1, 1
            slt  a0, t0, t1
            sltu a1, t0, t1
            slti a2, t0, 0
            sltiu a3, t0, 0
        """)
        self.assertEqual(reg(system, "a0"), 1)   # -1 < 1 signed
        self.assertEqual(reg(system, "a1"), 0)   # 0xFFFFFFFF > 1 unsigned
        self.assertEqual(reg(system, "a2"), 1)
        self.assertEqual(reg(system, "a3"), 0)

    def test_x0_is_hardwired_to_zero(self):
        system = run("""
        _start:
            li   t0, 1234
            add  x0, t0, t0
            addi x0, x0, 99
            add  a0, x0, x0
        """)
        self.assertEqual(reg(system, "zero"), 0)
        self.assertEqual(reg(system, "a0"), 0)

    def test_multiply_and_divide_semantics(self):
        system = run("""
        _start:
            li   t0, -7
            li   t1, 2
            mul  a0, t0, t1
            div  a1, t0, t1
            rem  a2, t0, t1
            divu a3, t0, t1
            li   t2, 0
            div  a4, t0, t2
            rem  a5, t0, t2
            li   t3, 0x80000000
            li   t4, -1
            div  a6, t3, t4
            rem  a7, t3, t4
            li   t5, 0x10000
            mulhu s2, t5, t5
        """)
        self.assertEqual(sreg(system, "a0"), -14)
        self.assertEqual(sreg(system, "a1"), -3)   # truncates toward zero
        self.assertEqual(sreg(system, "a2"), -1)
        self.assertEqual(reg(system, "a3"), (0xFFFFFFF9) // 2)
        self.assertEqual(sreg(system, "a4"), -1)   # divide by zero
        self.assertEqual(sreg(system, "a5"), -7)   # remainder by zero
        self.assertEqual(reg(system, "a6"), 0x80000000)  # overflow
        self.assertEqual(reg(system, "a7"), 0)
        self.assertEqual(reg(system, "s2"), 1)     # (2^16)^2 >> 32


class TestMemoryInstructions(unittest.TestCase):
    def test_load_store_widths_and_sign_extension(self):
        system = run("""
        _start:
            la   t0, buf
            li   t1, 0xFFFFFF85
            sw   t1, 0(t0)
            lb   a0, 0(t0)
            lbu  a1, 0(t0)
            lh   a2, 0(t0)
            lhu  a3, 0(t0)
            lw   a4, 0(t0)
            li   t2, 0x1234
            sh   t2, 8(t0)
            lhu  a5, 8(t0)
            li   t3, 0x7F
            sb   t3, 12(t0)
            lbu  a6, 12(t0)
            j    _halt
            .data
        buf:
            .space 32
        """)
        self.assertEqual(sreg(system, "a0"), -123)          # 0x85 sign extended
        self.assertEqual(reg(system, "a1"), 0x85)
        self.assertEqual(sreg(system, "a2"), -123)          # 0xFF85
        self.assertEqual(reg(system, "a3"), 0xFF85)
        self.assertEqual(reg(system, "a4"), 0xFFFFFF85)
        self.assertEqual(reg(system, "a5"), 0x1234)
        self.assertEqual(reg(system, "a6"), 0x7F)

    def test_store_then_load_through_the_cache_hierarchy(self):
        system = run("""
        _start:
            la   t0, buf
            li   t1, 0
            li   t2, 64
        fill:
            sw   t1, 0(t0)
            addi t0, t0, 4
            addi t1, t1, 1
            blt  t1, t2, fill
            la   t0, buf
            lw   a0, 0(t0)
            lw   a1, 128(t0)
            lw   a2, 252(t0)
            j    _halt
            .data
        buf:
            .space 512
        """)
        self.assertEqual(reg(system, "a0"), 0)
        self.assertEqual(reg(system, "a1"), 32)
        self.assertEqual(reg(system, "a2"), 63)

    def test_misaligned_load_traps(self):
        system = run("""
        _start:
            csrr t0, mtvec
            la   t1, handler
            csrw mtvec, t1
            la   t2, buf
            addi t2, t2, 1
            lw   a0, 0(t2)
            j    _halt
        handler:
            li   a1, 1
            csrr a2, mcause
            csrr a3, mepc
            j    _halt
            .data
        buf:
            .space 16
        """)
        self.assertEqual(reg(system, "a1"), 1, "handler did not run")
        self.assertEqual(reg(system, "a2"), CAUSE_LOAD_MISALIGNED)


class TestControlFlow(unittest.TestCase):
    def test_all_branch_conditions(self):
        system = run("""
        _start:
            li   s0, 0
            li   t0, 5
            li   t1, 5
            li   t2, -1
            beq  t0, t1, b1
            j    fail
        b1: ori  s0, s0, 1
            bne  t0, t2, b2
            j    fail
        b2: ori  s0, s0, 2
            blt  t2, t0, b3
            j    fail
        b3: ori  s0, s0, 4
            bge  t0, t1, b4
            j    fail
        b4: ori  s0, s0, 8
            bltu t0, t2, b5
            j    fail
        b5: ori  s0, s0, 16
            bgeu t2, t0, b6
            j    fail
        b6: ori  s0, s0, 32
            j    _halt
        fail:
            li   s0, -1
            j    _halt
        """)
        self.assertEqual(reg(system, "s0"), 63)

    def test_function_call_and_return(self):
        system = run("""
        _start:
            li   a0, 6
            li   a1, 7
            call multiply
            mv   s0, a0
            li   a0, 3
            li   a1, 4
            call multiply
            add  s0, s0, a0
            j    _halt
        multiply:
            mul  a0, a0, a1
            ret
        """)
        self.assertEqual(reg(system, "s0"), 42 + 12)

    def test_nested_loop_totals(self):
        system = run("""
        _start:
            li   s0, 0
            li   t0, 0
        outer:
            li   t1, 0
        inner:
            addi s0, s0, 1
            addi t1, t1, 1
            li   t2, 5
            blt  t1, t2, inner
            addi t0, t0, 1
            li   t3, 4
            blt  t0, t3, outer
            j    _halt
        """)
        self.assertEqual(reg(system, "s0"), 20)

    def test_jalr_computed_target(self):
        system = run("""
        _start:
            la   t0, table
            lw   t1, 4(t0)
            jalr ra, t1, 0
            j    _halt
        one:
            li   a0, 1
            j    _halt
        two:
            li   a0, 2
            j    _halt
            .data
        table:
            .word one, two
        """)
        self.assertEqual(reg(system, "a0"), 2)


class TestPipelineBehaviour(unittest.TestCase):
    def test_forwarding_removes_stalls_but_not_results(self):
        source = """
        _start:
            li   t0, 1
            add  t1, t0, t0
            add  t2, t1, t1
            add  t3, t2, t2
            add  a0, t3, t3
            j    _halt
        """
        fast = run(source, forwarding=True)
        slow = run(source, forwarding=False)
        self.assertEqual(reg(fast, "a0"), 16)
        self.assertEqual(reg(fast, "a0"), reg(slow, "a0"),
                         "forwarding must not change architectural results")

    def test_load_use_hazard_costs_a_bubble(self):
        system = run("""
        _start:
            la   t0, buf
            li   t1, 99
            sw   t1, 0(t0)
            lw   t2, 0(t0)
            addi a0, t2, 1
            j    _halt
            .data
        buf:
            .space 8
        """)
        self.assertEqual(reg(system, "a0"), 100)
        self.assertGreaterEqual(system.cores[0].stats.load_use_stalls, 1)

    def test_operands_survive_a_long_cache_miss(self):
        """Regression: an instruction waiting in EX while MEM is blocked on a
        cache miss must still see its producer's result. The forwarding
        sources behind it drain away during the stall."""
        system = run("""
        _start:
            la   s1, table           # auipc then addi: s1 is produced late
            lw   a0, 0(s1)           # cold miss, blocks MEM for ~50 cycles
            lw   a1, 4(s1)           # waits in EX the whole time
            lw   a2, 8(s1)
            add  a3, a1, a2
            j    _halt
            .data
            .align 5
        table:
            .word 1111, 2222, 3333
        """)
        self.assertEqual(reg(system, "a0"), 1111)
        self.assertEqual(reg(system, "a1"), 2222,
                         "stale operand latched before the producer wrote back")
        self.assertEqual(reg(system, "a2"), 3333)
        self.assertEqual(reg(system, "a3"), 2222 + 3333)

    def test_arithmetic_chain_across_a_miss_is_exact(self):
        system = run("""
        _start:
            la   s1, cell
            li   t0, 7
            addi t0, t0, 3           # t0 = 10, produced one cycle before use
            lw   a0, 0(s1)           # cold miss stalls everything behind it
            add  a1, t0, t0          # must be 20, not 14
            mul  a2, t0, a0
            j    _halt
            .data
            .align 5
        cell:
            .word 6
        """)
        self.assertEqual(reg(system, "a0"), 6)
        self.assertEqual(reg(system, "a1"), 20)
        self.assertEqual(reg(system, "a2"), 60)

    def test_branch_predictor_learns_a_hot_loop(self):
        source = """
        _start:
            li   t0, 200
        loop:
            addi t0, t0, -1
            bnez t0, loop
            j    _halt
        """
        system = run(source, branch_scheme="bimodal")
        stats = system.cores[0].bpu
        self.assertGreater(stats.accuracy, 0.9,
                           "a 200-iteration loop should be well predicted")

    def test_mispredict_does_not_corrupt_state(self):
        system = run("""
        _start:
            li   s0, 0
            li   t0, 0
            li   t1, 20
        loop:
            andi t2, t0, 1
            beqz t2, even
            addi s0, s0, 10
            j    next
        even:
            addi s0, s0, 1
        next:
            addi t0, t0, 1
            blt  t0, t1, loop
            j    _halt
        """)
        self.assertEqual(reg(system, "s0"), 10 * 10 + 10 * 1)

    def test_ipc_is_reasonable_on_a_hot_loop(self):
        system = run("""
        _start:
            li   t0, 500
        loop:
            addi t1, t1, 1
            addi t2, t2, 2
            addi t0, t0, -1
            bnez t0, loop
            j    _halt
        """)
        self.assertGreater(system.cores[0].stats.ipc, 0.7)
        self.assertLessEqual(system.cores[0].stats.ipc, 1.0)


class TestCache(unittest.TestCase):
    def test_lru_replacement_evicts_the_oldest(self):
        cache = Cache(CacheConfig(name="T", size=128, block_size=16, assoc=2))
        # 4 sets of 2 ways. Blocks 0 and 128 collide in set 0.
        cache.install(0x000, bytes(16), SHARED)
        cache.install(0x080, bytes(16), SHARED)
        cache.read_bytes(0x000, 4)               # make 0x000 the recent one
        evicted = cache.install(0x100, bytes(16), SHARED)
        self.assertIsNotNone(evicted)
        self.assertEqual(evicted[0], 0x080)
        self.assertIsNotNone(cache.find(0x000))
        self.assertIsNone(cache.find(0x080))

    def test_dirty_eviction_is_reported_for_writeback(self):
        cache = Cache(CacheConfig(name="T", size=64, block_size=16, assoc=1))
        cache.install(0x000, bytes(16), SHARED)
        cache.write_bytes(0x000, b"\x01\x02\x03\x04")
        evicted = cache.install(0x040, bytes(16), SHARED)
        self.assertIsNotNone(evicted)
        self.assertTrue(evicted[2], "eviction should be flagged dirty")
        self.assertEqual(cache.stats.writebacks, 1)

    def test_conflict_misses_appear_with_low_associativity(self):
        """Same capacity, different associativity, addresses chosen to
        collide in both: only the ways make the difference."""
        def cache(assoc):
            mem = MemoryConfig()
            mem.l1d = CacheConfig(name="L1D", size=1024, block_size=32,
                                  assoc=assoc)
            return mem
        source = """
        _start:
            la   t0, buf
            li   t3, 50
        loop:
            lw   t1, 0(t0)
            lw   t2, 1024(t0)
            lw   t4, 2048(t0)
            addi t3, t3, -1
            bnez t3, loop
            j    _halt
            .data
        buf:
            .space 3072
        """
        direct = run(source, memory=cache(1))
        four_way = run(source, memory=cache(4))
        self.assertGreater(direct.memsys.l1d[0].stats.misses,
                           10 * four_way.memsys.l1d[0].stats.misses,
                           "direct-mapped should thrash where 4-way does not")
        self.assertLess(four_way.memsys.l1d[0].stats.misses, 8,
                        "4-way should take only compulsory misses here")

    def test_cache_hit_rate_improves_on_reuse(self):
        system = run("""
        _start:
            la   t0, buf
            li   t3, 40
        loop:
            lw   t1, 0(t0)
            lw   t2, 4(t0)
            addi t3, t3, -1
            bnez t3, loop
            j    _halt
            .data
        buf:
            .space 64
        """)
        self.assertGreater(system.memsys.l1d[0].stats.hit_rate, 0.9)


class TestCoherence(unittest.TestCase):
    """Two cores sharing a line, driven by mhartid."""

    SHARED_SOURCE = """
    _start:
        csrr t0, mhartid
        bnez t0, core1
    core0:
        la   t1, shared
        li   t2, 111
        sw   t2, 0(t1)
        lw   a0, 0(t1)
        j    _halt
    core1:
        la   t1, shared
        lw   a0, 0(t1)
        li   t2, 222
        sw   t2, 0(t1)
        lw   a1, 0(t1)
        j    _halt
        .data
    shared:
        .word 0
    """

    def test_two_cores_generate_coherence_traffic(self):
        system = run(self.SHARED_SOURCE, cores=2)
        self.assertEqual(reg(system, "a0", core=0), 111)
        self.assertEqual(reg(system, "a1", core=1), 222)
        self.assertGreater(len(system.memsys.coherence_log), 0,
                           "sharing a line must produce snoop traffic")

    def test_write_invalidates_the_other_copy(self):
        system = run("""
        _start:
            csrr t0, mhartid
            la   t1, shared
            bnez t0, core1
            lw   a0, 0(t1)
            li   t2, 5
        spin0:
            lw   a1, 4(t1)
            beqz a1, spin0
            j    _halt
        core1:
            lw   a0, 0(t1)
            li   t2, 9
            sw   t2, 0(t1)
            li   t3, 1
            sw   t3, 4(t1)
            j    _halt
            .data
        shared:
            .word 0
            .word 0
        """, cores=2, cycles=60000)
        ops = {entry["op"] for entry in system.memsys.coherence_log}
        self.assertTrue({"BusRdX", "BusUpgr"} & ops,
                        "a write to a shared line must go out as RdX/Upgr")
        states = [entry["to"] for entry in system.memsys.coherence_log]
        self.assertIn(INVALID, states, "the other copy must be invalidated")

    def test_cores_have_independent_register_files(self):
        system = run("""
        _start:
            csrr a0, mhartid
            addi a1, a0, 100
            j    _halt
        """, cores=4)
        for i in range(4):
            self.assertEqual(reg(system, "a0", core=i), i)
            self.assertEqual(reg(system, "a1", core=i), 100 + i)


class TestTrapsAndInterrupts(unittest.TestCase):
    def test_ecall_enters_the_handler_and_mret_returns(self):
        system = run("""
        _start:
            la   t0, handler
            csrw mtvec, t0
            li   s0, 0
            ecall
            addi s0, s0, 1
            j    _halt
        handler:
            csrr a0, mcause
            csrr a1, mepc
            addi a1, a1, 4
            csrw mepc, a1
            li   s1, 7
            mret
        """)
        self.assertEqual(reg(system, "a0"), CAUSE_ECALL_M)
        self.assertEqual(reg(system, "s1"), 7, "handler did not run")
        self.assertEqual(reg(system, "s0"), 1, "mret did not resume")

    def test_illegal_instruction_traps(self):
        system = run("""
        _start:
            la   t0, handler
            csrw mtvec, t0
            csrr a0, 0x7ff
            j    _halt
        handler:
            csrr a1, mcause
            j    _halt
        """)
        self.assertEqual(reg(system, "a1"), CAUSE_ILLEGAL_INSTRUCTION)

    def test_timer_interrupt_fires_and_is_handled(self):
        system = run("""
        _start:
            la   t0, handler
            csrw mtvec, t0
            li   t1, 128
            csrw mie, t1                 # MTIE
            li   t1, 8
            csrs mstatus, t1             # MIE
            li   s0, 0
        spin:
            addi s0, s0, 1
            li   t2, 400
            blt  s0, t2, spin
            j    _halt
        handler:
            li   s1, 1
            csrr s2, mcause
            li   t3, 0xFFFFFFF
            li   t4, 0x2004000
            sw   t3, 0(t4)               # push mtimecmp far out
            mret
        """, timer_period=60, cycles=40000)
        self.assertEqual(reg(system, "s1"), 1, "timer handler never ran")
        self.assertEqual(reg(system, "s2"), 0x80000007,
                         "mcause should be a machine timer interrupt")
        self.assertGreaterEqual(system.cores[0].stats.interrupts, 1)

    def test_interrupt_does_not_lose_or_repeat_work(self):
        """The counter must be exact despite an interrupt mid-loop."""
        system = run("""
        _start:
            la   t0, handler
            csrw mtvec, t0
            li   t1, 128
            csrw mie, t1
            li   t1, 8
            csrs mstatus, t1
            la   s3, counter
            li   s0, 0
            li   t2, 300
        spin:
            lw   t4, 0(s3)
            addi t4, t4, 1
            sw   t4, 0(s3)
            addi s0, s0, 1
            blt  s0, t2, spin
            lw   a0, 0(s3)
            j    _halt
        handler:
            addi s1, s1, 1
            li   t3, 0xFFFFFFF
            li   t5, 0x2004000
            sw   t3, 0(t5)
            mret
            .data
        counter:
            .word 0
        """, timer_period=90, cycles=80000)
        self.assertEqual(reg(system, "a0"), 300,
                         "an interrupt must not duplicate or drop a store")
        self.assertGreaterEqual(reg(system, "s1"), 1)

    def test_software_interrupt_between_cores(self):
        system = run("""
        _start:
            csrr t0, mhartid
            la   t1, handler
            csrw mtvec, t1
            bnez t0, core1
        core0:
            li   t2, 0x2000004           # msip[1]
            li   t3, 1
            sw   t3, 0(t2)
            j    _halt
        core1:
            li   t1, 8
            csrw mie, t1                 # MSIE
            li   t1, 8
            csrs mstatus, t1
            wfi
            j    _halt
        handler:
            li   s1, 42
            li   t4, 0x2000004
            sw   x0, 0(t4)               # acknowledge the IPI
            mret
        """, cores=2, cycles=40000)
        self.assertEqual(reg(system, "s1", core=1), 42,
                         "core 1 should have taken the IPI")

    def test_wfi_parks_the_core(self):
        system = run("""
        _start:
            wfi
            j    _halt
        """, cycles=3000)
        self.assertGreater(system.cores[0].stats.wfi_cycles, 0)
        self.assertEqual(system.halt_reason,
                         "all cores parked with no interrupt source")


class TestAtomics(unittest.TestCase):
    """RV32A: LR/SC reservations and read-modify-write atomics."""

    def test_amo_operations_return_the_old_value(self):
        system = run("""
        _start:
            la   t0, cell
            li   t1, 100
            sw   t1, 0(t0)
            li   t2, 5
            amoadd.w  a0, t2, (t0)     # a0 = 100, cell = 105
            li   t2, 7
            amoswap.w a1, t2, (t0)     # a1 = 105, cell = 7
            li   t2, 0xF0
            amoor.w   a2, t2, (t0)     # a2 = 7,   cell = 0xF7
            li   t2, 0x0F
            amoand.w  a3, t2, (t0)     # a3 = 0xF7, cell = 0x07
            lw   a4, 0(t0)
            j    _halt
            .data
        cell:
            .word 0
        """)
        self.assertEqual(reg(system, "a0"), 100)
        self.assertEqual(reg(system, "a1"), 105)
        self.assertEqual(reg(system, "a2"), 7)
        self.assertEqual(reg(system, "a3"), 0xF7)
        self.assertEqual(reg(system, "a4"), 0x07)

    def test_lr_sc_succeeds_when_uncontended(self):
        system = run("""
        _start:
            la   t0, cell
            lr.w a0, (t0)
            li   t1, 77
            sc.w a1, t1, (t0)          # should succeed -> a1 = 0
            lw   a2, 0(t0)
            j    _halt
            .data
        cell:
            .word 5
        """)
        self.assertEqual(reg(system, "a0"), 5)
        self.assertEqual(reg(system, "a1"), 0, "sc.w should have succeeded")
        self.assertEqual(reg(system, "a2"), 77)

    def test_sc_fails_without_a_reservation(self):
        system = run("""
        _start:
            la   t0, cell
            li   t1, 99
            sc.w a0, t1, (t0)          # no lr.w first -> must fail
            lw   a1, 0(t0)
            j    _halt
            .data
        cell:
            .word 5
        """)
        self.assertEqual(reg(system, "a0"), 1, "sc.w should have failed")
        self.assertEqual(reg(system, "a1"), 5, "a failed sc.w must not store")

    def test_remote_write_kills_the_reservation(self):
        """Core 0 reserves, core 1 writes, core 0's sc.w must fail."""
        system = run("""
        _start:
            csrr t0, mhartid
            la   t1, cell
            la   t2, flag
            bnez t0, core1
        core0:
            lr.w a0, (t1)              # reserve
            li   t3, 1
            sw   t3, 0(t2)             # tell core 1 to go
        wait:
            lw   t4, 4(t2)             # spin until core 1 is done
            beqz t4, wait
            li   t5, 111
            sc.w a1, t5, (t1)          # reservation is gone -> must fail
            lw   a2, 0(t1)
            j    _halt
        core1:
        spin:
            lw   t4, 0(t2)
            beqz t4, spin
            li   t5, 222
            sw   t5, 0(t1)             # steal the line
            li   t5, 1
            sw   t5, 4(t2)
            j    _halt
            .data
            .align 5
        cell:
            .word 0
            .space 28
            .align 5
        flag:
            .word 0
            .word 0
        """, cores=2, cycles=60000)
        self.assertEqual(reg(system, "a1", core=0), 1,
                         "sc.w must fail after another core wrote the line")
        self.assertEqual(reg(system, "a2", core=0), 222,
                         "core 0 should see core 1's value, not its own")

    def test_four_cores_increment_a_counter_exactly(self):
        """The classic test: an atomic counter must not lose an update."""
        system = run("""
        _start:
            csrr s0, mhartid
            la   s1, counter
            li   s2, 25
        loop:
            beqz s2, done
            li   t0, 1
            amoadd.w t1, t0, (s1)
            addi s2, s2, -1
            j    loop
        done:
            j    _halt
            .data
            .align 5
        counter:
            .word 0
        """, cores=4, cycles=100000)
        addr = system.program.symbols["counter"]
        self.assertEqual(system.memsys.dram.read(addr, 4), 4 * 25)

    def test_spinlock_protects_a_non_atomic_critical_section(self):
        system = run("""
        _start:
            csrr s0, mhartid
            la   s1, lock
            la   s2, counter
            li   s3, 15
        outer:
            beqz s3, done
        acquire:
            lw   t0, 0(s1)
            bnez t0, acquire
            li   t1, 1
            amoswap.w t2, t1, (s1)
            bnez t2, acquire
            lw   t3, 0(s2)
            addi t3, t3, 1
            sw   t3, 0(s2)
            sw   x0, 0(s1)
            addi s3, s3, -1
            j    outer
        done:
            j    _halt
            .data
            .align 5
        lock:
            .word 0
            .space 28
            .align 5
        counter:
            .word 0
        """, cores=3, cycles=120000)
        addr = system.program.symbols["counter"]
        self.assertEqual(system.memsys.dram.read(addr, 4), 3 * 15,
                         "a lost update means the lock or MESI is broken")


class TestCoherenceInvariants(unittest.TestCase):
    def test_at_most_one_core_holds_a_line_modified(self):
        """MESI's core invariant, checked on every cycle of a real run."""
        system = run("""
        _start:
            csrr s0, mhartid
            la   s1, shared
            li   s2, 12
        loop:
            beqz s2, done
            li   t0, 1
            amoadd.w t1, t0, (s1)
            lw   t2, 0(s1)
            addi s2, s2, -1
            j    loop
        done:
            j    _halt
            .data
            .align 5
        shared:
            .word 0
        """, cores=3, cycles=120000)

        # Replay the per-cycle cache deltas and check the invariant.
        state = {}
        for record in system.tracer.records:
            for update in record.cache_updates:
                if update["cache"] != "L1D":
                    continue
                state[(update["core"], update["set"], update["way"])] = (
                    update["state"], update["addr"])
            owners = {}
            for (core, _s, _w), (line_state, addr) in state.items():
                if line_state == "M" and addr is not None:
                    owners.setdefault(addr, []).append(core)
            for addr, holders in owners.items():
                self.assertEqual(
                    len(holders), 1,
                    "cycle %d: 0x%08x Modified in cores %s"
                    % (record.cycle, addr, holders))

    def test_shared_lines_may_be_replicated(self):
        system = run("""
        _start:
            csrr s0, mhartid
            la   s1, table
            lw   a0, 0(s1)
            lw   a1, 4(s1)
            j    _halt
            .data
            .align 5
        table:
            .word 1234, 5678
        """, cores=4, cycles=60000)
        for i in range(4):
            self.assertEqual(reg(system, "a0", core=i), 1234)
            self.assertEqual(reg(system, "a1", core=i), 5678)


class TestTrace(unittest.TestCase):
    def test_trace_records_every_cycle_and_is_serialisable(self):
        import json
        system = run("""
        _start:
            li   a0, 3
            li   a1, 4
            add  a2, a0, a1
            j    _halt
        """)
        payload = system.tracer.to_dict()
        self.assertEqual(len(payload["cycles"]), system.cycle)
        blob = json.dumps(payload)
        self.assertGreater(len(blob), 100)
        kinds = {event["k"] for record in payload["cycles"]
                 for event in record.get("ev", [])}
        for expected in ("fetch", "alu", "regwrite", "retire", "mem"):
            self.assertIn(expected, kinds)

    def test_stage_view_never_double_books_an_instruction(self):
        system = run("""
        _start:
            li   t0, 4
        loop:
            addi t0, t0, -1
            bnez t0, loop
            j    _halt
        """)
        for record in system.tracer.records:
            seqs = [seq for seq in record.cores[0]["st"] if seq >= 0]
            self.assertEqual(len(seqs), len(set(seqs)),
                             "cycle %d has one instruction in two stages"
                             % record.cycle)

    def test_instruction_table_covers_every_stage_reference(self):
        system = run("""
        _start:
            li   t0, 3
        loop:
            addi t0, t0, -1
            bnez t0, loop
            j    _halt
        """)
        table = system.tracer.instrs
        for record in system.tracer.records:
            for core_id, view in enumerate(record.cores):
                for seq in view["st"]:
                    if seq >= 0:
                        self.assertIn("%d:%d" % (core_id, seq), table)


if __name__ == "__main__":
    unittest.main(verbosity=2)
