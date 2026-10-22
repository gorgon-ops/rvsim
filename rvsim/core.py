"""A five-stage in-order RISC-V core: IF, ID, EX, MEM, WB.

Forwarding and the load-use bubble it cannot remove, fetch-time prediction with a
two-cycle mispredict flush, multi-cycle mul/div, blocking cache misses, precise
traps committed in WB, and interrupts taken at an instruction boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import csr as csrmod
from .bpred import KIND_BRANCH, KIND_JUMP, KIND_RETURN, BranchUnit
from .csr import CSRFile, Trap
from .isa import ABI_NAMES, MASK32, Instr, decode, to_signed, to_unsigned
from .memory import is_mmio

STAGES = ["IF", "ID", "EX", "MEM", "WB"]


@dataclass
class CoreConfig:
    forwarding: bool = True
    branch_scheme: str = "gshare"
    bpred_entries: int = 64
    btb_entries: int = 32
    ras_depth: int = 8
    history_bits: int = 6
    mul_latency: int = 3
    div_latency: int = 12
    branch_penalty: int = 2      # informational; the flush does the work
    serialize_system: bool = True

    def describe(self) -> list:
        return [
            "pipeline: 5-stage in-order (IF ID EX MEM WB)",
            "forwarding: %s" % ("EX/MEM + MEM/WB" if self.forwarding else "disabled"),
            "branch prediction: %s, BTB %d entries, RAS %d deep"
            % (self.branch_scheme, self.btb_entries, self.ras_depth),
            "mul latency: %d, div latency: %d" % (self.mul_latency, self.div_latency),
        ]


@dataclass
class Slot:
    instr: Instr | None = None
    seq: int = -1
    pc: int = 0
    next_pc: int = 0
    pred_taken: bool = False
    pred_target: int = 0
    pred_info: dict = field(default_factory=dict)
    pred_kind: str = ""
    rs1_val: int = 0
    rs2_val: int = 0
    alu: int = 0
    mem_value: int = 0
    store_data: bytes = b""
    taken: bool = False
    target: int = 0
    trap: Trap | None = None
    mem_req: object = None
    ex_timer: int = 0
    fwd1: str = ""
    fwd2: str = ""
    operands_ready: bool = False
    id_rs1_val: int = 0
    id_rs2_val: int = 0
    squashed: bool = False

    @property
    def empty(self) -> bool:
        return self.instr is None

    def label(self) -> str:
        return self.instr.text() if self.instr else ""


BUBBLE = Slot()


def from_memory(instr) -> bool:
    return instr.is_load or instr.is_atomic


@dataclass
class CoreStats:
    cycles: int = 0
    retired: int = 0
    fetched: int = 0
    stall_cycles: int = 0
    load_use_stalls: int = 0
    icache_stalls: int = 0
    dcache_stalls: int = 0
    exec_stalls: int = 0
    serialize_stalls: int = 0
    flushes: int = 0
    branches: int = 0
    mispredicts: int = 0
    traps: int = 0
    interrupts: int = 0
    forwards: int = 0
    wfi_cycles: int = 0

    @property
    def ipc(self) -> float:
        return self.retired / self.cycles if self.cycles else 0.0

    @property
    def cpi(self) -> float:
        return self.cycles / self.retired if self.retired else 0.0

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["ipc"] = round(self.ipc, 4)
        d["cpi"] = round(self.cpi, 4)
        return d


class Core:
    def __init__(self, hart_id: int, memsys, config: CoreConfig | None = None,
                 emit=None, entry: int = 0):
        self.id = hart_id
        self.cfg = config or CoreConfig()
        self.mem = memsys
        self.emit = emit or (lambda *a, **k: None)
        self.csr = CSRFile(hart_id)
        self.bpu = BranchUnit(self.cfg.branch_scheme, self.cfg.bpred_entries,
                              self.cfg.btb_entries, self.cfg.ras_depth,
                              self.cfg.history_bits)
        self.regs = [0] * 32
        self.reg_writer = [None] * 32   # seq of the last writer, for the UI
        self.pc = entry
        self.stats = CoreStats()
        self.halted = False
        self.exit_code = None

        self.if_id = Slot()
        self.id_ex = Slot()
        self.ex_mem = Slot()
        self.mem_wb = Slot()

        self.fetch_req = None
        self.fetch_pc = entry
        self.next_seq = 0
        self.cycle = 0
        self.stall_reason = ""
        self.redirect = None
        self.retired_this_cycle = None
        self.reg_writes = []       # (reg, value) this cycle, for the trace
        self.csr_writes = []
        self.events = []
        self.trap_pending = None   # interrupt accepted, pipeline draining
        # Occupancy as it was during the cycle, captured before the latches move,
        # or every instruction would be labelled one stage early.
        self._latched = (Slot(), Slot(), Slot(), Slot())
        self.if_view = None

    # ------------------------------------------------------------------
    # Register file
    # ------------------------------------------------------------------

    def read_reg(self, index: int) -> int:
        return 0 if index == 0 else self.regs[index]

    def write_reg(self, index: int, value: int, seq: int = -1) -> None:
        if index == 0:
            return
        self.regs[index] = value & MASK32
        self.reg_writer[index] = seq
        self.reg_writes.append((index, value & MASK32))

    # ------------------------------------------------------------------
    # One clock
    # ------------------------------------------------------------------

    def tick(self, cycle: int) -> None:
        self.cycle = cycle
        self.stats.cycles += 1
        self.stall_reason = ""
        self.redirect = None
        self.reg_writes = []
        self.csr_writes = []
        self.retired_this_cycle = None
        if self.halted:
            self._latched = (Slot(), Slot(), Slot(), Slot())
            self.if_view = None
            return

        # Each stage works on what the latches hold now, before anything advances.
        self._latched = (self.if_id, self.id_ex, self.ex_mem, self.mem_wb)
        self.if_view = None

        # Taken at an instruction boundary: the front end stops, in-flight work
        # retires, and the trap commits with mepc at the next unfetched instruction.
        if self.trap_pending is None:
            cause = self.csr.pending_interrupt()
            if cause is not None:
                if self.csr.wfi and not self.csr.global_ie:
                    # WFI wakes on a pending interrupt even with interrupts disabled, but does
                    # not trap.
                    self.csr.wfi = False
                    self.emit("trap", core=self.id, kind="wfi-resume",
                              note="wfi resumed without trapping (MIE=0)")
                else:
                    self.trap_pending = cause
                    self.emit("trap", core=self.id, kind="interrupt-pending",
                              cause=cause, name=csrmod.cause_name(cause),
                              note="%s accepted; draining the pipeline"
                                   % csrmod.cause_name(cause))
        if self.trap_pending is not None and self._pipeline_empty():
            cause = self.trap_pending
            self.trap_pending = None
            handler = self._take_trap(cause, self.pc, 0, interrupt=True)
            self._flush_pipeline(handler)
            return

        # Every stage reads the start-of-cycle state, like real flip-flops.
        cur_if_id, cur_id_ex = self.if_id, self.id_ex
        cur_ex_mem, cur_mem_wb = self.ex_mem, self.mem_wb

        # --- WB ---------------------------------------------------------
        flush_all = self._writeback(cur_mem_wb)

        # --- MEM --------------------------------------------------------
        mem_ready = self._memory(cur_ex_mem)

        # --- EX ---------------------------------------------------------
        # Runs even when MEM is blocked: operands must be captured the cycle they are
        # available, before the forwarding sources drain away.
        ex_ready = self._execute(cur_id_ex, cur_ex_mem, cur_mem_wb,
                                 mem_ready) and mem_ready

        # --- ID ---------------------------------------------------------
        load_use = self._data_hazard(cur_if_id, cur_id_ex, cur_ex_mem)
        serialize = self._serialize_hazard(cur_if_id, cur_id_ex, cur_ex_mem,
                                           cur_mem_wb)
        id_ready = ex_ready and not load_use and not serialize
        decoded = self._decode(cur_if_id) if id_ready else None

        # --- IF ---------------------------------------------------------
        fetched, fetch_ready = self._fetch(id_ready)
        if_ready = id_ready and fetch_ready

        # --- latch updates ---------------------------------------------
        if flush_all is not None:
            self._flush_pipeline(flush_all)
            return

        if mem_ready:
            self.mem_wb = self._mem_result(cur_ex_mem)
        else:
            self.mem_wb = Slot()
            self.stats.dcache_stalls += 1
            self.stall_reason = "D-cache miss (%s)" % (
                cur_ex_mem.instr.text() if cur_ex_mem.instr else "memory")

        if mem_ready:
            if ex_ready:
                self.ex_mem = self._ex_result(cur_id_ex)
            else:
                self.ex_mem = Slot()
                if not cur_id_ex.empty:
                    self.stats.exec_stalls += 1
                    self.stall_reason = "multi-cycle %s in EX" % cur_id_ex.instr.name

        if ex_ready:
            if id_ready:
                self.id_ex = decoded
            else:
                self.id_ex = Slot()
                if load_use:
                    self.stats.load_use_stalls += 1
                    self.stall_reason = ("load-use interlock on %s" % load_use
                                         if self.cfg.forwarding
                                         else "RAW interlock on %s (no forwarding)"
                                              % load_use)
                elif serialize:
                    self.stats.serialize_stalls += 1
                    self.stall_reason = "waiting for serializing %s" % serialize

        if id_ready:
            if fetch_ready:
                self.if_id = fetched
            else:
                self.if_id = Slot()
                if not self.csr.wfi:
                    self.stats.icache_stalls += 1
                    self.stall_reason = self.stall_reason or "I-cache miss"

        if self.stall_reason:
            self.stats.stall_cycles += 1

        # After the latches moved, so the mispredicted instructions are the ones
        # dropped.
        if self.redirect is not None:
            self._squash_front(self.redirect)

    # ------------------------------------------------------------------
    # IF
    # ------------------------------------------------------------------

    def _fetch(self, may_advance: bool):
        if self.csr.wfi:
            self.stats.wfi_cycles += 1
            return Slot(), False
        if self.trap_pending is not None:
            return Slot(), False
        if self.fetch_req is None:
            self.fetch_pc = self.pc
            self.emit("fetch", core=self.id, pc=self.pc,
                      note="fetch 0x%08x" % self.pc)
            self.fetch_req = self.mem.issue(self.id, "fetch", self.pc, 4)
        # IF is occupied this cycle by the instruction about to get this sequence
        # number.
        self.if_view = {"seq": self.next_seq, "pc": self.fetch_pc,
                        "text": "<fetch>", "name": "fetch", "stage": "IF",
                        "cat": "fetch", "rd": "", "rs1": "", "rs2": "",
                        "trap": None, "fwd1": "", "fwd2": ""}
        if not self.fetch_req.done:
            return Slot(), False

        word = self.fetch_req.value
        pc = self.fetch_pc
        instr = decode(word, pc)
        instr.seq = self.next_seq
        instr.core = self.id
        slot = Slot(instr=instr, seq=self.next_seq, pc=pc)

        # Nothing is decoded yet, so only the BTB knows this PC is a control transfer.
        entry = self.bpu.btb.lookup(pc)
        next_pc = (pc + 4) & MASK32
        if entry is not None:
            target, kind = entry
            if kind == KIND_RETURN:
                # The same ret goes somewhere different for every caller.
                predicted = self.bpu.ras.pop()
                self.bpu.ras_used += 1
                taken = True
                target = predicted if predicted is not None else target
                info = {"scheme": "RAS", "depth": len(self.bpu.ras.stack),
                        "reason": "return address stack"}
            elif kind == KIND_JUMP:
                # Unconditional: always taken, and the direction predictor never trains on it.
                taken = True
                info = {"scheme": "BTB", "reason": "unconditional jump"}
            else:
                taken, info = self.bpu.predictor.predict(pc, target - pc)
            slot.pred_info = info
            slot.pred_kind = kind
            if taken:
                next_pc = target
                slot.pred_taken = True
                slot.pred_target = target
            self.emit("predict", core=self.id, pc=pc, taken=taken,
                      target=target, info=info, kind=kind,
                      note="BTB hit (%s) -> predict %s 0x%08x"
                           % (kind, "taken" if taken else "not-taken", target))
        slot.next_pc = next_pc

        if may_advance:
            self.next_seq += 1
            self.stats.fetched += 1
            self.pc = next_pc
            self.fetch_req = None
        return slot, True

    # ------------------------------------------------------------------
    # ID
    # ------------------------------------------------------------------

    def _decode(self, slot: Slot) -> Slot:
        if slot.empty:
            return Slot()
        out = Slot(instr=slot.instr, seq=slot.seq, pc=slot.pc,
                   next_pc=slot.next_pc, pred_taken=slot.pred_taken,
                   pred_target=slot.pred_target, pred_info=slot.pred_info,
                   pred_kind=slot.pred_kind)
        ins = slot.instr
        out.rs1_val = self.read_reg(ins.rs1)
        out.rs2_val = self.read_reg(ins.rs2)
        reads = []
        if ins.uses_rs1:
            reads.append((ins.rs1, out.rs1_val))
        if ins.uses_rs2:
            reads.append((ins.rs2, out.rs2_val))
        if reads:
            self.emit("regread", core=self.id, seq=slot.seq,
                      regs=[{"reg": r, "name": ABI_NAMES[r], "value": v}
                            for r, v in reads],
                      note="read " + ", ".join("%s=0x%08x" % (ABI_NAMES[r], v)
                                               for r, v in reads))
        if ins.illegal:
            out.trap = Trap(csrmod.CAUSE_ILLEGAL_INSTRUCTION, ins.raw)
        # Trained in decode, where the call is finally known. "call label" assembles
        # to auipc + jalr ra, so jalr counts too.
        if ins.name in ("jal", "jalr") and ins.rd in (1, 5):
            self.bpu.ras.push((slot.pc + 4) & MASK32)
            self.emit("predict", core=self.id, pc=slot.pc, kind="call",
                      note="call: pushed return address 0x%08x onto the RAS"
                           % ((slot.pc + 4) & MASK32))
        return out

    def _data_hazard(self, if_id: Slot, id_ex: Slot, ex_mem: Slot):
        """With forwarding, only a load in EX stalls. Without, any producer in EX or MEM
        must drain; WB is fine, as the register file writes before it is read.
        """
        if if_id.empty:
            return None
        consumer = if_id.instr
        if not consumer.reads:
            return None

        def blocks(slot: Slot, load_only: bool):
            if slot.empty:
                return None
            producer = slot.instr
            if not producer.writes_rd or producer.rd == 0:
                return None
            if load_only and not from_memory(producer):
                return None
            if producer.rd in consumer.reads:
                return ABI_NAMES[producer.rd]
            return None

        if self.cfg.forwarding:
            return blocks(id_ex, load_only=True)
        return blocks(id_ex, load_only=False) or blocks(ex_mem, load_only=False)

    def _serialize_hazard(self, if_id: Slot, *later: Slot):
        if not self.cfg.serialize_system or if_id.empty:
            return None
        for slot in later:
            if not slot.empty and (slot.instr.is_system or slot.instr.is_fence):
                return slot.instr.name
        return None

    # ------------------------------------------------------------------
    # EX
    # ------------------------------------------------------------------

    def _execute(self, slot: Slot, ex_mem: Slot, mem_wb: Slot,
                 mem_ready: bool) -> bool:
        if slot.empty:
            return True
        ins = slot.instr
        if slot.trap is not None:
            return True

        # ---- capture operands, exactly once ------------------------------
        # Forwarding only works while the producer is in EX/MEM or MEM/WB; during a
        # long stall in EX they drain away, so latch on the first cycle possible.
        if not slot.operands_ready:
            waiting_on = self._blocked_operand(slot, ex_mem, mem_ready)
            if waiting_on is not None:
                # The producer is stuck in MEM: nothing to capture yet.
                return False
            self._operands(slot, ex_mem, mem_wb, mem_ready)
            slot.operands_ready = True

        # Multi-cycle functional units hold EX for several cycles.
        if ins.is_muldiv:
            need = (self.cfg.div_latency
                    if ins.name.startswith(("div", "rem"))
                    else self.cfg.mul_latency)
            if slot.ex_timer < need - 1:
                slot.ex_timer += 1
                self.emit("exec", core=self.id, seq=slot.seq,
                          note="%s busy in EX (%d/%d cycles)"
                               % (ins.name, slot.ex_timer, need))
                return False

        if not mem_ready:
            # Computing now, with MEM blocked, would resolve the same branch or trap
            # twice.
            return False

        a, b = slot.rs1_val, slot.rs2_val
        try:
            self._alu(slot, a, b)
        except Trap as trap:
            slot.trap = trap
        return True

    def _blocked_operand(self, slot: Slot, ex_mem: Slot, mem_ready: bool):
        if mem_ready or ex_mem.empty:
            return None
        producer = ex_mem.instr
        if not producer.writes_rd or producer.rd == 0:
            return None
        if producer.rd in slot.instr.reads:
            return ABI_NAMES[producer.rd]
        return None

    def _operands(self, slot: Slot, ex_mem: Slot, mem_wb: Slot,
                  mem_ready: bool):
        """A fresh register-file read with forwarding on top: correct under long stalls, as
        any older producer has either written back or is still forwardable.
        """
        ins = slot.instr
        slot.id_rs1_val, slot.id_rs2_val = slot.rs1_val, slot.rs2_val
        a, b = self.read_reg(ins.rs1), self.read_reg(ins.rs2)
        slot.fwd1 = slot.fwd2 = ""
        if not self.cfg.forwarding:
            return a, b

        def source(reg: int):
            if reg == 0:
                return None
            if (mem_ready and not ex_mem.empty and ex_mem.instr.writes_rd
                    and ex_mem.instr.rd == reg and ex_mem.trap is None):
                producer = ex_mem.instr
                value = (ex_mem.mem_value if from_memory(producer)
                         else ex_mem.alu)
                return value, "EX/MEM"
            if (not mem_wb.empty and mem_wb.instr.writes_rd
                    and mem_wb.instr.rd == reg and mem_wb.trap is None):
                producer = mem_wb.instr
                value = (mem_wb.mem_value if from_memory(producer)
                         else mem_wb.alu)
                return value, "MEM/WB"
            return None

        if ins.uses_rs1:
            found = source(ins.rs1)
            if found:
                a, slot.fwd1 = found[0], found[1]
                self.stats.forwards += 1
                self.emit("forward", core=self.id, seq=slot.seq,
                          reg=ins.rs1, name=ABI_NAMES[ins.rs1],
                          value=a & MASK32, path=slot.fwd1,
                          note="forward %s from %s" % (ABI_NAMES[ins.rs1], slot.fwd1))
        if ins.uses_rs2:
            found = source(ins.rs2)
            if found:
                b, slot.fwd2 = found[0], found[1]
                self.stats.forwards += 1
                self.emit("forward", core=self.id, seq=slot.seq,
                          reg=ins.rs2, name=ABI_NAMES[ins.rs2],
                          value=b & MASK32, path=slot.fwd2,
                          note="forward %s from %s" % (ABI_NAMES[ins.rs2], slot.fwd2))
        slot.rs1_val, slot.rs2_val = a & MASK32, b & MASK32
        return a & MASK32, b & MASK32

    def _alu(self, slot: Slot, a: int, b: int) -> None:
        ins = slot.instr
        name = ins.name
        sa, sb = to_signed(a), to_signed(b)
        result = 0

        if name in ("lui",):
            result = ins.imm & MASK32
        elif name == "auipc":
            result = (slot.pc + ins.imm) & MASK32
        elif name in ("jal", "jalr"):
            result = (slot.pc + 4) & MASK32
            if name == "jal":
                target = (slot.pc + ins.imm) & MASK32
            else:
                target = (a + ins.imm) & ~1 & MASK32
            slot.taken, slot.target = True, target
            self._resolve_branch(slot, True, target)
        elif ins.is_branch:
            taken = {
                "beq": sa == sb, "bne": sa != sb, "blt": sa < sb,
                "bge": sa >= sb, "bltu": a < b, "bgeu": a >= b,
            }[name]
            target = (slot.pc + ins.imm) & MASK32 if taken else (slot.pc + 4) & MASK32
            slot.taken, slot.target = taken, target
            self._resolve_branch(slot, taken, target)
        elif ins.is_atomic:
            result = a & MASK32          # the address is rs1, no offset
            slot.store_data = (b & MASK32).to_bytes(4, "little")
            if result % 4 != 0:
                cause = (csrmod.CAUSE_LOAD_MISALIGNED if ins.name == "lr.w"
                         else csrmod.CAUSE_STORE_MISALIGNED)
                raise Trap(cause, result)
        elif ins.is_load or ins.is_store:
            result = (a + ins.imm) & MASK32
            if ins.is_store:
                slot.store_data = (b & ((1 << (8 * ins.mem_width)) - 1)).to_bytes(
                    ins.mem_width, "little")
            if result % ins.mem_width != 0:
                cause = (csrmod.CAUSE_STORE_MISALIGNED if ins.is_store
                         else csrmod.CAUSE_LOAD_MISALIGNED)
                raise Trap(cause, result)
        elif ins.is_system:
            result = a  # the effect commits in WB
        elif ins.is_fence:
            result = 0
        else:
            imm = ins.imm
            table = {
                "addi": lambda: sa + imm, "slti": lambda: int(sa < imm),
                "sltiu": lambda: int(a < (imm & MASK32)),
                "xori": lambda: a ^ (imm & MASK32),
                "ori": lambda: a | (imm & MASK32),
                "andi": lambda: a & (imm & MASK32),
                "slli": lambda: a << (imm & 31),
                "srli": lambda: a >> (imm & 31),
                "srai": lambda: sa >> (imm & 31),
                "add": lambda: sa + sb, "sub": lambda: sa - sb,
                "sll": lambda: a << (b & 31), "slt": lambda: int(sa < sb),
                "sltu": lambda: int(a < b), "xor": lambda: a ^ b,
                "srl": lambda: a >> (b & 31), "sra": lambda: sa >> (b & 31),
                "or": lambda: a | b, "and": lambda: a & b,
                "mul": lambda: sa * sb,
                "mulh": lambda: (sa * sb) >> 32,
                "mulhsu": lambda: (sa * b) >> 32,
                "mulhu": lambda: (a * b) >> 32,
                "div": lambda: _div(sa, sb), "divu": lambda: _divu(a, b),
                "rem": lambda: _rem(sa, sb), "remu": lambda: _remu(a, b),
                "nop": lambda: 0,
            }
            if name not in table:
                raise Trap(csrmod.CAUSE_ILLEGAL_INSTRUCTION, ins.raw)
            result = table[name]()
        slot.alu = result & MASK32
        if ins.writes_rd or ins.is_load or ins.is_store:
            self.emit("alu", core=self.id, seq=slot.seq, op=name,
                      a=a & MASK32, b=b & MASK32, result=slot.alu,
                      note="%s: 0x%08x op 0x%08x = 0x%08x"
                           % (name, a & MASK32, b & MASK32, slot.alu))

    @staticmethod
    def _transfer_kind(ins) -> str:
        if ins.is_branch:
            return KIND_BRANCH
        # jalr x0, 0(ra) is the canonical return.
        if ins.name == "jalr" and ins.rd == 0 and ins.rs1 in (1, 5) and ins.imm == 0:
            return KIND_RETURN
        return KIND_JUMP

    def _resolve_branch(self, slot: Slot, taken: bool, target: int) -> None:
        ins = slot.instr
        kind = self._transfer_kind(ins)
        if kind == KIND_BRANCH:
            self.stats.branches += 1
            self.bpu.predictions += 1
            self.bpu.predictor.update(slot.pc, taken)
        else:
            self.bpu.jumps += 1
        if taken:
            self.bpu.btb.update(slot.pc, target, kind)

        actual_next = target if taken else (slot.pc + 4) & MASK32
        mispredicted = actual_next != slot.next_pc
        if kind == KIND_RETURN and slot.pred_kind == KIND_RETURN and not mispredicted:
            self.bpu.ras_correct += 1
        if mispredicted:
            self.stats.mispredicts += 1
            if kind == KIND_BRANCH:
                self.bpu.mispredictions += 1
            else:
                self.bpu.jump_mispredictions += 1
            self.redirect = actual_next
            self.emit("branch", core=self.id, seq=slot.seq, pc=slot.pc,
                      taken=taken, target=actual_next,
                      predicted=slot.next_pc, mispredict=True,
                      note="MISPREDICT: predicted 0x%08x, actually 0x%08x"
                           % (slot.next_pc, actual_next))
        else:
            self.emit("branch", core=self.id, seq=slot.seq, pc=slot.pc,
                      taken=taken, target=actual_next,
                      predicted=slot.next_pc, mispredict=False,
                      note="prediction correct (%s)"
                           % ("taken" if taken else "not taken"))

    # ------------------------------------------------------------------
    # MEM
    # ------------------------------------------------------------------

    def _memory(self, slot: Slot) -> bool:
        if slot.empty or slot.trap is not None:
            return True
        ins = slot.instr
        if not (ins.is_load or ins.is_store or ins.is_atomic):
            return True
        if slot.mem_req is None:
            if ins.is_atomic:
                kind = {"lr.w": "lr", "sc.w": "sc"}.get(ins.name, "amo")
                slot.mem_req = self.mem.issue(
                    self.id, kind, slot.alu, 4, slot.store_data,
                    amo_op=ins.amo_op,
                    amo_operand=int.from_bytes(slot.store_data, "little"))
            else:
                kind = "store" if ins.is_store else "load"
                slot.mem_req = self.mem.issue(self.id, kind, slot.alu,
                                              ins.mem_width, slot.store_data)
        if not slot.mem_req.done:
            return False
        if ins.is_atomic:
            slot.mem_value = slot.mem_req.value & MASK32
            if ins.name == "sc.w":
                note = ("sc.w to 0x%08x %s" %
                        (slot.alu, "failed" if slot.mem_req.sc_failed
                         else "succeeded"))
            elif ins.name == "lr.w":
                note = "lr.w loaded 0x%08x and reserved the line" % slot.mem_value
            else:
                note = "%s returned old value 0x%08x" % (ins.name, slot.mem_value)
            self.emit("memdata", core=self.id, seq=slot.seq, addr=slot.alu,
                      value=slot.mem_value, kind="atomic", note=note)
        elif ins.is_load:
            raw = slot.mem_req.value
            width = ins.mem_width
            if ins.name in ("lb", "lh"):
                slot.mem_value = to_unsigned(_sext(raw, 8 * width))
            else:
                slot.mem_value = raw & MASK32
            self.emit("memdata", core=self.id, seq=slot.seq, addr=slot.alu,
                      value=slot.mem_value, kind="load",
                      note="loaded 0x%08x from 0x%08x" % (slot.mem_value, slot.alu))
        else:
            self.emit("memdata", core=self.id, seq=slot.seq, addr=slot.alu,
                      value=int.from_bytes(slot.store_data, "little"),
                      kind="store",
                      note="stored 0x%0*x to 0x%08x"
                           % (2 * ins.mem_width,
                              int.from_bytes(slot.store_data, "little"), slot.alu))
        return True

    # ------------------------------------------------------------------
    # WB -- the commit point
    # ------------------------------------------------------------------

    def _writeback(self, slot: Slot):
        self.csr.set(csrmod.MCYCLE, self.stats.cycles)
        if slot.empty:
            return None

        if slot.trap is not None:
            return self._take_trap(slot.trap.cause, slot.pc, slot.trap.tval)

        ins = slot.instr
        if ins.is_system:
            redirect = self._commit_system(slot)
            if redirect is not None:
                return redirect

        value = slot.mem_value if from_memory(ins) else slot.alu
        if ins.writes_rd and ins.rd != 0:
            self.write_reg(ins.rd, value, slot.seq)
            self.emit("regwrite", core=self.id, seq=slot.seq, reg=ins.rd,
                      name=ABI_NAMES[ins.rd], value=value & MASK32,
                      note="%s <- 0x%08x" % (ABI_NAMES[ins.rd], value & MASK32))
        self.stats.retired += 1
        self.csr.set(csrmod.MINSTRET, self.stats.retired)
        self.retired_this_cycle = slot
        self.emit("retire", core=self.id, seq=slot.seq, pc=slot.pc,
                  text=ins.text(), note="retired %s" % ins.text())
        return None

    def _commit_system(self, slot: Slot):
        ins = slot.instr
        if ins.name == "ecall":
            return self._take_trap(csrmod.CAUSE_ECALL_M, slot.pc, 0)
        if ins.name == "ebreak":
            return self._take_trap(csrmod.CAUSE_BREAKPOINT, slot.pc, slot.pc)
        if ins.name == "mret":
            target = self.csr.exit_trap()
            self.emit("trap", core=self.id, seq=slot.seq, kind="mret",
                      target=target,
                      note="mret -> 0x%08x (MIE restored from MPIE)" % target)
            self.stats.retired += 1
            self.retired_this_cycle = slot
            return target
        if ins.name == "wfi":
            self.csr.wfi = True
            self.stats.retired += 1
            self.retired_this_cycle = slot
            self.emit("trap", core=self.id, seq=slot.seq, kind="wfi",
                      note="wfi: core parked until an interrupt arrives")
            # Park at the following instruction and squash the front end.
            return (slot.pc + 4) & MASK32
        if ins.is_csr:
            try:
                old = self.csr.read(ins.csr)
                source = slot.rs1_val if not ins.name.endswith("i") else ins.imm
                if ins.name in ("csrrw", "csrrwi"):
                    new = source
                elif ins.name in ("csrrs", "csrrsi"):
                    new = old | source
                else:
                    new = old & ~source
                writes = not (ins.name in ("csrrs", "csrrc", "csrrsi", "csrrci")
                              and source == 0)
                if writes:
                    self.csr.write(ins.csr, new)
                    self.csr_writes.append((ins.csr, new & MASK32))
                    self.emit("csr", core=self.id, seq=slot.seq, csr=ins.csr,
                              name=csrmod.CSR_NAMES.get(ins.csr, "?"),
                              old=old, value=new & MASK32,
                              note="%s: 0x%08x -> 0x%08x"
                                   % (csrmod.CSR_NAMES.get(ins.csr, "?"), old,
                                      new & MASK32))
                slot.alu = old
            except Trap as trap:
                return self._take_trap(trap.cause, slot.pc, trap.tval)
        return None

    def _take_trap(self, cause: int, epc: int, tval: int,
                   interrupt: bool = False) -> int:
        handler = self.csr.enter_trap(cause, epc, tval)
        self.stats.traps += 1
        if interrupt:
            self.stats.interrupts += 1
        self.emit("trap", core=self.id, kind="interrupt" if interrupt else "exception",
                  cause=cause, name=csrmod.cause_name(cause), epc=epc,
                  tval=tval, handler=handler,
                  note="%s: %s -> handler 0x%08x"
                       % ("interrupt" if interrupt else "exception",
                          csrmod.cause_name(cause), handler))
        return handler

    # ------------------------------------------------------------------
    # Latch construction / flush
    # ------------------------------------------------------------------

    def _ex_result(self, slot: Slot) -> Slot:
        return slot if not slot.empty else Slot()

    def _mem_result(self, slot: Slot) -> Slot:
        return slot if not slot.empty else Slot()

    def _pipeline_empty(self) -> bool:
        return all(s.empty for s in (self.if_id, self.id_ex, self.ex_mem,
                                     self.mem_wb))

    def _flush_pipeline(self, target: int) -> None:
        dropped = sum(1 for s in (self.if_id, self.id_ex, self.ex_mem,
                                  self.mem_wb) if not s.empty)
        self.if_id = Slot()
        self.id_ex = Slot()
        self.ex_mem = Slot()
        self.mem_wb = Slot()
        self.fetch_req = None
        self.pc = target & MASK32
        self.stats.flushes += 1
        self.emit("flush", core=self.id, target=target, dropped=dropped,
                  note="pipeline flushed, fetch restarts at 0x%08x" % target)

    def _squash_front(self, target: int) -> None:
        dropped = sum(1 for s in (self.if_id, self.id_ex) if not s.empty)
        self.if_id = Slot()
        self.id_ex = Slot()
        self.fetch_req = None
        self.pc = target & MASK32
        self.stats.flushes += 1
        self.emit("flush", core=self.id, target=target, dropped=dropped,
                  note="squashed %d wrong-path instruction(s), refetch 0x%08x"
                       % (dropped, target))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stage_view(self) -> dict:
        """Latches as they were at the start of the cycle: IF/ID holds what ID works on now."""
        def describe(slot: Slot, stage: str):
            if slot.empty:
                return None
            ins = slot.instr
            return {
                "seq": slot.seq, "pc": slot.pc, "text": ins.text(),
                "name": ins.name, "stage": stage,
                "cat": category(ins),
                "rd": ABI_NAMES[ins.rd] if ins.writes_rd and ins.rd else "",
                "rs1": ABI_NAMES[ins.rs1] if ins.uses_rs1 else "",
                "rs2": ABI_NAMES[ins.rs2] if ins.uses_rs2 else "",
                "trap": slot.trap.cause if slot.trap else None,
                "fwd1": slot.fwd1, "fwd2": slot.fwd2,
            }
        if self.halted:
            return {stage: None for stage in STAGES}
        if_id, id_ex, ex_mem, mem_wb = self._latched
        return {
            "IF": self.if_view,
            "ID": describe(if_id, "ID"),
            "EX": describe(id_ex, "EX"),
            "MEM": describe(ex_mem, "MEM"),
            "WB": describe(mem_wb, "WB"),
        }

    def snapshot(self) -> dict:
        return {
            "id": self.id,
            "pc": self.pc,
            "regs": list(self.regs),
            "csrs": self.csr.snapshot(),
            "stages": self.stage_view(),
            "stall": self.stall_reason,
            "halted": self.halted,
            "wfi": self.csr.wfi,
        }


# --------------------------------------------------------------------------

def category(ins) -> str:
    """A coarse label the visualiser uses to narrate a stage."""
    if ins.name == "fetch":
        return "fetch"
    if ins.name == "nop":
        return "nop"
    if ins.is_load:
        return "load"
    if ins.is_store:
        return "store"
    if ins.is_branch:
        return "branch"
    if ins.is_jump:
        return "jump"
    if ins.is_muldiv:
        return "muldiv"
    if ins.is_atomic:
        return "atomic"
    if ins.is_csr:
        return "csr"
    if ins.is_system:
        return "system"
    if ins.name in ("lui", "auipc"):
        return "const"
    return "alu"


INT_MIN = -(1 << 31)


def _div(a: int, b: int) -> int:
    if b == 0:
        return -1
    if a == INT_MIN and b == -1:
        return INT_MIN
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


def _rem(a: int, b: int) -> int:
    if b == 0:
        return a
    if a == INT_MIN and b == -1:
        return 0
    r = abs(a) % abs(b)
    return -r if a < 0 else r


def _divu(a: int, b: int) -> int:
    return MASK32 if b == 0 else a // b


def _remu(a: int, b: int) -> int:
    return a if b == 0 else a % b


def _sext(value: int, width: int) -> int:
    sign = 1 << (width - 1)
    return (value & ((1 << width) - 1) ^ sign) - sign
