# -----------------------------
# Simple 5-stage pipeline simulator
# IF / ID / EX / MEM / WB
# RAW hazards, no forwarding
# -----------------------------

STAGES = ["IF", "ID", "EX", "MEM", "WB"]


class Instruction:
    def __init__(self, name, reads=None, writes=None):
        self.name = name
        self.reads = reads or set()
        self.writes = writes or set()
        self.timeline = {}  # cycle -> stage

    def __repr__(self):
        return self.name


NOP = Instruction("NOP")


def detect_raw_hazard(pipeline):
    """Detect RAW hazard between ID and EX/MEM"""
    id_instr = pipeline["ID"]
    if not id_instr:
        return False

    for stage in ("EX", "MEM"):
        instr = pipeline[stage]
        if instr and instr.writes & id_instr.reads:
            return True

    return False


def pipeline_empty(pipeline, pc, program_len):
    return pc >= program_len and all(pipeline[s] is None for s in STAGES)


def simulate(program):
    pipeline = {s: None for s in STAGES}
    pc = 0
    cycle = 1

    while not pipeline_empty(pipeline, pc, len(program)):
        stall = detect_raw_hazard(pipeline)

        # ---- Advance pipeline (WB -> MEM -> EX) ----
        pipeline["WB"] = pipeline["MEM"]
        pipeline["MEM"] = pipeline["EX"]

        if stall:
            pipeline["EX"] = None  # bubble
        else:
            pipeline["EX"] = pipeline["ID"]
            pipeline["ID"] = pipeline["IF"]
            pipeline["IF"] = program[pc] if pc < len(program) else None
            pc += 1

        # ---- Record pipeline state ----
        for stage, instr in pipeline.items():
            if instr:
                instr.timeline[cycle] = stage
            elif stage == "EX" and stall:
                # explicitly record bubble
                for i in program:
                    if cycle not in i.timeline:
                        i.timeline.setdefault(cycle, "--")

        cycle += 1

    return cycle - 1


def print_table(program, cycles):
    # Header
    print(" " * 10, end="")
    for c in range(1, cycles + 1):
        print(f"C{c:02}", end=" ")
    print()

    # Rows
    for i, instr in enumerate(program):
        print(f"I{i+1} ({instr.name:6}): ", end="")
        for c in range(1, cycles + 1):
            cell = instr.timeline.get(c, "  ")
            print(f"{cell:>3}", end=" ")
        print()


# -----------------------------
# Example program
# -----------------------------

program = [

    Instruction("LOAD", reads={"r2"}, writes={"r1"}),
    Instruction("ADD",  reads={"r1", "r4"}, writes={"r3"}),
    Instruction("SUB",  reads={"r3", "r6"}, writes={"r1"}),  

    Instruction("LOAD", reads={"r8"}, writes={"r7"}),
    Instruction("ADD",  reads={"r7", "r9"}, writes={"r10"}),

    Instruction("MUL",  reads={"r1", "r10"}, writes={"r3"}), 
    Instruction("ADD",  reads={"r3", "r5"}, writes={"r11"}),

    Instruction("LOAD", reads={"r12"}, writes={"r13"}),
    Instruction("ADD",  reads={"r13", "r14"}, writes={"r15"}),
    Instruction("SUB",  reads={"r15", "r16"}, writes={"r13"}),  

    Instruction("ADD",  reads={"r11", "r13"}, writes={"r17"}),
    Instruction("MUL",  reads={"r17", "r1"}, writes={"r18"}),
    Instruction("SUB",  reads={"r18", "r7"}, writes={"r1"}),   

    Instruction("ADD",  reads={"r20", "r21"}, writes={"r22"}),
    Instruction("MUL",  reads={"r22", "r23"}, writes={"r24"}),
    Instruction("SUB",  reads={"r24", "r25"}, writes={"r26"}),


    Instruction("ADD",  reads={"r26", "r3"}, writes={"r11"}), 
    Instruction("MUL",  reads={"r11", "r15"}, writes={"r27"}),

    Instruction("LOAD", reads={"r27"}, writes={"r28"}),
    Instruction("ADD",  reads={"r28", "r1"}, writes={"r3"}),   
    Instruction("SUB",  reads={"r3", "r10"}, writes={"r29"}),

    Instruction("ADD",  reads={"r30", "r31"}, writes={"r32"}),
    Instruction("MUL",  reads={"r32", "r33"}, writes={"r34"}),

    Instruction("ADD",  reads={"r29", "r34"}, writes={"r35"}),
    Instruction("SUB",  reads={"r35", "r22"}, writes={"r36"}),
    Instruction("MUL",  reads={"r36", "r24"}, writes={"r37"}),
]


cycles = simulate(program)
print_table(program, cycles)
