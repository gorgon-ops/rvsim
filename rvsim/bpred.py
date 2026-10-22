"""Branch prediction: direction predictors, a BTB and a return-address stack.

Each reports the state it used, so the visualiser can show why it predicted.
"""

from __future__ import annotations

MASK32 = 0xFFFFFFFF

STRONGLY_NOT_TAKEN = 0
WEAKLY_NOT_TAKEN = 1
WEAKLY_TAKEN = 2
STRONGLY_TAKEN = 3
COUNTER_LABELS = {0: "SN", 1: "WN", 2: "WT", 3: "ST"}


class StaticPredictor:
    def __init__(self, mode: str = "btfn"):
        self.mode = mode
        self.name = "static-" + mode

    def predict(self, pc: int, offset: int = 0):
        if self.mode == "taken":
            taken = True
        elif self.mode == "not-taken":
            taken = False
        else:
            taken = offset < 0  # loops branch backwards
        return taken, {"scheme": self.name, "reason": self.mode}

    def update(self, pc: int, taken: bool) -> None:
        pass

    def snapshot(self) -> dict:
        return {"scheme": self.name}


class BimodalPredictor:
    def __init__(self, entries: int = 64):
        self.entries = entries
        self.table = [WEAKLY_NOT_TAKEN] * entries
        self.name = "bimodal"

    def index(self, pc: int) -> int:
        return (pc >> 2) % self.entries

    def predict(self, pc: int, offset: int = 0):
        i = self.index(pc)
        counter = self.table[i]
        return counter >= WEAKLY_TAKEN, {
            "scheme": self.name, "index": i, "counter": counter,
            "state": COUNTER_LABELS[counter],
        }

    def update(self, pc: int, taken: bool) -> None:
        i = self.index(pc)
        if taken:
            self.table[i] = min(STRONGLY_TAKEN, self.table[i] + 1)
        else:
            self.table[i] = max(STRONGLY_NOT_TAKEN, self.table[i] - 1)

    def snapshot(self) -> dict:
        return {"scheme": self.name, "table": list(self.table)}


class GsharePredictor:
    """Global history XORed with the PC -- catches correlated branches."""

    def __init__(self, entries: int = 64, history_bits: int = 6):
        self.entries = entries
        self.history_bits = history_bits
        self.history = 0
        self.table = [WEAKLY_NOT_TAKEN] * entries
        self.name = "gshare"

    def index(self, pc: int) -> int:
        return ((pc >> 2) ^ self.history) % self.entries

    def predict(self, pc: int, offset: int = 0):
        i = self.index(pc)
        counter = self.table[i]
        return counter >= WEAKLY_TAKEN, {
            "scheme": self.name, "index": i, "counter": counter,
            "state": COUNTER_LABELS[counter],
            "history": format(self.history, "0%db" % self.history_bits),
        }

    def update(self, pc: int, taken: bool) -> None:
        i = self.index(pc)
        if taken:
            self.table[i] = min(STRONGLY_TAKEN, self.table[i] + 1)
        else:
            self.table[i] = max(STRONGLY_NOT_TAKEN, self.table[i] - 1)
        mask = (1 << self.history_bits) - 1
        self.history = ((self.history << 1) | int(taken)) & mask

    def snapshot(self) -> dict:
        return {"scheme": self.name, "history": self.history,
                "table": list(self.table)}


# A BTB entry records the kind of transfer at that PC: a jump is taken without
# asking the direction predictor, and a return is better served by the RAS.
KIND_BRANCH = "branch"
KIND_JUMP = "jump"
KIND_RETURN = "return"


class BTB:
    def __init__(self, entries: int = 32):
        self.entries = entries
        self.tags = [None] * entries
        self.targets = [0] * entries
        self.kinds = [KIND_BRANCH] * entries
        self.hits = 0
        self.misses = 0

    def index(self, pc: int) -> int:
        return (pc >> 2) % self.entries

    def lookup(self, pc: int):
        """A miss is an aliased or cold PC."""
        i = self.index(pc)
        if self.tags[i] == pc:
            self.hits += 1
            return self.targets[i], self.kinds[i]
        self.misses += 1
        return None

    def update(self, pc: int, target: int, kind: str = KIND_BRANCH) -> None:
        i = self.index(pc)
        self.tags[i] = pc
        self.targets[i] = target & MASK32
        self.kinds[i] = kind

    def snapshot(self) -> list:
        return [{"index": i, "pc": self.tags[i], "target": self.targets[i],
                 "kind": self.kinds[i]}
                for i in range(self.entries) if self.tags[i] is not None]


class ReturnAddressStack:
    """Predicts ``ret`` targets, which a BTB gets wrong for shared callees."""

    def __init__(self, depth: int = 8):
        self.depth = depth
        self.stack = []

    def push(self, addr: int) -> None:
        self.stack.append(addr & MASK32)
        if len(self.stack) > self.depth:
            self.stack.pop(0)

    def pop(self):
        return self.stack.pop() if self.stack else None

    def peek(self):
        return self.stack[-1] if self.stack else None

    def snapshot(self) -> list:
        return list(self.stack)


def make_predictor(scheme: str, entries: int = 64, history_bits: int = 6):
    scheme = (scheme or "bimodal").lower()
    if scheme in ("bimodal", "2bit"):
        return BimodalPredictor(entries)
    if scheme == "gshare":
        return GsharePredictor(entries, history_bits)
    if scheme in ("taken", "not-taken", "btfn", "static"):
        return StaticPredictor("btfn" if scheme == "static" else scheme)
    raise ValueError("unknown branch predictor %r" % scheme)


class BranchUnit:
    def __init__(self, scheme: str = "gshare", entries: int = 64,
                 btb_entries: int = 32, ras_depth: int = 8,
                 history_bits: int = 6):
        self.predictor = make_predictor(scheme, entries, history_bits)
        self.btb = BTB(btb_entries)
        self.ras = ReturnAddressStack(ras_depth)
        # Branches and jumps counted apart: a jump that hits in the BTB is always
        # right, which would flatter the direction predictor.
        self.predictions = 0
        self.mispredictions = 0
        self.jumps = 0
        self.jump_mispredictions = 0
        self.ras_used = 0
        self.ras_correct = 0

    @property
    def accuracy(self) -> float:
        if not self.predictions:
            return 1.0
        return (self.predictions - self.mispredictions) / self.predictions

    @property
    def jump_accuracy(self) -> float:
        if not self.jumps:
            return 1.0
        return (self.jumps - self.jump_mispredictions) / self.jumps

    def stats(self) -> dict:
        return {
            "scheme": self.predictor.name,
            "predictions": self.predictions,
            "mispredictions": self.mispredictions,
            "accuracy": round(self.accuracy, 4),
            "jumps": self.jumps,
            "jump_mispredictions": self.jump_mispredictions,
            "jump_accuracy": round(self.jump_accuracy, 4),
            "btb_hits": self.btb.hits,
            "btb_misses": self.btb.misses,
            "ras_used": self.ras_used,
            "ras_correct": self.ras_correct,
        }

    def snapshot(self) -> dict:
        return {"predictor": self.predictor.snapshot(),
                "btb": self.btb.snapshot(),
                "ras": self.ras.snapshot()}
