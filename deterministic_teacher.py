#!/usr/bin/env python3
"""
Deterministic teacher and reward machine for Kuhn Poker.

No LLM. All preferences are derived directly from game outcomes and
strategic quality, both encoded in the DOT machine symbols.

Two things are provided:
  1. DeterministicKuhnPokerTeacher  — drop-in replacement for LLMKuhnPokerTeacher,
     usable with symbolic_lstar.
  2. build_kuhn_reward_machine()    — returns the explicit 3-state ground-truth
     reward machine as a dict you can inspect or export.
"""

import re
import sys
import os
import random

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from dot_trace_generator import load_dot


# ---------------------------------------------------------------------------
# Strategic quality table
# Keyed by (p2_card, p1_visible_action, p2_action).
# P2 cannot see P1's card — their decision is evaluated on what they KNOW.
# ---------------------------------------------------------------------------
# Score: higher = better strategic play by P2.
# Calibrated to produce a total order across all complete hands.
_STRATEGIC_SCORE: dict = {
    # ---- P1 CHECKED first (P1 showed weakness) ----
    # P2 holds King: betting extracts value from a dominant hand
    ("King", "check", "bet"): 5,
    ("King", "check", "check"): 3,  # wins anyway, but missed value
    # P2 holds Queen: checking is safer (beats Jack, loses to King)
    ("Queen", "check", "check"): 3,
    ("Queen", "check", "bet"): 2,  # thin value / risky bluff
    # P2 holds Jack: bluffing has strategic value after P1 checked
    ("Jack", "check", "bet"): 3,  # correct bluff spot
    ("Jack", "check", "check"): 1,  # passive; never wins at showdown
    # ---- P1 BET first (P1 showed strength) ----
    # P2 holds King: must call — guaranteed to win
    ("King", "bet", "call"): 5,
    ("King", "bet", "fold"): -5,  # catastrophic: folding the best hand
    # P2 holds Queen: fold is correct (P1 range: King + bluff Jack)
    ("Queen", "bet", "fold"): 3,
    ("Queen", "bet", "call"): 1,  # marginal; can be right vs bluffs
    # P2 holds Jack: always fold — worst hand
    ("Jack", "bet", "fold"): 3,
    ("Jack", "bet", "call"): -3,  # calling with worst hand
    # ---- P2 RE-RAISED (P2B round: P2 responds to P1's raise) ----
    # P2 holds King: always call a re-raise
    ("King", "raise", "call"): 4,
    ("King", "raise", "fold"): -4,
    # P2 holds Queen: fold to re-raise (P1 re-raising range is very strong)
    ("Queen", "raise", "fold"): 3,
    ("Queen", "raise", "call"): 0,
    # P2 holds Jack: fold to re-raise
    ("Jack", "raise", "fold"): 2,
    ("Jack", "raise", "call"): -3,
}


def _p2_card_from_sym(sym: str) -> str:
    """Extract P2's card from a decoded symbol string."""
    parts = sym.split("_")
    # symbols are PHASE_P1card_P2card_...
    if len(parts) >= 3:
        return parts[2]
    return ""


def _p1_action_from_sym(sym: str) -> str:
    parts = sym.split("_")
    if len(parts) >= 4:
        return parts[3]
    return ""


def _strategic_score_for_sequence(seq: tuple) -> int:
    """
    Score P2's strategic decisions across a full hand.
    Sums contributions from each P2-decision symbol.
    """
    score = 0
    for sym in seq:
        parts = sym.split("_")
        phase = parts[0]

        if phase == "P2" and len(parts) >= 4:
            # P2_P1card_P2card_action[_outcome]
            p2_card = parts[2]
            action = parts[3]
            # P1's visible action leading here is "check" (P2 acts after P1 check)
            # or "bet" (P2 acts after P1 bet) — infer from DOT structure:
            # P2 phase always follows P1's first action, which is bet or check.
            # We look at the PREVIOUS symbol in the sequence instead.
            idx = seq.index(sym)
            p1_action = ""
            if idx > 0:
                prev = seq[idx - 1].split("_")
                if len(prev) >= 4:
                    p1_action = prev[3]  # P1's action: "check" or "bet"
            key = (p2_card, p1_action, action)
            score += _STRATEGIC_SCORE.get(key, 0)

        elif phase == "P2B" and len(parts) >= 4:
            # P2B_P1card_P2card_action[_outcome]
            p2_card = parts[2]
            action = parts[3]
            # P1B always raises (it's the only P1B action that leads to P2B)
            key = (p2_card, "raise", action)
            score += _STRATEGIC_SCORE.get(key, 0)

    return score


# ---------------------------------------------------------------------------
# Shared machine helpers (identical to llm_teacher helpers)
# ---------------------------------------------------------------------------


def _split_disjunction(inp: str) -> list:
    inp = inp.strip()
    if "|" not in inp:
        return [inp.strip("() ")]
    parts = re.split(r"\)\s*\|\s*\(", inp)
    return [p.strip("() ") for p in parts if p.strip("() ")]


def _true_aps(clause: str) -> frozenset:
    true = set()
    for tok in clause.split("&"):
        tok = tok.strip()
        if tok and not tok.startswith("!"):
            true.add(tok)
    return frozenset(true)


def _card(aps, hi, lo):
    if hi in aps and lo not in aps:
        return "King"
    if hi not in aps and lo in aps:
        return "Queen"
    return "Jack"


def _action(aps):
    a2, a1, a0 = "a2" in aps, "a1" in aps, "a0" in aps
    if not a2 and not a1 and a0:
        return "check"
    if not a2 and a1 and not a0:
        return "bet"
    if not a2 and a1 and a0:
        return "raise"
    if a2 and not a1 and not a0:
        return "call"
    if a2 and not a1 and a0:
        return "fold"
    return "none"


def _outcome_tag(aps):
    if "win1" in aps:
        return "W1"
    if "win2" in aps:
        return "W2"
    return "none"


def _to_symbol(inp_str: str, out_str: str) -> str:
    inp = _true_aps(inp_str)
    out = _true_aps(out_str)
    aps = inp | out
    c1 = _card(aps, "c1hi", "c1lo")
    c2 = _card(aps, "c2hi", "c2lo")
    act = _action(inp)
    res = _outcome_tag(out)
    if "deal" in inp:
        return f"DEAL_{c1}_{c2}"
    if "p1" in inp:
        return f"P1_{c1}_{c2}_{act}"
    if "p2" in inp:
        return f"P2_{c1}_{c2}_{act}_{res}"
    if "p1b" in inp:
        return f"P1B_{c1}_{c2}_{act}_{res}"
    if "p2b" in inp:
        return f"P2B_{c1}_{c2}_{act}_{res}"
    return "UNKNOWN"


def _prepare_machine(machine: dict) -> dict:
    """
    Clean and normalize a raw DOT-loaded automaton for use by the teacher.

    Performs two transformations on the transitions dict:

    1. Drop sink-state transitions — any edge whose destination is state "24"
       (the violation/error sink, reached when game assumptions are broken)
       is removed. Only valid play paths are kept.

    2. Flatten disjunctive edge labels — DOT edges can carry guards like
       "(a1 & !a2 & ...) | (a0 & !a2 & ...)", representing multiple concrete
       input patterns on one edge. Each disjunct is split into its own entry
       in the transition table, all mapping to the same (dst, out). This lets
       downstream code (_enumerate_all_hands) walk transitions one concrete
       clause at a time without evaluating Boolean formulas.

    Args:
        machine (dict): Raw automaton dict from load_dot(), with keys
            "states", "initial", "alphabet", and "transitions".

    Returns:
        dict: A copy of the input machine with a cleaned "transitions" dict.
            All other fields (states, initial, alphabet) are preserved unchanged.
    """
    clean: dict = {}
    for state, trans in machine["transitions"].items():
        for inp, (dst, out) in trans.items():
            if dst == "24":
                continue
            for clause in _split_disjunction(inp):
                clean.setdefault(state, {})[clause] = (dst, out)
    return {**machine, "transitions": clean}


def _enumerate_all_hands(machine: dict) -> list:
    initial = machine["initial"]
    trans = machine["transitions"]
    complete = []
    stack = [(initial, [])]
    while stack:
        state, path = stack.pop()
        for inp, (nxt, out) in trans.get(state, {}).items():
            sym = _to_symbol(inp, out)
            new_path = path + [sym]
            if nxt == initial:
                complete.append(tuple(new_path))
            elif len(new_path) < 10:
                stack.append((nxt, new_path))
    return list(set(complete))


def _eval_outcome(seq: tuple, P2_WINS=1, IN_PROG=0, P2_LOSES=-1) -> int:
    for sym in reversed(seq):
        if sym.endswith("_W2"):
            return P2_WINS
        if sym.endswith("_W1"):
            return P2_LOSES
    return IN_PROG


# ---------------------------------------------------------------------------
# Ground-truth reward machine
# ---------------------------------------------------------------------------


def build_kuhn_reward_machine(sigma_I: tuple) -> dict:
    """
    Build and return the minimal 3-state ground-truth reward machine for
    Kuhn Poker (P2-outcome signal).

    States:
      0  = IN_PROGRESS  (initial)
      1  = P2_WINS      (absorbing)
     -1  = P2_LOSES     (absorbing)

    Transitions:
      Any symbol ending _W2  → state  1
      Any symbol ending _W1  → state -1
      Anything else          → state  0  (resets; game phases are non-absorbing)
    """
    delta = {}
    for state in (0, 1, -1):
        delta[state] = {}
        for sym in sigma_I:
            if sym.endswith("_W2"):
                delta[state][sym] = 1
            elif sym.endswith("_W1"):
                delta[state][sym] = -1
            else:
                # Non-terminal symbols return to/stay at in-progress
                # unless we're already in a terminal state
                delta[state][sym] = state if state != 0 else 0

    return {
        "states": {0, 1, -1},
        "initial": 0,
        "sigma_I": sigma_I,
        "sigma_O": (-1, 0, 1),
        "delta": delta,
        "output": {0: 0, 1: 1, -1: -1},
        "state_names": {0: "IN_PROG", 1: "P2_WINS", -1: "P2_LOSES"},
    }


# ---------------------------------------------------------------------------
# DeterministicKuhnPokerTeacher
# ---------------------------------------------------------------------------


class DeterministicKuhnPokerTeacher:
    """
    Drop-in deterministic teacher for symbolic_lstar.

    Preferences are derived from:
      Primary   : game outcome  (P2_WINS > IN_PROG > P2_LOSES)
      Secondary : strategic quality score (higher = better P2 play)

    No LLM, no randomness in preferences.
    set_student() is accepted for API compatibility but unused — the teacher
    is already exact.
    """

    P2_WINS = 1
    IN_PROG = 0
    P2_LOSES = -1

    def __init__(self, dot_file: str, seq_sample_size: int = 200):
        raw = load_dot(dot_file)
        self.machine = _prepare_machine(raw)
        self.seq_sample_size = seq_sample_size

        # Enumerate all complete hands
        self.all_hands = _enumerate_all_hands(self.machine)
        print(f"[teacher] Enumerated {len(self.all_hands)} complete hands.")

        self.sigma_I = tuple(sorted({sym for hand in self.all_hands for sym in hand}))
        self.sigma_O = (self.P2_LOSES, self.IN_PROG, self.P2_WINS)

        # Pre-compute (outcome, strategic_score) for every hand
        self._scores: dict = {
            hand: (
                _eval_outcome(hand, self.P2_WINS, self.IN_PROG, self.P2_LOSES),
                _strategic_score_for_sequence(hand),
            )
            for hand in self.all_hands
        }

        # Expose the explicit reward machine
        self.reward_machine = build_kuhn_reward_machine(self.sigma_I)

        print(
            f"[teacher] Ready. {len(self.sigma_I)} symbols, "
            f"reward machine has {len(self.reward_machine['states'])} states."
        )

    # -----------------------------------------------------------------------
    # preference_query — fully deterministic
    # -----------------------------------------------------------------------

    def preference_query(self, s1: tuple, s2: tuple) -> int:
        """
        Compare two sequences by game outcome only.
        Outcome maps 1-to-1 onto sigma_O = {-1, 0, 1}, which is the
        constraint Z3 must satisfy.  Using strategic scores as a tiebreaker
        would create more than 3 distinct preference levels, making the
        SMT problem unsatisfiable.

        Returns  1 if s1 preferred,  -1 if s2 preferred,  0 if equal.
        """
        out1 = _eval_outcome(s1, self.P2_WINS, self.IN_PROG, self.P2_LOSES)
        out2 = _eval_outcome(s2, self.P2_WINS, self.IN_PROG, self.P2_LOSES)

        if out1 > out2:
            return 1
        if out1 < out2:
            return -1
        return 0

    # -----------------------------------------------------------------------
    # set_student — no-op (deterministic teacher needs no student fallback)
    # -----------------------------------------------------------------------

    def set_student(self, init_state, delta, output_fnc):
        pass  # API compatibility only

    # -----------------------------------------------------------------------
    # sample_sequences
    # -----------------------------------------------------------------------

    def sample_sequences(self, quantity: int) -> list:
        return [random.choice(self.all_hands) for _ in range(quantity)]

    # -----------------------------------------------------------------------
    # equivalence_query — exhaustive (game is small enough)
    # -----------------------------------------------------------------------

    def _run_hypothesis(self, seq, init_state, delta, output_fnc) -> int:
        q = init_state
        for sym in seq:
            q = delta.get(q, {}).get(sym, q)
        return output_fnc.get(q, self.IN_PROG)

    def equivalence_query(
        self, states, sigma_I, sigma_O, init_state, delta, output_fnc
    ):
        """
        Exhaustive check against every enumerated hand.
        Returns (True, None) or (False, (counterexample, correct_value)).
        """
        for seq in self.all_hands:
            teacher_val = _eval_outcome(seq, self.P2_WINS, self.IN_PROG, self.P2_LOSES)
            hyp_val = self._run_hypothesis(seq, init_state, delta, output_fnc)
            if teacher_val != hyp_val:
                return False, (seq, teacher_val)
        return True, None

    # -----------------------------------------------------------------------
    # Inspect helpers
    # -----------------------------------------------------------------------

    def print_all_hands(self):
        """Print every hand with its outcome and strategic score."""
        rows = sorted(self._scores.items(), key=lambda x: (-x[1][0], -x[1][1]))
        outcome_label = {1: "P2 wins", 0: "in-prog", -1: "P2 loses"}
        print(f"\n{'HAND':<80} {'OUTCOME':<12} {'STRAT':>6}")
        print("-" * 102)
        for hand, (out, strat) in rows:
            print(f"{' | '.join(hand):<80} {outcome_label[out]:<12} {strat:>6}")

    def print_reward_machine(self):
        """Print the explicit ground-truth reward machine."""
        rm = self.reward_machine
        print("\n=== Ground-Truth Kuhn Poker Reward Machine ===")
        print(f"States : {rm['state_names']}")
        print(f"Initial: {rm['initial']} ({rm['state_names'][rm['initial']]})")
        print(f"Output : {rm['output']}")
        print(f"\nTransitions (non-trivial only):")
        for state, trans in rm["delta"].items():
            for sym, nxt in sorted(trans.items()):
                if nxt != state:  # only print state changes
                    print(
                        f"  {rm['state_names'][state]:10} --[{sym}]--> {rm['state_names'][nxt]}"
                    )


# ---------------------------------------------------------------------------
# Smoke test / pretty-printer
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    teacher = DeterministicKuhnPokerTeacher(
        dot_file="Kuhn_Poker/kuhn_poker.dot",
        seq_sample_size=200,
    )

    teacher.print_all_hands()
    teacher.print_reward_machine()

    print(f"\n=== Preference query examples ===")
    ranked = sorted(teacher.all_hands, key=lambda h: teacher._scores[h], reverse=True)
    best, worst = ranked[0], ranked[-1]
    print(f"Best hand : {' | '.join(best)}")
    print(f"Worst hand: {' | '.join(worst)}")
    print(f"preference(best, worst) = {teacher.preference_query(best, worst)}")
    print(f"preference(worst, best) = {teacher.preference_query(worst, best)}")
    print(f"preference(best, best)  = {teacher.preference_query(best, best)}")
