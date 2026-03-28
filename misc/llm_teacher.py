#!/usr/bin/env python3
"""
LLM-backed teacher for Kuhn Poker, compatible with REMAP's symbolic_lstar.

Strategy: lazy, on-demand preference learning.
  - No upfront enumeration of all hands.
  - preference_query checks a local cache first.
  - If the query is unknown, falls back to the current learned student hypothesis.
  - If the student cannot answer (sequence not in its domain), prompts the LLM
    for just that one pair.
  - Goal: epsilon-closeness between the teacher (LLM-informed) and the student.

Usage:
    from llm_teacher import LLMKuhnPokerTeacher
    from lstar import symbolic_lstar

    teacher = LLMKuhnPokerTeacher(
        dot_file="Kuhn_Poker/kuhn_poker.dot",
        hoa_file="Kuhn_Poker/Kuhn_Poker.hoa",
        seq_sample_size=200,
    )
    hypothesis, data = symbolic_lstar(teacher.sigma_I, teacher.sigma_O, teacher)
"""

import re
import sys
import os
import random

import anthropic

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from dot_trace_generator import load_dot


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------
def _load_api_key() -> str:
    key_path = os.path.join(_HERE, ".env", "api_key")
    with open(key_path) as f:
        return f.read().strip()


# ---------------------------------------------------------------------------
# LLM prompt context
# ---------------------------------------------------------------------------
CONTEXT = (
    "You are evaluating Kuhn Poker hands. The deck has three cards: Jack < Queen < King.\n"
    "Each player antes 1 chip. Player 1 (P1) acts first: check or bet.\n"
    "Player 2 (P2) then responds based only on their OWN card and P1's visible action.\n"
    "You will see the FULL trace including both players' cards.\n"
    "Judge P2's play quality, given that P2 does NOT know P1's card when deciding.\n"
    "Hands where P2 wins are not automatically good — P2 may have won by luck.\n"
    "Hands where P2 loses are not automatically bad — P2 may have played correctly.\n"
    "Focus on whether P2's DECISION was strategically sound given their information."
)


class LLMKuhnPokerTeacher:
    """
    Lazy teacher for REMAP's symbolic_lstar.

    Answers preference queries on-demand:
      1. Cache hit            → return immediately (zero LLM calls).
      2. Student available    → use student hypothesis output (epsilon-closeness).
      3. Unknown to student   → ask LLM for this one pair (one LLM call, then cache).

    Call set_student() after each hypothesis update so the teacher can use it
    as a cheap oracle before falling back to the LLM.
    """

    P2_WINS = 1
    IN_PROG = 0
    P2_LOSES = -1

    def __init__(
        self,
        dot_file: str,
        hoa_file: str,
        seq_sample_size: int,
        model: str = "claude-haiku-4-5-20251001",
    ):
        raw = load_dot(dot_file)
        self.machine = self._prepare_machine(raw)
        self.hoa_file = hoa_file
        self.seq_sample_size = seq_sample_size
        self.model = model
        self.client = anthropic.Anthropic(api_key=_load_api_key())

        # Build sigma_I from all transitions (no full hand enumeration needed)
        self.sigma_I = self._build_sigma_I()
        self.sigma_O = (self.P2_LOSES, self.IN_PROG, self.P2_WINS)

        # Lazy preference cache: (s1, s2) -> int
        self._pref_cache: dict = {}

        # Current learned student: (init_state, delta, output_fnc) or None
        self._student = None

        # Counters for efficiency analysis
        self._llm_calls = 0
        self._student_hits = 0
        self._cache_hits = 0

        print(f"[teacher] Ready. sigma_I has {len(self.sigma_I)} symbols.")

    # -----------------------------------------------------------------------
    # Student registration  (call this after each hypothesis update)
    # -----------------------------------------------------------------------

    def set_student(self, init_state, delta, output_fnc):
        """Register the current learned hypothesis as a cheap preference oracle."""
        self._student = (init_state, delta, output_fnc)

    # -----------------------------------------------------------------------
    # Machine preparation
    # -----------------------------------------------------------------------

    def _prepare_machine(self, machine: dict) -> dict:
        """Remove sink transitions and expand disjunctive labels."""
        clean: dict = {}
        for state, trans in machine["transitions"].items():
            for inp, (dst, out) in trans.items():
                if dst == "24":
                    continue
                for clause in self._split_disjunction(inp):
                    clean.setdefault(state, {})[clause] = (dst, out)
        return {**machine, "transitions": clean}

    def _split_disjunction(self, inp: str) -> list:
        inp = inp.strip()
        if "|" not in inp:
            return [inp.strip("() ")]
        parts = re.split(r"\)\s*\|\s*\(", inp)
        return [p.strip("() ") for p in parts if p.strip("() ")]

    # -----------------------------------------------------------------------
    # AP / label parsing
    # -----------------------------------------------------------------------

    def _true_aps(self, clause: str) -> frozenset:
        true = set()
        for tok in clause.split("&"):
            tok = tok.strip()
            if tok and not tok.startswith("!"):
                true.add(tok)
        return frozenset(true)

    # -----------------------------------------------------------------------
    # Decode helpers
    # -----------------------------------------------------------------------

    def _card(self, aps: frozenset, hi: str, lo: str) -> str:
        if hi in aps and lo not in aps:
            return "King"
        if hi not in aps and lo in aps:
            return "Queen"
        return "Jack"

    def _action(self, aps: frozenset) -> str:
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

    def _outcome_tag(self, aps: frozenset) -> str:
        if "win1" in aps:
            return "W1"
        if "win2" in aps:
            return "W2"
        return "none"

    def _to_symbol(self, inp_str: str, out_str: str) -> str:
        inp = self._true_aps(inp_str)
        out = self._true_aps(out_str)
        aps = inp | out
        c1 = self._card(aps, "c1hi", "c1lo")
        c2 = self._card(aps, "c2hi", "c2lo")
        act = self._action(inp)
        res = self._outcome_tag(out)
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

    # -----------------------------------------------------------------------
    # Build sigma_I from machine transitions (no full enumeration)
    # -----------------------------------------------------------------------

    def _build_sigma_I(self) -> tuple:
        """Collect every symbol reachable in the DOT machine."""
        seen = set()
        for state, trans in self.machine["transitions"].items():
            for inp, (dst, out) in trans.items():
                seen.add(self._to_symbol(inp, out))
        return tuple(sorted(seen))

    # -----------------------------------------------------------------------
    # Human-readable decode for LLM prompt
    # -----------------------------------------------------------------------

    def _trace_to_text(self, seq: tuple) -> str:
        lines = []
        for sym in seq:
            p = sym.split("_")
            phase = p[0]
            if phase == "DEAL" and len(p) >= 3:
                lines.append(f"  Deal    : P1 gets {p[1]}, P2 gets {p[2]}")
            elif phase == "P1" and len(p) >= 4:
                lines.append(f"  P1      : {p[3]}  (P1={p[1]}, P2={p[2]})")
            elif phase == "P2" and len(p) >= 5:
                out = {"W1": "-> P1 wins", "W2": "-> P2 wins"}.get(p[4], "-> continues")
                lines.append(f"  P2      : {p[3]} {out}  (P2 holds {p[2]})")
            elif phase == "P1B" and len(p) >= 5:
                out = {"W1": "-> P1 wins", "W2": "-> P2 wins"}.get(p[4], "-> continues")
                lines.append(f"  P1 resp : {p[3]} {out}")
            elif phase == "P2B" and len(p) >= 5:
                out = {"W1": "-> P1 wins", "W2": "-> P2 wins"}.get(p[4], "-> continues")
                lines.append(f"  P2 resp : {p[3]} {out}")
        return "\n".join(lines) if lines else "  (empty trace)"

    # -----------------------------------------------------------------------
    # Student helper — run a sequence through the current hypothesis
    # -----------------------------------------------------------------------

    def _student_output(self, seq: tuple):
        """
        Run seq through the current student hypothesis.
        Returns the output value, or None if any symbol is missing from delta.
        """
        if self._student is None:
            return None
        init_state, delta, output_fnc = self._student
        q = init_state
        for sym in seq:
            nxt = delta.get(q, {}).get(sym, None)
            if nxt is None:
                return None
            q = nxt
        return output_fnc.get(q, None)

    # -----------------------------------------------------------------------
    # LLM pairwise comparison — called only when cache and student both miss
    # -----------------------------------------------------------------------

    def _llm_compare(self, s1: tuple, s2: tuple) -> int:
        """
        Ask the LLM to compare exactly two hands.
        Returns  1 if s1 has better P2 play, -1 if s2 does, 0 if equal.
        One LLM call; result is cached by the caller.
        """
        prompt = (
            f"{CONTEXT}\n\n"
            "Compare the P2 play quality in these two Kuhn Poker hands:\n\n"
            f"Hand A:\n{self._trace_to_text(s1)}\n\n"
            f"Hand B:\n{self._trace_to_text(s2)}\n\n"
            "Which hand shows BETTER strategic play by P2, or are they equivalent?\n"
            "Consider only whether P2's decision was correct given P2's own card and "
            "P1's visible action — not the final outcome.\n"
            "Reply with ONLY one of: A, B, or EQUAL"
        )

        print(f"[teacher] LLM compare: querying for new pair...")
        for attempt in range(4):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=10,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                break
            except Exception as e:
                if attempt == 3:
                    raise
                wait = 2**attempt
                print(f"[teacher] LLM error ({e}), retrying in {wait}s...")
                import time

                time.sleep(wait)

        raw = resp.content[0].text.strip().upper()
        # Be conservative: only trust clean single-token answers
        if raw.startswith("A") and "B" not in raw:
            return 1
        if raw.startswith("B") and "A" not in raw:
            return -1
        return 0

    # -----------------------------------------------------------------------
    # preference_query — lazy, three-tier fallback
    # -----------------------------------------------------------------------

    def preference_query(self, s1: tuple, s2: tuple) -> int:
        """
        Compare two sequences.  Returns  1 if s1 is preferred, -1 if s2 is,
        0 if equal.

        Tier 1: cache hit               → O(1), no LLM.
        Tier 2: student hypothesis      → cheap oracle, no LLM.
        Tier 3: LLM pairwise query      → one API call, then cached.
        """
        # Tier 1: cache
        key = (s1, s2)
        if key in self._pref_cache:
            self._cache_hits += 1
            return self._pref_cache[key]

        # Tier 2: student
        v1 = self._student_output(s1)
        v2 = self._student_output(s2)
        if v1 is not None and v2 is not None:
            self._student_hits += 1
            if v1 > v2:
                pref = 1
            elif v1 < v2:
                pref = -1
            else:
                pref = 0
            self._pref_cache[key] = pref
            self._pref_cache[(s2, s1)] = -pref
            return pref

        # Tier 3: LLM
        self._llm_calls += 1
        pref = self._llm_compare(s1, s2)
        self._pref_cache[key] = pref
        self._pref_cache[(s2, s1)] = -pref
        return pref

    # -----------------------------------------------------------------------
    # sample_sequences — random walks on the DOT machine
    # -----------------------------------------------------------------------

    def sample_sequences(self, quantity: int) -> list:
        """
        Generate complete hands via random walks on the machine.
        A hand is complete when a transition returns to the initial state.
        """
        initial = self.machine["initial"]
        trans = self.machine["transitions"]
        results = []
        attempts = 0

        while len(results) < quantity and attempts < quantity * 20:
            attempts += 1
            state = initial
            path = []
            for _ in range(10):  # depth limit
                choices = list(trans.get(state, {}).items())
                if not choices:
                    break
                inp, (nxt, out) = random.choice(choices)
                sym = self._to_symbol(inp, out)
                path.append(sym)
                if nxt == initial:
                    results.append(tuple(path))
                    break
                state = nxt

        # Pad with repeats if random walks were unlucky
        if results:
            while len(results) < quantity:
                results.append(random.choice(results))

        return results

    # -----------------------------------------------------------------------
    # Equivalence query — sampled, no LLM calls
    # -----------------------------------------------------------------------

    def _eval_outcome(self, seq: tuple) -> int:
        for sym in reversed(seq):
            if sym.endswith("_W2"):
                return self.P2_WINS
            if sym.endswith("_W1"):
                return self.P2_LOSES
        return self.IN_PROG

    def _run_hypothesis(self, seq, init_state, delta, output_fnc) -> int:
        q = init_state
        for sym in seq:
            q = delta.get(q, {}).get(sym, q)
        return output_fnc.get(q, self.IN_PROG)

    def equivalence_query(
        self, states, sigma_I, sigma_O, init_state, delta, output_fnc
    ):
        """
        Sample-based equivalence check.
        Returns (True, None) or (False, (counterexample, correct_value)).
        """
        seqs = self.sample_sequences(self.seq_sample_size)
        for seq in seqs:
            teacher_val = self._eval_outcome(seq)
            hyp_val = self._run_hypothesis(seq, init_state, delta, output_fnc)
            if teacher_val != hyp_val:
                return False, (seq, teacher_val)
        return True, None


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading teacher (lazy — no upfront LLM calls)...")
    teacher = LLMKuhnPokerTeacher(
        dot_file="Kuhn_Poker/kuhn_poker.dot",
        hoa_file="Kuhn_Poker/Kuhn_Poker.hoa",
        seq_sample_size=200,
    )

    print(f"\nsigma_I : {len(teacher.sigma_I)} symbols")
    print(f"sigma_O : {teacher.sigma_O}")

    print("\n=== Sampling 5 hands ===")
    for hand in teacher.sample_sequences(5):
        print(teacher._trace_to_text(hand))
        print()

    print("=== Preference query test (will call LLM if cache empty) ===")
    hands = teacher.sample_sequences(2)
    s1, s2 = hands[0], hands[1]
    pref = teacher.preference_query(s1, s2)
    label = {1: "Hand A better", -1: "Hand B better", 0: "equal"}[pref]
    print(f"  preference_query = {pref}  ({label})")
    print(f"  cache size: {len(teacher._pref_cache)}")
    # Second call should be cache hit
    pref2 = teacher.preference_query(s1, s2)
    print(f"  repeated query   = {pref2}  (from cache, no LLM call)")
