"""
Toy 'compiler optimization' demo — a stripped-down compiler_gym-style
task showing how the L* + preference-oracle setup applies.

WHAT COMPILER_GYM ACTUALLY DOES
-------------------------------
Compiler_gym frames compiler pass ordering as an RL environment:
    state   = some abstraction of LLVM IR (instruction count, features)
    action  = which optimization pass to apply next
    reward  = improvement in code size / runtime
    episode = a sequence of pass selections terminating in an output binary
A learned policy maps state observations to next-pass selections.

WHY THIS MATCHES THE L* SETUP
-----------------------------
- It's a sequential decision problem with a finite action alphabet.
- Many real-world compiler heuristics ARE finite-state pattern matchers
  ("if I see this opcode pair, apply this rewrite") — exactly what a
  Mealy machine encodes.
- Preferences over pass sequences are easier to obtain than precise
  rewards: humans (or empirical measurements) can rank "sequence A
  produced shorter code than sequence B" without quantifying by how
  much.
- Mealy machines can be model-checked, certified, deployed to
  resource-constrained targets — properties that matter in compiler
  contexts.

THE TOY
-------
A program is a sequence of micro-ops drawn from the alphabet:

    LOAD x      : push x onto stack
    ADD0        : add literal 0          (semantically a NOP)
    ADD1        : add literal 1
    SUB1        : subtract 1
    MUL0        : multiply by 0          (collapses stack to 0)
    MUL1        : multiply by 1          (semantically a NOP)
    STORE       : pop and write
    NOP         : do nothing

The optimizer scans the program with a 2-token window and at each step
picks one of:

    SKIP        : advance window by 1, no rewrite
    DROP_RIGHT  : remove the right token of the window, then advance
    DROP_BOTH   : remove both tokens of the window
    FOLD        : replace ADD1,SUB1 → NOP (cancels)

Quality = -len(final_program).  Shorter is better.

THE PREFERENCE ORACLE
---------------------
oracle.compare(trace_a, trace_b)
    Replays each trace's actions on a shared starting program;
    returns 't1' iff the program produced by trace_a is shorter,
    'equal' if same length, else 't2'.  No scalar reward exposed.

This is exactly the same compare-only API your game oracles export,
so the SAME learner code drives this demo with no algorithmic changes.

WHAT L* WOULD LEARN
-------------------
The optimal optimizer for these rules is a tiny finite-state machine:

    on (X, ADD0)         → DROP_RIGHT     # add 0 is identity
    on (X, MUL1)         → DROP_RIGHT     # mul 1 is identity
    on (X, NOP)          → DROP_RIGHT     # nop is removable
    on (MUL0, STORE)     → DROP_BOTH      # storing zero is dead
    on (ADD1, SUB1)      → FOLD           # cancels
    on (SUB1, ADD1)      → FOLD           # cancels
    on anything else     → SKIP

Roughly 4–7 distinct Mealy states depending on equivalence-class
merging.  Compare with a Q-table (BT+Q baseline): one entry per
(window, action) ≈ 8² × 4 = 256 cells, none of them merged.

That compactness gap is the artifact-totality story — made concrete on
a concrete optimization task.

This file:
    - defines the environment
    - defines the preference oracle
    - shows one membership query and one equivalence-style preference
      check end-to-end
    - prints what a hand-derived optimal Mealy machine looks like
"""

from __future__ import annotations
from dataclasses import dataclass


# ----------------------------------------------------------------------
# Token alphabet
# ----------------------------------------------------------------------

LOAD, ADD0, ADD1, SUB1, MUL0, MUL1, STORE, NOP = (
    'LOAD', 'ADD0', 'ADD1', 'SUB1', 'MUL0', 'MUL1', 'STORE', 'NOP'
)
TOKENS = (LOAD, ADD0, ADD1, SUB1, MUL0, MUL1, STORE, NOP)

# Action alphabet for the optimizer
SKIP, DROP_RIGHT, DROP_BOTH, FOLD = 'SKIP', 'DROP_RIGHT', 'DROP_BOTH', 'FOLD'
ACTIONS = (SKIP, DROP_RIGHT, DROP_BOTH, FOLD)


# ----------------------------------------------------------------------
# Environment — apply an optimizer trace to a starting program
# ----------------------------------------------------------------------

@dataclass
class OptimizerState:
    program: tuple
    cursor:  int

    def window(self) -> tuple:
        if self.cursor + 1 >= len(self.program):
            return None
        return (self.program[self.cursor], self.program[self.cursor + 1])

    def is_done(self) -> bool:
        return self.window() is None


def apply_action(state: OptimizerState, action: str) -> OptimizerState:
    p, i = list(state.program), state.cursor
    if action == SKIP:
        return OptimizerState(tuple(p), i + 1)
    if action == DROP_RIGHT:
        return OptimizerState(tuple(p[:i + 1] + p[i + 2:]), i)
    if action == DROP_BOTH:
        return OptimizerState(tuple(p[:i] + p[i + 2:]), i)
    if action == FOLD:
        # cancel ADD1,SUB1 and SUB1,ADD1 by replacing with NOP and dropping it
        if state.window() in {(ADD1, SUB1), (SUB1, ADD1)}:
            return OptimizerState(tuple(p[:i] + p[i + 2:]), i)
        return OptimizerState(tuple(p), i + 1)   # ill-formed FOLD = SKIP
    raise ValueError(f'unknown action {action}')


def run(program: tuple, action_trace: tuple) -> tuple:
    """Apply each action in sequence; return the final program."""
    s = OptimizerState(program=program, cursor=0)
    for a in action_trace:
        if s.is_done():
            break
        s = apply_action(s, a)
    return s.program


# ----------------------------------------------------------------------
# Preference oracle  —  same API as your game oracles
# ----------------------------------------------------------------------

class CompilerPreferenceOracle:
    """
    compare(trace_a, trace_b) returns 't1' if trace_a yields a strictly
    shorter program than trace_b on the held-out benchmark `program`,
    'equal' on tie, else 't2'.

    Identical signature to PreferenceOracle.compare in the game setup —
    a learner that drives one drives the other.
    """
    def __init__(self, program: tuple) -> None:
        self.program = program

    def compare(self, trace_a: tuple, trace_b: tuple) -> str:
        la = len(run(self.program, trace_a))
        lb = len(run(self.program, trace_b))
        if la < lb: return 't1'
        if lb < la: return 't2'
        return 'equal'


# ----------------------------------------------------------------------
# Hand-derived optimal Mealy machine
#   input  alphabet : 2-token window observations  (≤ 64 distinct)
#   output alphabet : ACTIONS
#   states          : 1 (a stateless rewriter — every decision is local)
# ----------------------------------------------------------------------

def optimal_action(window: tuple) -> str:
    if window is None:                    return SKIP
    L, R = window
    if R in (ADD0, MUL1, NOP):            return DROP_RIGHT
    if (L, R) in {(MUL0, STORE)}:         return DROP_BOTH
    if (L, R) in {(ADD1, SUB1), (SUB1, ADD1)}: return FOLD
    return SKIP


def optimize_to_fixpoint(program: tuple) -> tuple:
    """Apply the optimal Mealy policy until no more rewrites fire."""
    prev = None
    while program != prev:
        prev = program
        s = OptimizerState(program=program, cursor=0)
        out_actions: list = []
        while not s.is_done():
            a = optimal_action(s.window())
            out_actions.append(a)
            s = apply_action(s, a)
        program = run(prev, tuple(out_actions))
    return program


# ----------------------------------------------------------------------
# Demo: membership-query and preference-query in one place
# ----------------------------------------------------------------------

def demo() -> None:
    benchmark = (LOAD, ADD0, ADD1, SUB1, MUL1, STORE, NOP)
    oracle    = CompilerPreferenceOracle(benchmark)

    print('=== TOY COMPILER DEMO ===')
    print(f'Benchmark program: {benchmark}   (len={len(benchmark)})')
    print()

    # ---- Membership query analogue ----
    # "If the optimizer applies action sequence T to the benchmark,
    #  what is the resulting program?"
    trace_a = (SKIP, DROP_RIGHT, FOLD, DROP_RIGHT, SKIP)
    out_a   = run(benchmark, trace_a)
    print(f'Membership query:')
    print(f'  trace = {trace_a}')
    print(f'  →     {out_a}   (len={len(out_a)})')
    print()

    # ---- A second action sequence to compare against ----
    trace_b = (SKIP, SKIP, SKIP, SKIP, SKIP)            # do-nothing
    out_b   = run(benchmark, trace_b)
    print(f'  trace = {trace_b}')
    print(f'  →     {out_b}   (len={len(out_b)})')
    print()

    # ---- Preference query (oracle.compare) ----
    pref = oracle.compare(trace_a, trace_b)
    print(f'oracle.compare(trace_a, trace_b) = {pref!r}   '
          f'(t1 means trace_a preferred — produced shorter code)')
    print()

    # ---- What the optimal Mealy machine produces ----
    optimal = optimize_to_fixpoint(benchmark)
    print(f'Optimal Mealy machine output: {optimal}   (len={len(optimal)})')
    print()

    # ---- Show the whole transition table ----
    print('Optimal Mealy transition table  (window → action):')
    seen = set()
    for L in TOKENS:
        for R in TOKENS:
            a = optimal_action((L, R))
            key = (L, R, a)
            if a != SKIP and key not in seen:
                print(f'    ({L:>5}, {R:>5})   →   {a}')
                seen.add(key)
    print('    (everything else)         →   SKIP')
    print()
    print('That whole rewriter is one Mealy state — a stateless transducer over')
    print('the 64-element window alphabet.  L* would discover this from preference')
    print('queries alone, with no access to len() or any scalar reward.')


if __name__ == '__main__':
    demo()
