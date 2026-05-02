"""
Tally wrapper for preference oracles.

Wraps any oracle exposing compare() and preferred_move() and counts every
call. Used to compare the per-game preference-oracle cost of the L*+MCTS
pipeline (queries paid up front during learning) versus a pure-MCTS
baseline that pays per-move (queries scale linearly with games played).

Forwards every other attribute to the wrapped oracle so it remains a
drop-in replacement.
"""

from __future__ import annotations


class CountingOracle:

    def __init__(self, inner) -> None:
        self._inner                = inner
        self.compare_calls         = 0
        self.preferred_move_calls  = 0

    @property
    def total_queries(self) -> int:
        return self.compare_calls + self.preferred_move_calls

    # --------------------------------------------------------------
    # Counted methods
    # --------------------------------------------------------------

    def compare(self, t1, t2):
        self.compare_calls += 1
        return self._inner.compare(t1, t2)

    def preferred_move(self, prefix):
        self.preferred_move_calls += 1
        return self._inner.preferred_move(prefix)

    # --------------------------------------------------------------
    # Forward everything else (e.g. self.nfa, internal _cache) to the
    # wrapped oracle so this works as a drop-in.
    # --------------------------------------------------------------

    def __getattr__(self, name):
        return getattr(self._inner, name)
