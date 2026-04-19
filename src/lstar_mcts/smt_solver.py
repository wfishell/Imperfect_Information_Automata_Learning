"""
SMT-based value assigner.

Collects pairwise preference orderings over traces and uses Z3 to find
a consistent numeric assignment satisfying all constraints.

The outside world only feeds in ordinal facts:
    add('t1', 't2', 't1')    →  value(t1) > value(t2)
    add('t1', 't2', 't2')    →  value(t2) > value(t1)
    add('t1', 't2', 'equal') →  value(t1) = value(t2)

solve() returns a dict  trace_key → float  in the range [0, 1],
normalised so the least preferred trace gets 0 and the most preferred gets 1.
"""

from __future__ import annotations
from fractions import Fraction
import z3


class SMTValueAssigner:
    """
    Incremental SMT solver that maps traces to consistent numeric values.

    Traces are identified by their action sequences (tuples of strings).
    Values returned are always normalised to [0, 1].
    """

    def __init__(self) -> None:
        self._solver   = z3.Optimize()
        self._vars: dict[tuple, z3.ArithRef] = {}
        self._constraints: list[tuple] = []   # for inspection / debugging
        self._last_values: dict[tuple, float] = {}
        self._n = 0                            # counter for unique var names

    # ------------------------------------------------------------------
    # Variable management
    # ------------------------------------------------------------------

    def _var(self, trace: list[str] | tuple) -> z3.ArithRef:
        key = tuple(trace)
        if key not in self._vars:
            self._n += 1
            v = z3.Real(f'v{self._n}')
            self._vars[key] = v
            # Keep all values in [0, 100]
            self._solver.add(v >= 0)
            self._solver.add(v <= 100)
        return self._vars[key]

    # ------------------------------------------------------------------
    # Adding preferences
    # ------------------------------------------------------------------

    def add(self, trace1: list[str], trace2: list[str], preference: str) -> None:
        """
        Record a pairwise preference.

        Parameters
        ----------
        trace1, trace2 : action sequences
        preference     : 't1'   → trace1 preferred
                         't2'   → trace2 preferred
                         'equal'→ no preference
        """
        v1 = self._var(trace1)
        v2 = self._var(trace2)
        key = (tuple(trace1), tuple(trace2), preference)
        self._constraints.append(key)

        if preference == 't1':
            self._solver.add(v1 > v2)
        elif preference == 't2':
            self._solver.add(v2 > v1)
        elif preference == 'equal':
            self._solver.add(v1 == v2)
        else:
            raise ValueError(f'Unknown preference: {preference!r}')

    # ------------------------------------------------------------------
    # Solving
    # ------------------------------------------------------------------

    def solve(self) -> dict[tuple, float] | None:
        """
        Find a consistent numeric assignment for all traces seen so far.

        Returns a dict  trace_key → float ∈ [0, 1]  (normalised),
        or None if the constraints are unsatisfiable.
        """
        result = self._solver.check()
        if result != z3.sat:
            return None

        model = self._solver.model()
        raw: dict[tuple, float] = {}
        for key, var in self._vars.items():
            z3_val = model[var]
            if z3_val is None:
                raw[key] = 0.0
            else:
                raw[key] = float(Fraction(str(z3_val)))

        # Normalise to [0, 1]
        if not raw:
            return {}
        lo, hi = min(raw.values()), max(raw.values())
        if hi == lo:
            self._last_values = {k: 0.5 for k in raw}
        else:
            self._last_values = {k: (v - lo) / (hi - lo) for k, v in raw.items()}

        return dict(self._last_values)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def value(self, trace: list[str] | tuple) -> float | None:
        """Return the last solved value for *trace*, or None if unseen."""
        return self._last_values.get(tuple(trace))

    def is_satisfiable(self) -> bool:
        return self._solver.check() == z3.sat

    def n_constraints(self) -> int:
        return len(self._constraints)

    def known_traces(self) -> list[tuple]:
        return list(self._vars.keys())


# ------------------------------------------------------------------
# Quick demo
# ------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    from src.game.game_generator import generate_tree, compute_trace_scores
    from src.game.game_nfa import GameNFA
    from src.lstar_mcts.preference_oracle import PreferenceOracle

    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    seed  = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    root   = generate_tree(depth, seed=seed)
    nfa    = GameNFA(root)
    oracle = PreferenceOracle(nfa)

    smt = SMTValueAssigner()

    # Feed all pairwise preferences for terminal traces
    traces = [path for path, _ in compute_trace_scores(root)]
    print(f'Adding {len(traces)*(len(traces)-1)//2} pairwise preferences '
          f'over {len(traces)} terminal traces...')

    for i, t1 in enumerate(traces):
        for t2 in traces[i + 1:]:
            pref = oracle.compare(t1, t2)
            smt.add(t1, t2, pref)

    values = smt.solve()
    if values is None:
        print('UNSATISFIABLE — constraint contradiction')
    else:
        print(f'\nNormalised SMT values (0=worst, 1=best):')
        sorted_traces = sorted(values.items(), key=lambda x: -x[1])
        for trace_key, val in sorted_traces:
            trace_str = ' → '.join(trace_key)
            print(f'  {trace_str:35s}  {val:.4f}')

        # Sanity check: verify all preferences are respected
        violations = 0
        for t1, t2, pref in smt._constraints:
            v1, v2 = values.get(t1, 0), values.get(t2, 0)
            if pref == 't1' and not v1 > v2:
                violations += 1
            elif pref == 't2' and not v2 > v1:
                violations += 1
            elif pref == 'equal' and v1 != v2:
                violations += 1
        print(f'\nConstraint violations: {violations} / {smt.n_constraints()}')
