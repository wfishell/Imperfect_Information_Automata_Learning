"""
Custom Mealy L* implementation with targeted table invalidation.

Unlike AALpy's run_Lstar, this implementation allows the SUL to change
mid-run. When update_strategy() is called on the SUL, the caller can
call lstar.invalidate(p1_prefix) to mark only the affected table rows
as stale. Only those rows are re-queried before the next hypothesis is
built — the rest of the table remains valid.

Table structure:
  S : set of prefix tuples  (P1 input sequences)
  E : set of suffix tuples  (P1 input sequences)
  T : (s, e) -> tuple of P2 outputs when running s+e through the SUL

Two rows are equivalent if T[s1][e] == T[s2][e] for all e in E.

Closed    : for every s in S and a in alphabet, row(s+a) matches some row in S
Consistent: if row(s1)==row(s2) then for all a, row(s1+a)==row(s2+a)
"""

from __future__ import annotations
from src.lstar_mcts.game_sul import GameSUL


class MealyState:
    def __init__(self, state_id: str, row: tuple):
        self.state_id   = state_id
        self.row        = row
        self.transitions: dict[str, tuple[str, 'MealyState']] = {}  # input -> (output, next_state)

    def __repr__(self):
        return f'State({self.state_id})'


class MealyMachine:
    """Minimal Mealy machine compatible with find_cex interface."""

    def __init__(self, states: list[MealyState], initial: MealyState):
        self.states        = states
        self.initial_state = initial
        self.current_state = initial

    def reset_to_initial(self):
        self.current_state = self.initial_state

    def step(self, inp: str) -> str | None:
        t = self.current_state.transitions.get(inp)
        if t is None:
            return None
        output, next_state = t
        self.current_state = next_state
        return output

    def save(self, path: str):
        lines = ['digraph MealyMachine {', '  rankdir=LR;']
        for state in self.states:
            shape = 'doublecircle' if state is self.initial_state else 'circle'
            lines.append(f'  {state.state_id} [shape={shape}];')
        for state in self.states:
            for inp, (out, dst) in state.transitions.items():
                lines.append(f'  {state.state_id} -> {dst.state_id} [label="{inp}/{out}"];')
        lines.append('}')
        with open(path + '.dot', 'w') as f:
            f.write('\n'.join(lines))


class MealyLStar:

    def __init__(self, alphabet: list[str], sul: GameSUL, eq_oracle, verbose: bool = False):
        self.alphabet  = alphabet
        self.sul       = sul
        self.oracle    = eq_oracle
        self.verbose   = verbose

        self.S: list[tuple] = [()]                          # prefixes (epsilon always first)
        self.E: list[tuple] = [(a,) for a in alphabet]     # suffixes (single inputs to start)

        # T[(s, e)] = tuple of P2 outputs for the e-portion of query s+e
        self.T: dict[tuple, tuple] = {}

        # stale entries that need re-querying before next hypothesis
        self._stale: set[tuple] = set()

    # ------------------------------------------------------------------
    # SUL queries
    # ------------------------------------------------------------------

    def _query(self, p1_sequence: tuple) -> tuple:
        """Run a full membership query and return all P2 outputs."""
        self.sul.pre()
        outputs = tuple(self.sul.step(a) for a in p1_sequence)
        self.sul.post()
        return outputs

    def _fill_entry(self, s: tuple, e: tuple):
        full    = s + e
        outputs = self._query(full)
        self.T[(s, e)] = outputs[len(s):]   # only the e-portion
        self._stale.discard((s, e))

    def _fill_table(self):
        """Fill all missing or stale entries for S ∪ S·Σ × E."""
        upper = set(self.S)
        lower = {s + (a,) for s in self.S for a in self.alphabet}
        for s in upper | lower:
            for e in self.E:
                if (s, e) not in self.T or (s, e) in self._stale:
                    self._fill_entry(s, e)

    # ------------------------------------------------------------------
    # Invalidation
    # ------------------------------------------------------------------

    def invalidate(self, p1_prefix: tuple):
        """
        Mark stale every table entry whose query passes through p1_prefix.
        Uses a conservative approximation: any (s, e) where s+e starts
        with p1_prefix is potentially affected.
        """
        n = len(p1_prefix)
        for (s, e) in list(self.T):
            full = s + e
            if len(full) >= n and full[:n] == p1_prefix:
                self._stale.add((s, e))

    # ------------------------------------------------------------------
    # Row access
    # ------------------------------------------------------------------

    def _row(self, s: tuple) -> tuple:
        return tuple(self.T.get((s, e)) for e in self.E)

    def _all_rows(self) -> set[tuple]:
        return {self._row(s) for s in self.S}

    # ------------------------------------------------------------------
    # Closedness
    # ------------------------------------------------------------------

    def _close(self):
        changed = True
        while changed:
            changed = False
            for s in list(self.S):
                for a in self.alphabet:
                    sa = s + (a,)
                    if self._row(sa) not in self._all_rows():
                        if sa not in self.S:
                            self.S.append(sa)
                            self._fill_table()
                            changed = True

    # ------------------------------------------------------------------
    # Consistency
    # ------------------------------------------------------------------

    def _make_consistent(self):
        changed = True
        while changed:
            changed = False
            for i, s1 in enumerate(self.S):
                for s2 in self.S[i + 1:]:
                    if self._row(s1) != self._row(s2):
                        continue
                    for a in self.alphabet:
                        for e in list(self.E):
                            v1 = self.T.get((s1 + (a,), e))
                            v2 = self.T.get((s2 + (a,), e))
                            if v1 != v2:
                                new_e = (a,) + e
                                if new_e not in self.E:
                                    self.E.append(new_e)
                                    self._fill_table()
                                    changed = True

    # ------------------------------------------------------------------
    # Hypothesis construction
    # ------------------------------------------------------------------

    def _build_hypothesis(self) -> MealyMachine:
        # Map each distinct row to a state
        row_to_state: dict[tuple, MealyState] = {}
        for s in self.S:
            r = self._row(s)
            if r not in row_to_state:
                sid = f'q{len(row_to_state)}'
                row_to_state[r] = MealyState(sid, r)

        # Wire transitions
        for s in self.S:
            src = row_to_state[self._row(s)]
            for a in self.alphabet:
                sa  = s + (a,)
                out = self.T.get((s, (a,)))
                output     = out[0] if out else None
                dst        = row_to_state[self._row(sa)]
                src.transitions[a] = (output, dst)

        initial = row_to_state[self._row(())]
        states  = list(row_to_state.values())

        if self.verbose:
            print(f'  Hypothesis: {len(states)} states')

        return MealyMachine(states, initial)

    # ------------------------------------------------------------------
    # CEX processing
    # ------------------------------------------------------------------

    def _process_cex(self, cex: list[str]):
        """Add all prefixes of the CEX to S."""
        for i in range(1, len(cex) + 1):
            prefix = tuple(cex[:i])
            if prefix not in self.S:
                self.S.append(prefix)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> MealyMachine:
        self._fill_table()

        round_num = 0
        while True:
            # Alternate close/consistent until both hold simultaneously
            prev = None
            while (len(self.S), len(self.E)) != prev:
                prev = (len(self.S), len(self.E))
                self._close()
                self._make_consistent()

            hypothesis = self._build_hypothesis()
            round_num += 1
            print(f'Hypothesis {round_num}: {len(hypothesis.states)} states.')

            cex = self.oracle.find_cex(hypothesis)

            if cex is None:
                return hypothesis

            # Invalidate all entries whose query passes through any prefix of the CEX
            for i in range(1, len(cex) + 1):
                self.invalidate(tuple(cex[:i]))

            self._process_cex(cex)
            self._fill_table()
