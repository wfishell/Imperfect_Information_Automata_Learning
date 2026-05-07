"""
Optimal P1 player for Dots and Boxes (full minimax solve).

Full minimax assuming optimal P2:
    leaf:        state.value (terminal box-differential)
    P2 to move:  max over children      (P2 maximises)
    P1 to move:  min over children      (P1 minimises)

The DaB game tree on a 2x2 board is small enough to enumerate (a few
thousand reachable states). On 3x2 or larger the cache grows fast but
remains tractable for offline solving — minutes of construction time
on a laptop.

Forced-pass states are handled in the recursion: their only child is
PASS, so the minimax value equals the child's value.

Usage:
    from src.eval.dots_and_boxes.p1_optimal import OptimalP1
    p1 = OptimalP1(rows=2, cols=2, seed=0)   # solves at construction
    action = p1.pick(state)
"""

from __future__ import annotations
import random
from src.game.dots_and_boxes.board     import DotsAndBoxesState, PASS
from src.game.dots_and_boxes.game_nfa  import DotsAndBoxesNFA


class OptimalP1:

    def __init__(self,
                 rows: int = 2,
                 cols: int = 2,
                 seed: int | None = None) -> None:
        self.rng     = random.Random(seed)
        self._cache: dict[tuple, float] = {}
        nfa          = DotsAndBoxesNFA(rows=rows, cols=cols)
        self._solve(nfa.root)

    def _state_key(self, state: DotsAndBoxesState) -> tuple:
        # Same key shape as DotsAndBoxesOracle._minimax — distinguishes
        # forced_pass states (which have a different action set) from
        # otherwise-identical normal states.
        return (state.edges, state.player,
                state.p1_boxes, state.p2_boxes,
                state.forced_pass)

    def _solve(self, state: DotsAndBoxesState) -> float:
        key = self._state_key(state)
        if key in self._cache:
            return self._cache[key]

        if state.is_terminal():
            v = float(state.value)
        else:
            child_vals = [self._solve(c) for c in state.children.values()]
            v = max(child_vals) if state.player == 'P2' else min(child_vals)
        self._cache[key] = v
        return v

    def pick(self, state: DotsAndBoxesState):
        if state.player != 'P1':
            raise ValueError(f'OptimalP1 called on a {state.player} state')
        if state.is_terminal():
            raise ValueError('OptimalP1 called on a terminal state')

        # Forced-pass: only PASS is legal.
        if state.forced_pass:
            return PASS

        scored = [(action, self._cache[self._state_key(child)])
                  for action, child in state.children.items()]
        min_v      = min(v for _, v in scored)
        candidates = [action for action, v in scored if v == min_v]
        return self.rng.choice(candidates)


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------

if __name__ == '__main__':
    from src.game.dots_and_boxes.game_nfa import DotsAndBoxesNFA

    nfa = DotsAndBoxesNFA(rows=2, cols=2)
    print('Solving 2x2 DaB minimax (cache may take a few seconds)...')
    p1  = OptimalP1(rows=2, cols=2, seed=42)
    print(f'Solved. Cache size: {len(p1._cache)} unique states.')

    print('\nOptimal P1 at empty 2x2 board:')
    scored = [(a, p1._cache[p1._state_key(c)])
              for a, c in nfa.root.children.items()]
    for a, v in sorted(scored, key=lambda kv: kv[1])[:5]:
        print(f'  edge {a}: minimax value = {v:+.4f}')
    print(f'  pick: {p1.pick(nfa.root)}  (smallest minimax)')
