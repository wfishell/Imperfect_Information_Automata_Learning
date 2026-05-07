"""
Optimal P1 player for Nim (full minimax solve).

Caches minimax values for every reachable state from the given pile
configuration:
    leaf:        state.value (terminal)
    P2 to move:  max over children      (P2 maximises)
    P1 to move:  min over children      (P1 minimises)

Nim is theoretically tractable via Nim-sum (XOR of piles), but we use
full minimax here for consistency with how OptimalP2 would be computed
via NimOracle's _minimax. Either approach yields the same picks under
optimal play.

For typical small configurations like (1,2,3), (1,3,5), (2,3,4),
solving completes in milliseconds — only a few thousand reachable
states.

Usage:
    from src.eval.nim.p1_optimal import OptimalP1
    p1 = OptimalP1(piles=(1, 2, 3), seed=0)   # solves at construction
    action = p1.pick(state)
"""

from __future__ import annotations
import random
from src.game.nim.board     import NimState
from src.game.nim.game_nfa  import NimNFA


class OptimalP1:

    def __init__(self,
                 piles: tuple = (1, 2, 3),
                 seed: int | None = None) -> None:
        self.rng     = random.Random(seed)
        self._cache: dict[tuple, float] = {}
        nfa          = NimNFA(piles=piles)
        self._solve(nfa.root)

    @staticmethod
    def _state_key(state: NimState) -> tuple:
        return (state.piles, state.player)

    def _solve(self, state: NimState) -> float:
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

    def pick(self, state: NimState):
        if state.player != 'P1':
            raise ValueError(f'OptimalP1 called on a {state.player} state')
        if state.is_terminal():
            raise ValueError('OptimalP1 called on a terminal state')

        scored = [
            (action, self._cache[self._state_key(child)])
            for action, child in state.children.items()
        ]
        min_v      = min(v for _, v in scored)
        candidates = [action for action, v in scored if v == min_v]
        return self.rng.choice(candidates)


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------

if __name__ == '__main__':
    from src.game.nim.game_nfa import NimNFA

    piles = (1, 2, 3)
    nfa   = NimNFA(piles=piles)
    p1    = OptimalP1(piles=piles, seed=42)

    print(f'Solved Nim {piles}. Cache size: {len(p1._cache)} unique states.')
    print()
    print(f'Optimal P1 at piles={piles}:')
    scored = [(a, p1._cache[p1._state_key(c)])
              for a, c in nfa.root.children.items()]
    for a, v in sorted(scored, key=lambda kv: kv[1])[:5]:
        child_piles = nfa.root.children[a].piles
        print(f'  {a} → piles={child_piles}: minimax = {v:+.2f}')
    print(f'  pick: {p1.pick(nfa.root)}  (smallest minimax = best for P1)')
