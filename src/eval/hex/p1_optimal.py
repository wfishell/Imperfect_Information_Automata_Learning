"""
Optimal P1 (X) player for Hex (full minimax solve).

Caches minimax values for every reachable state from the empty board:
    leaf:        state.value (terminal, ±1 by winner)
    P2 to move:  max over children      (P2 maximises)
    P1 to move:  min over children      (P1 minimises)

Hex is famously a first-player-win game on any size board (Nash, 1949)
— so under optimal play by both sides, OptimalP1 will win every game.
On a 3×3 the game tree is small (a few hundred to a few thousand
reachable states); on 4×4 it grows but is still tractable in seconds
to a few minutes.

Usage:
    from src.eval.hex.p1_optimal import OptimalP1
    p1 = OptimalP1(size=3, seed=0)        # solves the game tree at construction
    action = p1.pick(state)
"""

from __future__ import annotations
import random
from src.game.hex.board     import HexState
from src.game.hex.game_nfa  import HexNFA


class OptimalP1:

    def __init__(self,
                 size: int = 3,
                 seed: int | None = None) -> None:
        self.rng     = random.Random(seed)
        self._cache: dict[tuple, float] = {}
        nfa          = HexNFA(size=size)
        self._solve(nfa.root)

    @staticmethod
    def _state_key(state: HexState) -> tuple:
        return (state.board, state.player)

    def _solve(self, state: HexState) -> float:
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

    def pick(self, state: HexState) -> int:
        if state.player != 'P1':
            raise ValueError(f'OptimalP1 called on a {state.player} state')
        if state.is_terminal():
            raise ValueError('OptimalP1 called on a terminal state')

        scored = [
            (cell, self._cache[self._state_key(child)])
            for cell, child in state.children.items()
        ]
        min_v      = min(v for _, v in scored)
        candidates = [cell for cell, v in scored if v == min_v]
        return self.rng.choice(candidates)


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------

if __name__ == '__main__':
    from src.game.hex.game_nfa import HexNFA

    size = 3
    nfa  = HexNFA(size=size)
    print(f'Solving Hex {size}×{size} minimax (may take a few seconds)...')
    p1   = OptimalP1(size=size, seed=42)
    print(f'Solved. Cache size: {len(p1._cache)} unique states.')

    print(f'\nOptimal P1 at empty {size}×{size} board:')
    scored = [(c, p1._cache[p1._state_key(child)])
              for c, child in nfa.root.children.items()]
    for cell, v in sorted(scored, key=lambda kv: kv[1])[:5]:
        print(f'  cell {cell}: minimax = {v:+.2f}')
    print(f'  pick: {p1.pick(nfa.root)}  (smallest minimax = best for P1)')
    print(f'\nHex is a first-player-win game; OptimalP1 wins every game vs OptimalP2.')
