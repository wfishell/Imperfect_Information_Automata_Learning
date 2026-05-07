"""
Random P1 (X) player for Hex.

Picks an empty cell uniformly at random. Floor baseline.

Usage:
    from src.eval.hex.p1_random import RandomP1
    p1 = RandomP1(seed=42)
    action = p1.pick(state)
"""

from __future__ import annotations
import random
from src.game.hex.board import HexState


class RandomP1:

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def pick(self, state: HexState) -> int:
        if state.player != 'P1':
            raise ValueError(f'RandomP1 called on a {state.player} state')
        if state.is_terminal():
            raise ValueError('RandomP1 called on a terminal state')
        return self.rng.choice(list(state.children.keys()))


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------

if __name__ == '__main__':
    from src.game.hex.game_nfa import HexNFA

    nfa = HexNFA(size=3)
    p1  = RandomP1(seed=42)

    print('Random P1 sample picks at empty 3×3 board:')
    print(f'  legal cells (count = {len(nfa.root.children)})')
    for _ in range(3):
        print(f'  pick: {p1.pick(nfa.root)}')
