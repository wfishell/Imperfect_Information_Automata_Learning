"""
Random P1 player for Nim.

Picks uniformly over legal (pile_index, count) actions. Floor baseline.

Usage:
    from src.eval.nim.p1_random import RandomP1
    p1 = RandomP1(seed=42)
    action = p1.pick(state)         # returns e.g. (1, 2)  → take 2 from pile 1
"""

from __future__ import annotations
import random
from src.game.nim.board import NimState


class RandomP1:

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def pick(self, state: NimState):
        if state.player != 'P1':
            raise ValueError(f'RandomP1 called on a {state.player} state')
        if state.is_terminal():
            raise ValueError('RandomP1 called on a terminal state')
        return self.rng.choice(list(state.children.keys()))


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------

if __name__ == '__main__':
    from src.game.nim.game_nfa import NimNFA

    nfa = NimNFA(piles=(1, 2, 3))
    p1  = RandomP1(seed=42)

    print('Random P1 sample picks at piles=(1,2,3):')
    legal = list(nfa.root.children.keys())
    print(f'  legal actions ({len(legal)}): {legal}')
    for _ in range(3):
        print(f'  pick: {p1.pick(nfa.root)}')
