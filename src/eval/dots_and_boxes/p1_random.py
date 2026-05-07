"""
Random P1 player for Dots and Boxes.

Picks uniformly over legal actions. On a forced_pass state (where P2
just completed a box and earned the extra turn) the only legal action
is PASS, so this naturally returns PASS without any special-casing —
state.children at a forced-pass state contains only the PASS edge.

Usage:
    from src.eval.dots_and_boxes.p1_random import RandomP1
    p1 = RandomP1(seed=42)
    action = p1.pick(state)
"""

from __future__ import annotations
import random
from src.game.dots_and_boxes.board import DotsAndBoxesState, PASS


class RandomP1:

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def pick(self, state: DotsAndBoxesState):
        if state.player != 'P1':
            raise ValueError(f'RandomP1 called on a {state.player} state')
        if state.is_terminal():
            raise ValueError('RandomP1 called on a terminal state')
        # On forced_pass states, children is just {PASS: ...} so this
        # returns PASS deterministically. Otherwise picks among edges.
        return self.rng.choice(list(state.children.keys()))


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------

if __name__ == '__main__':
    from src.game.dots_and_boxes.game_nfa import DotsAndBoxesNFA

    nfa = DotsAndBoxesNFA(rows=2, cols=2)
    p1  = RandomP1(seed=42)

    print('Random P1 sample picks at empty 2x2 board:')
    legal = list(nfa.root.children.keys())
    print(f'  legal actions: {legal}')
    for _ in range(3):
        print(f'  pick: {p1.pick(nfa.root)}')
