"""
Random P1 (X) player for Tic-Tac-Toe.

Picks a legal action (empty square) uniformly at random. Floor baseline.

Usage:
    from src.eval.tic_tac_toe.p1_random import RandomP1
    p1 = RandomP1(seed=42)
    action = p1.pick(state)
"""

from __future__ import annotations
import random
from src.game.tic_tac_toe.board import TicTacToeState


class RandomP1:

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def pick(self, state: TicTacToeState) -> int:
        if state.player != 'P1':
            raise ValueError(f'RandomP1 called on a {state.player} state')
        if state.is_terminal():
            raise ValueError('RandomP1 called on a terminal state')
        return self.rng.choice(list(state.children.keys()))


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------

if __name__ == '__main__':
    from src.game.tic_tac_toe.game_nfa import TicTacToeNFA

    nfa = TicTacToeNFA()
    p1  = RandomP1(seed=42)

    print('Random P1 sample picks at empty board:')
    for _ in range(3):
        print(f'  {p1.pick(nfa.root)}')
