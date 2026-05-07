"""
Greedy P1 (X) player for Tic-Tac-Toe.

Mirror of the depth-0 heuristic the TicTacToeOracle uses for P2 (O):
    h(board) = (O_open_line_score − X_open_line_score) / 8

P2 (greedy) MAXIMISES this score (more O-favourable lines).
P1 (greedy) MINIMISES it — the smaller (= more X-favourable, fewer
O lines), the better for P1.

For each legal action a, evaluate h(child_board); pick the action with
the smallest h. Ties broken by RNG.

This is one-step adversarial: it doesn't anticipate P2's response or
look further down the tree. Globally suboptimal — a smarter P2 will
exploit greedy P1's myopia (e.g., let P1 set up a double-threat trap
that greedy P1 misses).

Usage:
    from src.eval.tic_tac_toe.p1_greedy import GreedyP1
    p1 = GreedyP1(seed=0)
    action = p1.pick(state)
"""

from __future__ import annotations
import random
from src.game.tic_tac_toe.board               import TicTacToeState
from src.game.tic_tac_toe.preference_oracle   import TicTacToeOracle


class GreedyP1:

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def pick(self, state: TicTacToeState) -> int:
        if state.player != 'P1':
            raise ValueError(f'GreedyP1 called on a {state.player} state')
        if state.is_terminal():
            raise ValueError('GreedyP1 called on a terminal state')

        # Score each child board by P2's heuristic; pick the lowest.
        scored = [
            (sq, TicTacToeOracle._heuristic(child.board))
            for sq, child in state.children.items()
        ]
        min_h = min(h for _, h in scored)
        candidates = [sq for sq, h in scored if h == min_h]
        return self.rng.choice(candidates)


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------

if __name__ == '__main__':
    from src.game.tic_tac_toe.game_nfa import TicTacToeNFA

    nfa = TicTacToeNFA()
    p1  = GreedyP1(seed=42)

    print('Greedy P1 at empty board:')
    scored = [(sq, TicTacToeOracle._heuristic(c.board))
              for sq, c in nfa.root.children.items()]
    for sq, h in sorted(scored, key=lambda kv: kv[1]):
        print(f'  square {sq}: heuristic = {h:+.4f}')
    print(f'  pick: {p1.pick(nfa.root)}  (smallest heuristic)')
