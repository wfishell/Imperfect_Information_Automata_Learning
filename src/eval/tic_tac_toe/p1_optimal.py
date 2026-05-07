"""
Optimal P1 (X) player for Tic-Tac-Toe.

Full minimax assuming optimal P2 (O) response. Caches minimax values
across all reachable states (the TTT game tree is small enough). Uses
the same recursion the TicTacToeOracle uses internally:
    leaf:       state.value
    P2 to move: max over children      (P2 maximises terminal value)
    P1 to move: min over children      (P1 minimises)

Terminal values are −1 (X wins, i.e., P1 wins → bad for O), 0 (draw),
+1 (O wins). So P1 wants the minimum.

Under optimal play by both sides Tic-Tac-Toe is a draw → minimax value
at the empty board is 0. OptimalP1 will find a draw against OptimalP2;
against weaker P2 it can win.

Usage:
    from src.eval.tic_tac_toe.p1_optimal import OptimalP1
    p1 = OptimalP1()        # solves the TTT tree at construction
    action = p1.pick(state)
"""

from __future__ import annotations
import random
from src.game.tic_tac_toe.board     import TicTacToeState
from src.game.tic_tac_toe.game_nfa  import TicTacToeNFA


class OptimalP1:

    def __init__(self, seed: int | None = None) -> None:
        self._cache: dict[tuple, int] = {}
        self.rng = random.Random(seed)
        # Solve the entire game tree once.
        nfa = TicTacToeNFA()
        self._solve(nfa.root)

    def _solve(self, state: TicTacToeState) -> int:
        key = (state.board, state.player)
        if key in self._cache:
            return self._cache[key]
        if state.is_terminal():
            v = state.value
        else:
            cs = [self._solve(c) for c in state.children.values()]
            v = max(cs) if state.player == 'P2' else min(cs)
        self._cache[key] = v
        return v

    def pick(self, state: TicTacToeState) -> int:
        if state.player != 'P1':
            raise ValueError(f'OptimalP1 called on a {state.player} state')
        if state.is_terminal():
            raise ValueError('OptimalP1 called on a terminal state')

        scored = [
            (sq, self._cache[(child.board, child.player)])
            for sq, child in state.children.items()
        ]
        min_v = min(v for _, v in scored)
        candidates = [sq for sq, v in scored if v == min_v]
        return self.rng.choice(candidates)


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------

if __name__ == '__main__':
    from src.game.tic_tac_toe.game_nfa import TicTacToeNFA

    nfa = TicTacToeNFA()
    p1  = OptimalP1(seed=42)

    print('Optimal P1 at empty board:')
    scored = [(sq, p1._cache[(c.board, c.player)])
              for sq, c in nfa.root.children.items()]
    for sq, v in sorted(scored, key=lambda kv: kv[1]):
        print(f'  square {sq}: minimax = {v}')
    print(f'  pick: {p1.pick(nfa.root)}  (smallest minimax — opens with optimal play)')
    print()
    print('TTT under optimal P1 vs optimal P2 → draw (minimax = 0 from root).')
