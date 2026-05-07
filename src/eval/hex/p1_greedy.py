"""
Greedy P1 (X) player for Hex.

Mirror of the frontier-connectivity heuristic HexOracle uses for P2 (O):
    h(state) = (o_reach - x_reach) / size² × 0.9

Where {token}_reach is the count of {token}-coloured cells reachable
(by same-colour adjacency) from {token}'s starting edge:
  X (P1) starts from the top row, trying to reach the bottom.
  O (P2) starts from the left col, trying to reach the right.

P2 (greedy) MAXIMISES h — more O-reach, fewer X-reach.
P1 (greedy) MINIMISES h — more X-reach, fewer O-reach.

For each legal move, evaluate h on the resulting child; pick the cell
that minimises h. Ties broken by RNG.

This is one-step adversarial: it advances P1's connection greedily but
ignores threats P2 can build in parallel — so e.g. greedy P1 misses
"bridge" moves that secure two-step connections, and is exploitable by
a smarter P2.

Usage:
    from src.eval.hex.p1_greedy import GreedyP1
    p1 = GreedyP1(seed=0)
    action = p1.pick(state)
"""

from __future__ import annotations
import random
from src.game.hex.board               import HexState
from src.game.hex.preference_oracle   import HexOracle


class GreedyP1:

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def pick(self, state: HexState) -> int:
        if state.player != 'P1':
            raise ValueError(f'GreedyP1 called on a {state.player} state')
        if state.is_terminal():
            raise ValueError('GreedyP1 called on a terminal state')

        scored = [
            (cell, HexOracle._heuristic(child))
            for cell, child in state.children.items()
        ]
        min_h      = min(h for _, h in scored)
        candidates = [cell for cell, h in scored if h == min_h]
        return self.rng.choice(candidates)


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------

if __name__ == '__main__':
    from src.game.hex.game_nfa import HexNFA

    nfa = HexNFA(size=3)
    p1  = GreedyP1(seed=42)

    print('Greedy P1 at empty 3×3 board:')
    scored = [(c, HexOracle._heuristic(child))
              for c, child in nfa.root.children.items()]
    for cell, h in sorted(scored, key=lambda kv: kv[1])[:5]:
        print(f'  cell {cell}: heuristic = {h:+.4f}')
    print(f'  pick: {p1.pick(nfa.root)}  (smallest heuristic)')
