"""
Greedy P1 player for Nim.

Mirror of the largest-pile heuristic NimOracle uses for P2:
    h(state) = (max(piles) / sum(piles))     scaled to ±0.9, signed by player

P2 (greedy) MAXIMISES h — prefers states where one big pile dominates.
P1 (greedy) MINIMISES h — prefers states where piles are evened out.

For each legal action, evaluate h on the resulting child; pick the
action with the smallest h. Ties broken by RNG.

This is one-step adversarial: it ignores Nim's actual game theory
(which depends on Nim-sum / XOR of pile sizes), so greedy P1 routinely
hands P2 winning positions. That's the failure mode L*+MCTS lookahead
exploits.

Usage:
    from src.eval.nim.p1_greedy import GreedyP1
    p1 = GreedyP1(seed=0)
    action = p1.pick(state)
"""

from __future__ import annotations
import random
from src.game.nim.board               import NimState
from src.game.nim.preference_oracle   import NimOracle


class GreedyP1:

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def pick(self, state: NimState):
        if state.player != 'P1':
            raise ValueError(f'GreedyP1 called on a {state.player} state')
        if state.is_terminal():
            raise ValueError('GreedyP1 called on a terminal state')

        scored = [
            (action, NimOracle._heuristic(child))
            for action, child in state.children.items()
        ]
        min_h      = min(h for _, h in scored)
        candidates = [action for action, h in scored if h == min_h]
        return self.rng.choice(candidates)


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------

if __name__ == '__main__':
    from src.game.nim.game_nfa import NimNFA

    nfa = NimNFA(piles=(1, 2, 3))
    p1  = GreedyP1(seed=42)

    print('Greedy P1 at piles=(1,2,3):')
    scored = [(a, NimOracle._heuristic(c))
              for a, c in nfa.root.children.items()]
    for a, h in sorted(scored, key=lambda kv: kv[1])[:5]:
        # action a = (pile_idx, count); show resulting piles too
        child_piles = nfa.root.children[a].piles
        print(f'  {a} → piles={child_piles}: heuristic = {h:+.4f}')
    print(f'  pick: {p1.pick(nfa.root)}  (smallest heuristic)')
