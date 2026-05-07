"""
Greedy P1 player for Dots and Boxes.

Mirror of the depth-0 heuristic the DotsAndBoxesOracle uses for P2:
    h(state) = (p2_boxes - p1_boxes) / total_boxes

P2 (greedy) MAXIMISES h (more P2 boxes is better).
P1 (greedy) MINIMISES h — fewer P2 boxes / more P1 boxes is better.

For each legal P1 action, evaluate h on the resulting child; pick the
action with the smallest h. Ties broken by RNG.

This is one-step adversarial: it doesn't anticipate the chain dynamics
that dominate DaB endgames, so greedy P1 routinely sets up long P2
chains. That's the whole point — global lookahead beats this.

Forced-pass states return PASS (only legal action).

Usage:
    from src.eval.dots_and_boxes.p1_greedy import GreedyP1
    p1 = GreedyP1(seed=0)
    action = p1.pick(state)
"""

from __future__ import annotations
import random
from src.game.dots_and_boxes.board               import DotsAndBoxesState, PASS
from src.game.dots_and_boxes.preference_oracle   import DotsAndBoxesOracle


class GreedyP1:

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def pick(self, state: DotsAndBoxesState):
        if state.player != 'P1':
            raise ValueError(f'GreedyP1 called on a {state.player} state')
        if state.is_terminal():
            raise ValueError('GreedyP1 called on a terminal state')

        # Forced-pass: only PASS is legal.
        if state.forced_pass:
            return PASS

        # Score each child by P2's heuristic; pick the lowest (best for P1).
        scored = [
            (action, DotsAndBoxesOracle._heuristic(child))
            for action, child in state.children.items()
        ]
        min_h      = min(h for _, h in scored)
        candidates = [action for action, h in scored if h == min_h]
        return self.rng.choice(candidates)


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------

if __name__ == '__main__':
    from src.game.dots_and_boxes.game_nfa import DotsAndBoxesNFA

    nfa = DotsAndBoxesNFA(rows=2, cols=2)
    p1  = GreedyP1(seed=42)

    print('Greedy P1 at empty 2x2 board:')
    scored = [(a, DotsAndBoxesOracle._heuristic(c))
              for a, c in nfa.root.children.items()]
    for a, h in sorted(scored, key=lambda kv: kv[1])[:5]:
        print(f'  edge {a}: heuristic = {h:+.4f}')
    print(f'  pick: {p1.pick(nfa.root)}  (smallest heuristic = best for P1)')
