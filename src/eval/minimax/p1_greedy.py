"""
Greedy P1 player for the minimax game (one-step adversarial heuristic).

P1 is the adversary minimising P2's cumulative trace value. The greedy
heuristic picks the action whose IMMEDIATE child has the smallest
node.value — i.e., one-step lookahead, no consideration of P2's response
or downstream consequences.

This is *locally* optimal for P1 (best one-step move) but globally
suboptimal: a smarter P2 can exploit greedy P1's myopia.

Tie-breaking: random over actions tied for the lowest immediate value
(seed for reproducibility).

Usage:
    from src.eval.minimax.p1_greedy import GreedyP1
    p1 = GreedyP1(seed=0)
    action = p1.pick(current_node)
"""

from __future__ import annotations
import random
from src.game.minimax.game_generator import GameNode


class GreedyP1:

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def pick(self, node: GameNode) -> str:
        if node.player != 'P1':
            raise ValueError(f'GreedyP1 called on a {node.player} node')
        if node.is_terminal():
            raise ValueError('GreedyP1 called on a terminal node')

        # Lowest immediate child value = adversarial one-step choice.
        items     = list(node.children.items())
        min_val   = min(c.value for _, c in items)
        best_args = [a for a, c in items if c.value == min_val]
        return self.rng.choice(best_args)


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------

if __name__ == '__main__':
    from src.game.minimax.game_generator import generate_tree

    root = generate_tree(depth=4, seed=0)
    p1   = GreedyP1(seed=42)

    print('Greedy P1 at root:')
    print(f'  children: {[(a, c.value) for a, c in root.children.items()]}')
    print(f'  pick    : {p1.pick(root)}  (smallest immediate value)')
