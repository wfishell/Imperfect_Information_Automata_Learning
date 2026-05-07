"""
Random P1 player for the minimax game.

At each P1 node, picks a legal action uniformly at random. P1 here is
the environment / adversary; "random" means *no strategy* — purely
uniform over node.children. This is the floor baseline.

Usage:
    from src.eval.minimax.p1_random import RandomP1
    p1 = RandomP1(seed=42)
    action = p1.pick(current_node)
"""

from __future__ import annotations
import random
from src.game.minimax.game_generator import GameNode


class RandomP1:

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def pick(self, node: GameNode) -> str:
        if node.player != 'P1':
            raise ValueError(f'RandomP1 called on a {node.player} node')
        if node.is_terminal():
            raise ValueError('RandomP1 called on a terminal node')
        return self.rng.choice(list(node.children.keys()))


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------

if __name__ == '__main__':
    from src.game.minimax.game_generator import generate_tree

    root = generate_tree(depth=4, seed=0)
    p1   = RandomP1(seed=42)

    # Take 3 P1 actions — should differ across calls (random)
    print('Random P1 sample picks at root:')
    for _ in range(3):
        print(f'  {p1.pick(root)}')
