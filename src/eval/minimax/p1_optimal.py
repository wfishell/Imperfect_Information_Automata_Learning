"""
Optimal P1 player for the minimax game (perfect adversary).

P1 minimises P2's cumulative trace value, assuming P2 plays optimally
in response. Computed by full minimax over the game tree:

    minimax(node):
        if terminal:    return node.value
        if P2 to move:  return node.value + max_a minimax(node.children[a])
        if P1 to move:  return node.value + min_a minimax(node.children[a])

OptimalP1 caches the minimax value at every node up front (one DFS over
the tree) so .pick() is O(branching) per call.

Usage:
    from src.eval.minimax.p1_optimal import OptimalP1
    p1 = OptimalP1(root)        # solves the tree at construction
    action = p1.pick(current_node)
"""

from __future__ import annotations
import random
from src.game.minimax.game_generator import GameNode


class OptimalP1:

    def __init__(self, root: GameNode, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self._cache: dict[int, float] = {}
        self._solve(root)

    def _solve(self, node: GameNode) -> float:
        key = id(node)
        if key in self._cache:
            return self._cache[key]

        if node.is_terminal():
            v = float(node.value)
        else:
            child_values = [self._solve(c) for c in node.children.values()]
            if node.player == 'P2':
                v = node.value + max(child_values)   # P2 maximises
            else:                                    # P1 minimises
                v = node.value + min(child_values)
        self._cache[key] = v
        return v

    def pick(self, node: GameNode) -> str:
        if node.player != 'P1':
            raise ValueError(f'OptimalP1 called on a {node.player} node')
        if node.is_terminal():
            raise ValueError('OptimalP1 called on a terminal node')
        # Pick the action whose child has the smallest minimax value;
        # break ties uniformly at random for variability across runs.
        scored     = [(a, self._cache[id(c)]) for a, c in node.children.items()]
        min_v      = min(v for _, v in scored)
        candidates = [a for a, v in scored if v == min_v]
        return self.rng.choice(candidates)


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------

if __name__ == '__main__':
    from src.game.minimax.game_generator import generate_tree

    root = generate_tree(depth=4, seed=0)
    p1   = OptimalP1(root)

    print('Optimal P1 at root:')
    print(f'  child minimax values: '
          f'{[(a, round(p1._cache[id(c)], 2)) for a, c in root.children.items()]}')
    print(f'  pick                : {p1.pick(root)}  (smallest minimax value)')
