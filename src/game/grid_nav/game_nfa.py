"""
NFA wrapper around GridNavState.

Exposes the L*-compatible API used by the rest of the pipeline:
    root, alphabet, p1_alphabet, p2_alphabet,
    get_node(trace), is_terminal(trace), current_player(trace),
    p1_legal_inputs(trace), p2_legal_moves(trace),
    step(trace, action).

Construction:
    GridNavNFA(grid_size=5, k=3, seed=42)              # random obstacles
    GridNavNFA(grid_size=5, obstacles=[(1,1),(2,3)])   # explicit
    GridNavNFA(grid_size=6, goal=(5,5), k=3)           # custom goal
"""

from __future__ import annotations

import random
from collections import deque
from itertools   import product

from src.game.grid_nav.board import (
    GridNavState, ACTIONS, DELTAS, make_observation,
)


_DIR_BUCKETS = ('AT', 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW')


class GridNavNFA:

    def __init__(
        self,
        grid_size: int                       = 5,
        obstacles: list | tuple | None       = None,
        k:         int                       = 3,
        seed:      int | None                = None,
        max_moves: int                       = 30,
        goal:      tuple[int, int] | None    = None,
        start:     tuple[int, int]           = (0, 0),
    ) -> None:
        self.grid_size = grid_size
        self.start     = start
        self.goal      = goal if goal is not None else (grid_size - 1, grid_size - 1)
        self.max_moves = max_moves

        if obstacles is None:
            obstacles = self._sample_obstacles(k, seed)
        self.obstacles = frozenset(obstacles)

        if not self._reachable(self.start, self.goal, self.obstacles):
            raise ValueError(
                f'goal {self.goal} unreachable from start {self.start} with '
                f'obstacles {sorted(self.obstacles)}'
            )

        self.root = GridNavState(
            car_pos    = self.start,
            obstacles  = self.obstacles,
            grid_size  = self.grid_size,
            goal       = self.goal,
            max_moves  = self.max_moves,
            move_count = 0,
            player     = 'P1',
            last_obs   = None,
        )

        # P1 alphabet for L*: restrict to observations the environment
        # ACTUALLY emits in this layout (≈ 1 per reachable car cell).
        # The NFA is still permissive at P1 nodes (accepts all 144),
        # but L* only iterates over the reachable subset, which keeps
        # MCTS rollout combinatorics tractable.
        self.p1_alphabet = self._enumerate_reachable_observations()
        self.p2_alphabet = list(ACTIONS)
        self.alphabet    = self.p1_alphabet

    # ------------------------------------------------------------------
    # Sampling + reachability
    # ------------------------------------------------------------------

    def _sample_obstacles(self, k: int, seed: int | None) -> tuple:
        rng = random.Random(seed)
        candidates = [
            (x, y) for x in range(self.grid_size)
                   for y in range(self.grid_size)
                   if (x, y) != self.start and (x, y) != self.goal
        ]
        for _ in range(200):
            picks = tuple(rng.sample(candidates, k))
            if self._reachable(self.start, self.goal, frozenset(picks)):
                return picks
        raise RuntimeError(
            f'could not sample {k} obstacles with reachable goal in 200 tries'
        )

    def _reachable(self, src, dst, obstacles) -> bool:
        q = deque([src])
        seen = {src}
        while q:
            cx, cy = q.popleft()
            if (cx, cy) == dst:
                return True
            for ddx, ddy in DELTAS.values():
                nx, ny = cx + ddx, cy + ddy
                if (0 <= nx < self.grid_size and
                    0 <= ny < self.grid_size and
                    (nx, ny) not in obstacles and
                    (nx, ny) not in seen):
                    seen.add((nx, ny))
                    q.append((nx, ny))
        return False

    @staticmethod
    def _all_observation_symbols() -> list[str]:
        bits = [''.join(t) for t in product('01', repeat=4)]   # 16 patterns
        return [f'{d}|{b}' for d in _DIR_BUCKETS for b in bits]

    def _enumerate_reachable_observations(self) -> list[str]:
        """
        Observations the environment actually emits at any cell
        reachable from start (one observation per reachable car cell).
        Layout-specific; used to restrict L*'s input alphabet so MCTS
        rollouts don't blow up over the full 144-symbol space.
        """
        seen = set()
        frontier = [self.start]
        visited  = {self.start}
        while frontier:
            new_frontier = []
            for pos in frontier:
                seen.add(make_observation(
                    pos, self.obstacles, self.grid_size, self.goal))
                for ddx, ddy in DELTAS.values():
                    nx, ny = pos[0] + ddx, pos[1] + ddy
                    if (0 <= nx < self.grid_size and 0 <= ny < self.grid_size
                        and (nx, ny) not in self.obstacles
                        and (nx, ny) not in visited):
                        visited.add((nx, ny))
                        new_frontier.append((nx, ny))
            frontier = new_frontier
        return sorted(seen)

    # ------------------------------------------------------------------
    # L*-compatible navigation API
    # ------------------------------------------------------------------

    def get_node(self, trace: list) -> GridNavState | None:
        node = self.root
        for symbol in trace:
            if symbol not in node.children:
                return None
            node = node.children[symbol]
        return node

    def is_terminal(self, trace: list) -> bool:
        node = self.get_node(trace)
        return node is not None and node.is_terminal()

    def current_player(self, trace: list) -> str | None:
        node = self.get_node(trace)
        if node is None or node.is_terminal():
            return None
        return node.player

    def p1_legal_inputs(self, trace: list) -> list:
        node = self.get_node(trace)
        if node is None or node.player != 'P1' or node.is_terminal():
            return []
        return list(node.children.keys())

    def p2_legal_moves(self, trace: list) -> list:
        node = self.get_node(trace)
        if node is None or node.player != 'P2' or node.is_terminal():
            return []
        return list(node.children.keys())

    def step(self, trace: list, action) -> list | None:
        node = self.get_node(trace)
        if node is None or action not in node.children:
            return None
        return trace + [action]

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def render(self, car_pos: tuple[int, int] | None = None) -> str:
        n = self.grid_size
        car = car_pos if car_pos is not None else self.start
        rows = []
        for y in range(n - 1, -1, -1):
            row = []
            for x in range(n):
                if (x, y) == car:
                    row.append(' C ')
                elif (x, y) == self.goal:
                    row.append(' G ')
                elif (x, y) in self.obstacles:
                    row.append(' # ')
                else:
                    row.append(' . ')
            rows.append('|' + '|'.join(row) + '|')
        return '\n'.join(rows)


# ----------------------------------------------------------------------
# Quick demo
# ----------------------------------------------------------------------

if __name__ == '__main__':
    nfa = GridNavNFA(grid_size=5, k=3, seed=42)
    print(f'Start: {nfa.start}   Goal: {nfa.goal}   Obstacles: {sorted(nfa.obstacles)}')
    print(nfa.render())
    print()
    print(f'P1 alphabet size: {len(nfa.p1_alphabet)}   '
          f'P2 alphabet: {nfa.p2_alphabet}   max_moves={nfa.max_moves}')
    print(f'Initial observation: {list(nfa.root.children.keys())[0]}')
