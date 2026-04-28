from __future__ import annotations
from collections import deque

from src.game.hex.game_nfa import HexNFA
from src.game.hex.board import HexState, _neighbors, X, O


class HexOracle:

    def __init__(self, nfa: HexNFA, depth: int | None = None) -> None:
        self.nfa   = nfa
        self.depth = depth                 # None → unbounded (globally optimal)
        self._cache: dict[tuple, float] = {}

    # ------------------------------------------------------------------
    # Public API — mirrors TicTacToeOracle interface
    # ------------------------------------------------------------------

    def preferred_move(self, prefix: list) -> int | None:
        """Return the best P2 (O) cell from the state reached by prefix."""
        state = self.nfa.get_node(prefix)
        if state is None or state.player != 'P2' or state.is_terminal():
            return None
        return max(state.children,
                   key=lambda cell: self._minimax(state.children[cell], self.depth))

    def compare(self, trace1: list, trace2: list) -> str:
        v1 = self._trace_value(trace1)
        v2 = self._trace_value(trace2)
        if v1 > v2:
            return 't1'
        if v2 > v1:
            return 't2'
        return 'equal'

    # ------------------------------------------------------------------
    # Internal minimax
    # ------------------------------------------------------------------

    def _trace_value(self, trace: list) -> float:
        state = self.nfa.get_node(trace)
        if state is None:
            return -1.0
        return self._minimax(state, self.depth)

    def _minimax(self, state: HexState, depth: int | None) -> float:
        key = (state.board, state.player, depth)
        if key in self._cache:
            return self._cache[key]

        if state.is_terminal():
            result = float(state.value)
        elif depth == 0:
            result = self._heuristic(state)
        elif state.player == 'P2':
            next_depth = None if depth is None else depth - 1
            result = max(self._minimax(c, next_depth) for c in state.children.values())
        else:
            next_depth = None if depth is None else depth - 1
            result = min(self._minimax(c, next_depth) for c in state.children.values())

        self._cache[key] = result
        return result

    @staticmethod
    def _heuristic(state: HexState) -> float:
        """
        Frontier connectivity heuristic from P2 (O) perspective.

        For each player, count cells of their color reachable (via same-color
        adjacency) from their starting edge.  A larger frontier means more
        progress toward a winning connection.

        P1 (X) starts from the top row, trying to reach the bottom row.
        P2 (O) starts from the left col, trying to reach the right col.

        Normalized to (-1, 1) — consistent with terminal values ±1.
        """
        size  = state.size
        board = state.board
        total = size * size

        def frontier_size(token: int, starts: list[int]) -> int:
            owned   = [c for c in starts if board[c] == token]
            visited: set  = set(owned)
            queue:   deque = deque(owned)
            while queue:
                cell = queue.popleft()
                for nb in _neighbors(cell, size):
                    if nb not in visited and board[nb] == token:
                        visited.add(nb)
                        queue.append(nb)
            return len(visited)

        x_starts = list(range(size))               # top row cells
        o_starts = [r * size for r in range(size)] # left col cells

        x_reach = frontier_size(X, x_starts)
        o_reach = frontier_size(O, o_starts)

        if total == 0:
            return 0.0
        return (o_reach - x_reach) / total * 0.9
