from __future__ import annotations
from src.game.nim.game_nfa import NimNFA
from src.game.nim.board import NimState


class NimOracle:

    def __init__(self, nfa: NimNFA, depth: int | None = None) -> None:
        self.nfa   = nfa
        self.depth = depth
        self._cache: dict[tuple, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preferred_move(self, prefix: list) -> tuple | None:
        state = self.nfa.get_node(prefix)
        if state is None or state.player != 'P2' or state.is_terminal():
            return None
        return max(state.children,
                   key=lambda mv: self._minimax(state.children[mv], self.depth))

    def compare(self, trace1: list, trace2: list) -> str:
        v1 = self._trace_value(trace1)
        v2 = self._trace_value(trace2)
        if v1 > v2: return 't1'
        if v2 > v1: return 't2'
        return 'equal'

    # ------------------------------------------------------------------
    # Internal minimax
    # ------------------------------------------------------------------

    def _trace_value(self, trace: list) -> float:
        state = self.nfa.get_node(trace)
        if state is None:
            return -1.0
        return self._minimax(state, self.depth)

    def _minimax(self, state: NimState, depth: int | None) -> float:
        key = (state.piles, state.player, depth)
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
    def _heuristic(state: NimState) -> float:
        nim_xor = 0
        for p in state.piles:
            nim_xor ^= p
        # nim_xor != 0 → player to move is in a winning position
        # Return from P2's perspective; ±0.5 stays strictly inside terminal ±1
        if state.player == 'P2':
            return 0.5 if nim_xor != 0 else -0.5
        else:
            return -0.5 if nim_xor != 0 else 0.5
