from __future__ import annotations
from src.game.dots_and_boxes.game_nfa import DotsAndBoxesNFA
from src.game.dots_and_boxes.board import DotsAndBoxesState


class DotsAndBoxesOracle:

    def __init__(self, nfa: DotsAndBoxesNFA, depth: int | None = None) -> None:
        self.nfa   = nfa
        self.depth = depth                 # None → unbounded (globally optimal)
        self._cache: dict[tuple, float] = {}

    # ------------------------------------------------------------------
    # Public API — mirrors TicTacToeOracle interface
    # ------------------------------------------------------------------

    def preferred_move(self, prefix: list) -> int | str | None:
        """Return the best P2 move from the state reached by prefix.

        At a forced-pass P2 state the only legal move is PASS, which is
        returned directly.  At a real P2 state the best edge is returned.
        """
        state = self.nfa.get_node(prefix)
        if state is None or state.player != 'P2' or state.is_terminal():
            return None
        return max(state.children, key=lambda e: self._minimax(state.children[e], self.depth))

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

    def _minimax(self, state: DotsAndBoxesState, depth: int | None) -> float:
        # forced_pass is included in the key: a forced-pass state has a
        # different value than a normal state with the same edges/player/boxes
        # (forced-pass has only one PASS child, normal has real choices).
        key = (state.edges, state.player, state.p1_boxes, state.p2_boxes,
               state.forced_pass, depth)
        if key in self._cache:
            return self._cache[key]

        if state.is_terminal():
            result = float(state.value)
        elif depth == 0:
            result = self._heuristic(state)
        elif state.player == 'P2':
            next_depth = None if depth is None else depth - 1
            result = max(self._minimax(child, next_depth) for child in state.children.values())
        else:
            next_depth = None if depth is None else depth - 1
            result = min(self._minimax(child, next_depth) for child in state.children.values())

        self._cache[key] = result
        return result

    @staticmethod
    def _heuristic(state: DotsAndBoxesState) -> float:
        """
        Box score differential from P2's perspective, normalized to (-1, 1).

        (p2_boxes - p1_boxes) / total_boxes

        Captures the current lead without lookahead.  Consistent in scale
        with terminal values ±1 so compare() gives real signal at the cutoff.
        """
        total = state.total_boxes
        if total == 0:
            return 0.0
        return (state.p2_boxes - state.p1_boxes) / total
