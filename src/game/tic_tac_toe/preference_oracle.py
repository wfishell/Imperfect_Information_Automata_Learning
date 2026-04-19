from __future__ import annotations
from src.game.tic_tac_toe.game_nfa import TicTacToeNFA
from src.game.tic_tac_toe.board import TicTacToeState


class TicTacToeOracle:

    def __init__(self, nfa: TicTacToeNFA) -> None:
        self.nfa    = nfa
        self._cache: dict[tuple, int] = {}   # (board, player) → minimax value

    # ------------------------------------------------------------------
    # Public API — mirrors PreferenceOracle interface
    # ------------------------------------------------------------------

    def preferred_move(self, prefix: list) -> int | None:
        """Return the best P2 (O) move from the state reached by prefix."""
        state = self.nfa.get_node(prefix)
        if state is None or state.player != 'P2' or state.is_terminal():
            return None
        return max(state.children, key=lambda sq: self._minimax(state.children[sq]))

    def compare(self, trace1: list, trace2: list) -> str:
        """
        Compare two traces from P2's perspective.

        Uses minimax value of the resulting state — works for both
        terminal traces and non-terminal traces (e.g. at depth_n < 9).
        """
        v1 = self._trace_value(trace1)
        v2 = self._trace_value(trace2)
        if v1 > v2:   return 't1'
        if v2 > v1:   return 't2'
        return 'equal'

    # ------------------------------------------------------------------
    # Internal minimax
    # ------------------------------------------------------------------

    def _trace_value(self, trace: list) -> int:
        state = self.nfa.get_node(trace)
        if state is None:
            return -1   # illegal trace — treat as worst outcome for P2
        return self._minimax(state)

    def _minimax(self, state: TicTacToeState) -> int:
        key = (state.board, state.player)
        if key in self._cache:
            return self._cache[key]

        if state.is_terminal():
            result = state.value
        elif state.player == 'P2':   # O maximises
            result = max(self._minimax(child) for child in state.children.values())
        else:                        # P1 minimises (from O's perspective)
            result = min(self._minimax(child) for child in state.children.values())

        self._cache[key] = result
        return result
