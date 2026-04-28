from __future__ import annotations
from src.game.tic_tac_toe.game_nfa import TicTacToeNFA
from src.game.tic_tac_toe.board import TicTacToeState, LINES, EMPTY, X, O


class TicTacToeOracle:

    def __init__(self, nfa: TicTacToeNFA, depth: int | None = None) -> None:
        self.nfa   = nfa
        self.depth = depth                       # None → unbounded (globally optimal)
        self._cache: dict[tuple, int] = {}       # (board, player, remaining_depth) → value

    # ------------------------------------------------------------------
    # Public API — mirrors PreferenceOracle interface
    # ------------------------------------------------------------------

    def preferred_move(self, prefix: list) -> int | None:
        """Return the best P2 (O) move from the state reached by prefix."""
        state = self.nfa.get_node(prefix)
        if state is None or state.player != 'P2' or state.is_terminal():
            return None
        return max(state.children, key=lambda sq: self._minimax(state.children[sq], self.depth))

    # Question: TODO: Should compare be from P2's perspective or just the better move?
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
        return self._minimax(state, self.depth)

    def _minimax(self, state: TicTacToeState, depth: int | None) -> int:
        key = (state.board, state.player, depth)
        if key in self._cache:
            return self._cache[key]

        if state.is_terminal():
            result = state.value
        elif depth == 0:
            result = self._heuristic(state.board)
        elif state.player == 'P2':
            next_depth = None if depth is None else depth - 1
            result = max(self._minimax(child, next_depth) for child in state.children.values())
        else:
            next_depth = None if depth is None else depth - 1
            result = min(self._minimax(child, next_depth) for child in state.children.values())

        self._cache[key] = result
        return result

    @staticmethod
    def _heuristic(board: tuple) -> float:
        """
        Weighted open-lines heuristic from O's perspective.

        For each of the 8 lines, if it contains only O pieces (and empties),
        its contribution is (O_pieces_in_line / 3).  If it contains only X
        pieces (and empties), that same weight is subtracted.  Mixed lines
        (both X and O) are dead and contribute nothing.

        The raw score is normalized by the maximum possible value (8 * 1.0 = 8)
        so the result stays in (-1, 1), consistent with terminal values ±1.
        """
        o_score = 0.0
        x_score = 0.0

        for a, b, c in LINES:
            cells = (board[a], board[b], board[c])
            has_x = X in cells
            has_o = O in cells

            if has_o and not has_x:
                o_score += cells.count(O) / 3.0
            elif has_x and not has_o:
                x_score += cells.count(X) / 3.0

        max_score = len(LINES)   # 8 lines × max weight 1.0
        return (o_score - x_score) / max_score
