from __future__ import annotations
from src.game.dots_and_boxes.board import DotsAndBoxesState, PASS


class DotsAndBoxesNFA:

    def __init__(self, rows: int = 2, cols: int = 2) -> None:
        self.root = DotsAndBoxesState(rows=rows, cols=cols)

    # ------------------------------------------------------------------
    # Core navigation — used by MCTSEquivalenceOracle and GameSUL
    # ------------------------------------------------------------------

    def get_node(self, trace: list) -> DotsAndBoxesState | None:
        """Replay trace from root, returning the resulting state or None if illegal.

        PASS is a real child key on forced-pass states, so it is navigated
        like any other action rather than treated as a no-op.
        """
        state = self.root
        for action in trace:
            if state is None or state.is_terminal():
                return None
            if action not in state.children:
                return None
            state = state.children[action]
        return state

    # ------------------------------------------------------------------
    # Player-specific move queries
    # ------------------------------------------------------------------

    def p1_legal_inputs(self, trace: list) -> list:
        """P1's legal inputs at the state reached by trace.

        Returns [PASS] at a forced-pass P1 state, the real edge indices at
        a normal P1 state, and [] when it is P2's turn or the game is over.
        """
        state = self.get_node(trace)
        if state is None or state.is_terminal() or state.player != 'P1':
            return []
        return list(state.children.keys())

    def p2_legal_moves(self, trace: list) -> list:
        """P2's legal moves at the state reached by trace.

        Returns [PASS] at a forced-pass P2 state, real edge indices at a
        normal P2 state, and [] when it is P1's turn or the game is over.
        """
        state = self.get_node(trace)
        if state is None or state.player != 'P2' or state.is_terminal():
            return []
        return list(state.children.keys())

    def is_terminal(self, trace: list) -> bool:
        state = self.get_node(trace)
        return state is not None and state.is_terminal()

    def current_player(self, trace: list) -> str | None:
        state = self.get_node(trace)
        if state is None or state.is_terminal():
            return None
        return state.player

    # ------------------------------------------------------------------
    # Full P1 alphabet — includes PASS for forced-pass states
    # ------------------------------------------------------------------

    @property
    def p1_alphabet(self) -> list:
        return list(self.root.children.keys()) + [PASS]
