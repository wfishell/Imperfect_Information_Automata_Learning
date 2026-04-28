from __future__ import annotations
from src.game.hex.board import HexState


class HexNFA:

    def __init__(self, size: int = 3) -> None:
        self.root = HexState(size=size)

    # ------------------------------------------------------------------
    # Core navigation — used by MCTSEquivalenceOracle and GameSUL
    # ------------------------------------------------------------------

    def get_node(self, trace: list) -> HexState | None:
        """Replay trace from root; return resulting state or None if illegal."""
        state = self.root
        for action in trace:
            if state is None or state.is_terminal():
                return None
            if action not in state.children:
                return None
            state = state.children[action]
        return state

    # ------------------------------------------------------------------
    # Player-specific move queries — mirrors TicTacToeNFA interface
    # ------------------------------------------------------------------

    def p1_legal_inputs(self, trace: list) -> list:
        state = self.get_node(trace)
        if state is None or state.player != 'P1' or state.is_terminal():
            return []
        return list(state.children.keys())

    def p2_legal_moves(self, trace: list) -> list:
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
