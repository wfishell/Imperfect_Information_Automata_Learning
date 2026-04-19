from __future__ import annotations
from src.game.nim.board import NimState


class NimNFA:

    def __init__(self, piles: tuple = (1, 2, 3)) -> None:
        self.root     = NimState(piles=piles)
        self.alphabet = list(self.root.children.keys())

    # ------------------------------------------------------------------
    # Core navigation
    # ------------------------------------------------------------------

    def get_node(self, trace: list) -> NimState | None:
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
