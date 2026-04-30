from __future__ import annotations
from src.game.dots_and_boxes.board import DotsAndBoxesState

PASS = 'PASS'


class DotsAndBoxesNFA:

    def __init__(self, rows: int = 2, cols: int = 2) -> None:
        self.root = DotsAndBoxesState(rows=rows, cols=cols)

    # ------------------------------------------------------------------
    # Core navigation — used by MCTSEquivalenceOracle and GameSUL
    # ------------------------------------------------------------------

    def get_node(self, trace: list) -> DotsAndBoxesState | None:
        """Replay trace from root, returning the resulting state or None if illegal.
        PASS actions are no-ops — they advance the model turn without changing game state."""
        state = self.root
        for action in trace:
            if state is None or state.is_terminal():
                return None
            if action == PASS: # Pass only legal when a player has an extra turn.
                continue
            if action not in state.children: # Illegal Move
                return None
            state = state.children[action]
        return state
    
    # TODO: Known Issue - Pass only acts as a no-op. I want it to behave more like a valid state/transition such that we can reuse our standard SUL and MCTS Oracle.

    # ------------------------------------------------------------------
    # Player-specific move queries — mirrors TicTacToeNFA interface
    # ------------------------------------------------------------------

    def p1_legal_inputs(self, trace: list) -> list:
        state = self.get_node(trace)
        if state is None or state.is_terminal():
            return []
        if state.player == 'P2':
            return [PASS]   # P2 earned extra turn — P1's only legal input is PASS
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
