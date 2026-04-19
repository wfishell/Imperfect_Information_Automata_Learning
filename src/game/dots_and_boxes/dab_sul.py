"""
DotsAndBoxesSUL — SUL for Dots and Boxes with PASS mechanics.

Dots and Boxes breaks the strict P1→P2 alternation assumed by GameSUL:
a player earns an extra turn by completing a box.  We model this as:

  - P2 earns extra turn → P1's only legal input is PASS (no-op).
    P2 then responds with their extra move as normal.
  - P1 earns extra turn → P2's output for that step is PASS (no-op).
    P1 inputs their extra move on the next step call.

This keeps the one-to-one P1-input → P2-output contract that L* requires,
while faithfully encoding consecutive moves.

Internally we track _real_trace (only actual edge draws, no PASS) so that
oracle.preferred_move always receives a valid game trace.
"""

from __future__ import annotations
from aalpy.base import SUL

from src.game.dots_and_boxes.game_nfa import DotsAndBoxesNFA, PASS
from src.game.dots_and_boxes.preference_oracle import DotsAndBoxesOracle


class DotsAndBoxesSUL(SUL):

    def __init__(self, nfa: DotsAndBoxesNFA, oracle: DotsAndBoxesOracle) -> None:
        super().__init__()
        self.nfa    = nfa
        self.oracle = oracle

        self._state      = nfa.root   # current game state
        self._real_trace: list = []   # edge draws only — no PASS

        # trace_at_p2_node (tuple of real edges) → overridden P2 response
        self._overrides: dict[tuple, int] = {}

    # ------------------------------------------------------------------
    # AALpy SUL interface
    # ------------------------------------------------------------------

    def pre(self) -> None:
        self._state      = self.nfa.root
        self._real_trace = []

    def post(self) -> None:
        pass

    def step(self, p1_input) -> int | str | None:
        """
        Process one P1 input and return P2's response.

        p1_input == PASS : P2 earned an extra turn; P1 does nothing.
                           We skip advancing the game state and ask P2 to move.
        p1_input == edge : Normal P1 move.  Advance game state.
                           If P1 completed a box (still P1's turn), return PASS.
                           Otherwise get P2's preferred move and advance.
        """
        if p1_input == PASS:
            # P2 earned an extra turn — game state already has player=='P2'
            pass  # don't advance game state
        else:
            if p1_input not in self._state.children:
                return PASS
            self._state = self._state.children[p1_input]
            self._real_trace.append(p1_input)

        if self._state.is_terminal():
            return PASS

        if self._state.player == 'P1':
            # P1 completed a box and keeps their turn — P2 must pass
            return PASS

        # P2's turn — get preferred move (check overrides first)
        override_key = tuple(self._real_trace)
        if override_key in self._overrides:
            p2_response = self._overrides[override_key]
        else:
            p2_response = self.oracle.preferred_move(self._real_trace)

        if p2_response is None or p2_response not in self._state.children:
            return PASS

        self._state = self._state.children[p2_response]
        self._real_trace.append(p2_response)
        return p2_response

    # ------------------------------------------------------------------
    # Strategy overrides — called by MCTSEquivalenceOracle
    # ------------------------------------------------------------------

    def update_strategy(self, trace_at_p2_node: list, new_response) -> None:
        key = tuple(trace_at_p2_node)
        self._overrides[key] = new_response

    def current_strategy(self, trace_at_p2_node: list):
        key = tuple(trace_at_p2_node)
        if key in self._overrides:
            return self._overrides[key]
        return self.oracle.preferred_move(trace_at_p2_node)

    def p1_inputs_from_trace(self, full_trace: list) -> list:
        """Extract P1 inputs (even indices) from interleaved trace including PASS."""
        return [full_trace[i] for i in range(0, len(full_trace), 2)]
