"""
Custom SUL (System Under Learning) for the minimax game.

Input alphabet  : P1's moves  (e.g. 'A', 'B')
Output alphabet : P2's moves  (e.g. 'X', 'Y')

Each step() receives one P1 input and returns P2's preferred response given
the full trace history seen so far.  The SUL tracks the interleaved trace
(P1_move, P2_move, P1_move, P2_move, ...) so the history-conditioned oracle
always has the full context.

Strategy overrides: when MCTS finds a better P2 response at some prefix, it
calls update_strategy() to register the improvement.  Subsequent membership
queries at that prefix will return the improved response instead of the
greedy oracle answer.  This is what creates the discrepancy between the
old hypothesis and the new SUL that AALpy uses as a counterexample.
"""

from aalpy.base import SUL
from src.game.minimax.game_nfa import GameNFA
from src.lstar_mcts.preference_oracle import PreferenceOracle


class GameSUL(SUL):

    def __init__(self, nfa: GameNFA, oracle: PreferenceOracle) -> None:
        super().__init__()
        self.nfa    = nfa
        self.oracle = oracle

        # Full interleaved trace for current query: [P1, P2, P1, P2, ...]
        self._trace: list[str] = []

        # prefix (after P1 input, i.e. at a P2 node) → override P2 response
        # key   = tuple of the trace UP TO AND INCLUDING the P1 input
        # value = P2 action that MCTS found to be better
        self._overrides: dict[tuple, str] = {}

        # Membership query cache: tuple(p1_prefix) → p2_response_at_that_step
        # Key = P1 inputs seen so far at each step; value = P2 response.
        # Cleared when a strategy override changes the SUL's answers.
        self._cache: dict[tuple, str] = {}
        self._current_p1: list[str] = []

    # ------------------------------------------------------------------
    # AALpy SUL interface
    # ------------------------------------------------------------------

    def pre(self) -> None:
        """Reset trace at the start of each membership query."""
        self._trace = []
        self._current_p1 = []

    def post(self) -> None:
        pass

    def step(self, p1_input: str) -> str | None:
        """
        Process one P1 input and return P2's response.

        Cache key = tuple of P1 inputs seen so far (including this one).
        This lets lookups work incrementally — each step checks only the
        prefix it has accumulated, matching how the cache was populated.
        """
        self._current_p1.append(p1_input)
        cache_key = tuple(self._current_p1)

        if cache_key in self._cache:
            p2_response = self._cache[cache_key]
            self._trace.append(p1_input)
            self._trace.append(p2_response)
            return p2_response

        self._trace.append(p1_input)

        p2_moves = self.nfa.p2_legal_moves(self._trace)
        if not p2_moves:
            return None

        interleaved_key = tuple(self._trace)
        if interleaved_key in self._overrides:
            p2_response = self._overrides[interleaved_key]
        else:
            p2_response = self.oracle.preferred_move(self._trace)

        self._trace.append(p2_response)
        self._cache[cache_key] = p2_response
        return p2_response

    # ------------------------------------------------------------------
    # Strategy update (called by MCTS oracle)
    # ------------------------------------------------------------------

    def update_strategy(self, trace_at_p2_node: list[str], new_response: str) -> None:
        """
        Override the P2 response at the state reached by trace_at_p2_node.
        Clears the membership query cache since SUL answers have changed.
        """
        key = tuple(trace_at_p2_node)
        if self._overrides.get(key) != new_response:
            self._overrides[key] = new_response
            self._cache.clear()   # invalidate: answers may have changed

    def current_strategy(self, trace_at_p2_node: list[str]) -> str | None:
        """Return whatever P2 would do at this state under the current strategy."""
        key = tuple(trace_at_p2_node)
        if key in self._overrides:
            return self._overrides[key]
        return self.oracle.preferred_move(trace_at_p2_node)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def p1_inputs_from_trace(self, full_trace: list[str]) -> list[str]:
        """Extract just P1's inputs from an interleaved trace."""
        return [full_trace[i] for i in range(0, len(full_trace), 2)]
