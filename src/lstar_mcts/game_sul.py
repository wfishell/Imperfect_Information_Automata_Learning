"""
Custom SUL (System Under Learning)

Input alphabet  : P1's moves
Output alphabet : P2's moves

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
from src.lstar_mcts.table_b import TableB


class GameSUL(SUL):

    def __init__(self, nfa, oracle, table_b: TableB) -> None:
        super().__init__()
        self.nfa    = nfa
        self.oracle = oracle
        self.table_b=table_b

        # Full interleaved trace for current query: [P1, P2, P1, P2, ...]
        self._trace: list[str] = []

        # prefix (after P1 input, i.e. at a P2 node) → override P2 response
        # key   = tuple of the trace UP TO AND INCLUDING the P1 input
        # value = P2 action that MCTS found to be better
        self._overrides: dict[tuple, str] = {}

        # Spec-derived overrides — set by .patch() and protected from
        # further update_strategy calls. Any prefix in this set is
        # immutable: the SafetyEqOracle has determined the correct
        # response from a verified safety property.
        self._spec_locked: set[tuple] = set()

        # Spec-derived STATE-keyed overrides — set by .patch_state().
        # Keyed by an env-state hash (provided by `state_key_fn`), so
        # any input sequence that ends up at that env state gets the
        # patched answer regardless of how it got there. This is what
        # makes "patch the boundary states once" actually work — without
        # it, every distinct prefix to the same boundary state would
        # need its own patch.
        self._state_overrides: dict = {}
        self._spec_locked_states: set = set()
        # Optional callable: env_state -> hashable key. SafetyEqOracle
        # sets this at construction time. If None, state-keyed overrides
        # are bypassed (only prefix-keyed _overrides apply).
        self.state_key_fn = None

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

        # State-keyed overrides take priority — these are spec-derived
        # patches that should fire at a particular env state regardless
        # of how the input sequence got there.
        p2_response = None
        if self._state_overrides and self.state_key_fn is not None:
            env_state = self.nfa.get_node(self._trace)
            if env_state is not None:
                env_key = self.state_key_fn(env_state)
                if env_key in self._state_overrides:
                    p2_response = self._state_overrides[env_key]

        if p2_response is None:
            interleaved_key = tuple(self._trace)
            if interleaved_key in self._overrides:
                p2_response = self._overrides[interleaved_key]
            else:
                p2_response = self.oracle.preferred_move(self._trace)

        for move in p2_moves:
            if move == p2_response:
                self.table_b.record_visit(self._trace, move)
            else:
                self.table_b._get(self._trace,move)
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

        Spec-locked prefixes (set via `patch`) AND spec-locked env states
        (set via `patch_state`) cannot be overwritten — a verified safety
        property always wins over MCTS preference updates. The latter
        check prevents MCTS from "improving" preferences by routing the
        trajectory around safety boundaries.
        """
        key = tuple(trace_at_p2_node)

        # Prefix-keyed lock.
        if key in self._spec_locked:
            return

        # State-keyed lock — if the env state at this prefix is in the
        # safety-locked set, MCTS may not overwrite the SUL's answer
        # there or at any prefix that lands at the same state.
        if self._spec_locked_states and self.state_key_fn is not None:
            env_state = self.nfa.get_node(trace_at_p2_node)
            if env_state is not None:
                env_key = self.state_key_fn(env_state)
                if env_key in self._spec_locked_states:
                    return

        if self._overrides.get(key) != new_response:
            self._overrides[key] = new_response
            self._cache.clear()

    def patch_state(self, state_key, safe_response: str) -> None:
        """
        Spec-derived STATE-keyed override. Fires at a particular env
        state regardless of which input sequence got there. Requires
        `self.state_key_fn` to have been set (so `step()` can compute
        the key for the current trace).

        Used by SafetyEqOracle: enumerate the few boundary states once,
        install one patch per state. Any prefix L* explores that ends
        up at a boundary state will receive the patched answer.
        """
        self._state_overrides[state_key] = safe_response
        self._spec_locked_states.add(state_key)
        self._cache.clear()

    def patch(self, trace_at_p2_node: list[str], safe_response: str) -> None:
        """
        Spec-derived override at the state reached by trace_at_p2_node.
        Sets the response AND marks the prefix as immutable so subsequent
        oracle calls and MCTS strategy-updates cannot overwrite it.

        Called by SafetyEqOracle when the model-checker finds a state
        where the controller's emitted action would violate G(gas > 0).
        """
        key = tuple(trace_at_p2_node)
        self._overrides[key] = safe_response
        self._spec_locked.add(key)
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
