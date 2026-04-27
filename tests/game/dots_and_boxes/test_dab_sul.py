"""
Tests for DotsAndBoxesSUL (src/game/dots_and_boxes/dab_sul.py)
Run: python -m pytest tests/game/dots_and_boxes/test_dab_sul.py -v

2×2 grid edge layout:
  box(0,0): edges 0, 2, 6, 7
  box(0,1): edges 1, 3, 7, 8
  box(1,0): edges 2, 4, 9, 10
  box(1,1): edges 3, 5, 10, 11

PASS mechanic:
  P2 earns extra turn → P1 inputs PASS, P2 responds with their extra move.
  P1 earns extra turn → P2's output for that step is PASS.
"""

import pytest
from src.game.dots_and_boxes.board import DotsAndBoxesState
from src.game.dots_and_boxes.game_nfa import DotsAndBoxesNFA, PASS
from src.game.dots_and_boxes.preference_oracle import DotsAndBoxesOracle
from src.game.dots_and_boxes.dab_sul import DotsAndBoxesSUL


@pytest.fixture
def sul():
    nfa    = DotsAndBoxesNFA()
    oracle = DotsAndBoxesOracle(nfa)
    return DotsAndBoxesSUL(nfa, oracle)


def run_steps(sul: DotsAndBoxesSUL, p1_inputs: list) -> list:
    """Helper: call pre(), step through p1_inputs, return list of P2 outputs."""
    sul.pre()
    return [sul.step(p1) for p1 in p1_inputs]


# ---------------------------------------------------------------------------
# pre / post — reset behaviour
# ---------------------------------------------------------------------------

class TestReset:
    def test_state_resets_to_root(self, sul):
        sul.pre()
        sul.step(0)
        sul.pre()
        assert sul._state is sul.nfa.root

    def test_real_trace_clears(self, sul):
        sul.pre()
        sul.step(0)
        sul.pre()
        assert sul._real_trace == []

    def test_pre_is_idempotent(self, sul):
        sul.pre()
        sul.pre()
        assert sul._state is sul.nfa.root
        assert sul._real_trace == []

    def test_step_after_pre_matches_fresh_sul(self, sul):
        # Run a query, reset, then confirm first step gives the same output.
        sul.pre()
        first_output = sul.step(0)
        sul.pre()
        assert sul.step(0) == first_output


# ---------------------------------------------------------------------------
# step — normal alternation (no box completions)
# ---------------------------------------------------------------------------

class TestStepNormalAlternation:
    def test_first_step_returns_legal_edge(self, sul):
        # P1 plays edge 0 (no box completion possible on first move)
        sul.pre()
        p2 = sul.step(0)
        assert p2 != PASS
        assert p2 in set(range(12)) - {0}

    def test_real_trace_has_both_edges_after_one_step(self, sul):
        sul.pre()
        p2 = sul.step(0)
        assert sul._real_trace[0] == 0
        assert sul._real_trace[1] == p2
        assert len(sul._real_trace) == 2

    def test_state_is_back_to_p1_turn_after_step(self, sul):
        # P2 responds without completing a box → turn returns to P1
        sul.pre()
        sul.step(0)
        assert sul._state.player == 'P1'

    def test_p2_does_not_replay_p1_edge(self, sul):
        sul.pre()
        p2 = sul.step(0)
        assert p2 != 0

    def test_trace_grows_by_two_per_step(self, sul):
        sul.pre()
        sul.step(0)
        assert len(sul._real_trace) == 2
        # pick a still-available edge rather than hardcoding, since P2
        # may have already claimed edge 1 in its response to step 0
        p1_second = next(e for e in range(12) if e in sul._state.children)
        sul.step(p1_second)
        assert len(sul._real_trace) == 4

    def test_state_is_not_root_after_step(self, sul):
        sul.pre()
        sul.step(0)
        assert sul._state is not sul.nfa.root


# ---------------------------------------------------------------------------
# step — PASS mechanic when P2 completes a box
# ---------------------------------------------------------------------------

class TestStepP2ExtraTurn:
    """
    box(0,0) borders: 0, 2, 6, 7.
    We force P2 to play 2 (via override) after P1 plays 0, giving _real_trace=[0,2].
    P1 then plays 6; the oracle is guaranteed to play 7 (completing box(0,0)),
    confirmed by test_preference_oracle::test_takes_box_immediately.
    After that step, state.player=='P2' — P2 earned an extra turn.
    """

    def _drive_to_p2_extra_turn(self, sul) -> int:
        """Set up the SUL so P2 just completed box(0,0) and holds the turn.
        Returns the output of the completing step (should be 7)."""
        sul.pre()
        sul._overrides[(0,)] = 2          # force P2 to play 2 after P1 plays 0
        sul.step(0)                        # P1:0 → P2:2 (override). _real_trace=[0,2]
        return sul.step(6)                 # P1:6 → P2:7 (oracle). _real_trace=[0,2,6,7]

    def test_completing_step_returns_completing_edge(self, sul):
        assert self._drive_to_p2_extra_turn(sul) == 7

    def test_state_player_is_p2_after_completion(self, sul):
        self._drive_to_p2_extra_turn(sul)
        assert sul._state.player == 'P2'

    def test_pass_input_returns_legal_edge(self, sul):
        self._drive_to_p2_extra_turn(sul)
        extra = sul.step(PASS)
        assert extra != PASS
        assert extra in set(range(12))

    def test_extra_move_appended_to_real_trace(self, sul):
        self._drive_to_p2_extra_turn(sul)
        sul.step(PASS)
        # _real_trace should now contain [0, 2, 6, 7, <extra_move>]
        assert len(sul._real_trace) == 5

    def test_extra_move_not_already_drawn(self, sul):
        self._drive_to_p2_extra_turn(sul)
        extra = sul.step(PASS)
        assert extra not in {0, 2, 6, 7}  # all four already drawn


# ---------------------------------------------------------------------------
# step — PASS output when P1 completes a box
# ---------------------------------------------------------------------------

class TestStepP1ExtraTurn:
    """
    box(0,0) borders: 0, 2, 6, 7.
    P2 is forced via overrides to play neutral edges (4, 5, 8) so that
    edges 0, 2, 6 accumulate as P1 moves. P1 then plays 7, completing
    box(0,0). step() must return PASS immediately — P1 keeps their turn
    and P2 gets no response for that step.
    """

    def _drive_to_p1_extra_turn(self, sul) -> int | str:
        """Drive the SUL to a state where P1 just completed box(0,0).
        Returns the output of the completing step (should be PASS)."""
        sul.pre()
        sul._overrides[(0,)]            = 4
        sul._overrides[(0, 4, 2)]       = 5
        sul._overrides[(0, 4, 2, 5, 6)] = 8
        sul.step(0)   # P1:0 → P2:4.  _real_trace=[0,4]
        sul.step(2)   # P1:2 → P2:5.  _real_trace=[0,4,2,5]
        sul.step(6)   # P1:6 → P2:8.  _real_trace=[0,4,2,5,6,8]
        return sul.step(7)  # P1:7 completes box(0,0)

    def test_completing_step_returns_pass(self, sul):
        assert self._drive_to_p1_extra_turn(sul) == PASS

    def test_state_player_is_p1_after_completion(self, sul):
        self._drive_to_p1_extra_turn(sul)
        assert sul._state.player == 'P1'

    def test_p1_score_incremented(self, sul):
        self._drive_to_p1_extra_turn(sul)
        assert sul._state.p1_boxes == 1

    def test_p1_completing_edge_in_real_trace(self, sul):
        self._drive_to_p1_extra_turn(sul)
        assert 7 in sul._real_trace

    def test_no_p2_response_appended_after_completion(self, sul):
        # P2 played 3 responses (edges 4, 5, 8) and P1 played 4 edges (0,2,6,7)
        # step() returns PASS before asking P2, so trace length is 7 not 8
        self._drive_to_p1_extra_turn(sul)
        assert len(sul._real_trace) == 7


# ---------------------------------------------------------------------------
# step — terminal state handling
# ---------------------------------------------------------------------------

class TestStepTerminal:
    def test_step_on_already_terminal_returns_pass(self, sul):
        # Inject a terminal state directly — all edges drawn, no children
        sul.pre()
        sul._state = DotsAndBoxesState(edges=(True,) * 12, p1_boxes=2, p2_boxes=2)
        assert sul.step(0) == PASS

    def test_step_does_not_advance_state_on_terminal(self, sul):
        sul.pre()
        terminal = DotsAndBoxesState(edges=(True,) * 12, p1_boxes=2, p2_boxes=2)
        sul._state = terminal
        sul.step(0)
        assert sul._state is terminal

    def test_step_does_not_modify_real_trace_on_terminal(self, sul):
        sul.pre()
        sul._state = DotsAndBoxesState(edges=(True,) * 12, p1_boxes=2, p2_boxes=2)
        sul.step(0)
        assert sul._real_trace == []

    def test_last_edge_makes_state_terminal(self, sul):
        # Inject a state with only edge 11 remaining — P1 draws it
        sul.pre()
        sul._state = DotsAndBoxesState(
            edges=(True,) * 11 + (False,), p1_boxes=2, p2_boxes=1, player='P1'
        )
        result = sul.step(11)
        assert result == PASS
        assert sul._state.is_terminal()


# ---------------------------------------------------------------------------
# strategy overrides
# ---------------------------------------------------------------------------

class TestStrategyOverrides:
    def test_update_strategy_registers_override(self, sul):
        sul.update_strategy([0], 5)
        assert sul._overrides[(0,)] == 5

    def test_update_strategy_overwrites_existing(self, sul):
        sul.update_strategy([0], 5)
        sul.update_strategy([0], 3)
        assert sul._overrides[(0,)] == 3

    def test_current_strategy_returns_override_when_set(self, sul):
        sul.update_strategy([0], 5)
        assert sul.current_strategy([0]) == 5

    def test_current_strategy_falls_back_to_oracle(self, sul):
        # No override at [0,2,6] — oracle is guaranteed to return 7
        assert sul.current_strategy([0, 2, 6]) == 7

    def test_step_uses_override(self, sul):
        # Force P2 to play edge 3 after P1 plays edge 0
        sul.pre()
        sul.update_strategy([0], 3)
        result = sul.step(0)
        assert result == 3

    def test_overrides_persist_across_pre(self, sul):
        # pre() resets game state but must NOT clear overrides —
        # MCTS patches are durable across membership queries
        sul.update_strategy([0], 5)
        sul.pre()
        assert (0,) in sul._overrides
        assert sul._overrides[(0,)] == 5


# ---------------------------------------------------------------------------
# p1_inputs_from_trace
# ---------------------------------------------------------------------------

class TestP1InputsFromTrace:
    pass
