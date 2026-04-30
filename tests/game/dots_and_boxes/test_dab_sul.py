"""
Integration tests for Dots and Boxes via GameSUL.

DotsAndBoxesSUL has been removed; GameSUL now works directly because the
forced-pass encoding in DotsAndBoxesNFA preserves strict P1/P2 alternation.

2×2 grid edge layout:
  box(0,0): edges 0, 2, 6, 7
  box(0,1): edges 1, 3, 7, 8
  box(1,0): edges 2, 4, 9, 10
  box(1,1): edges 3, 5, 10, 11

Interleaved trace contract: [P1_input, P2_output, P1_input, P2_output, ...]
  - P1 sends PASS when forced (P2 completed a box).
  - P2 output is PASS when forced (P1 completed a box).
"""

import pytest
from src.game.dots_and_boxes.game_nfa import DotsAndBoxesNFA, PASS
from src.game.dots_and_boxes.preference_oracle import DotsAndBoxesOracle
from src.lstar_mcts.game_sul import GameSUL
from src.lstar_mcts.table_b import TableB


@pytest.fixture
def sul():
    nfa    = DotsAndBoxesNFA()
    oracle = DotsAndBoxesOracle(nfa)
    return GameSUL(nfa=nfa, oracle=oracle, table_b=TableB())


def run_steps(sul: GameSUL, p1_inputs: list) -> list:
    sul.pre()
    return [sul.step(p1) for p1 in p1_inputs]


# ---------------------------------------------------------------------------
# pre / post — reset behaviour
# ---------------------------------------------------------------------------

class TestReset:
    def test_trace_clears_on_pre(self, sul):
        sul.pre()
        sul.step(0)
        sul.pre()
        assert sul._trace == []

    def test_pre_is_idempotent(self, sul):
        sul.pre()
        sul.pre()
        assert sul._trace == []

    def test_step_after_pre_is_consistent(self, sul):
        sul.pre()
        first = sul.step(0)
        sul.pre()
        assert sul.step(0) == first


# ---------------------------------------------------------------------------
# Normal alternation (no box completions)
# ---------------------------------------------------------------------------

class TestNormalAlternation:
    def test_first_step_returns_legal_edge(self, sul):
        sul.pre()
        p2 = sul.step(0)
        assert p2 != PASS
        assert p2 in set(range(12)) - {0}

    def test_p2_does_not_replay_p1_edge(self, sul):
        sul.pre()
        p2 = sul.step(0)
        assert p2 != 0

    def test_trace_grows_by_two_per_step(self, sul):
        sul.pre()
        sul.step(0)
        assert len(sul._trace) == 2
        p1_second = next(e for e in range(12) if e in sul.nfa.get_node(sul._trace).children)
        sul.step(p1_second)
        assert len(sul._trace) == 4


# ---------------------------------------------------------------------------
# P2 earns extra turn (P2 completes a box)
# ---------------------------------------------------------------------------

class TestP2ExtraTurn:
    """
    Force P2 to play edges 0,2,6,7 via strategy overrides so P2 completes
    box(0,0).  After that step, P1's input is PASS and the SUL should return
    P2's extra-turn move (a real edge, not PASS).
    """

    def _drive_to_p2_extra(self, sul) -> int:
        """Drive SUL to state where P2 just completed box(0,0).
        Returns P2's completing-move output (should be 7)."""
        sul.pre()
        # Force P2 to play 2 after P1 plays 0
        sul.update_strategy([0], 2)
        sul.step(0)        # P1:0 → P2:2 (override).  trace=[0,2]
        return sul.step(6) # P1:6 → P2:7 (oracle).    trace=[0,2,6,7]

    def test_completing_step_returns_completing_edge(self, sul):
        assert self._drive_to_p2_extra(sul) == 7

    def test_p1_pass_returns_real_edge(self, sul):
        self._drive_to_p2_extra(sul)
        extra = sul.step(PASS)   # P1 forced pass; P2 takes extra turn
        assert extra != PASS
        assert extra in set(range(12))

    def test_extra_move_not_already_drawn(self, sul):
        self._drive_to_p2_extra(sul)
        extra = sul.step(PASS)
        assert extra not in {0, 2, 6, 7}

    def test_trace_contains_pass(self, sul):
        self._drive_to_p2_extra(sul)
        sul.step(PASS)
        assert PASS in sul._trace


# ---------------------------------------------------------------------------
# P1 earns extra turn (P1 completes a box)
# ---------------------------------------------------------------------------

class TestP1ExtraTurn:
    """
    Force P2 to play neutral edges so that P1 can complete box(0,0).
    When P1 draws the completing edge, step() must return PASS (P2 is
    forced to pass so P1 gets another turn).
    """

    def _drive_to_p1_extra(self, sul) -> object:
        """Drive SUL to state where P1 just completed box(0,0).
        Returns the output of the completing step (should be PASS)."""
        sul.pre()
        sul.update_strategy([0],             4)
        sul.update_strategy([0, 4, 2],       5)
        sul.update_strategy([0, 4, 2, 5, 6], 8)
        sul.step(0)   # P1:0 → P2:4
        sul.step(2)   # P1:2 → P2:5
        sul.step(6)   # P1:6 → P2:8
        return sul.step(7)   # P1:7 completes box(0,0)

    def test_completing_step_returns_pass(self, sul):
        assert self._drive_to_p1_extra(sul) == PASS

    def test_after_pass_p1_can_move_again(self, sul):
        self._drive_to_p1_extra(sul)
        # Now P1 sends their extra-turn move; SUL should give a real P2 response
        state = sul.nfa.get_node(sul._trace)
        assert state.forced_pass is False or state.player == 'P1'
        # Pick any undrawn edge that is P1's turn
        extra_p1 = next(e for e in range(12) if e not in {0, 2, 4, 5, 6, 7, 8})
        result = sul.step(extra_p1)
        assert result != PASS or sul.nfa.get_node(sul._trace).is_terminal()


# ---------------------------------------------------------------------------
# Strategy overrides
# ---------------------------------------------------------------------------

class TestStrategyOverrides:
    def test_update_strategy_registers_override(self, sul):
        sul.update_strategy([0], 5)
        assert sul._overrides[(0,)] == 5

    def test_step_uses_override(self, sul):
        sul.pre()
        sul.update_strategy([0], 3)
        result = sul.step(0)
        assert result == 3

    def test_overrides_persist_across_pre(self, sul):
        sul.update_strategy([0], 5)
        sul.pre()
        assert (0,) in sul._overrides
        assert sul._overrides[(0,)] == 5

    def test_current_strategy_returns_override(self, sul):
        sul.update_strategy([0], 5)
        assert sul.current_strategy([0]) == 5

    def test_current_strategy_falls_back_to_oracle(self, sul):
        # No override; oracle with 3 sides of box(0,0) drawn picks 7
        assert sul.current_strategy([0, 2, 6]) == 7


# ---------------------------------------------------------------------------
# p1_inputs_from_trace
# ---------------------------------------------------------------------------

class TestP1InputsFromTrace:
    def test_empty_trace_returns_empty(self, sul):
        assert sul.p1_inputs_from_trace([]) == []

    def test_single_step(self, sul):
        assert sul.p1_inputs_from_trace([0, 3]) == [0]

    def test_multiple_steps(self, sul):
        assert sul.p1_inputs_from_trace([0, 3, 1, 5]) == [0, 1]

    def test_pass_at_even_index_is_included(self, sul):
        assert sul.p1_inputs_from_trace([PASS, 3]) == [PASS]
