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
    pass


# ---------------------------------------------------------------------------
# step — PASS mechanic when P2 completes a box
# ---------------------------------------------------------------------------

class TestStepP2ExtraTurn:
    pass


# ---------------------------------------------------------------------------
# step — PASS output when P1 completes a box
# ---------------------------------------------------------------------------

class TestStepP1ExtraTurn:
    pass


# ---------------------------------------------------------------------------
# step — terminal state handling
# ---------------------------------------------------------------------------

class TestStepTerminal:
    pass


# ---------------------------------------------------------------------------
# strategy overrides
# ---------------------------------------------------------------------------

class TestStrategyOverrides:
    pass


# ---------------------------------------------------------------------------
# p1_inputs_from_trace
# ---------------------------------------------------------------------------

class TestP1InputsFromTrace:
    pass
