"""
Tests for DotsAndBoxesOracle (src/game/dots_and_boxes/preference_oracle.py)
Run: python -m pytest tests/game/dots_and_boxes/test_preference_oracle.py -v

2×2 grid edge layout:
  box(0,0): edges 0, 2, 6, 7
  box(0,1): edges 1, 3, 7, 8
  box(1,0): edges 2, 4, 9, 10
  box(1,1): edges 3, 5, 10, 11
"""

import pytest
from src.game.dots_and_boxes.game_nfa import DotsAndBoxesNFA, PASS
from src.game.dots_and_boxes.preference_oracle import DotsAndBoxesOracle
from src.game.dots_and_boxes.board import DotsAndBoxesState


@pytest.fixture
def oracle():
    return DotsAndBoxesOracle(DotsAndBoxesNFA())


@pytest.fixture
def oracle_depth1():
    return DotsAndBoxesOracle(DotsAndBoxesNFA(), depth=1)


def make_terminal_trace() -> list:
    """Return a trace that plays all 12 edges and reaches a terminal state.
    Handles forced-pass states by inserting PASS into the trace."""
    nfa = DotsAndBoxesNFA()
    state = nfa.root
    remaining = list(range(12))
    trace = []
    while not state.is_terminal():
        if state.forced_pass:
            trace.append(PASS)
            state = state.children[PASS]
            continue
        e = next(e for e in remaining if e in state.children)
        state = state.children[e]
        trace.append(e)
        remaining.remove(e)
    return trace


# ---------------------------------------------------------------------------
# preferred_move — take a box immediately
# ---------------------------------------------------------------------------

class TestPreferredMove:
    def test_takes_box_immediately(self, oracle):
        # box(0,0) has 3 sides drawn: edges 0, 2, 6 → P2 must play 7 to claim it
        # Sequence: P1:0, P2:2, P1:6 → P2 to move
        assert oracle.preferred_move([0, 2, 6]) == 7

    def test_returns_legal_move(self, oracle):
        move = oracle.preferred_move([0])   # P2 to move, many options
        assert move in set(range(12)) - {0}

    def test_takes_a_completing_move(self, oracle):
        # box(0,0) needs edge 7, box(1,0) needs edge 4 — both one step away
        # Sequence: P1:0, P2:9, P1:2, P2:10, P1:6 → P2 to move
        assert oracle.preferred_move([0, 9, 2, 10, 6]) in {4, 7}


# ---------------------------------------------------------------------------
# preferred_move — boundary conditions
# ---------------------------------------------------------------------------

class TestPreferredMoveBoundary:
    def test_returns_none_on_p1_turn(self, oracle):
        assert oracle.preferred_move([]) is None

    def test_returns_none_on_terminal(self, oracle):
        assert oracle.preferred_move(make_terminal_trace()) is None

    def test_returns_none_on_invalid_trace(self, oracle):
        assert oracle.preferred_move([0, 0]) is None   # illegal repeated edge


# ---------------------------------------------------------------------------
# minimax values
# ---------------------------------------------------------------------------

class TestMinimax:
    def test_p2_win_beats_draw(self, oracle):
        all_drawn = (True,) * 12
        p2_win = DotsAndBoxesState(edges=all_drawn, p1_boxes=1, p2_boxes=3)
        draw   = DotsAndBoxesState(edges=all_drawn, p1_boxes=2, p2_boxes=2)
        assert oracle._minimax(p2_win, None) == 1.0
        assert oracle._minimax(draw,   None) == 0.0
        assert oracle._minimax(p2_win, None) > oracle._minimax(draw, None)

    def test_p2_win_beats_p1_win(self, oracle):
        all_drawn = (True,) * 12
        p2_win = DotsAndBoxesState(edges=all_drawn, p1_boxes=1, p2_boxes=3)
        p1_win = DotsAndBoxesState(edges=all_drawn, p1_boxes=3, p2_boxes=1)
        assert oracle._minimax(p2_win, None) > oracle._minimax(p1_win, None)

    def test_equal_draws(self, oracle):
        draw = DotsAndBoxesState(edges=(True,) * 12, p1_boxes=2, p2_boxes=2)
        assert oracle._minimax(draw, None) == oracle._minimax(draw, None)


# ---------------------------------------------------------------------------
# heuristic
# ---------------------------------------------------------------------------

class TestHeuristic:
    def test_zero_on_empty_board(self):
        assert DotsAndBoxesOracle._heuristic(DotsAndBoxesState()) == 0.0

    def test_positive_when_p2_leads(self):
        assert DotsAndBoxesOracle._heuristic(DotsAndBoxesState(p2_boxes=2, p1_boxes=0)) > 0.0

    def test_negative_when_p1_leads(self):
        assert DotsAndBoxesOracle._heuristic(DotsAndBoxesState(p2_boxes=0, p1_boxes=2)) < 0.0

    def test_range(self):
        s_max = DotsAndBoxesState(p2_boxes=4, p1_boxes=0)
        s_min = DotsAndBoxesState(p2_boxes=0, p1_boxes=4)
        assert DotsAndBoxesOracle._heuristic(s_max) == pytest.approx(1.0)
        assert DotsAndBoxesOracle._heuristic(s_min) == pytest.approx(-1.0)

    def test_normalized_by_total_boxes(self):
        s = DotsAndBoxesState(p2_boxes=1, p1_boxes=0)   # 2×2 → total=4
        assert DotsAndBoxesOracle._heuristic(s) == pytest.approx(1 / 4)


# ---------------------------------------------------------------------------
# Bounded depth — depth=1 uses heuristic at cutoff
# ---------------------------------------------------------------------------

class TestBoundedDepth:
    def test_depth1_distinguishes_box_leads(self, oracle_depth1):
        s_p2_ahead = DotsAndBoxesState(p2_boxes=2, p1_boxes=0)
        s_equal    = DotsAndBoxesState(p2_boxes=1, p1_boxes=1)
        assert oracle_depth1._minimax(s_p2_ahead, 1) >= oracle_depth1._minimax(s_equal, 1)

    def test_depth_none_matches_full_minimax(self, oracle):
        assert oracle.preferred_move([0, 2, 6]) == 7
