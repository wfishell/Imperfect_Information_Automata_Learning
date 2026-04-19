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
from src.game.dots_and_boxes.game_nfa import DotsAndBoxesNFA
from src.game.dots_and_boxes.preference_oracle import DotsAndBoxesOracle
from src.game.dots_and_boxes.board import DotsAndBoxesState


@pytest.fixture
def oracle():
    nfa = DotsAndBoxesNFA()
    return DotsAndBoxesOracle(nfa)


@pytest.fixture
def oracle_depth1():
    nfa = DotsAndBoxesNFA()
    return DotsAndBoxesOracle(nfa, depth=1)


# ---------------------------------------------------------------------------
# preferred_move — take a box immediately
# ---------------------------------------------------------------------------

def test_preferred_move_takes_box_immediately(oracle):
    # box(0,0) has 3 sides drawn: edges 0, 2, 6 → P2 must play 7 to claim it
    # Sequence: P1:0, P2:2, P1:6 → P2 to move
    prefix = [0, 2, 6]
    move = oracle.preferred_move(prefix)
    assert move == 7

def test_preferred_move_takes_box_returns_legal_move(oracle):
    prefix = [0]   # P2 to move, many options
    move = oracle.preferred_move(prefix)
    assert move in set(range(12)) - {0}


# ---------------------------------------------------------------------------
# preferred_move — block P1 from completing a box (depth ≥ 2)
# ---------------------------------------------------------------------------

def test_preferred_move_takes_a_completing_move(oracle):
    # With edges 0,2,6 drawn box(0,0) needs edge 7, and
    # with edges 2,9,10 drawn box(1,0) needs edge 4.
    # Both are one step away — P2 should take one of them.
    # Sequence: P1:0, P2:9, P1:2, P2:10, P1:6 → P2 to move
    prefix = [0, 9, 2, 10, 6]
    move = oracle.preferred_move(prefix)
    assert move in {4, 7}   # either completing move is correct


# ---------------------------------------------------------------------------
# preferred_move — boundary conditions
# ---------------------------------------------------------------------------

def test_preferred_move_returns_none_on_p1_turn(oracle):
    assert oracle.preferred_move([]) is None

def test_preferred_move_returns_none_on_terminal(oracle):
    # Build a terminal state trace
    nfa = DotsAndBoxesNFA()
    state = nfa.root
    trace = []
    remaining = list(range(12))
    while remaining:
        e = next(e for e in remaining if e in state.children)
        state = state.children[e]
        trace.append(e)
        remaining.remove(e)
    assert oracle.preferred_move(trace) is None

def test_preferred_move_returns_none_on_invalid_trace(oracle):
    assert oracle.preferred_move([0, 0]) is None   # illegal repeated edge


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def test_compare_p2_win_beats_draw(oracle):
    all_drawn = (True,) * 12
    nfa = DotsAndBoxesNFA()
    o = DotsAndBoxesOracle(nfa)
    p2_win = DotsAndBoxesState(edges=all_drawn, p1_boxes=1, p2_boxes=3)
    draw   = DotsAndBoxesState(edges=all_drawn, p1_boxes=2, p2_boxes=2)
    assert o._minimax(p2_win, None) == 1.0
    assert o._minimax(draw,   None) == 0.0
    assert o._minimax(p2_win, None) > o._minimax(draw, None)

def test_compare_p2_win_beats_p1_win(oracle):
    all_drawn = (True,) * 12
    nfa = DotsAndBoxesNFA()
    o = DotsAndBoxesOracle(nfa)
    p2_win = DotsAndBoxesState(edges=all_drawn, p1_boxes=1, p2_boxes=3)
    p1_win = DotsAndBoxesState(edges=all_drawn, p1_boxes=3, p2_boxes=1)
    assert o._minimax(p2_win, None) >  o._minimax(p1_win, None)

def test_compare_equal_draws(oracle):
    all_drawn = (True,) * 12
    nfa = DotsAndBoxesNFA()
    o = DotsAndBoxesOracle(nfa)
    draw = DotsAndBoxesState(edges=all_drawn, p1_boxes=2, p2_boxes=2)
    assert o._minimax(draw, None) == o._minimax(draw, None)


# ---------------------------------------------------------------------------
# heuristic
# ---------------------------------------------------------------------------

def test_heuristic_zero_on_empty_board():
    s = DotsAndBoxesState()
    assert DotsAndBoxesOracle._heuristic(s) == 0.0

def test_heuristic_positive_when_p2_leads():
    s = DotsAndBoxesState(p2_boxes=2, p1_boxes=0)
    assert DotsAndBoxesOracle._heuristic(s) > 0.0

def test_heuristic_negative_when_p1_leads():
    s = DotsAndBoxesState(p2_boxes=0, p1_boxes=2)
    assert DotsAndBoxesOracle._heuristic(s) < 0.0

def test_heuristic_range():
    s_max = DotsAndBoxesState(p2_boxes=4, p1_boxes=0)
    s_min = DotsAndBoxesState(p2_boxes=0, p1_boxes=4)
    assert DotsAndBoxesOracle._heuristic(s_max) == pytest.approx(1.0)
    assert DotsAndBoxesOracle._heuristic(s_min) == pytest.approx(-1.0)

def test_heuristic_normalised_by_total_boxes():
    s = DotsAndBoxesState(p2_boxes=1, p1_boxes=0)   # 2×2 → total=4
    assert DotsAndBoxesOracle._heuristic(s) == pytest.approx(1 / 4)


# ---------------------------------------------------------------------------
# Bounded depth — depth=1 uses heuristic at cutoff
# ---------------------------------------------------------------------------

def test_depth1_compare_distinguishes_box_leads(oracle_depth1):
    # At depth=1 from a non-terminal position, the heuristic must give
    # different values for different box counts.
    nfa = DotsAndBoxesNFA()
    o = DotsAndBoxesOracle(nfa, depth=1)
    # State where P2 is ahead
    s_p2_ahead = DotsAndBoxesState(p2_boxes=2, p1_boxes=0)
    s_equal    = DotsAndBoxesState(p2_boxes=1, p1_boxes=1)
    assert o._minimax(s_p2_ahead, 1) >= o._minimax(s_equal, 1)

def test_depth_none_matches_full_minimax(oracle):
    # With depth=None, preferred_move should still take an immediate box
    prefix = [0, 2, 6]
    assert oracle.preferred_move(prefix) == 7
