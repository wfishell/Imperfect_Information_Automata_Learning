"""
Tests for DotsAndBoxesState (src/game/dots_and_boxes/board.py)
Run: python -m pytest tests/game/dots_and_boxes/test_board.py -v

2×2 box grid edge layout (R=2, C=2, 12 edges total):

  Horizontal edges (0–5):
    0  top    of box(0,0)
    1  top    of box(0,1)
    2  bottom of box(0,0) / top    of box(1,0)
    3  bottom of box(0,1) / top    of box(1,1)
    4  bottom of box(1,0)
    5  bottom of box(1,1)

  Vertical edges (6–11):
    6  left   of box(0,0)
    7  right  of box(0,0) / left  of box(0,1)
    8  right  of box(0,1)
    9  left   of box(1,0)
   10  right  of box(1,0) / left  of box(1,1)
   11  right  of box(1,1)

  Box borders:
    box(0,0): edges 0, 2, 6, 7
    box(0,1): edges 1, 3, 7, 8
    box(1,0): edges 2, 4, 9, 10
    box(1,1): edges 3, 5, 10, 11
"""

import pytest
from src.game.dots_and_boxes.board import (
    DotsAndBoxesState,
    PASS,
    _h_edge,
    _v_edge,
    _box_borders,
    _adjacent_boxes,
    _boxes_completed_by
    )


class TestEdgeIndexing:
    def test_h_edge_indices(self):
        # 2 x 2 Grid:
        assert _h_edge(0, 0, 2) == 0
        assert _h_edge(0, 1, 2) == 1
        assert _h_edge(1, 0, 2) == 2
        assert _h_edge(1, 1, 2) == 3
        assert _h_edge(2, 0, 2) == 4
        assert _h_edge(2, 1, 2) == 5

    def test_v_edge_indices(self):
        # 2 x 2 Grid, vertical edges start at index 6:
        assert _v_edge(0, 0, 2, 2) == 6
        assert _v_edge(0, 1, 2, 2) == 7
        assert _v_edge(0, 2, 2, 2) == 8
        assert _v_edge(1, 0, 2, 2) == 9
        assert _v_edge(1, 1, 2, 2) == 10
        assert _v_edge(1, 2, 2, 2) == 11

    def test_box_querying(self):
        # 2 x 2 Grid:
        assert _box_borders(0, 0, 2, 2) == (0, 2, 6, 7)
        assert _box_borders(0, 1, 2, 2) == (1, 3, 7, 8)
        assert _box_borders(1, 0, 2, 2) == (2, 4, 9, 10)
        assert _box_borders(1, 1, 2, 2) == (3, 5, 10, 11)

    def test_adjacent_boxes(self):
        # 2 x 2 Grid:
        assert _adjacent_boxes(0, 2, 2) == [(0, 0)]
        assert _adjacent_boxes(7, 2, 2) == [(0, 0), (0, 1)]
        assert _adjacent_boxes(10, 2, 2) == [(1, 0), (1, 1)]

class TestBoxesCompletedBy:
    def test_no_boxes_completed(self):
        edges = (True, False, True, False, False, False,
                 False, False, False, False, False, False)
        assert _boxes_completed_by(edges, 7, 2, 2) == 0

    def test_one_box_completed(self):
        edges = (True, False, True, False, False, False,
                 True, False, False, False, False, False)
        assert _boxes_completed_by(edges, 7, 2, 2) == 1

    def test_two_boxes_completed(self):
        edges = (True, True, True, True, False, False,
                 True, True, True, False, False, False)
        assert _boxes_completed_by(edges, 7, 2, 2) == 2

# ---------------------------------------------------------------------------
# DotsAndBoxesState Functionality Tests
# ---------------------------------------------------------------------------

def make_state(moves: list) -> DotsAndBoxesState:
    state = DotsAndBoxesState()
    for move in moves:
        state = state.children[move]
    return state


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def test_initial_all_edges_undrawn():
    s = DotsAndBoxesState()
    assert all(not e for e in s.edges)

def test_initial_player_is_p1():
    assert DotsAndBoxesState().player == 'P1'

def test_initial_scores_zero():
    s = DotsAndBoxesState()
    assert s.p1_boxes == 0 and s.p2_boxes == 0

def test_initial_children_count_2x2():
    assert len(DotsAndBoxesState().children) == 12

def test_initial_not_terminal():
    assert not DotsAndBoxesState().is_terminal()

def test_initial_winner_none():
    assert DotsAndBoxesState().winner() is None

def test_total_boxes_2x2():
    assert DotsAndBoxesState().total_boxes == 4

def test_initial_not_forced_pass():
    assert DotsAndBoxesState().forced_pass is False


# ---------------------------------------------------------------------------
# Children / transitions — no box completion
# ---------------------------------------------------------------------------

def test_children_decrease_after_each_non_completing_move():
    # Edges 0,1,2,3 don't complete any box on their own
    s = make_state([0])
    assert len(s.children) == 11

def test_player_alternates_when_no_box_completed():
    s = DotsAndBoxesState()
    assert s.player == 'P1'
    s = s.children[0]   # P1 draws edge 0 — no box complete
    assert s.player == 'P2'
    s = s.children[1]   # P2 draws edge 1 — no box complete
    assert s.player == 'P1'

def test_drawn_edge_not_in_children():
    s = make_state([0])
    assert 0 not in s.children

def test_children_keys_are_undrawn_edges():
    moves = [0, 1, 2]
    s = make_state(moves)
    assert set(s.children.keys()) == set(range(12)) - set(moves)


# ---------------------------------------------------------------------------
# Box completion — forced-pass encoding
# ---------------------------------------------------------------------------

def test_p2_box_completion_creates_p1_forced_pass():
    # box(0,0) borders: 0, 2, 6, 7
    # P1:0, P2:2, P1:6 — three sides drawn, P2 to move
    # P2 draws 7 → completes box(0,0) → P1 is now forced to pass
    s = make_state([0, 2, 6, 7])
    assert s.forced_pass is True
    assert s.player == 'P1'

def test_p2_box_completion_increments_p2_score():
    s = make_state([0, 2, 6, 7])   # P2 completes box(0,0)
    assert s.p2_boxes == 1
    assert s.p1_boxes == 0

def test_forced_pass_only_child_is_pass():
    s = make_state([0, 2, 6, 7])   # P1-forced state
    assert set(s.children.keys()) == {PASS}

def test_forced_pass_leads_to_real_p2_turn():
    s_forced = make_state([0, 2, 6, 7])           # P1-forced
    s_real   = s_forced.children[PASS]             # P2's extra real turn
    assert s_real.player == 'P2'
    assert s_real.forced_pass is False

def test_forced_pass_real_turn_has_correct_children():
    s_forced = make_state([0, 2, 6, 7])
    s_real   = s_forced.children[PASS]
    # 4 edges drawn, 8 remain
    assert len(s_real.children) == 8
    assert 7 not in s_real.children

def test_p1_completes_box_creates_p2_forced_pass():
    # box(0,0): 0,2,6,7. Arrange so P1 draws the 4th side.
    # P1:0, P2:9, P1:2, P2:3, P1:6, P2:1, P1:7
    s = make_state([0, 9, 2, 3, 6, 1, 7])
    assert s.p1_boxes == 1
    assert s.player == 'P2'
    assert s.forced_pass is True

def test_p1_box_forced_pass_leads_to_real_p1_turn():
    s_forced = make_state([0, 9, 2, 3, 6, 1, 7])  # P2-forced
    s_real   = s_forced.children[PASS]              # P1's extra real turn
    assert s_real.player == 'P1'
    assert s_real.forced_pass is False


# ---------------------------------------------------------------------------
# Double box completion (one move completes two boxes)
# ---------------------------------------------------------------------------

def test_double_box_completion():
    # Edge 7 is shared by box(0,0) and box(0,1).
    # box(0,0) needs: 0,2,6,7   box(0,1) needs: 1,3,7,8
    # Draw {0,1,2,3,6,8} first (6 non-completing moves), then P1 draws 7.
    # Sequence: P1:0, P2:1, P1:2, P2:3, P1:6, P2:8 → P1 to move
    s = make_state([0, 1, 2, 3, 6, 8, 7])
    assert s.p1_boxes == 2
    assert s.player == 'P2'         # P2 is forced to pass
    assert s.forced_pass is True

def test_double_box_increments_score_by_two():
    s = make_state([0, 1, 2, 3, 6, 8])
    assert s.player == 'P1'
    s2 = s.children[7]
    assert s2.p1_boxes == s.p1_boxes + 2


# ---------------------------------------------------------------------------
# Terminal detection
# ---------------------------------------------------------------------------

def test_not_terminal_with_edges_remaining():
    assert not make_state([0, 1, 2]).is_terminal()

def test_terminal_when_all_edges_drawn():
    s = DotsAndBoxesState()
    edges_left = list(range(12))
    while not s.is_terminal():
        if s.forced_pass:
            s = s.children[PASS]
            continue
        e = next((e for e in edges_left if e in s.children), None)
        if e is None:
            break
        s = s.children[e]
        edges_left.remove(e)
    assert s.is_terminal()

def test_terminal_all_edges_true():
    all_drawn = (True,) * 12
    s = DotsAndBoxesState(edges=all_drawn, p1_boxes=2, p2_boxes=2)
    assert s.is_terminal()

def test_terminal_children_empty():
    all_drawn = (True,) * 12
    s = DotsAndBoxesState(edges=all_drawn, p1_boxes=2, p2_boxes=2)
    assert s.children == {}


# ---------------------------------------------------------------------------
# Winner and value
# ---------------------------------------------------------------------------

def test_winner_p1():
    s = DotsAndBoxesState(edges=(True,) * 12, p1_boxes=3, p2_boxes=1)
    assert s.winner() == 'P1'
    assert s.value == -1

def test_winner_p2():
    s = DotsAndBoxesState(edges=(True,) * 12, p1_boxes=1, p2_boxes=3)
    assert s.winner() == 'P2'
    assert s.value == 1

def test_winner_draw():
    s = DotsAndBoxesState(edges=(True,) * 12, p1_boxes=2, p2_boxes=2)
    assert s.winner() == 'draw'
    assert s.value == 0

def test_winner_none_on_non_terminal():
    assert DotsAndBoxesState().winner() is None

def test_non_terminal_value_zero():
    assert make_state([0, 1]).value == 0


# ---------------------------------------------------------------------------
# Parameterized grid sizes
# ---------------------------------------------------------------------------

def test_initial_children_count_3x3():
    s = DotsAndBoxesState(rows=3, cols=3)
    # horizontal: 4*3=12, vertical: 3*4=12, total=24
    assert len(s.children) == 24

def test_total_boxes_3x3():
    assert DotsAndBoxesState(rows=3, cols=3).total_boxes == 9
