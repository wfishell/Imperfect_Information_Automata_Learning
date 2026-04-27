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
    _h_edge,
    _v_edge
    )


# New Unit Tests

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


# Original LLM Generated Tests

def make_state(moves: list[int]) -> DotsAndBoxesState:
    state = DotsAndBoxesState()
    for move in moves:
        state = state.children[move]
        # DEBUG
        print(f"After move {move}:\n player={state.player},\n "
              f"p1_boxes={state.p1_boxes}, p2_boxes={state.p2_boxes},\n edges={state.edges}\n")

    # DEBUG
    print(f"Made state with moves {moves}:\n states:")

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
# Box completion — single box
# ---------------------------------------------------------------------------

def test_single_box_completion_keeps_same_player():
    # box(0,0) borders: 0, 2, 6, 7
    # P1:0, P2:2, P1:6 — three sides drawn, P2 to move
    # P2 draws 7 → completes box(0,0), P2 keeps turn
    s = make_state([0, 2, 6, 7])
    assert s.player == 'P2'

def test_single_box_completion_increments_score():
    s = make_state([0, 2, 6, 7])   # P2 completes box(0,0)
    assert s.p2_boxes == 1
    assert s.p1_boxes == 0

def test_p1_completes_box_increments_p1_score():
    # box(0,0): 0,2,6,7. Arrange so P1 draws the 4th side.
    # P1:0, P2:1, P1:2, P2:3, P1:6 — 5 edges drawn, P2 to move
    # Need P1 to draw edge 7. Let P2 draw something neutral first.
    # P1:0, P2:9, P1:2, P2:3, P1:6 → P2 to move → P2:1 → P1 to move → P1:7
    s = make_state([0, 9, 2, 3, 6, 1, 7])
    # After [0,9,2,3,6]: P2 to move (no completions yet)
    # After [1]: P1 to move
    # After [7]: box(0,0) complete (edges 0,2,6,7 all drawn), P1 keeps turn
    assert s.p1_boxes == 1
    assert s.player == 'P1'

def test_box_completion_reduces_children_count():
    # After completing a box the same player moves again from same edge pool
    s_before = make_state([0, 2, 6])   # 3 sides of box(0,0) drawn, P2 to move
    s_after  = s_before.children[7]    # P2 draws 7 → completes box, P2 moves again
    # 4 edges drawn total, 8 remain
    assert len(s_after.children) == 8


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
    assert s.player == 'P1'   # P1 drew the completing edge, keeps turn

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
    # Draw all 12 edges in a specific order where no one gets extra turns
    # to keep the sequence valid.
    # Any order works as long as we don't try to replay an already-drawn edge.
    # Use a sequence that avoids completing boxes until forced.
    # Edges that share no box: 0,4,5,6,8,9,11 (7 edges, no box complete until late)
    # Actually just replay a known full game.
    # Draw all edges avoiding early completions where possible.
    # Simplest: replay all 12 in order, accepting that some player gets extra turns.
    s = DotsAndBoxesState()
    edges_left = list(range(12))
    while edges_left:
        e = edges_left[0]
        if e in s.children:
            s = s.children[e]
            edges_left.remove(e)
        else:
            # edge already drawn (shouldn't happen in a fresh replay)
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
# Parameterised grid sizes
# ---------------------------------------------------------------------------

def test_initial_children_count_3x3():
    s = DotsAndBoxesState(rows=3, cols=3)
    # horizontal: 4*3=12, vertical: 3*4=12, total=24
    assert len(s.children) == 24

def test_total_boxes_3x3():
    assert DotsAndBoxesState(rows=3, cols=3).total_boxes == 9
