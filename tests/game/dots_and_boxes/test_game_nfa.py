"""
Tests for DotsAndBoxesNFA (src/game/dots_and_boxes/game_nfa.py)
Run: python -m pytest tests/game/dots_and_boxes/test_game_nfa.py -v

Edge layout for the default 2×2 grid — see test_board.py for full reference.
  box(0,0): edges 0, 2, 6, 7
  box(0,1): edges 1, 3, 7, 8
  box(1,0): edges 2, 4, 9, 10
  box(1,1): edges 3, 5, 10, 11
"""

import pytest
from src.game.dots_and_boxes.game_nfa import DotsAndBoxesNFA


@pytest.fixture
def nfa():
    return DotsAndBoxesNFA()


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

def test_root_is_initial_state(nfa):
    assert nfa.root.player == 'P1'
    assert not any(nfa.root.edges)

def test_root_has_12_children_2x2(nfa):
    assert len(nfa.root.children) == 12

def test_alphabet_is_all_12_edges(nfa):
    assert set(nfa.root.children.keys()) == set(range(12))


# ---------------------------------------------------------------------------
# get_node
# ---------------------------------------------------------------------------

def test_get_node_empty_trace(nfa):
    assert nfa.get_node([]) is nfa.root

def test_get_node_one_move(nfa):
    state = nfa.get_node([0])
    assert state is not None
    assert state.edges[0] is True
    assert state.player == 'P2'

def test_get_node_two_moves(nfa):
    state = nfa.get_node([0, 1])
    assert state.edges[0] is True
    assert state.edges[1] is True
    assert state.player == 'P1'

def test_get_node_illegal_repeated_edge_returns_none(nfa):
    assert nfa.get_node([0, 0]) is None

def test_get_node_illegal_edge_in_sequence_returns_none(nfa):
    assert nfa.get_node([0, 1, 0]) is None


# ---------------------------------------------------------------------------
# Extra-turn mechanic — get_node must handle consecutive same-player moves
# ---------------------------------------------------------------------------

def test_get_node_same_player_after_box_completion(nfa):
    # box(0,0): edges 0,2,6,7
    # P1:0, P2:2, P1:6 → P2 to move; P2:7 completes box, P2 keeps turn
    state = nfa.get_node([0, 2, 6, 7])
    assert state is not None
    assert state.player == 'P2'
    assert state.p2_boxes == 1

def test_get_node_after_double_completion(nfa):
    # P1:0, P2:1, P1:2, P2:3, P1:6, P2:8 → P1 to move
    # P1:7 completes box(0,0) and box(0,1) → P1 keeps turn, p1_boxes=2
    state = nfa.get_node([0, 1, 2, 3, 6, 8, 7])
    assert state is not None
    assert state.p1_boxes == 2
    assert state.player == 'P1'


# ---------------------------------------------------------------------------
# p1_legal_inputs / p2_legal_moves
# ---------------------------------------------------------------------------

def test_p1_legal_inputs_at_root(nfa):
    assert set(nfa.p1_legal_inputs([])) == set(range(12))

def test_p2_legal_moves_after_one_p1_move(nfa):
    assert set(nfa.p2_legal_moves([0])) == set(range(12)) - {0}

def test_p1_legal_inputs_empty_on_p2_turn(nfa):
    assert nfa.p1_legal_inputs([0]) == []

def test_p2_legal_moves_empty_on_p1_turn(nfa):
    assert nfa.p2_legal_moves([]) == []

def test_p1_legal_inputs_shrink_after_moves(nfa):
    assert set(nfa.p1_legal_inputs([0, 1])) == set(range(12)) - {0, 1}

def test_p2_legal_moves_available_after_box_completion(nfa):
    # After P2 completes box(0,0), P2 still has legal moves
    moves = nfa.p2_legal_moves([0, 2, 6, 7])
    assert len(moves) == 8   # 12 - 4 drawn edges
    assert 7 not in moves    # already drawn


# ---------------------------------------------------------------------------
# is_terminal / current_player
# ---------------------------------------------------------------------------

def test_is_terminal_false_on_empty(nfa):
    assert not nfa.is_terminal([])

def test_is_terminal_false_mid_game(nfa):
    assert not nfa.is_terminal([0, 1, 2])

def test_current_player_p1_at_root(nfa):
    assert nfa.current_player([]) == 'P1'

def test_current_player_p2_after_p1_move(nfa):
    assert nfa.current_player([0]) == 'P2'

def test_current_player_same_after_box_completion(nfa):
    # P2 completes box(0,0) and keeps turn
    assert nfa.current_player([0, 2, 6, 7]) == 'P2'

def test_current_player_none_on_terminal(nfa):
    all_drawn = tuple(range(12))
    # Build a terminal state by drawing all edges
    state = nfa.root
    remaining = list(range(12))
    trace = []
    while remaining:
        e = next(e for e in remaining if e in state.children)
        state = state.children[e]
        trace.append(e)
        remaining.remove(e)
    assert nfa.current_player(trace) is None


# ---------------------------------------------------------------------------
# 3×3 grid
# ---------------------------------------------------------------------------

def test_3x3_root_has_24_children():
    nfa3 = DotsAndBoxesNFA(rows=3, cols=3)
    assert len(nfa3.root.children) == 24

def test_3x3_alphabet_is_all_24_edges():
    nfa3 = DotsAndBoxesNFA(rows=3, cols=3)
    assert set(nfa3.root.children.keys()) == set(range(24))
