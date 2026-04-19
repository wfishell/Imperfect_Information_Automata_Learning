"""
Tests for NimState (src/game/nim/board.py)
Run: python -m pytest tests/game/nim/test_board.py -v
"""
import pytest
from src.game.nim.board import NimState, _apply_move


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def root():
    return NimState(piles=(1, 2, 3), player='P1')


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def test_initial_not_terminal(root):
    assert not root.is_terminal()

def test_initial_player_p1(root):
    assert root.player == 'P1'

def test_initial_winner_none(root):
    assert root.winner() is None

def test_initial_piles(root):
    assert root.piles == (1, 2, 3)


# ---------------------------------------------------------------------------
# Children count and structure
# ---------------------------------------------------------------------------

def test_children_count_equals_sum_piles(root):
    # sum([1,2,3]) = 6
    assert len(root.children) == 6

def test_children_keys_are_tuples(root):
    for key in root.children:
        assert isinstance(key, tuple)
        assert len(key) == 2

def test_children_keys_cover_all_legal_moves(root):
    expected = {(0,1), (1,1),(1,2), (2,1),(2,2),(2,3)}
    assert set(root.children.keys()) == expected

def test_child_pile_reduced_correctly(root):
    child = root.children[(2, 2)]
    assert child.piles == (1, 2, 1)

def test_child_pile_takes_entire_pile(root):
    child = root.children[(1, 2)]
    assert child.piles == (1, 0, 3)


# ---------------------------------------------------------------------------
# Player alternation
# ---------------------------------------------------------------------------

def test_player_alternates_after_move(root):
    child = root.children[(0, 1)]
    assert child.player == 'P2'

def test_player_alternates_twice(root):
    grandchild = root.children[(0, 1)].children[(1, 1)]
    assert grandchild.player == 'P1'


# ---------------------------------------------------------------------------
# Terminal detection
# ---------------------------------------------------------------------------

def test_terminal_when_all_piles_zero():
    s = NimState(piles=(0, 0, 0), player='P1')
    assert s.is_terminal()

def test_not_terminal_partial_piles():
    s = NimState(piles=(0, 0, 1), player='P2')
    assert not s.is_terminal()

def test_terminal_has_no_children():
    s = NimState(piles=(0, 0, 0), player='P1')
    assert s.children == {}


# ---------------------------------------------------------------------------
# Winner and value
# ---------------------------------------------------------------------------

def test_winner_none_on_non_terminal(root):
    assert root.winner() is None

def test_winner_p1_when_p2_to_move_at_terminal():
    # P2 is to move but piles are empty → P1 took last object → P1 wins
    s = NimState(piles=(0, 0, 0), player='P2')
    assert s.winner() == 'P1'

def test_winner_p2_when_p1_to_move_at_terminal():
    # P1 is to move but piles are empty → P2 took last object → P2 wins
    s = NimState(piles=(0, 0, 0), player='P1')
    assert s.winner() == 'P2'

def test_value_positive_when_p2_wins():
    s = NimState(piles=(0, 0, 0), player='P1')  # P2 wins
    assert s.value == 1

def test_value_negative_when_p1_wins():
    s = NimState(piles=(0, 0, 0), player='P2')  # P1 wins
    assert s.value == -1


# ---------------------------------------------------------------------------
# Single-pile state
# ---------------------------------------------------------------------------

def test_single_pile_children_count():
    s = NimState(piles=(3,), player='P1')
    assert len(s.children) == 3

def test_single_pile_keys():
    s = NimState(piles=(3,), player='P1')
    assert set(s.children.keys()) == {(0,1), (0,2), (0,3)}

def test_pile_of_one_reaches_terminal():
    s = NimState(piles=(1,), player='P1')
    child = s.children[(0, 1)]
    assert child.is_terminal()

def test_pile_of_one_winner_is_p1():
    s = NimState(piles=(1,), player='P1')
    child = s.children[(0, 1)]
    # P2 is now to move at empty piles → P1 won
    assert child.winner() == 'P1'


# ---------------------------------------------------------------------------
# Multi-pile: only target pile changes
# ---------------------------------------------------------------------------

def test_other_piles_unchanged(root):
    child = root.children[(1, 1)]
    assert child.piles[0] == root.piles[0]
    assert child.piles[2] == root.piles[2]


# ---------------------------------------------------------------------------
# _apply_move helper
# ---------------------------------------------------------------------------

def test_apply_move_basic():
    assert _apply_move((1, 2, 3), 1, 2) == (1, 0, 3)

def test_apply_move_entire_pile():
    assert _apply_move((3,), 0, 3) == (0,)
