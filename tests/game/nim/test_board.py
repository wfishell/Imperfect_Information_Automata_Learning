"""
Tests for NimState (src/game/nim/board.py)
Run: python -m pytest tests/game/nim/test_board.py -v
"""
import pytest
from src.game.nim.board import NimState, _apply_move


@pytest.fixture
def root():
    return NimState(piles=(1, 2, 3), player='P1')


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_not_terminal(self, root):
        assert not root.is_terminal()

    def test_player_is_p1(self, root):
        assert root.player == 'P1'

    def test_winner_is_none(self, root):
        assert root.winner() is None

    def test_piles(self, root):
        assert root.piles == (1, 2, 3)


# ---------------------------------------------------------------------------
# Children count and structure
# ---------------------------------------------------------------------------

class TestChildren:
    def test_count_equals_sum_of_piles(self, root):
        # sum([1,2,3]) = 6
        assert len(root.children) == 6

    def test_keys_are_two_element_tuples(self, root):
        for key in root.children:
            assert isinstance(key, tuple)
            assert len(key) == 2

    def test_keys_cover_all_legal_moves(self, root):
        expected = {(0, 1), (1, 1), (1, 2), (2, 1), (2, 2), (2, 3)}
        assert set(root.children.keys()) == expected

    def test_target_pile_reduced_correctly(self, root):
        assert root.children[(2, 2)].piles == (1, 2, 1)

    def test_entire_pile_can_be_taken(self, root):
        assert root.children[(1, 2)].piles == (1, 0, 3)

    def test_other_piles_unchanged(self, root):
        child = root.children[(1, 1)]
        assert child.piles[0] == root.piles[0]
        assert child.piles[1] == root.piles[1] - 1
        assert child.piles[2] == root.piles[2]


# ---------------------------------------------------------------------------
# Player alternation
# ---------------------------------------------------------------------------

class TestPlayerAlternation:
    def test_alternates_after_one_move(self, root):
        assert root.children[(0, 1)].player == 'P2'

    def test_alternates_after_two_moves(self, root):
        assert root.children[(0, 1)].children[(1, 1)].player == 'P1'


# ---------------------------------------------------------------------------
# Terminal detection
# ---------------------------------------------------------------------------

class TestTerminal:
    def test_terminal_when_all_piles_zero(self):
        assert NimState(piles=(0, 0, 0), player='P1').is_terminal()

    def test_not_terminal_with_partial_piles(self):
        assert not NimState(piles=(0, 0, 1), player='P2').is_terminal()

    def test_terminal_has_no_children(self):
        assert NimState(piles=(0, 0, 0), player='P1').children == {}


# ---------------------------------------------------------------------------
# Winner and value
# ---------------------------------------------------------------------------

class TestWinnerAndValue:
    def test_winner_none_on_non_terminal(self, root):
        assert root.winner() is None

    def test_winner_p1_when_p2_to_move_at_terminal(self):
        # P2 is to move but piles are empty → P1 took last object → P1 wins
        assert NimState(piles=(0, 0, 0), player='P2').winner() == 'P1'

    def test_winner_p2_when_p1_to_move_at_terminal(self):
        # P1 is to move but piles are empty → P2 took last object → P2 wins
        assert NimState(piles=(0, 0, 0), player='P1').winner() == 'P2'

    def test_value_positive_when_p2_wins(self):
        assert NimState(piles=(0, 0, 0), player='P1').value == 1

    def test_value_negative_when_p1_wins(self):
        assert NimState(piles=(0, 0, 0), player='P2').value == -1


# ---------------------------------------------------------------------------
# Single-pile states
# ---------------------------------------------------------------------------

class TestSinglePile:
    def test_children_count(self):
        assert len(NimState(piles=(3,), player='P1').children) == 3

    def test_children_keys(self):
        assert set(NimState(piles=(3,), player='P1').children.keys()) == {(0, 1), (0, 2), (0, 3)}

    def test_pile_of_one_reaches_terminal(self):
        child = NimState(piles=(1,), player='P1').children[(0, 1)]
        assert child.is_terminal()

    def test_pile_of_one_winner_is_p1(self):
        child = NimState(piles=(1,), player='P1').children[(0, 1)]
        assert child.winner() == 'P1'


# ---------------------------------------------------------------------------
# _apply_move helper
# ---------------------------------------------------------------------------

class TestApplyMove:
    def test_basic(self):
        assert _apply_move((1, 2, 3), 1, 2) == (1, 0, 3)

    def test_entire_pile(self):
        assert _apply_move((3,), 0, 3) == (0,)
