"""
Tests for TicTacToeNFA (src/game/tic_tac_toe/game_nfa.py)
Run: python -m pytest tests/game/tic_tac_toe/test_game_nfa.py -v
"""
import pytest
from src.game.tic_tac_toe.game_nfa import TicTacToeNFA


@pytest.fixture
def nfa():
    return TicTacToeNFA()


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

class TestNFARoot:
    def test_is_initial_state(self, nfa):
        assert nfa.root.player == 'P1'
        assert all(sq == 0 for sq in nfa.root.board)

    def test_has_nine_children(self, nfa):
        assert len(nfa.root.children) == 9

    def test_alphabet_is_all_nine_squares(self, nfa):
        assert set(nfa.root.children.keys()) == set(range(9))


# ---------------------------------------------------------------------------
# get_node
# ---------------------------------------------------------------------------

class TestGetNode:
    def test_empty_trace_returns_root(self, nfa):
        assert nfa.get_node([]) is nfa.root

    def test_one_move(self, nfa):
        state = nfa.get_node([4])
        assert state is not None
        assert state.board[4] == 1   # X played center
        assert state.player == 'P2'

    def test_two_moves(self, nfa):
        state = nfa.get_node([4, 0])
        assert state.board[4] == 1   # X center
        assert state.board[0] == 2   # O top-left
        assert state.player == 'P1'

    def test_illegal_repeated_square_returns_none(self, nfa):
        assert nfa.get_node([4, 4]) is None

    def test_illegal_occupied_square_returns_none(self, nfa):
        assert nfa.get_node([0, 1, 0]) is None   # X tries to replay square 0


# ---------------------------------------------------------------------------
# p1_legal_inputs / p2_legal_moves
# ---------------------------------------------------------------------------

class TestLegalInputs:
    def test_p1_legal_inputs_at_root(self, nfa):
        assert set(nfa.p1_legal_inputs([])) == set(range(9))

    def test_p2_legal_moves_after_one_move(self, nfa):
        assert set(nfa.p2_legal_moves([4])) == set(range(9)) - {4}

    def test_p1_legal_inputs_empty_on_p2_turn(self, nfa):
        assert nfa.p1_legal_inputs([4]) == []

    def test_p2_legal_moves_empty_on_p1_turn(self, nfa):
        assert nfa.p2_legal_moves([]) == []

    def test_p1_legal_inputs_shrink_after_moves(self, nfa):
        assert set(nfa.p1_legal_inputs([4, 0])) == set(range(9)) - {4, 0}


# ---------------------------------------------------------------------------
# is_terminal / current_player
# ---------------------------------------------------------------------------

class TestTerminalAndCurrentPlayer:
    def test_is_terminal_false_on_empty(self, nfa):
        assert not nfa.is_terminal([])

    def test_is_terminal_true_on_p1_win(self, nfa):
        # X wins top row: X:0,1,2  O:3,4
        assert nfa.is_terminal([0, 3, 1, 4, 2])

    def test_is_terminal_true_on_p2_win(self, nfa):
        # O wins top row
        assert nfa.is_terminal([3, 0, 4, 1, 8, 2])

    def test_current_player_p1_at_root(self, nfa):
        assert nfa.current_player([]) == 'P1'

    def test_current_player_p2_after_one_move(self, nfa):
        assert nfa.current_player([4]) == 'P2'

    def test_current_player_none_on_terminal(self, nfa):
        assert nfa.current_player([0, 3, 1, 4, 2]) is None
