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

def test_root_is_initial_state(nfa):
    assert nfa.root.player == 'P1'
    assert all(sq == 0 for sq in nfa.root.board)

def test_root_has_nine_children(nfa):
    assert len(nfa.root.children) == 9

def test_alphabet_is_all_nine_squares(nfa):
    assert set(nfa.root.children.keys()) == set(range(9))


# ---------------------------------------------------------------------------
# get_node
# ---------------------------------------------------------------------------

def test_get_node_empty_trace(nfa):
    state = nfa.get_node([])
    assert state is nfa.root

def test_get_node_one_move(nfa):
    state = nfa.get_node([4])
    assert state is not None
    assert state.board[4] == 1   # X played center
    assert state.player == 'P2'

def test_get_node_two_moves(nfa):
    state = nfa.get_node([4, 0])
    assert state is not None
    assert state.board[4] == 1   # X center
    assert state.board[0] == 2   # O top-left
    assert state.player == 'P1'

def test_get_node_illegal_move_returns_none(nfa):
    assert nfa.get_node([4, 4]) is None   # square 4 played twice

def test_get_node_illegal_move_on_occupied_returns_none(nfa):
    assert nfa.get_node([0, 1, 0]) is None   # X tries to replay square 0


# ---------------------------------------------------------------------------
# p1_legal_inputs / p2_legal_moves
# ---------------------------------------------------------------------------

def test_p1_legal_inputs_at_root(nfa):
    inputs = nfa.p1_legal_inputs([])
    assert set(inputs) == set(range(9))

def test_p2_legal_moves_after_one_move(nfa):
    moves = nfa.p2_legal_moves([4])
    assert set(moves) == set(range(9)) - {4}

def test_p1_legal_inputs_empty_on_p2_turn(nfa):
    assert nfa.p1_legal_inputs([4]) == []

def test_p2_legal_moves_empty_on_p1_turn(nfa):
    assert nfa.p2_legal_moves([]) == []

def test_p1_legal_inputs_shrink_correctly(nfa):
    inputs = nfa.p1_legal_inputs([4, 0])
    assert set(inputs) == set(range(9)) - {4, 0}


# ---------------------------------------------------------------------------
# is_terminal / current_player
# ---------------------------------------------------------------------------

def test_is_terminal_false_on_empty(nfa):
    assert not nfa.is_terminal([])

def test_is_terminal_true_on_x_win(nfa):
    # X wins top row: X:0,1,2  O:3,4
    assert nfa.is_terminal([0, 3, 1, 4, 2])

def test_is_terminal_true_on_o_win(nfa):
    # O wins top row
    assert nfa.is_terminal([3, 0, 4, 1, 8, 2])

def test_current_player_p1_at_root(nfa):
    assert nfa.current_player([]) == 'P1'

def test_current_player_p2_after_one_move(nfa):
    assert nfa.current_player([4]) == 'P2'

def test_current_player_none_on_terminal(nfa):
    assert nfa.current_player([0, 3, 1, 4, 2]) is None
