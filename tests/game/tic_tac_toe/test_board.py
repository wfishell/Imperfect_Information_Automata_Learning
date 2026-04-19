"""
Tests for TicTacToeState (src/game/tic_tac_toe/board.py)
Run: python -m pytest tests/game/tic_tac_toe/test_board.py -v
"""
import pytest
from src.game.tic_tac_toe.board import TicTacToeState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state(moves: list[int]) -> TicTacToeState:
    """Replay a sequence of moves from the initial state."""
    state = TicTacToeState()
    for move in moves:
        state = state.children[move]
    return state


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def test_initial_board_all_empty():
    state = TicTacToeState()
    assert all(sq == 0 for sq in state.board)

def test_initial_player_is_p1():
    state = TicTacToeState()
    assert state.player == 'P1'

def test_initial_children_count():
    state = TicTacToeState()
    assert len(state.children) == 9

def test_initial_not_terminal():
    state = TicTacToeState()
    assert not state.is_terminal()

def test_initial_winner_is_none():
    state = TicTacToeState()
    assert state.winner() is None


# ---------------------------------------------------------------------------
# Children / transitions
# ---------------------------------------------------------------------------

def test_children_after_one_move():
    state = make_state([4])
    assert len(state.children) == 8

def test_children_after_two_moves():
    state = make_state([4, 0])
    assert len(state.children) == 7

def test_children_keys_are_unoccupied_squares():
    state = make_state([4, 0, 8])
    occupied = {4, 0, 8}
    assert set(state.children.keys()) == set(range(9)) - occupied

def test_player_alternates():
    state = TicTacToeState()
    assert state.player == 'P1'
    state = state.children[0]
    assert state.player == 'P2'
    state = state.children[1]
    assert state.player == 'P1'


# ---------------------------------------------------------------------------
# Terminal detection — X wins (all 8 lines)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("moves", [
    [0, 3, 1, 4, 2],        # X wins top row
    [3, 0, 4, 1, 5],        # X wins middle row
    [6, 0, 7, 1, 8],        # X wins bottom row
    [0, 1, 3, 2, 6],        # X wins left col
    [1, 0, 4, 2, 7],        # X wins middle col
    [2, 0, 5, 1, 8],        # X wins right col
    [0, 1, 4, 2, 8],        # X wins diagonal \
    [2, 0, 4, 1, 6],        # X wins diagonal /
])
def test_x_wins(moves):
    state = make_state(moves)
    assert state.is_terminal()
    assert state.winner() == 'P1'
    assert state.value == -1


# ---------------------------------------------------------------------------
# Terminal detection — O wins (all 8 lines)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("moves", [
    [3, 0, 4, 1, 8, 2],     # O wins top row
    [0, 3, 1, 4, 2, 5],     # O wins middle row
    [0, 6, 1, 7, 2, 8],     # O wins bottom row
    [1, 0, 2, 3, 8, 6],     # O wins left col
    [0, 1, 3, 4, 6, 7],     # O wins middle col
    [0, 2, 1, 5, 3, 8],     # O wins right col
    [1, 0, 2, 4, 3, 8],     # O wins diagonal \
    [0, 2, 1, 4, 3, 6],     # O wins diagonal /
])
def test_o_wins(moves):
    state = make_state(moves)
    assert state.is_terminal()
    assert state.winner() == 'P2'
    assert state.value == 1


# ---------------------------------------------------------------------------
# Draw
# ---------------------------------------------------------------------------

def test_draw():
    # X: 0,2,5,6,7  O: 1,3,4,8  — no winner, board full
    state = make_state([0, 1, 2, 3, 5, 4, 6, 8, 7])
    assert state.is_terminal()
    assert state.winner() == 'draw'
    assert state.value == 0


# ---------------------------------------------------------------------------
# Value on non-terminal
# ---------------------------------------------------------------------------

def test_non_terminal_value_is_zero():
    state = make_state([4, 0])
    assert not state.is_terminal()
    assert state.value == 0
