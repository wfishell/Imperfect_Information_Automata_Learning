"""
Tests for TicTacToeState (src/game/tic_tac_toe/board.py)
Run: python -m pytest tests/game/tic_tac_toe/test_board.py -v
"""
import pytest
from src.game.tic_tac_toe.board import TicTacToeState


def make_state(moves: list[int]) -> TicTacToeState:
    """Replay a sequence of moves from the initial state."""
    state = TicTacToeState()
    for move in moves:
        state = state.children[move]
    return state


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_board_all_empty(self):
        assert all(sq == 0 for sq in TicTacToeState().board)

    def test_player_is_p1(self):
        assert TicTacToeState().player == 'P1'

    def test_children_count(self):
        assert len(TicTacToeState().children) == 9

    def test_not_terminal(self):
        assert not TicTacToeState().is_terminal()

    def test_winner_is_none(self):
        assert TicTacToeState().winner() is None


# ---------------------------------------------------------------------------
# Children / transitions
# ---------------------------------------------------------------------------

class TestChildren:
    def test_count_after_one_move(self):
        assert len(make_state([4]).children) == 8

    def test_count_after_two_moves(self):
        assert len(make_state([4, 0]).children) == 7

    def test_keys_are_unoccupied_squares(self):
        occupied = {4, 0, 8}
        assert set(make_state([4, 0, 8]).children.keys()) == set(range(9)) - occupied

    def test_player_alternates(self):
        s = TicTacToeState()
        assert s.player == 'P1'
        s = s.children[0]
        assert s.player == 'P2'
        s = s.children[1]
        assert s.player == 'P1'


# ---------------------------------------------------------------------------
# Terminal detection — P1 (X) wins all 8 lines
# ---------------------------------------------------------------------------

class TestP1Wins:
    @pytest.mark.parametrize("moves", [
        [0, 3, 1, 4, 2],        # top row
        [3, 0, 4, 1, 5],        # middle row
        [6, 0, 7, 1, 8],        # bottom row
        [0, 1, 3, 2, 6],        # left col
        [1, 0, 4, 2, 7],        # middle col
        [2, 0, 5, 1, 8],        # right col
        [0, 1, 4, 2, 8],        # diagonal \
        [2, 0, 4, 1, 6],        # diagonal /
    ])
    def test_p1_wins(self, moves):
        state = make_state(moves)
        assert state.is_terminal()
        assert state.winner() == 'P1'
        assert state.value == -1


# ---------------------------------------------------------------------------
# Terminal detection — P2 (O) wins all 8 lines
# ---------------------------------------------------------------------------

class TestP2Wins:
    @pytest.mark.parametrize("moves", [
        [3, 0, 4, 1, 8, 2],     # top row
        [0, 3, 1, 4, 8, 5],     # middle row
        [0, 6, 1, 7, 3, 8],     # bottom row
        [1, 0, 2, 3, 8, 6],     # left col
        [0, 1, 2, 4, 3, 7],     # middle col
        [0, 2, 1, 5, 3, 8],     # right col
        [1, 0, 2, 4, 3, 8],     # diagonal \
        [0, 2, 1, 4, 3, 6],     # diagonal /
    ])
    def test_p2_wins(self, moves):
        state = make_state(moves)
        assert state.is_terminal()
        assert state.winner() == 'P2'
        assert state.value == 1


# ---------------------------------------------------------------------------
# Draw and non-terminal value
# ---------------------------------------------------------------------------

class TestDrawAndValue:
    def test_draw(self):
        # X: 0,2,5,6,7  O: 1,3,4,8  — full board, no winner
        state = make_state([0, 1, 2, 3, 5, 4, 6, 8, 7])
        assert state.is_terminal()
        assert state.winner() == 'draw'
        assert state.value == 0

    def test_non_terminal_value_is_zero(self):
        state = make_state([4, 0])
        assert not state.is_terminal()
        assert state.value == 0
