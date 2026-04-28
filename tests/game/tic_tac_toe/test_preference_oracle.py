"""
Tests for TicTacToeOracle (src/game/tic_tac_toe/preference_oracle.py)
Run: python -m pytest tests/game/tic_tac_toe/test_preference_oracle.py -v
"""
import pytest
from src.game.tic_tac_toe.game_nfa import TicTacToeNFA
from src.game.tic_tac_toe.board import X, O, EMPTY
from src.game.tic_tac_toe.preference_oracle import TicTacToeOracle


@pytest.fixture
def oracle():
    return TicTacToeOracle(TicTacToeNFA())


# ---------------------------------------------------------------------------
# preferred_move — optimal play
# ---------------------------------------------------------------------------

class TestPreferredMove:
    def test_takes_winning_move(self, oracle):
        # O has 0,4 and can win by playing 8 (diagonal 0-4-8)
        # Trace: X:1, O:0, X:2, O:4, X:3 — O's turn, wins at 8
        assert oracle.preferred_move([1, 0, 2, 4, 3]) == 8

    def test_blocks_p1_win(self, oracle):
        # X has 0,1 and wins at 2 — O must block
        # Trace: X:0, O:4, X:1 — O must play 2
        assert oracle.preferred_move([0, 4, 1]) == 2

    def test_returns_legal_move(self, oracle):
        move = oracle.preferred_move([4])
        assert move in set(range(9)) - {4}


# ---------------------------------------------------------------------------
# preferred_move — boundary conditions
# ---------------------------------------------------------------------------

class TestPreferredMoveBoundary:
    def test_returns_none_on_p1_turn(self, oracle):
        assert oracle.preferred_move([]) is None

    def test_returns_none_on_terminal(self, oracle):
        # X wins top row
        assert oracle.preferred_move([0, 3, 1, 4, 2]) is None


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

class TestCompare:
    def test_p2_win_beats_draw(self, oracle):
        o_win = [3, 0, 4, 1, 8, 2]          # O wins top row
        draw  = [0, 1, 2, 3, 5, 4, 6, 8, 7]
        assert oracle.compare(o_win, draw) == 't1'

    def test_draw_beats_p1_win(self, oracle):
        draw  = [0, 1, 2, 3, 5, 4, 6, 8, 7]
        x_win = [0, 3, 1, 4, 2]
        assert oracle.compare(draw, x_win) == 't1'

    def test_p2_win_beats_p1_win(self, oracle):
        assert oracle.compare([3, 0, 4, 1, 8, 2], [0, 3, 1, 4, 2]) == 't1'

    def test_p1_win_loses_to_p2_win(self, oracle):
        assert oracle.compare([0, 3, 1, 4, 2], [3, 0, 4, 1, 8, 2]) == 't2'

    def test_equal_draws(self, oracle):
        draw = [0, 1, 2, 3, 5, 4, 6, 8, 7]
        assert oracle.compare(draw, draw) == 'equal'


# ---------------------------------------------------------------------------
# _heuristic
# ---------------------------------------------------------------------------

class TestHeuristic:
    def test_empty_board_is_zero(self):
        assert TicTacToeOracle._heuristic((EMPTY,) * 9) == 0.0

    def test_positive_when_o_has_open_lines(self):
        # O occupies center (index 4) — contributes to 4 open lines
        board = (EMPTY,) * 4 + (O,) + (EMPTY,) * 4
        assert TicTacToeOracle._heuristic(board) > 0.0

    def test_negative_when_x_has_open_lines(self):
        # X occupies center — contributes to 4 open lines for X
        board = (EMPTY,) * 4 + (X,) + (EMPTY,) * 4
        assert TicTacToeOracle._heuristic(board) < 0.0

    def test_mixed_line_scores_less_than_clear_line(self):
        # A line with both X and O is dead — clear O line scores higher than mixed
        clear_o = (O, O, EMPTY) + (EMPTY,) * 6   # top row open for O
        mixed   = (X, O, EMPTY) + (EMPTY,) * 6   # top row blocked
        assert TicTacToeOracle._heuristic(clear_o) > TicTacToeOracle._heuristic(mixed)

    def test_stays_inside_terminal_range(self):
        # Heuristic should stay strictly inside (-1, 1)
        board = (O, O, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY)
        h = TicTacToeOracle._heuristic(board)
        assert -1.0 < h < 1.0
