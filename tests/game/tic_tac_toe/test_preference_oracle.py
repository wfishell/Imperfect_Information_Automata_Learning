"""
Tests for TicTacToeOracle (src/game/tic_tac_toe/preference_oracle.py)
Run: python -m pytest tests/game/tic_tac_toe/test_preference_oracle.py -v
"""
import pytest
from src.game.tic_tac_toe.game_nfa import TicTacToeNFA
from src.game.tic_tac_toe.preference_oracle import TicTacToeOracle


@pytest.fixture
def oracle():
    nfa = TicTacToeNFA()
    return TicTacToeOracle(nfa)


# ---------------------------------------------------------------------------
# preferred_move
# ---------------------------------------------------------------------------

def test_preferred_move_takes_winning_move(oracle):
    # O has 0,4 and can win by playing 8 (diagonal 0-4-8)
    # Trace: X:1, O:0, X:2, O:4, X:3 — now O's turn, O wins at 8
    prefix = [1, 0, 2, 4, 3]
    move = oracle.preferred_move(prefix)
    assert move == 8

def test_preferred_move_blocks_x_win(oracle):
    # X has 0,1 and will win at 2 — O must block
    # Trace: X:0, O:4, X:1  — O must play 2
    prefix = [0, 4, 1]
    move = oracle.preferred_move(prefix)
    assert move == 2

def test_preferred_move_returns_none_on_p1_turn(oracle):
    assert oracle.preferred_move([]) is None

def test_preferred_move_returns_none_on_terminal(oracle):
    # X wins top row
    assert oracle.preferred_move([0, 3, 1, 4, 2]) is None

def test_preferred_move_returns_legal_move(oracle):
    prefix = [4]
    move = oracle.preferred_move(prefix)
    assert move in set(range(9)) - {4}


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def test_compare_o_win_beats_draw(oracle):
    # trace1: O wins; trace2: draw
    o_win  = [3, 0, 4, 1, 8, 2]        # O wins top row
    draw   = [0, 1, 2, 3, 5, 4, 6, 8, 7]
    assert oracle.compare(o_win, draw) == 't1'

def test_compare_draw_beats_x_win(oracle):
    draw  = [0, 1, 2, 3, 5, 4, 6, 8, 7]
    x_win = [0, 3, 1, 4, 2]             # X wins top row
    assert oracle.compare(draw, x_win) == 't1'

def test_compare_o_win_beats_x_win(oracle):
    o_win = [3, 0, 4, 1, 8, 2]
    x_win = [0, 3, 1, 4, 2]
    assert oracle.compare(o_win, x_win) == 't1'

def test_compare_x_win_loses_to_o_win(oracle):
    x_win = [0, 3, 1, 4, 2]
    o_win = [3, 0, 4, 1, 8, 2]
    assert oracle.compare(x_win, o_win) == 't2'

def test_compare_equal_draws(oracle):
    draw1 = [0, 1, 2, 3, 5, 4, 6, 8, 7]
    draw2 = [0, 1, 2, 3, 5, 4, 6, 8, 7]
    assert oracle.compare(draw1, draw2) == 'equal'
