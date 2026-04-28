"""
Tests for HexOracle (src/game/hex/preference_oracle.py)
Run: python -m pytest tests/game/hex/test_preference_oracle.py -v

3×3 board layout:
    0 1 2      P1 (X) connects top row {0,1,2} → bottom row {6,7,8}
     3 4 5     P2 (O) connects left col {0,3,6} → right col {2,5,8}
      6 7 8
"""
import pytest
from src.game.hex.game_nfa import HexNFA
from src.game.hex.preference_oracle import HexOracle
from src.game.hex.board import HexState, EMPTY, X, O


@pytest.fixture
def nfa():
    return HexNFA()


@pytest.fixture
def oracle(nfa):
    return HexOracle(nfa)


@pytest.fixture
def oracle_depth2(nfa):
    return HexOracle(nfa, depth=2)


# ---------------------------------------------------------------------------
# preferred_move — boundary conditions
# ---------------------------------------------------------------------------

class TestPreferredMoveBoundary:
    def test_returns_none_on_p1_turn(self, oracle):
        assert oracle.preferred_move([]) is None

    def test_returns_none_on_terminal_p1_win(self, oracle):
        # P1 wins at [1,0,4,3,7]
        assert oracle.preferred_move([1, 0, 4, 3, 7]) is None

    def test_returns_none_on_terminal_p2_win(self, oracle):
        # P2 wins at [7,3,2,4,0,5]
        assert oracle.preferred_move([7, 3, 2, 4, 0, 5]) is None

    def test_returns_none_on_invalid_trace(self, oracle):
        assert oracle.preferred_move([4, 4]) is None


# ---------------------------------------------------------------------------
# preferred_move — correctness
# ---------------------------------------------------------------------------

class TestPreferredMove:
    def test_returns_legal_move(self, oracle):
        move = oracle.preferred_move([4])
        assert move is not None
        assert move in set(range(9)) - {4}

    def test_takes_winning_move(self, oracle):
        # O has cells 3 and 4; playing 5 completes the left→right connection.
        # Trace: P1:7, P2:3, P1:2, P2:4, P1:0 — P2's turn, wins by playing 5.
        move = oracle.preferred_move([7, 3, 2, 4, 0])
        assert move == 5

    def test_returns_legal_move_on_p2_first_turn(self, oracle):
        move = oracle.preferred_move([0])
        assert move in set(range(9)) - {0}

    def test_depth2_returns_legal_move(self, oracle_depth2):
        move = oracle_depth2.preferred_move([4])
        assert move is not None
        assert move in set(range(9)) - {4}


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

class TestCompare:
    def test_p2_win_beats_p1_win(self, oracle):
        p2_win = [7, 3, 2, 4, 0, 5]    # O wins
        p1_win = [1, 0, 4, 3, 7]        # X wins
        assert oracle.compare(p2_win, p1_win) == 't1'

    def test_p1_win_loses_to_p2_win(self, oracle):
        p2_win = [7, 3, 2, 4, 0, 5]
        p1_win = [1, 0, 4, 3, 7]
        assert oracle.compare(p1_win, p2_win) == 't2'

    def test_equal_same_trace(self, oracle):
        trace = [7, 3, 2, 4, 0, 5]
        assert oracle.compare(trace, trace) == 'equal'

    def test_ordering_consistent_with_trace_value(self, oracle):
        t1 = [7, 3, 2, 4, 0, 5]   # P2 wins
        t2 = [1, 0, 4, 3, 7]      # P1 wins
        v1 = oracle._trace_value(t1)
        v2 = oracle._trace_value(t2)
        result = oracle.compare(t1, t2)
        if v1 > v2:
            assert result == 't1'
        elif v2 > v1:
            assert result == 't2'
        else:
            assert result == 'equal'


# ---------------------------------------------------------------------------
# _heuristic
# ---------------------------------------------------------------------------

class TestHeuristic:
    def test_empty_board_is_zero(self):
        state = HexState(size=3)
        assert HexOracle._heuristic(state) == 0.0

    def test_positive_when_o_has_left_col_cells(self):
        # O occupies left col cell 3 → frontier grows from left side
        board = list((EMPTY,) * 9)
        board[3] = O
        state = HexState(size=3, board=tuple(board), player='P1')
        assert HexOracle._heuristic(state) > 0.0

    def test_negative_when_x_has_top_row_cells(self):
        # X occupies top row cell 1 → frontier grows from top
        board = list((EMPTY,) * 9)
        board[1] = X
        state = HexState(size=3, board=tuple(board), player='P2')
        assert HexOracle._heuristic(state) < 0.0

    def test_stays_inside_terminal_range(self):
        # Several non-terminal boards — heuristic must stay strictly in (-1, 1)
        boards = [
            [O, O, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
            [X, X, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
            [O, EMPTY, EMPTY, O, EMPTY, EMPTY, O, EMPTY, EMPTY],
        ]
        for b in boards:
            state = HexState(size=3, board=tuple(b), player='P1')
            if not state.is_terminal():
                h = HexOracle._heuristic(state)
                assert -1.0 < h < 1.0

    def test_more_o_frontier_beats_more_x_frontier(self):
        # O has left col cells, X has nothing → positive heuristic
        board_o = [EMPTY] * 9
        board_o[0] = board_o[3] = board_o[6] = O
        state_o = HexState(size=3, board=tuple(board_o), player='P1')

        # X has top row cells, O has nothing → negative heuristic
        board_x = [EMPTY] * 9
        board_x[0] = board_x[1] = board_x[2] = X
        state_x = HexState(size=3, board=tuple(board_x), player='P2')

        assert HexOracle._heuristic(state_o) > HexOracle._heuristic(state_x)


# ---------------------------------------------------------------------------
# Bounded depth
# ---------------------------------------------------------------------------

class TestBoundedDepth:
    def test_depth2_preferred_move_is_legal(self, oracle_depth2):
        move = oracle_depth2.preferred_move([4])
        state = oracle_depth2.nfa.get_node([4])
        assert move in state.children

    def test_depth0_returns_legal_move(self, nfa):
        oracle_d0 = HexOracle(nfa, depth=0)
        move = oracle_d0.preferred_move([4])
        assert move is not None
        assert move in set(range(9)) - {4}
