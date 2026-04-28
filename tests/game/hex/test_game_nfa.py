"""
Tests for HexNFA (src/game/hex/game_nfa.py)
Run: python -m pytest tests/game/hex/test_game_nfa.py -v
"""
import pytest
from src.game.hex.game_nfa import HexNFA


@pytest.fixture
def nfa():
    return HexNFA()


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

class TestNFARoot:
    def test_is_initial_state(self, nfa):
        assert nfa.root.player == 'P1'
        assert all(sq == 0 for sq in nfa.root.board)

    def test_has_nine_children(self, nfa):
        assert len(nfa.root.children) == 9

    def test_alphabet_is_all_nine_cells(self, nfa):
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
        assert state.board[4] == 1   # X placed at center
        assert state.player == 'P2'

    def test_two_moves(self, nfa):
        state = nfa.get_node([4, 0])
        assert state.board[4] == 1   # X center
        assert state.board[0] == 2   # O top-left
        assert state.player == 'P1'

    def test_illegal_occupied_cell_returns_none(self, nfa):
        assert nfa.get_node([4, 4]) is None

    def test_illegal_replay_returns_none(self, nfa):
        assert nfa.get_node([0, 1, 0]) is None

    def test_beyond_terminal_returns_none(self, nfa):
        # P1 wins at [1, 0, 4, 3, 7]; any further move should return None
        win_trace = [1, 0, 4, 3, 7]
        assert nfa.get_node(win_trace + [2]) is None


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

    def test_p1_legal_inputs_shrink_after_two_moves(self, nfa):
        assert set(nfa.p1_legal_inputs([4, 0])) == set(range(9)) - {4, 0}

    def test_p1_legal_inputs_empty_on_terminal(self, nfa):
        assert nfa.p1_legal_inputs([1, 0, 4, 3, 7]) == []

    def test_p2_legal_moves_empty_on_terminal(self, nfa):
        assert nfa.p2_legal_moves([7, 3, 2, 4, 0, 5]) == []


# ---------------------------------------------------------------------------
# is_terminal / current_player
# ---------------------------------------------------------------------------

class TestTerminalAndCurrentPlayer:
    def test_is_terminal_false_on_empty(self, nfa):
        assert not nfa.is_terminal([])

    def test_is_terminal_true_on_p1_win(self, nfa):
        # X connects top→bottom via cells 1,4,7
        assert nfa.is_terminal([1, 0, 4, 3, 7])

    def test_is_terminal_true_on_p2_win(self, nfa):
        # O connects left→right via cells 3,4,5
        assert nfa.is_terminal([7, 3, 2, 4, 0, 5])

    def test_current_player_p1_at_root(self, nfa):
        assert nfa.current_player([]) == 'P1'

    def test_current_player_p2_after_one_move(self, nfa):
        assert nfa.current_player([4]) == 'P2'

    def test_current_player_p1_after_two_moves(self, nfa):
        assert nfa.current_player([4, 0]) == 'P1'

    def test_current_player_none_on_terminal(self, nfa):
        assert nfa.current_player([1, 0, 4, 3, 7]) is None

    def test_current_player_none_on_invalid_trace(self, nfa):
        assert nfa.current_player([4, 4]) is None


# ---------------------------------------------------------------------------
# 2×2 board
# ---------------------------------------------------------------------------

class TestSmallNFA:
    def test_2x2_initial_children(self):
        nfa2 = HexNFA(size=2)
        assert len(nfa2.root.children) == 4
        assert set(nfa2.root.children.keys()) == {0, 1, 2, 3}

    def test_2x2_p1_win(self):
        nfa2 = HexNFA(size=2)
        # X at 0,2: 0 is top row, 2 is bottom row, they are neighbors
        assert nfa2.is_terminal([0, 1, 2])
        state = nfa2.get_node([0, 1, 2])
        assert state.winner() == 'P1'
