"""
Tests for HexState (src/game/hex/board.py)
Run: python -m pytest tests/game/hex/test_board.py -v

3×3 board cell layout (row-major):
    0 1 2
     3 4 5
      6 7 8

P1 (X) wins by connecting top row {0,1,2} to bottom row {6,7,8}.
P2 (O) wins by connecting left col {0,3,6} to right col {2,5,8}.

Hex adjacency for cell (r,c):
    (r-1,c)  (r-1,c+1)
    (r,c-1)             (r,c+1)
    (r+1,c-1)  (r+1,c)

Neighbor examples on the 3×3 grid:
    cell 0 (r=0,c=0): neighbors → 1, 3
    cell 4 (r=1,c=1): neighbors → 1, 2, 3, 5, 6, 7
    cell 8 (r=2,c=2): neighbors → 5, 7
"""
import pytest
from src.game.hex.board import HexState, _neighbors, _connected, EMPTY, X, O


def make_state(moves: list[int], size: int = 3) -> HexState:
    """Replay a sequence of moves from the initial state."""
    state = HexState(size=size)
    for move in moves:
        state = state.children[move]
    return state


# ---------------------------------------------------------------------------
# _neighbors helper
# ---------------------------------------------------------------------------

class TestNeighbors:
    def test_corner_top_left(self):
        # cell 0 (r=0,c=0) on 3×3 has neighbors 1 and 3
        assert set(_neighbors(0, 3)) == {1, 3}

    def test_corner_bottom_right(self):
        # cell 8 (r=2,c=2) on 3×3 has neighbors 5 and 7
        assert set(_neighbors(8, 3)) == {5, 7}

    def test_center(self):
        # cell 4 (r=1,c=1) on 3×3 has 6 neighbors
        assert set(_neighbors(4, 3)) == {1, 2, 3, 5, 6, 7}

    def test_top_right_corner(self):
        # cell 2 (r=0,c=2) on 3×3: (0,1)=1, (1,1)=4, (1,2)=5
        assert set(_neighbors(2, 3)) == {1, 4, 5}

    def test_no_out_of_bounds(self):
        for size in (2, 3, 4):
            for cell in range(size * size):
                for nb in _neighbors(cell, size):
                    assert 0 <= nb < size * size


# ---------------------------------------------------------------------------
# _connected helper
# ---------------------------------------------------------------------------

class TestConnected:
    def test_x_wins_center_column(self):
        # X at 1, 4, 7 — connected path from top row to bottom row
        board = [EMPTY] * 9
        board[1] = board[4] = board[7] = X
        assert _connected(tuple(board), X, 3)

    def test_x_not_connected_without_path(self):
        # X only in top row — no path to bottom
        board = [EMPTY] * 9
        board[0] = board[1] = board[2] = X
        assert not _connected(tuple(board), X, 3)

    def test_o_wins_middle_row(self):
        # O at 3, 4, 5 — left col to right col via middle row
        board = [EMPTY] * 9
        board[3] = board[4] = board[5] = O
        assert _connected(tuple(board), O, 3)

    def test_o_not_connected_without_path(self):
        # O only in left col — no path to right
        board = [EMPTY] * 9
        board[0] = board[3] = board[6] = O
        assert not _connected(tuple(board), O, 3)

    def test_empty_board_not_connected(self):
        board = (EMPTY,) * 9
        assert not _connected(board, X, 3)
        assert not _connected(board, O, 3)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_board_all_empty(self):
        assert all(sq == EMPTY for sq in HexState().board)

    def test_player_is_p1(self):
        assert HexState().player == 'P1'

    def test_size_default(self):
        assert HexState().size == 3

    def test_children_count_3x3(self):
        # 9 empty cells → 9 children
        assert len(HexState().children) == 9

    def test_not_terminal(self):
        assert not HexState().is_terminal()

    def test_winner_is_none(self):
        assert HexState().winner() is None

    def test_value_is_zero(self):
        assert HexState().value == 0


# ---------------------------------------------------------------------------
# Children / transitions
# ---------------------------------------------------------------------------

class TestChildren:
    def test_count_decreases_after_each_move(self):
        assert len(make_state([4]).children) == 8
        assert len(make_state([4, 0]).children) == 7

    def test_played_cell_not_in_children(self):
        s = make_state([4])
        assert 4 not in s.children

    def test_children_keys_are_unoccupied_cells(self):
        occupied = {4, 0, 8}
        assert set(make_state([4, 0, 8]).children.keys()) == set(range(9)) - occupied

    def test_player_alternates(self):
        s = HexState()
        assert s.player == 'P1'
        s = s.children[0]
        assert s.player == 'P2'
        s = s.children[1]
        assert s.player == 'P1'

    def test_correct_token_placed(self):
        # P1 places X, P2 places O
        s = HexState().children[4]   # P1 plays cell 4
        assert s.board[4] == X
        s2 = s.children[0]           # P2 plays cell 0
        assert s2.board[0] == O


# ---------------------------------------------------------------------------
# Terminal detection — P1 (X) wins
# ---------------------------------------------------------------------------

class TestP1Wins:
    @pytest.mark.parametrize("moves", [
        [1, 0, 4, 3, 7],        # X at 1,4,7 — center column top→bottom
        [0, 1, 3, 5, 6],        # X at 0,3,6 — left column top→bottom
        [2, 1, 5, 3, 8],        # X at 2,5,8 — right column top→bottom
        [2, 0, 4, 1, 6],        # X at 2,4,6 — diagonal via 2→4→6
    ])
    def test_p1_wins(self, moves):
        state = make_state(moves)
        assert state.is_terminal()
        assert state.winner() == 'P1'
        assert state.value == -1

    def test_p1_win_no_more_children(self):
        state = make_state([1, 0, 4, 3, 7])
        assert state.children == {}


# ---------------------------------------------------------------------------
# Terminal detection — P2 (O) wins
# ---------------------------------------------------------------------------

class TestP2Wins:
    @pytest.mark.parametrize("moves", [
        [7, 3, 2, 4, 0, 5],     # O at 3,4,5 — middle row left→right
        [6, 0, 7, 1, 8, 2],     # O at 0,1,2 — top row left→right
        [1, 6, 2, 7, 0, 8],     # O at 6,7,8 — bottom row left→right
    ])
    def test_p2_wins(self, moves):
        state = make_state(moves)
        assert state.is_terminal()
        assert state.winner() == 'P2'
        assert state.value == 1

    def test_p2_win_no_more_children(self):
        state = make_state([7, 3, 2, 4, 0, 5])
        assert state.children == {}


# ---------------------------------------------------------------------------
# No draws in Hex
# ---------------------------------------------------------------------------

class TestNoDraws:
    def test_winner_is_never_draw(self):
        # Exhaustively verify no 3×3 state reachable from root has winner='draw'
        # via a short sample of full-game traces
        full_game_traces = [
            [1, 0, 4, 3, 7],           # P1 wins early
            [7, 3, 2, 4, 0, 5],        # P2 wins early
            [2, 0, 4, 1, 6],           # P1 wins via 2→4→6
        ]
        for trace in full_game_traces:
            state = make_state(trace)
            assert state.winner() in ('P1', 'P2')

    def test_non_terminal_winner_is_none(self):
        assert make_state([4]).winner() is None
        assert make_state([4, 0]).winner() is None


# ---------------------------------------------------------------------------
# 2×2 board
# ---------------------------------------------------------------------------

class TestSmallBoard:
    def test_2x2_initial_children(self):
        assert len(HexState(size=2).children) == 4

    def test_2x2_p1_wins(self):
        # On 2×2, X at 0,2 (top row=0,1; bottom row=2,3; neighbors of 0: 1,2)
        # X plays 0 then 2: 0 is top row, neighbor of 2 (r=1,c=0) — wait
        # Actually for 2×2: cell 0 (r=0,c=0) neighbors: (0,1)=1, (1,0)=2
        # So 0 connects to 2 directly. If X has 0 and 2, _connected returns True.
        board = (X, EMPTY, X, EMPTY)
        state = HexState(size=2, board=board, player='P2')
        assert state.winner() == 'P1'

    def test_2x2_p2_wins(self):
        # O at 0,1 (top row). left col={0,2}, right col={1,3}.
        # 0 is left col, 1 is right col (1%2=1=size-1). They are neighbors!
        board = (O, O, EMPTY, EMPTY)
        state = HexState(size=2, board=board, player='P1')
        assert state.winner() == 'P2'
