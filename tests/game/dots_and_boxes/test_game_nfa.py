"""
Tests for DotsAndBoxesNFA (src/game/dots_and_boxes/game_nfa.py)
Run: python -m pytest tests/game/dots_and_boxes/test_game_nfa.py -v

Edge layout for the default 2×2 grid — see test_board.py for full reference.
  box(0,0): edges 0, 2, 6, 7
  box(0,1): edges 1, 3, 7, 8
  box(1,0): edges 2, 4, 9, 10
  box(1,1): edges 3, 5, 10, 11
"""

import pytest
from src.game.dots_and_boxes.game_nfa import DotsAndBoxesNFA, PASS


@pytest.fixture
def nfa():
    return DotsAndBoxesNFA()


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

class TestNFARoot:
    def test_root_is_initial_state(self, nfa):
        assert nfa.root.player == 'P1'
        assert not any(nfa.root.edges)

    def test_root_has_12_children_2x2(self, nfa):
        assert len(nfa.root.children) == 12

    def test_alphabet_is_all_12_edges(self, nfa):
        assert set(nfa.root.children.keys()) == set(range(12))

    def test_p1_alphabet_includes_pass(self, nfa):
        assert PASS in nfa.p1_alphabet
        assert set(nfa.p1_alphabet) == set(range(12)) | {PASS}


# ---------------------------------------------------------------------------
# get_node
# ---------------------------------------------------------------------------

class TestGetNode:
    def test_empty_trace_returns_root(self, nfa):
        assert nfa.get_node([]) is nfa.root

    def test_one_move(self, nfa):
        state = nfa.get_node([0])
        assert state is not None
        assert state.edges[0] is True
        assert state.player == 'P2'

    def test_two_moves(self, nfa):
        state = nfa.get_node([0, 1])
        assert state.edges[0] is True
        assert state.edges[1] is True
        assert state.player == 'P1'

    def test_illegal_repeated_edge_returns_none(self, nfa):
        assert nfa.get_node([0, 0]) is None

    def test_illegal_edge_in_sequence_returns_none(self, nfa):
        assert nfa.get_node([0, 1, 0]) is None

    def test_pass_navigates_forced_pass_state(self, nfa):
        # After P2 completes box(0,0), get_node with PASS should advance
        # to P2's real extra-turn state.
        state = nfa.get_node([0, 2, 6, 7, PASS])
        assert state is not None
        assert state.player == 'P2'
        assert state.forced_pass is False

    def test_pass_illegal_on_non_forced_state(self, nfa):
        # PASS is not a valid move when the state is not forced-pass
        assert nfa.get_node([0, PASS]) is None


# ---------------------------------------------------------------------------
# Forced-pass mechanic
# ---------------------------------------------------------------------------

class TestForcedPassMechanic:
    def test_p2_box_completion_yields_p1_forced_pass(self, nfa):
        # P2 completes box(0,0) via edges 0,2,6,7 → state is P1-forced
        state = nfa.get_node([0, 2, 6, 7])
        assert state is not None
        assert state.player == 'P1'
        assert state.forced_pass is True
        assert state.p2_boxes == 1

    def test_p1_box_completion_yields_p2_forced_pass(self, nfa):
        # Arrange so P1 draws the 4th side of box(0,0)
        state = nfa.get_node([0, 1, 2, 3, 6, 8, 7])
        assert state is not None
        assert state.player == 'P2'
        assert state.forced_pass is True
        assert state.p1_boxes == 2

    def test_double_completion_also_forced(self, nfa):
        # Edge 7 completes both box(0,0) and box(0,1) for P1
        state = nfa.get_node([0, 1, 2, 3, 6, 8, 7])
        assert state.p1_boxes == 2
        assert state.forced_pass is True


# ---------------------------------------------------------------------------
# p1_legal_inputs / p2_legal_moves
# ---------------------------------------------------------------------------

class TestLegalInputs:
    def test_p1_legal_inputs_at_root(self, nfa):
        assert set(nfa.p1_legal_inputs([])) == set(range(12))

    def test_p2_legal_moves_after_one_p1_move(self, nfa):
        assert set(nfa.p2_legal_moves([0])) == set(range(12)) - {0}

    def test_p1_legal_inputs_pass_when_forced(self, nfa):
        # After P2 completes box(0,0), P1 is forced to pass
        assert nfa.p1_legal_inputs([0, 2, 6, 7]) == [PASS]

    def test_p1_legal_inputs_empty_on_p2_turn(self, nfa):
        # When it is P2's real turn, P1 has no legal inputs
        assert nfa.p1_legal_inputs([0]) == []

    def test_p2_legal_moves_empty_on_p1_turn(self, nfa):
        assert nfa.p2_legal_moves([]) == []

    def test_p2_legal_moves_pass_when_forced(self, nfa):
        # After P1 completes two boxes, P2 is forced to pass
        assert nfa.p2_legal_moves([0, 1, 2, 3, 6, 8, 7]) == [PASS]

    def test_p1_legal_inputs_shrink_after_moves(self, nfa):
        assert set(nfa.p1_legal_inputs([0, 1])) == set(range(12)) - {0, 1}

    def test_p2_legal_moves_after_forced_pass(self, nfa):
        # After P2 completes box(0,0) and P1 passes, P2 has real moves
        moves = nfa.p2_legal_moves([0, 2, 6, 7, PASS])
        assert len(moves) == 8   # 12 - 4 drawn edges
        assert 7 not in moves    # already drawn


# ---------------------------------------------------------------------------
# is_terminal / current_player
# ---------------------------------------------------------------------------

class TestTerminalAndCurrentPlayer:
    def test_is_terminal_false_on_empty(self, nfa):
        assert not nfa.is_terminal([])

    def test_is_terminal_false_mid_game(self, nfa):
        assert not nfa.is_terminal([0, 1, 2])

    def test_current_player_p1_at_root(self, nfa):
        assert nfa.current_player([]) == 'P1'

    def test_current_player_p2_after_p1_move(self, nfa):
        assert nfa.current_player([0]) == 'P2'

    def test_current_player_p1_forced_after_p2_box(self, nfa):
        # P2 completes box(0,0); P1 is the forced-pass player next
        assert nfa.current_player([0, 2, 6, 7]) == 'P1'
        state = nfa.get_node([0, 2, 6, 7])
        assert state.forced_pass is True

    def test_current_player_none_on_terminal(self, nfa):
        state = nfa.root
        remaining = list(range(12))
        trace = []
        while not state.is_terminal():
            if state.forced_pass:
                state = state.children[PASS]
                trace.append(PASS)
                continue
            e = next(e for e in remaining if e in state.children)
            state = state.children[e]
            trace.append(e)
            remaining.remove(e)
        assert nfa.current_player(trace) is None


# ---------------------------------------------------------------------------
# 3×3 grid
# ---------------------------------------------------------------------------

class TestGridSizes:
    def test_3x3_root_has_24_children(self):
        nfa3 = DotsAndBoxesNFA(rows=3, cols=3)
        assert len(nfa3.root.children) == 24

    def test_3x3_alphabet_is_all_24_edges(self):
        nfa3 = DotsAndBoxesNFA(rows=3, cols=3)
        assert set(nfa3.root.children.keys()) == set(range(24))

    def test_3x3_p1_alphabet_includes_pass(self):
        nfa3 = DotsAndBoxesNFA(rows=3, cols=3)
        assert PASS in nfa3.p1_alphabet
