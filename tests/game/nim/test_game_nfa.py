"""
Tests for NimNFA (src/game/nim/game_nfa.py)
Run: python -m pytest tests/game/nim/test_game_nfa.py -v
"""
import pytest
from src.game.nim.game_nfa import NimNFA


@pytest.fixture
def nfa():
    return NimNFA(piles=(1, 2, 3))


# ---------------------------------------------------------------------------
# Root / alphabet
# ---------------------------------------------------------------------------

def test_root_is_initial_state(nfa):
    assert nfa.root.piles == (1, 2, 3)
    assert nfa.root.player == 'P1'

def test_alphabet_equals_root_children_keys(nfa):
    assert set(nfa.alphabet) == set(nfa.root.children.keys())

def test_alphabet_size_equals_sum_piles(nfa):
    assert len(nfa.alphabet) == 6   # sum([1,2,3])

def test_alphabet_contains_all_moves(nfa):
    expected = {(0,1), (1,1),(1,2), (2,1),(2,2),(2,3)}
    assert set(nfa.alphabet) == expected


# ---------------------------------------------------------------------------
# get_node
# ---------------------------------------------------------------------------

def test_get_node_empty_trace_returns_root(nfa):
    assert nfa.get_node([]) is nfa.root

def test_get_node_single_move(nfa):
    state = nfa.get_node([(2, 3)])
    assert state.piles == (1, 2, 0)
    assert state.player == 'P2'

def test_get_node_two_moves(nfa):
    state = nfa.get_node([(2, 3), (1, 2)])
    assert state.piles == (1, 0, 0)
    assert state.player == 'P1'

def test_get_node_illegal_move_returns_none(nfa):
    # pile 0 has size 1; can't remove 2
    assert nfa.get_node([(0, 2)]) is None

def test_get_node_exhausted_pile_returns_none(nfa):
    # pile 1 taken, then try to take from it again
    assert nfa.get_node([(1, 2), (1, 1)]) is None

def test_get_node_past_terminal_returns_none(nfa):
    # take everything then try another move
    full_clear = [(0,1), (1,2), (2,3)]
    # after P1 takes pile0, P2 takes pile1, P1 takes pile2 → terminal
    trace = [(0,1), (1,2), (2,3), (0,1)]
    assert nfa.get_node(trace) is None


# ---------------------------------------------------------------------------
# p1_legal_inputs
# ---------------------------------------------------------------------------

def test_p1_legal_inputs_at_root(nfa):
    moves = nfa.p1_legal_inputs([])
    assert set(moves) == {(0,1),(1,1),(1,2),(2,1),(2,2),(2,3)}

def test_p1_legal_inputs_empty_on_p2_turn(nfa):
    assert nfa.p1_legal_inputs([(0, 1)]) == []

def test_p1_legal_inputs_empty_on_terminal(nfa):
    # drain all piles: P1→(0,1), P2→(1,2), P1→(2,3) → terminal, P2 to move
    assert nfa.p1_legal_inputs([(0,1),(1,2),(2,3)]) == []


# ---------------------------------------------------------------------------
# p2_legal_moves
# ---------------------------------------------------------------------------

def test_p2_legal_moves_after_first_move(nfa):
    moves = nfa.p2_legal_moves([(0, 1)])
    # pile 0 now 0; legal: pile1 (1,2 remain) + pile2 (1,2,3)
    expected = {(1,1),(1,2),(2,1),(2,2),(2,3)}
    assert set(moves) == expected

def test_p2_legal_moves_empty_on_p1_turn(nfa):
    assert nfa.p2_legal_moves([]) == []

def test_p2_legal_moves_empty_on_terminal(nfa):
    assert nfa.p2_legal_moves([(0,1),(1,2),(2,3)]) == []


# ---------------------------------------------------------------------------
# is_terminal / current_player
# ---------------------------------------------------------------------------

def test_is_terminal_false_at_root(nfa):
    assert not nfa.is_terminal([])

def test_is_terminal_true_when_all_taken(nfa):
    assert nfa.is_terminal([(0,1),(1,2),(2,3)])

def test_current_player_p1_at_root(nfa):
    assert nfa.current_player([]) == 'P1'

def test_current_player_p2_after_p1_move(nfa):
    assert nfa.current_player([(0,1)]) == 'P2'

def test_current_player_none_on_terminal(nfa):
    assert nfa.current_player([(0,1),(1,2),(2,3)]) is None


# ---------------------------------------------------------------------------
# Different pile configuration
# ---------------------------------------------------------------------------

def test_single_pile_alphabet():
    nfa = NimNFA(piles=(3,))
    assert set(nfa.alphabet) == {(0,1),(0,2),(0,3)}

def test_larger_piles_alphabet_size():
    nfa = NimNFA(piles=(2, 3, 4))
    assert len(nfa.alphabet) == 9   # sum([2,3,4])
