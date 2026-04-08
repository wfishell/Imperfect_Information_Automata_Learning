"""
Tests for the Dots and Boxes 2x2 NFA.
"""
import pytest
from dots_and_boxes import (
    DotsAndBoxesNFA, INITIAL_STATE, EDGE_NAMES, BOXES,
    transition, legal_moves, is_terminal, winner, NUM_EDGES,
)


@pytest.fixture
def nfa():
    return DotsAndBoxesNFA()


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------

def test_initial_state(nfa):
    assert nfa.initial_state.claimed == frozenset()
    assert nfa.initial_state.player == 0
    assert nfa.initial_state.score == (0, 0)


def test_alphabet_size(nfa):
    assert len(nfa.alphabet) == 12


def test_all_edges_legal_at_start(nfa):
    moves = nfa.legal_moves_at(nfa.initial_state)
    assert sorted(moves) == list(range(12))


def test_no_legal_moves_at_terminal(nfa):
    # Build a terminal state by claiming all edges in some valid order
    state = nfa.initial_state
    for e in range(12):
        if not is_terminal(state):
            state = transition(state, e)
    # After all 12 edges, no legal moves
    assert nfa.legal_moves_at(state) == []


def test_dead_state_has_no_moves(nfa):
    assert nfa.legal_moves_at(None) == []


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------

def test_illegal_move_returns_none(nfa):
    state = nfa.initial_state
    state2 = nfa.delta(state, 0)
    # Claiming edge 0 again is illegal
    assert nfa.delta(state2, 0) is None


def test_legal_move_removes_from_available(nfa):
    state = nfa.delta(nfa.initial_state, 0)
    assert 0 not in nfa.legal_moves_at(state)
    assert len(nfa.legal_moves_at(state)) == 11


def test_player_switches_without_box(nfa):
    # Claiming a single edge that does not complete a box switches player
    state = nfa.delta(nfa.initial_state, 0)  # claim H00 — no box yet
    assert state.player == 1


def test_player_stays_on_box_completion(nfa):
    # Box A = {0, 2, 6, 7}. Claim 3 of its edges, then the 4th.
    state = nfa.initial_state
    for e in [0, 2, 6]:          # P1 claims 3 edges, alternating with P2
        # track who actually moves
        state = transition(state, e)
    # Now claim edge 7 — whoever's turn it is completes Box A
    player_before = state.player
    state = transition(state, 7)
    assert state.player == player_before   # same player goes again
    assert state.score[player_before] == 1  # that player scored


def test_score_increments_on_box(nfa):
    # Claim all 4 edges of Box A: {0, 2, 6, 7}
    # Go in an order that keeps it simple: P1 takes 0, P2 takes 2, P1 takes 6, P2 takes 7
    state = nfa.initial_state
    state = transition(state, 0)  # P1
    state = transition(state, 2)  # P2
    state = transition(state, 6)  # P1
    state = transition(state, 7)  # P2 completes Box A → P2 scores
    assert state.score == (0, 1)
    assert state.player == 1      # P2 goes again


# ---------------------------------------------------------------------------
# Box definitions
# ---------------------------------------------------------------------------

def test_all_boxes_have_four_edges():
    for box in BOXES:
        assert len(box) == 4


def test_box_edges_are_valid_indices():
    for box in BOXES:
        for e in box:
            assert 0 <= e < NUM_EDGES


def test_shared_edges():
    # V01 (edge 7) is shared by Box A and Box B
    assert 7 in BOXES[0] and 7 in BOXES[1]
    # H10 (edge 2) is shared by Box A and Box C
    assert 2 in BOXES[0] and 2 in BOXES[2]
    # V11 (edge 10) is shared by Box C and Box D
    assert 10 in BOXES[2] and 10 in BOXES[3]
    # H11 (edge 3) is shared by Box B and Box D
    assert 3 in BOXES[1] and 3 in BOXES[3]


# ---------------------------------------------------------------------------
# Acceptance — valid complete games
# ---------------------------------------------------------------------------

def test_accepts_simple_sequential_game(nfa):
    # Claim all 12 edges in order 0..11 — always legal since no repeats
    word = list(range(12))
    assert nfa.accepts(word)


def test_rejects_repeated_edge(nfa):
    word = [0, 0] + list(range(1, 11))
    assert not nfa.accepts(word)


def test_rejects_incomplete_game(nfa):
    word = list(range(11))   # only 11 edges
    assert not nfa.accepts(word)


def test_rejects_empty_word(nfa):
    assert not nfa.accepts([])


def test_run_length(nfa):
    word = list(range(12))
    states = nfa.run(word)
    assert len(states) == 13  # initial + one per symbol


def test_terminal_after_12_moves(nfa):
    word = list(range(12))
    states = nfa.run(word)
    assert is_terminal(states[-1])


# ---------------------------------------------------------------------------
# Winner
# ---------------------------------------------------------------------------

def test_winner_p1(nfa):
    # Force P1 to complete all 4 boxes
    # Simplest: P1 takes edge 0, P2 takes some non-box edge,
    # then engineer so P1 completes every box.
    # Instead just check the winner() function directly on constructed scores.
    from dots_and_boxes import GameState
    state = GameState(frozenset(range(12)), player=0, score=(3, 1))
    assert winner(state) == 0


def test_winner_p2(nfa):
    from dots_and_boxes import GameState
    state = GameState(frozenset(range(12)), player=0, score=(1, 3))
    assert winner(state) == 1


def test_draw(nfa):
    from dots_and_boxes import GameState
    state = GameState(frozenset(range(12)), player=0, score=(2, 2))
    assert winner(state) is None


# ---------------------------------------------------------------------------
# Language size sanity check
# ---------------------------------------------------------------------------

def test_language_is_nonempty(nfa):
    # At minimum, the simple sequential game is accepted
    assert nfa.accepts(list(range(12)))


def test_all_permutations_without_box_bonus_are_valid(nfa):
    """
    Any permutation of 0..11 where NO box is ever completed mid-game
    is a valid game. Check a few.
    """
    import itertools
    # Edges that appear in NO box: none — all edges border at least one box.
    # But games without box completions mid-sequence are still valid as long as
    # they complete all edges. Just check a handful of permutations.
    count = 0
    for perm in itertools.permutations(range(12)):
        if nfa.accepts(list(perm)):
            count += 1
        if count >= 10:
            break
    assert count >= 10  # many valid games exist


# ---------------------------------------------------------------------------
# Display (smoke test — just ensure no crash)
# ---------------------------------------------------------------------------

def test_display_does_not_crash(nfa, capsys):
    state = nfa.delta(nfa.initial_state, 0)
    nfa.display_state(state)
    nfa.display_state(None)
    captured = capsys.readouterr()
    assert len(captured.out) > 0
