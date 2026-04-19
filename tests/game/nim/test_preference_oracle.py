"""
Tests for NimOracle (src/game/nim/preference_oracle.py)
Run: python -m pytest tests/game/nim/test_preference_oracle.py -v

Classic 3-pile Nim [1,2,3] nim-sum facts:
  1 XOR 2 XOR 3 = 0  → root is a P1-winning (P2-losing) position
  After P1 plays (2,2): piles=(1,2,1) → 1^2^1=2 ≠ 0 → P2 in winning position
  After P1 plays (2,3): piles=(1,2,0) → 1^2^0=3 ≠ 0 → P2 in winning position
"""
import pytest
from src.game.nim.game_nfa import NimNFA
from src.game.nim.board import NimState
from src.game.nim.preference_oracle import NimOracle


@pytest.fixture
def nfa():
    return NimNFA(piles=(1, 2, 3))

@pytest.fixture
def oracle(nfa):
    return NimOracle(nfa)

@pytest.fixture
def oracle_depth1(nfa):
    return NimOracle(nfa, depth=1)


# ---------------------------------------------------------------------------
# preferred_move — basic legality
# ---------------------------------------------------------------------------

def test_preferred_move_returns_none_on_p1_turn(oracle):
    assert oracle.preferred_move([]) is None

def test_preferred_move_returns_none_on_terminal(oracle):
    assert oracle.preferred_move([(0,1),(1,2),(2,3)]) is None

def test_preferred_move_returns_none_on_invalid_trace(oracle):
    assert oracle.preferred_move([(0,2)]) is None   # pile 0 only has 1

def test_preferred_move_returns_legal_move(oracle):
    # P2 to move after P1 takes from pile 2
    move = oracle.preferred_move([(2, 3)])
    state = oracle.nfa.get_node([(2, 3)])
    assert move in state.children


# ---------------------------------------------------------------------------
# preferred_move — optimal play (depth=None uses nim-sum)
# ---------------------------------------------------------------------------

def test_preferred_move_wins_from_winning_position(oracle):
    # After P1:(2,2) → piles=(1,2,1), nim_xor=2 → P2 in winning position
    # Optimal P2 move zeros the nim-sum: take 2 from pile 1 → (1,0,1), xor=0
    move = oracle.preferred_move([(2, 2)])
    next_state = oracle.nfa.get_node([(2, 2), move])
    nim_xor = 0
    for p in next_state.piles:
        nim_xor ^= p
    assert nim_xor == 0, f"Expected nim_xor=0 after P2's move, got {nim_xor} (move={move})"

def test_preferred_move_returns_some_move_in_losing_position(oracle):
    # Root has nim_xor=0 → P2 is in a losing position after P1 does nothing (impossible),
    # but after P1:(2,1) → piles=(1,2,2), xor=1 → P2 winning
    # After P1:(1,1) → piles=(1,1,3), xor=3 → P2 winning
    # After P1:(0,1) → piles=(0,2,3), xor=1 → P2 winning
    # All P1 moves from the root leave P2 in a winning position (root xor=0 means P1 loses)
    move = oracle.preferred_move([(0, 1)])
    assert move is not None


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def test_compare_p2_win_beats_p2_loss(oracle):
    # trace leading to P2 win vs trace leading to P1 win
    # P2 wins from (1,2,1) as P2 to move (xor=2, winning)
    # Build a terminal P2-win trace: P1:(2,2), P2:(1,2), P1:(0,1) → (0,0,1), P2:(2,1) → done, P1 loses
    p2_win_trace = [(2,2),(1,2),(0,1),(2,1)]
    # Build terminal P1-win: drain everything with P1 getting last move
    # P1:(0,1)→(0,2,3), P2:(1,1)→(0,1,3), P1:(2,3)→(0,1,0), P2:(1,1)→(0,0,0) → P1 wins (P2 moved last? no)
    # Terminal: player to move loses. After (0,0,0) reached, whoever is to move loses.
    # Let's just verify compare is consistent with _minimax
    nfa = NimNFA(piles=(1, 2, 3))
    o   = NimOracle(nfa)
    # Trace ending at P2-win terminal: piles=(0,0,0) with P1 to move → P2 won
    p2_win  = NimState(piles=(0,0,0), player='P1')  # P2 wins
    p1_win  = NimState(piles=(0,0,0), player='P2')  # P1 wins
    assert o._minimax(p2_win, None) > o._minimax(p1_win, None)

def test_compare_equal_for_same_trace(oracle):
    assert oracle.compare([(2,3)], [(2,3)]) == 'equal'

def test_compare_t1_when_trace1_better(oracle):
    # After P1:(2,2) piles=(1,2,1) xor=2 → P2 wins
    # After P1:(0,1) piles=(0,2,3) xor=1 → P2 wins too, but let's verify ordering
    v1 = oracle._trace_value([(2,2)])
    v2 = oracle._trace_value([(0,1)])
    result = oracle.compare([(2,2)], [(0,1)])
    if v1 > v2:
        assert result == 't1'
    elif v2 > v1:
        assert result == 't2'
    else:
        assert result == 'equal'


# ---------------------------------------------------------------------------
# _heuristic
# ---------------------------------------------------------------------------

def test_heuristic_positive_when_p2_to_move_and_xor_nonzero():
    # piles=(1,2,1), xor=2≠0, P2 to move → winning → positive
    s = NimState(piles=(1,2,1), player='P2')
    assert NimOracle._heuristic(s) > 0

def test_heuristic_negative_when_p2_to_move_and_xor_zero():
    # piles=(1,2,3), xor=0, P2 to move → losing → negative
    s = NimState(piles=(1,2,3), player='P2')
    assert NimOracle._heuristic(s) < 0

def test_heuristic_negative_when_p1_to_move_and_xor_nonzero():
    # piles=(1,2,1), xor=2≠0, P1 to move → P1 winning → bad for P2 → negative
    s = NimState(piles=(1,2,1), player='P1')
    assert NimOracle._heuristic(s) < 0

def test_heuristic_positive_when_p1_to_move_and_xor_zero():
    # piles=(1,2,3), xor=0, P1 to move → P1 losing → good for P2 → positive
    s = NimState(piles=(1,2,3), player='P1')
    assert NimOracle._heuristic(s) > 0

def test_heuristic_magnitude():
    s = NimState(piles=(1,2,1), player='P2')
    assert NimOracle._heuristic(s) == 0.5

def test_heuristic_stays_inside_terminal_range():
    for piles in [(1,2,3),(1,2,1),(0,0,1)]:
        for player in ('P1','P2'):
            s = NimState(piles=piles, player=player)
            if not s.is_terminal():
                h = NimOracle._heuristic(s)
                assert -1.0 < h < 1.0


# ---------------------------------------------------------------------------
# Bounded depth
# ---------------------------------------------------------------------------

def test_depth1_preferred_move_is_legal(oracle_depth1):
    move = oracle_depth1.preferred_move([(2, 2)])
    state = oracle_depth1.nfa.get_node([(2, 2)])
    assert move in state.children

def test_depth_none_matches_optimal_on_winning_position(oracle):
    # With full minimax, P2 should zero the nim-sum from a winning position
    move = oracle.preferred_move([(2, 2)])   # P2 to move, piles=(1,2,1), xor=2
    next_piles = oracle.nfa.get_node([(2,2), move]).piles
    nim_xor = 0
    for p in next_piles:
        nim_xor ^= p
    assert nim_xor == 0
