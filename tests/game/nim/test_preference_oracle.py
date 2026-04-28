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
# preferred_move — boundary conditions
# ---------------------------------------------------------------------------

class TestPreferredMoveBoundary:
    def test_returns_none_on_p1_turn(self, oracle):
        assert oracle.preferred_move([]) is None

    def test_returns_none_on_terminal(self, oracle):
        assert oracle.preferred_move([(0, 1), (1, 2), (2, 3)]) is None

    def test_returns_none_on_invalid_trace(self, oracle):
        assert oracle.preferred_move([(0, 2)]) is None   # pile 0 only has 1


# ---------------------------------------------------------------------------
# preferred_move — legality and optimal play
# ---------------------------------------------------------------------------

class TestPreferredMove:
    def test_returns_legal_move(self, oracle):
        move = oracle.preferred_move([(2, 3)])
        state = oracle.nfa.get_node([(2, 3)])
        assert move in state.children

    def test_zeros_nim_sum_from_winning_position(self, oracle):
        # After P1:(2,2) → piles=(1,2,1), xor=2 → P2 in winning position
        # Optimal P2 move zeros the nim-sum: take 2 from pile 1 → (1,0,1), xor=0
        move = oracle.preferred_move([(2, 2)])
        next_state = oracle.nfa.get_node([(2, 2), move])
        nim_xor = 0
        for p in next_state.piles:
            nim_xor ^= p
        assert nim_xor == 0

    def test_returns_some_move_from_losing_position(self, oracle):
        # Root xor=0 → all P1 moves leave P2 in a winning position
        move = oracle.preferred_move([(0, 1)])
        assert move is not None


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

class TestCompare:
    def test_p2_win_beats_p1_win(self, oracle):
        p2_win = NimState(piles=(0, 0, 0), player='P1')   # P2 wins
        p1_win = NimState(piles=(0, 0, 0), player='P2')   # P1 wins
        assert oracle._minimax(p2_win, None) > oracle._minimax(p1_win, None)

    def test_equal_for_same_trace(self, oracle):
        assert oracle.compare([(2, 3)], [(2, 3)]) == 'equal'

    def test_ordering_consistent_with_trace_value(self, oracle):
        v1 = oracle._trace_value([(2, 2)])
        v2 = oracle._trace_value([(0, 1)])
        result = oracle.compare([(2, 2)], [(0, 1)])
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
    def test_positive_when_p2_to_move(self):
        # P2 to move with non-empty piles → always positive
        assert NimOracle._heuristic(NimState(piles=(1, 2, 3), player='P2')) > 0

    def test_negative_when_p1_to_move(self):
        # P1 to move → always negative (score is from P2's perspective)
        assert NimOracle._heuristic(NimState(piles=(1, 2, 3), player='P1')) < 0

    def test_dominant_pile_scores_higher_than_balanced(self):
        # (3,0,0): max/total=1.0 → higher score than (1,1,1): max/total=1/3
        s_dominant = NimState(piles=(3, 0, 0), player='P2')
        s_balanced  = NimState(piles=(1, 1, 1), player='P2')
        assert NimOracle._heuristic(s_dominant) > NimOracle._heuristic(s_balanced)

    def test_magnitude(self):
        # piles=(1,2,1): max=2, total=4, (2/4)*0.9 = 0.45
        assert NimOracle._heuristic(NimState(piles=(1, 2, 1), player='P2')) == pytest.approx(0.45)

    def test_stays_inside_terminal_range(self):
        # 0.9 scaling ensures max/total=1.0 edge case never reaches ±1.0
        for piles in [(1, 2, 3), (1, 2, 1), (0, 0, 1)]:
            for player in ('P1', 'P2'):
                s = NimState(piles=piles, player=player)
                if not s.is_terminal():
                    assert -1.0 < NimOracle._heuristic(s) < 1.0


# ---------------------------------------------------------------------------
# Bounded depth
# ---------------------------------------------------------------------------

class TestBoundedDepth:
    def test_depth1_preferred_move_is_legal(self, oracle_depth1):
        move = oracle_depth1.preferred_move([(2, 2)])
        state = oracle_depth1.nfa.get_node([(2, 2)])
        assert move in state.children

    def test_depth_none_zeros_nim_sum_on_winning_position(self, oracle):
        # Full minimax should zero the nim-sum from a winning position
        move = oracle.preferred_move([(2, 2)])   # piles=(1,2,1), xor=2
        next_piles = oracle.nfa.get_node([(2, 2), move]).piles
        nim_xor = 0
        for p in next_piles:
            nim_xor ^= p
        assert nim_xor == 0
