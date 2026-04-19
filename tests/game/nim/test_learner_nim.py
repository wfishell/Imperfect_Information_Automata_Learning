"""
End-to-end tests for Nim L* learner (src/scripts/learner_nim.py)
Run: python -m pytest tests/game/nim/test_learner_nim.py -v
"""
import pytest
from aalpy.learning_algs import run_Lstar

from src.game.nim.game_nfa import NimNFA
from src.game.nim.preference_oracle import NimOracle
from src.lstar_mcts.game_sul import GameSUL
from src.lstar_mcts.table_b import TableB
from src.lstar_mcts.mcts_oracle import MCTSEquivalenceOracle
from src.scripts.learner_nim import evaluate_vs_random


@pytest.fixture(scope='module')
def learned_model():
    """Run L* once on [1,2,3] piles and share across tests in this module."""
    nfa     = NimNFA(piles=(1, 2, 3))
    oracle  = NimOracle(nfa)
    sul     = GameSUL(nfa, oracle)
    table_b = TableB()
    eq      = MCTSEquivalenceOracle(
        sul=sul, nfa=nfa, oracle=oracle, table_b=table_b,
        depth_N=5, K=100, epsilon=0.05, verbose=False,
    )

    model = run_Lstar(
        alphabet=nfa.alphabet,
        sul=sul,
        eq_oracle=eq,
        automaton_type='mealy',
        print_level=0,
        cache_and_non_det_check=False,
    )
    return model, nfa


# ---------------------------------------------------------------------------
# Basic sanity
# ---------------------------------------------------------------------------

def test_model_has_states(learned_model):
    model, _ = learned_model
    assert len(model.states) >= 1

def test_model_has_initial_state(learned_model):
    model, _ = learned_model
    assert model.initial_state is not None


# ---------------------------------------------------------------------------
# evaluate_vs_random output contract
# ---------------------------------------------------------------------------

def test_evaluate_counts_sum_to_n_games(learned_model):
    model, nfa = learned_model
    losses, draws, wins = evaluate_vs_random(model, nfa, n_games=50, seed=42)
    assert losses + draws + wins == 50

def test_evaluate_counts_non_negative(learned_model):
    model, nfa = learned_model
    losses, draws, wins = evaluate_vs_random(model, nfa, n_games=50, seed=42)
    assert losses >= 0 and draws >= 0 and wins >= 0


# ---------------------------------------------------------------------------
# Strategy quality
# ---------------------------------------------------------------------------

def test_beats_random_more_than_it_loses(learned_model):
    """Learned P2 should win more than it loses against random P1."""
    model, nfa = learned_model
    losses, draws, wins = evaluate_vs_random(model, nfa, n_games=100, seed=0)
    assert wins > losses, f"Expected wins > losses, got wins={wins} losses={losses}"

def test_win_rate_above_threshold(learned_model):
    """Learned P2 should win more than 80% of games against random P1."""
    model, nfa = learned_model
    losses, draws, wins = evaluate_vs_random(model, nfa, n_games=100, seed=0)
    assert wins >= 80, f"Win rate too low: {wins}/100"
