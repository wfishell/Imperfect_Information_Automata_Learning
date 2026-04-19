"""
End-to-end tests for Dots and Boxes L* learner (src/scripts/learner_dab.py)
Run: python -m pytest tests/game/dots_and_boxes/test_learner_dab.py -v

These tests are slower (~seconds) — they run the full L* loop.
"""
import random
import pytest
from aalpy.learning_algs import run_Lstar

from src.game.dots_and_boxes.game_nfa import DotsAndBoxesNFA, PASS
from src.game.dots_and_boxes.preference_oracle import DotsAndBoxesOracle
from src.game.dots_and_boxes.dab_sul import DotsAndBoxesSUL
from src.lstar_mcts.table_b import TableB
from src.lstar_mcts.mcts_oracle import MCTSEquivalenceOracle
from src.scripts.learner_dab import evaluate_vs_random


@pytest.fixture(scope='module')
def learned_model():
    """Run L* once on 2×2 grid and share across tests in this module."""
    nfa     = DotsAndBoxesNFA(rows=2, cols=2)
    oracle  = DotsAndBoxesOracle(nfa)
    sul     = DotsAndBoxesSUL(nfa, oracle)
    table_b = TableB()
    eq      = MCTSEquivalenceOracle(
        sul=sul, nfa=nfa, oracle=oracle, table_b=table_b,
        depth_N=5, K=100, epsilon=0.05, verbose=False,
    )
    p1_inputs = list(nfa.root.children.keys()) + [PASS]

    model = run_Lstar(
        alphabet=p1_inputs,
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
# Strategy quality vs random P1
# ---------------------------------------------------------------------------

def test_beats_random_more_than_it_loses(learned_model):
    """
    Play 100 games: random P1 vs learned P2.
    P2 should win more games than it loses.
    Extra turns are handled via PASS: P1 inputs PASS when P2 earned an extra turn.
    """
    model, nfa = learned_model
    losses, draws, wins = evaluate_vs_random(model, nfa, n_games=100, seed=0)
    assert wins > losses, f"Expected wins > losses, got wins={wins} losses={losses}"

def test_loss_rate_below_threshold(learned_model):
    """Learned P2 should lose fewer than 20% of games against random P1."""
    model, nfa = learned_model
    losses, draws, wins = evaluate_vs_random(model, nfa, n_games=100, seed=0)
    assert losses < 20, f"Loss rate too high: {losses}/100"
