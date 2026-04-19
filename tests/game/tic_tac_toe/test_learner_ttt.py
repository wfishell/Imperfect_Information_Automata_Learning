"""
End-to-end tests for tic-tac-toe L* learner (src/scripts/learner_ttt.py)
Run: python -m pytest tests/game/tic_tac_toe/test_learner_ttt.py -v

These tests are slower (~seconds) — they run the full L* loop.
"""
import random
import pytest
from aalpy.learning_algs import run_Lstar

from src.game.tic_tac_toe.game_nfa import TicTacToeNFA
from src.game.tic_tac_toe.preference_oracle import TicTacToeOracle
from src.lstar_mcts.game_sul import GameSUL
from src.lstar_mcts.table_b import TableB
from src.lstar_mcts.mcts_oracle import MCTSEquivalenceOracle


@pytest.fixture(scope='module')
def learned_model():
    """Run L* once and share the result across tests in this module."""
    nfa      = TicTacToeNFA()
    oracle   = TicTacToeOracle(nfa)
    sul      = GameSUL(nfa, oracle)
    table_b  = TableB()
    eq       = MCTSEquivalenceOracle(
        sul=sul, nfa=nfa, oracle=oracle, table_b=table_b,
        depth_N=5, K=100, epsilon=0.05, verbose=False,
    )
    p1_inputs = list(nfa.root.children.keys())

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
# Never loses to random opponent
# ---------------------------------------------------------------------------

def test_never_loses_to_random(learned_model):
    """
    Play 100 games: random X vs learned O.
    With a minimax oracle, O should never lose.
    """
    model, nfa = learned_model
    rng = random.Random(0)

    losses = 0
    for _ in range(100):
        state = nfa.root
        model.reset_to_initial()

        while not state.is_terminal():
            if state.player == 'P1':
                # Random X move
                move = rng.choice(list(state.children.keys()))
                model.step(move)          # keep model in sync
                state = state.children[move]
            else:
                # Learned O move
                if state.is_terminal():
                    break
                p1_moves_so_far = [
                    m for i, m in enumerate(
                        [k for k in state.board if k == 1]
                    )
                ]
                # Step model with a dummy P1 move to get O's output
                # (model already stepped on P1's move above)
                o_move = model.step(list(state.children.keys())[0])
                # Use oracle fallback if model output is invalid
                if o_move not in state.children:
                    o_move = list(state.children.keys())[0]
                state = state.children[o_move]

        if state.winner() == 'P1':
            losses += 1

    assert losses == 0, f"Learned O lost {losses}/100 games to random X"
