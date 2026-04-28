"""
Tests for run_lstar_mcts (src/lstar_mcts/learner.py)
Run: python -m pytest tests/lstar_mcts/test_learner.py -v

These tests verify the API contract of run_lstar_mcts() — return types,
model structure, and query accounting — using a small minimax game (depth=2)
so the full learning loop completes quickly.  Strategy quality is intentionally
out of scope here; game-specific learner scripts carry those tests.
"""
import pytest

from src.game.minimax.game_generator import generate_tree
from src.game.minimax.game_nfa import GameNFA
from src.game.minimax.preference_oracle import PreferenceOracle
from src.lstar_mcts.learner import run_lstar_mcts
from src.lstar_mcts.game_sul import GameSUL
from src.lstar_mcts.mcts_oracle import MCTSEquivalenceOracle
from src.lstar_mcts.table_b import TableB
from src.lstar_mcts.custom_lstar import MealyMachine


@pytest.fixture(scope='module')
def learning_result():
    """Run the full L* + MCTS loop once on a depth-2 minimax tree."""
    root      = generate_tree(depth=2, seed=0)
    nfa       = GameNFA(root)
    oracle    = PreferenceOracle(nfa)
    p1_inputs = list(root.children.keys())
    return run_lstar_mcts(nfa, oracle, p1_inputs, depth_n=2, K=50), nfa, p1_inputs


# ---------------------------------------------------------------------------
# Return value contract
# ---------------------------------------------------------------------------

class TestReturnContract:
    def test_returns_four_elements(self, learning_result):
        result, _, _ = learning_result
        assert len(result) == 4

    def test_model_is_mealy_machine(self, learning_result):
        (model, *_), _, _ = learning_result
        assert isinstance(model, MealyMachine)

    def test_sul_is_game_sul(self, learning_result):
        (_, sul, *_), _, _ = learning_result
        assert isinstance(sul, GameSUL)

    def test_mcts_is_equivalence_oracle(self, learning_result):
        (_, _, mcts, _), _, _ = learning_result
        assert isinstance(mcts, MCTSEquivalenceOracle)

    def test_table_b_is_table_b(self, learning_result):
        (_, _, _, table_b), _, _ = learning_result
        assert isinstance(table_b, TableB)


# ---------------------------------------------------------------------------
# Model structure
# ---------------------------------------------------------------------------

class TestModelStructure:
    def test_has_at_least_one_state(self, learning_result):
        (model, *_), _, _ = learning_result
        assert len(model.states) >= 1

    def test_has_initial_state(self, learning_result):
        (model, *_), _, _ = learning_result
        assert model.initial_state is not None

    def test_initial_state_in_states(self, learning_result):
        (model, *_), _, _ = learning_result
        assert model.initial_state in model.states

    def test_every_state_has_transition_for_every_input(self, learning_result):
        (model, *_), _, p1_inputs = learning_result
        for state in model.states:
            for inp in p1_inputs:
                assert inp in state.transitions, \
                    f'State {state} missing transition for input {inp!r}'

    def test_all_transition_targets_are_valid_states(self, learning_result):
        (model, *_), _, _ = learning_result
        valid = set(model.states)
        for state in model.states:
            for inp, (out, dst) in state.transitions.items():
                assert dst in valid


# ---------------------------------------------------------------------------
# Query accounting
# ---------------------------------------------------------------------------

class TestQueryAccounting:
    def test_membership_cache_populated(self, learning_result):
        # MealyLStar calls sul.pre()/step() directly, not sul.query(), so
        # AALpy's num_queries counter stays 0.  The internal cache is the
        # correct proxy: it accumulates one entry per membership query step.
        (_, sul, *_), _, _ = learning_result
        assert len(sul._cache) > 0

    def test_equivalence_queries_positive(self, learning_result):
        (_, _, mcts, _), _, _ = learning_result
        assert mcts.num_queries > 0


# ---------------------------------------------------------------------------
# Output legality
# ---------------------------------------------------------------------------

class TestOutputLegality:
    def test_outputs_are_legal_p2_moves(self, learning_result):
        """Step through every P1 input from root; each output must be a
        valid P2 move at that point in the NFA."""
        (model, *_), nfa, p1_inputs = learning_result
        model.reset_to_initial()
        trace = []
        for p1 in p1_inputs:
            output = model.step(p1)
            trace.append(p1)
            legal = nfa.p2_legal_moves(trace)
            if legal:   # non-terminal after P1 move
                assert output in legal, \
                    f'Illegal P2 output {output!r} at trace {trace}; legal={legal}'
            trace.append(output)
