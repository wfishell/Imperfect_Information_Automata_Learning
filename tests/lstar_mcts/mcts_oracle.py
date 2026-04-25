"""
Unit tests for MCTSEquivalenceOracle (src/lstar_mcts/mcts_oracle.py).

Run Instructions:

'pytest tests/lstar_mcts/mcts_oracle.py -v'

"""

import pytest
from unittest.mock import MagicMock
from src.lstar_mcts.mcts_oracle import MCTSEquivalenceOracle
from src.game.minimax.game_generator import generate_tree
from src.game.minimax.game_nfa import GameNFA
from src.lstar_mcts.preference_oracle import PreferenceOracle
from src.lstar_mcts.game_sul import GameSUL
from src.lstar_mcts.table_b import TableB


@pytest.fixture
def oracle():
    root  = generate_tree(depth=2, seed=42)
    nfa   = GameNFA(root)
    pref  = PreferenceOracle(nfa)
    sul   = GameSUL(nfa, pref)
    table = TableB()
    return MCTSEquivalenceOracle(sul, nfa, pref, table, depth_N=2)

# Will
class TestCheckForImprovement:
    """Tests for MCTSEquivalenceOracle._check_for_improvement"""
    pass


class TestFindCex:
    """Tests for MCTSEquivalenceOracle.find_cex"""
    pass

# Will
class TestRollout:
    """Tests for MCTSEquivalenceOracle._rollout"""
    pass


class TestBudgetCheck:
    """Tests for MCTSEquivalenceOracle._budget_check"""
    pass

# Will
class TestShadowTrace:
    """Tests for MCTSEquivalenceOracle._shadow_trace"""
    pass

# Will
class TestCollectTableALeaves:
    """Tests for MCTSEquivalenceOracle._collect_table_a_leaves"""
    pass


class TestHypothesisOutput:
    """Tests for MCTSEquivalenceOracle._hypothesis_output
    
    Asserts that the method:

    1. Strips player 2 inputs from trace.
    2. Resets the hypothesis to initial state.
    3. Steps through the hypothesis using the player 1 inputs, returning the final output.

    Specifically looks at minimax as test:
    ```
        Hypothesis: digraph learnedModel {
            s0 [label="s0"];
            s1 [label="s1"];
            s2 [label="s2"];
            s0 -> s1 [label="A/Y"];
            s0 -> s1 [label="B/X"];
            s1 -> s2 [label="A/Y"];
            s1 -> s2 [label="B/X"];
            s2 -> s2 [label="A/None"];
            s2 -> s2 [label="B/None"];
            __start0 [shape=none, label=""];
            __start0 -> s0 [label=""];
        }
    ```
    """

    def test_returns_last_step_output(self, oracle):
        hypothesis = MagicMock()

        # Simulate Hypothesis - 1
        hypothesis.step.side_effect = ['X']

        result = oracle._hypothesis_output(hypothesis, ['B'])
        assert result == 'X'

        # Simulate Hypothesis - 2
        hypothesis.step.side_effect = ['X', 'Y']

        result = oracle._hypothesis_output(hypothesis, ['B', 'X', 'A'])
        assert result == 'Y'


    def test_returns_none_on_exception(self, oracle):
        hypothesis = MagicMock()

        # Simulate Hypothesis that raises an exception
        hypothesis.step.side_effect = Exception("Hypothesis error")

        result = oracle._hypothesis_output(hypothesis, ['B', 'X', 'A'])
        assert result is None


class TestPruneAndUpdateTableB:
    """Tests for MCTSEquivalenceOracle._prune_and_update_table_b"""
    pass
