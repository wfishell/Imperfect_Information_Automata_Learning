"""
Unit tests for MCTSEquivalenceOracle (src/lstar_mcts/mcts_oracle.py).

Run Instructions:

'pytest tests/lstar_mcts/mcts_oracle.py -v'

"""

import pytest
from src.lstar_mcts.mcts_oracle import MCTSEquivalenceOracle


class TestCheckForImprovement:
    """Tests for MCTSEquivalenceOracle._check_for_improvement"""
    pass


class TestFindCex:
    """Tests for MCTSEquivalenceOracle.find_cex"""
    pass


class TestRollout:
    """Tests for MCTSEquivalenceOracle._rollout"""
    pass


class TestBudgetCheck:
    """Tests for MCTSEquivalenceOracle._budget_check"""
    pass


class TestShadowTrace:
    """Tests for MCTSEquivalenceOracle._shadow_trace"""
    pass


class TestCollectTableALeaves:
    """Tests for MCTSEquivalenceOracle._collect_table_a_leaves"""
    pass


class TestHypothesisOutput:
    """Tests for MCTSEquivalenceOracle._hypothesis_output"""
    pass


class TestCollectDepthNLeaves:
    """Tests for MCTSEquivalenceOracle._collect_depth_n_leaves"""
    pass


class TestPruneAndUpdateTableB:
    """Tests for MCTSEquivalenceOracle._prune_and_update_table_b"""
    pass
