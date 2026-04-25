"""
Unit tests for src/game/minimax/game_generator.py.

Run Instructions:

'pytest tests/game/minimax/game_generator.py -v'

"""

import pytest
from src.game.minimax.game_generator import (
    GameNode,
    generate_tree,
    compute_trace_scores,
    print_tree,
    tree_to_dict,
    tree_from_dict,
)


class TestGameNode:
    """Tests for GameNode dataclass and its is_terminal method"""
    
    def test_fields(self):
        node = GameNode(value=10, player='P1', depth=2)
        assert node.value == 10
        assert node.player == 'P1'
        assert node.depth == 2
        assert node.children == {}


class TestGenerateTree:
    """Tests for generate_tree"""
    pass


class TestComputeTraceScores:
    """Tests for compute_trace_scores"""
    pass


class TestPrintTree:
    """Tests for print_tree"""
    pass


class TestTreeToDict:
    """Tests for tree_to_dict"""
    pass


class TestTreeFromDict:
    """Tests for tree_from_dict"""
    pass
