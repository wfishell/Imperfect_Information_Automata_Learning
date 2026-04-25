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
    
    def test_is_terminal(self):
        node = GameNode(value=5, player='P2', depth=1)
        assert node.is_terminal() == True
        
        node.children['A'] = GameNode(value=3, player='P1', depth=2)
        assert node.is_terminal() == False

    def test_children_not_shared(self):
        # Ensure the dict default_factory creates a new dict for each instance
        node1 = GameNode(value=1, player='P1', depth=0)
        node2 = GameNode(value=2, player='P1', depth=0)
        node1.children['A'] = GameNode(value=3, player='P2', depth=1)
        assert 'A' not in node2.children


class TestGenerateTree:
    """Tests for generate_tree"""

    def test_root_is_p1_at_depth_zero(self):
        root = generate_tree(depth=2, seed=0)
        assert root.player == 'P1'
        assert root.depth == 0

    def test_players_alternate(self):
        root = generate_tree(depth=2, seed=0)
        for child in root.children.values():
            assert child.player == 'P2'
            for grandchild in child.children.values():
                assert grandchild.player == 'P1'

    def test_leaves_are_terminal_at_correct_depth(self):
        root = generate_tree(depth=2, seed=0)
        traces = compute_trace_scores(root)
        for _, _ in traces:
            pass  # just ensure no exception
        # All leaves reachable — verify via structure
        def check_leaves(node):
            if node.is_terminal():
                assert node.depth == 2
            for child in node.children.values():
                check_leaves(child)
        check_leaves(root)

    def test_branching_factor(self):
        root = generate_tree(depth=2, branching=3, seed=0)
        assert len(root.children) == 3
        for child in root.children.values():
            assert len(child.children) == 3

    def test_p1_actions_are_letters(self):
        root = generate_tree(depth=1, seed=0)
        assert set(root.children.keys()) == {'A', 'B'}

    def test_p2_actions_are_letters(self):
        root = generate_tree(depth=2, seed=0)
        for child in root.children.values():
            assert set(child.children.keys()) == {'X', 'Y'}

    def test_node_values_in_range(self):
        root = generate_tree(depth=3, seed=0)
        def check(node):
            assert 0 <= node.value <= 10
            for child in node.children.values():
                check(child)
        check(root)

    def test_same_seed_produces_same_tree(self):
        r1 = generate_tree(depth=2, seed=42)
        r2 = generate_tree(depth=2, seed=42)
        assert tree_to_dict(r1) == tree_to_dict(r2)

    def test_different_seeds_produce_different_trees(self):
        r1 = generate_tree(depth=2, seed=1)
        r2 = generate_tree(depth=2, seed=2)
        assert tree_to_dict(r1) != tree_to_dict(r2)


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
