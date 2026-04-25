"""
Unit tests for GameNFA (src/game/minimax/game_nfa.py).

Run Instructions:

'pytest tests/game/minimax/game_nfa.py -v'

"""

import pytest
from src.game.minimax.game_nfa import GameNFA
from src.game.minimax.game_generator import GameNode, generate_tree


@pytest.fixture
def nfa():
    root = generate_tree(depth=2, seed=42)
    return GameNFA(root)


class TestGetNode:
    """Tests for GameNFA.get_node"""
    pass


class TestIsTerminal:
    """Tests for GameNFA.is_terminal"""
    pass


class TestCurrentPlayer:
    """Tests for GameNFA.current_player"""
    pass


class TestP2LegalMoves:
    """Tests for GameNFA.p2_legal_moves"""
    pass


class TestP2Left:
    """Tests for GameNFA.p2_left"""
    pass


class TestP2Right:
    """Tests for GameNFA.p2_right"""
    pass


class TestP1LegalInputs:
    """Tests for GameNFA.p1_legal_inputs"""
    pass


class TestStep:
    """Tests for GameNFA.step"""
    pass


class TestDescribeState:
    """Tests for GameNFA.describe_state"""
    pass
