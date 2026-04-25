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
    """
    Tree Structure w/ seed=42, depth=2:

    ```
    - Root: P1, val=10                                                   
    - ['A'] → P2, val=1                                                  
    - ['B'] → P2, val=3                                                  
    - ['A','X'] → P1, val=0, terminal                                    
    - ['A','Y'] → P1, val=4, terminal                                    
    - ['B','X'] → P1, val=3, terminal                                    
    - ['B','Y'] → P1, val=2, terminal                                  
    ```

    """
    root = generate_tree(depth=2, seed=42)
    return GameNFA(root)


class TestGetNode:
    """Tests for GameNFA.get_node"""

    def test_valid_trace(self, nfa):
        # Case 1
        trace = ['B']
        node = nfa.get_node(trace)
        assert node is not None
        assert node.player == 'P2'
        assert node.value == 3

        # Case 2
        trace = ['A', 'X']
        node = nfa.get_node(trace)
        assert node is not None
        assert node.player == 'P1'
        assert node.value == 0

    def test_invalid_trace(self, nfa):
        trace = ['C']
        node = nfa.get_node(trace)
        assert node is None


class TestIsTerminal:
    """Tests for GameNFA.is_terminal"""
    
    def test_terminal_state(self, nfa):
        trace = ['A', 'X']
        assert nfa.is_terminal(trace) == True

    def test_non_terminal_state(self, nfa):
        trace = ['A']
        assert nfa.is_terminal(trace) == False


class TestCurrentPlayer:
    """Tests for GameNFA.current_player"""

    def test_valid_trace(self, nfa):
        # DEBUG
        print(f"Tree: {generate_tree(depth=2, seed=42)}")

        # P1 Case
        trace = ['A']
        assert nfa.current_player(trace) == 'P2'
    
        # P2 Case
        trace = ['A', 'X']
        assert nfa.current_player(trace) == None
        # Terminal State → No current player (It would be P2 if the game continued)

    def test_invalid_trace(self, nfa):
        trace = ['C']
        assert nfa.current_player(trace) is None


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
