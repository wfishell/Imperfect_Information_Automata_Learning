"""
Simple NFA for P2 (the system player) in a minimax game tree.

States  = nodes in the game tree, identified by the trace prefix that reaches them.
Alphabet = P1 inputs (A, B, ...) interleaved with P2 responses (X, Y, ...).

At every P2 state, P2's legal moves are simply the children of that node:
  - left  child  (first  action, e.g. X)
  - right child  (second action, e.g. Y)

At every P1 state, P1's input determines which branch is taken — P2 has no choice.
"""

from src.game.minimax.game_generator import GameNode


class GameNFA:
    """
    NFA representing P2's legal moves at each reachable game state.

    The NFA is derived directly from the game tree.  A 'state' is a
    GameNode; it is uniquely identified by the trace (sequence of actions)
    that leads to it from the root.
    """

    def __init__(self, root: GameNode) -> None:
        self.root = root

    # ------------------------------------------------------------------
    # Core navigation
    # ------------------------------------------------------------------

    def get_node(self, trace: list[str]) -> GameNode | None:
        """Return the GameNode reached by following *trace* from the root."""
        node = self.root
        for action in trace:
            if action not in node.children:
                return None
            node = node.children[action]
        return node

    def is_terminal(self, trace: list[str]) -> bool:
        node = self.get_node(trace)
        return node is not None and node.is_terminal()

    def current_player(self, trace: list[str]) -> str | None:
        """Return 'P1' or 'P2' for the player to move, or None if terminal."""
        node = self.get_node(trace)
        if node is None or node.is_terminal():
            return None
        return node.player

    # ------------------------------------------------------------------
    # P2 legal moves
    # ------------------------------------------------------------------

    def p2_legal_moves(self, trace: list[str]) -> list[str]:
        """
        Return P2's legal moves at the state reached by *trace*.
        Returns [] if the state is not a P2 node or is terminal.
        """
        node = self.get_node(trace)
        if node is None or node.player != 'P2' or node.is_terminal():
            return []
        return list(node.children.keys())

    def p2_left(self, trace: list[str]) -> str | None:
        """Return P2's left-child action at *trace*, or None."""
        moves = self.p2_legal_moves(trace)
        return moves[0] if moves else None

    def p2_right(self, trace: list[str]) -> str | None:
        """Return P2's right-child action at *trace*, or None."""
        moves = self.p2_legal_moves(trace)
        return moves[1] if len(moves) > 1 else None

    # ------------------------------------------------------------------
    # P1 legal inputs
    # ------------------------------------------------------------------

    def p1_legal_inputs(self, trace: list[str]) -> list[str]:
        """
        Return the P1 inputs available at the state reached by *trace*.
        Returns [] if the state is not a P1 node or is terminal.
        """
        node = self.get_node(trace)
        if node is None or node.player != 'P1' or node.is_terminal():
            return []
        return list(node.children.keys())

    # ------------------------------------------------------------------
    # Transition
    # ------------------------------------------------------------------

    def step(self, trace: list[str], action: str) -> list[str] | None:
        """
        Take *action* from the state identified by *trace*.
        Returns the new trace, or None if the action is not legal.
        """
        node = self.get_node(trace)
        if node is None or action not in node.children:
            return None
        return trace + [action]

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def describe_state(self, trace: list[str]) -> str:
        node = self.get_node(trace)
        if node is None:
            return 'INVALID'
        trace_str = ' → '.join(trace) if trace else 'ε'
        if node.is_terminal():
            return f'{trace_str}  [{node.player}] val={node.value}  TERMINAL'
        moves = self.p2_legal_moves(trace) or self.p1_legal_inputs(trace)
        return (f'{trace_str}  [{node.player}] val={node.value}  '
                f'legal={moves}')


# ------------------------------------------------------------------
# Quick demo
# ------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    from src.game.minimax.game_generator import generate_tree, print_tree

    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    seed  = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    root = generate_tree(depth, seed=seed)
    nfa  = GameNFA(root)

    print_tree(root)
    print()

    # Walk a few traces and show legal moves at each step
    sample_traces = [
        [],
        ['A'],
        ['A', 'X'],
        ['A', 'X', 'A'],
        ['B'],
        ['B', 'Y'],
        ['B', 'Y', 'B'],
    ]

    print('State descriptions:')
    for t in sample_traces:
        print(' ', nfa.describe_state(t))
