"""
Preference oracle for P2 (the system player).

The oracle is a black box.  Internally it uses cumulative node-value sums
to rank moves, but the public API exposes only:

  - preferred_move(prefix)  →  the action P2 chooses at this decision point
  - compare(trace1, trace2) →  which trace P2 prefers  ('t1', 't2', or 'equal')

No numeric scores are ever returned.  The SMT solver (elsewhere) derives
consistent numeric assignments from the collected pairwise orderings.
"""

from src.game.game_nfa import GameNFA


class PreferenceOracle:

    def __init__(self, nfa: GameNFA) -> None:
        self.nfa = nfa

    # ------------------------------------------------------------------
    # Internal scoring — never exposed
    # ------------------------------------------------------------------

    def _score(self, trace: list[str]) -> int:
        node = self.nfa.root
        total = node.value
        for action in trace:
            node = node.children[action]
            total += node.value
        return total

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preferred_move(self, prefix: list[str]) -> str | None:
        """
        At the P2 decision point reached by *prefix*, return the action
        P2 chooses — the move leading to the highest cumulative score.

        Used for L* membership queries.
        Returns None if *prefix* is not a P2 decision point.
        """
        moves = self.nfa.p2_legal_moves(prefix)
        if not moves:
            return None
        return max(moves, key=lambda m: self._score(prefix + [m]))

    def compare(self, trace1: list[str], trace2: list[str]) -> str:
        """
        Ordinal comparison of two traces.

        Returns
        -------
        't1'    trace1 is preferred
        't2'    trace2 is preferred
        'equal' no preference
        """
        s1 = self._score(trace1)
        s2 = self._score(trace2)
        if s1 > s2:
            return 't1'
        if s2 > s1:
            return 't2'
        return 'equal'


# ------------------------------------------------------------------
# Quick demo
# ------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    from src.game.game_generator import generate_tree, print_tree, compute_trace_scores
    from src.game.game_nfa import GameNFA

    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    seed  = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    root   = generate_tree(depth, seed=seed)
    nfa    = GameNFA(root)
    oracle = PreferenceOracle(nfa)

    print_tree(root)
    print()

    # Show P2's choice at every non-terminal P2 decision point
    print('P2 choices (membership query answers):')
    visited = set()
    for path, _ in compute_trace_scores(root):
        for i in range(len(path) + 1):
            prefix = tuple(path[:i])
            if prefix in visited:
                continue
            visited.add(prefix)

            node = nfa.get_node(list(prefix))
            if node is None or node.player != 'P2' or node.is_terminal():
                continue

            choice = oracle.preferred_move(list(prefix))
            prefix_str = ' → '.join(prefix) if prefix else 'ε'
            print(f'  {prefix_str:25s}  P2 chooses: {choice}')

    print()
    print('Pairwise comparisons (for MCTS):')
    traces = [path for path, _ in compute_trace_scores(root)]
    for i in range(min(4, len(traces))):
        for j in range(i + 1, min(4, len(traces))):
            result = oracle.compare(traces[i], traces[j])
            t1 = ' → '.join(traces[i])
            t2 = ' → '.join(traces[j])
            winner = t1 if result == 't1' else (t2 if result == 't2' else 'equal')
            print(f'  [{t1}]  vs  [{t2}]  →  {result}  (preferred: {winner})')
