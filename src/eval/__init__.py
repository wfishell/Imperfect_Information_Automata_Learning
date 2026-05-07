"""
Evaluation harness for L*+MCTS-learned controllers.

Subpackages provide P1 (environment) player strategies for each game.
Each player exposes a class with `.pick(node) -> action` so it can be
plugged into a uniform evaluation loop:

    from src.eval.minimax.p1_random import RandomP1

    p1 = RandomP1(seed=0)
    action = p1.pick(current_node)

Three P1 strategies per game:
  - random : uniform over legal actions.
  - greedy : one-step adversarial heuristic.
  - optimal: full minimax-style tree solve (perfect adversary).
"""
