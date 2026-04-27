"""
Visualized strategy learner — identical algorithm to learner.py,
with Rich terminal output at each step.

Usage:
    python -m src.scripts.random_game.learner_viz 4
    python -m src.scripts.random_game.learner_viz 4 --seed 42 --depth-n 2 --K 100
    python -m src.scripts.random_game.learner_viz 6 --tree-depth 3   # show 3 levels of game tree
"""

import argparse
from pathlib import Path
from aalpy.learning_algs import run_Lstar

from src.game.minimax.game_generator import generate_tree, print_tree
from src.game.minimax.game_nfa import GameNFA
from src.game.minimax.preference_oracle import PreferenceOracle
from src.lstar_mcts.game_sul import GameSUL
from src.lstar_mcts.table_b import TableB
from src.lstar_mcts.mcts_oracle import MCTSEquivalenceOracle
from src.scripts.minimax.learner import evaluate
from src.viz.visualizer import Visualizer


def main():
    parser = argparse.ArgumentParser(
        description='L* + MCTS strategy learner with Rich visualisation.'
    )
    parser.add_argument('depth',      type=int,            help='Game tree depth')
    parser.add_argument('--seed',     type=int, default=0, help='Random seed')
    parser.add_argument('--depth-n',  dest='depth_n', type=int, default=None,
                        help='MCTS search depth N (default: game depth)')
    parser.add_argument('--K',        type=int,   default=200,
                        help='MCTS rollout budget per equivalence query')
    parser.add_argument('--epsilon',  type=float, default=0.05,
                        help='Min advantage to accept a counterexample')
    parser.add_argument('--tree-depth', dest='tree_depth', type=int, default=3,
                        help='How many levels of the game tree to display (default: 3)')
    args = parser.parse_args()

    depth_n = args.depth_n if args.depth_n is not None else args.depth
    viz     = Visualizer()

    # ------------------------------------------------------------------
    # Build game and show tree
    # ------------------------------------------------------------------
    root = generate_tree(args.depth, seed=args.seed)
    viz.show_game_tree(root, max_depth=args.tree_depth)

    # ------------------------------------------------------------------
    # Component setup  (identical to learner.py)
    # ------------------------------------------------------------------
    nfa     = GameNFA(root)
    oracle  = PreferenceOracle(nfa)
    sul     = GameSUL(nfa, oracle)
    table_b = TableB()

    eq_oracle = MCTSEquivalenceOracle(
        sul=sul, nfa=nfa, oracle=oracle, table_b=table_b,
        depth_N=depth_n, K=args.K, epsilon=args.epsilon,
        verbose=False,
    )

    p1_inputs = list(root.children.keys())

    # ------------------------------------------------------------------
    # Main loop  (identical logic to learner.py, with viz hooks)
    # ------------------------------------------------------------------
    round_num = 0
    model     = None

    while True:
        round_num += 1
        viz.show_round_header(round_num)

        # L* convergence
        model = run_Lstar(
            alphabet=p1_inputs,
            sul=sul,
            eq_oracle=eq_oracle,
            automaton_type='mealy',
            print_level=0,
            cache_and_non_det_check=False,
        )
        viz.show_hypothesis(model, p1_inputs)

        # MCTS rollouts
        for _ in range(args.K):
            eq_oracle._rollout(model)

        viz.show_table_b(eq_oracle.table_b)
        viz.show_deviations(eq_oracle._deviation_leaves)

        # Check for improvement
        improvement = eq_oracle._check_for_improvement(model)

        # Scores
        scores = evaluate(model, root)
        viz.show_scores(scores, round_num)

        viz.show_improvement(improvement)

        if improvement is None:
            break

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    viz.show_final_summary(model, sul, eq_oracle, table_b)

    diagrams_dir = Path(__file__).parents[1] / 'viz' / 'diagrams'
    diagrams_dir.mkdir(parents=True, exist_ok=True)
    save_path = diagrams_dir / f'learned_strategy_d{args.depth}_s{args.seed}'
    model.save(str(save_path))
    viz.console.print(f"[dim]Saved: {save_path}.dot[/dim]")


if __name__ == '__main__':
    main()
