"""
Main entry point: learn P2's strategy automaton for a random minimax game.

Usage:
    python -m src.scripts.minimax.learner 4
    python -m src.scripts.minimax.learner 4 --seed 42 --depth-n 2 --K 100 --verbose

Outer loop:
  1. Run L* to convergence on the current SUL  -> hypothesis
  2. Run K MCTS rollouts against the hypothesis
       - If majority found: update SUL strategy at CE subtree, go to 1
       - If no majority:    done
"""

import argparse
from pathlib import Path

from src.game.minimax.game_generator import generate_tree, print_tree
from src.game.minimax.game_nfa import GameNFA
from src.game.minimax.preference_oracle import PreferenceOracle
from src.lstar_mcts.game_sul import GameSUL
from src.lstar_mcts.table_b import TableB
from src.lstar_mcts.mcts_oracle import MCTSEquivalenceOracle
from src.lstar_mcts.custom_lstar import MealyLStar


def main():
    parser = argparse.ArgumentParser(
        description='Learn P2 strategy automaton via L* + MCTS equivalence oracle.'
    )
    parser.add_argument('depth',     type=int,            help='Game tree depth')
    parser.add_argument('--seed',    type=int, default=0, help='Random seed')
    parser.add_argument('--depth-n', dest='depth_n', type=int, default=None,
                        help='MCTS search depth N (default: game depth)')
    parser.add_argument('--K',       type=int, default=200,
                        help='MCTS rollout budget per equivalence query')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    depth_n = args.depth_n if args.depth_n is not None else args.depth

    # ---------------------------------------------------------------
    # STEP 1: BUILD THE GAME TREE
    # ---------------------------------------------------------------
    print(f'Game tree  depth={args.depth}  seed={args.seed}')
    root    = generate_tree(args.depth, seed=args.seed)
    nfa     = GameNFA(root)
    oracle  = PreferenceOracle(nfa)
    table_b = TableB()
    sul     = GameSUL(nfa, oracle, table_b)
    p1_inputs = list(root.children.keys())

    mcts = MCTSEquivalenceOracle(
        sul     = sul,
        nfa     = nfa,
        oracle  = oracle,
        table_b = table_b,
        depth_N = depth_n,
        K       = args.K,
        verbose = args.verbose,
    )

    print(f'depth_n={depth_n}  K={args.K}')
    print()

    lstar = MealyLStar(
        alphabet  = p1_inputs,
        sul       = sul,
        eq_oracle = mcts,
        verbose   = args.verbose,
    )
    model = lstar.run()

    print()
    print('Learned automaton:')
    print(f'  States     : {len(model.states)}')
    print(f'  Membership : {sul.num_queries}')
    print(f'  Equivalence: {mcts.num_queries}')
    print()

    scores = evaluate(model, root, nfa)
    print(f'Score (mean over all P1 sequences):')
    print(f'  Optimal   : {scores["optimal_mean"]:.2f}')
    print(f'  Learned   : {scores["learned_mean"]:.2f}')
    print(f'  Random    : {scores["random_mean"]:.2f}')
    print(f'  Normalised: {scores["normalised"]:.3f}  (0=random, 1=optimal)')
    print()
    print(table_b.summary())

    out_dir = Path(__file__).parents[3] / 'outputs'
    out_dir.mkdir(exist_ok=True)
    model.save(str(out_dir / 'learned_strategy'))
    print(f'Saved: {out_dir / "learned_strategy.dot"}')


def evaluate(model, root, nfa) -> dict:
    import random as rng
    from src.game.minimax.game_generator import GameNode

    def optimal_score(node: GameNode) -> float:
        if node.is_terminal():
            return node.value
        child_scores = {a: optimal_score(c) for a, c in node.children.items()}
        if node.player == 'P2':
            return node.value + max(child_scores.values())
        return node.value + sum(child_scores.values()) / len(child_scores)

    def all_p1_sequences(node: GameNode, seq: list) -> list:
        if node.is_terminal():
            return [list(seq)]
        if node.player == 'P1':
            return [s for a, c in node.children.items()
                    for s in all_p1_sequences(c, seq + [a])]
        return [s for c in node.children.values()
                for s in all_p1_sequences(c, seq)]

    def score_for_strategy(p1_sequence: list, p2_choice_fn) -> float:
        node   = root
        trace  = []
        total  = node.value
        p1_idx = 0
        while not node.is_terminal():
            if node.player == 'P1':
                if p1_idx >= len(p1_sequence):
                    break
                action = p1_sequence[p1_idx]
                p1_idx += 1
            else:
                action = p2_choice_fn(trace)
            if action not in node.children:
                break
            trace.append(action)
            node   = node.children[action]
            total += node.value
        return total

    p1_seqs        = all_p1_sequences(root, [])
    optimal_scores = []
    learned_scores = []
    random_scores  = []

    for p1_seq in p1_seqs:
        def opt_p2(trace):
            node = nfa.get_node(trace)
            return None if node is None else max(
                node.children, key=lambda a: optimal_score(node.children[a]))

        model.reset_to_initial()
        p2_outputs = [model.step(p) for p in p1_seq]
        p2_iter    = iter(p2_outputs)

        def learned_p2(trace, it=p2_iter):
            return next(it, None)

        def rand_p2(trace):
            node = nfa.get_node(trace)
            return None if node is None else rng.choice(list(node.children.keys()))

        optimal_scores.append(score_for_strategy(p1_seq, opt_p2))
        learned_scores.append(score_for_strategy(p1_seq, learned_p2))
        random_scores.append(score_for_strategy(p1_seq, rand_p2))

    opt_mean = sum(optimal_scores) / len(optimal_scores)
    lrn_mean = sum(learned_scores) / len(learned_scores)
    rnd_mean = sum(random_scores)  / len(random_scores)

    normalised = (1.0 if opt_mean == rnd_mean
                  else (lrn_mean - rnd_mean) / (opt_mean - rnd_mean))

    return {
        'optimal_mean': opt_mean,
        'learned_mean': lrn_mean,
        'random_mean':  rnd_mean,
        'normalised':   normalised,
        'n_sequences':  len(p1_seqs),
    }


if __name__ == '__main__':
    main()
