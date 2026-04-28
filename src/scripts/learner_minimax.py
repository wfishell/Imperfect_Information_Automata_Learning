"""
Learn P2's strategy automaton for a random minimax game via L* + MCTS.

Usage:
    python -m src.scripts.learner_minimax --depth 4
    python -m src.scripts.learner_minimax --depth 4 --seed 42 --depth-n 2 --K 200
    python -m src.scripts.learner_minimax --depth 4 --viz
"""

import argparse
import random
from pathlib import Path

from src.game.minimax.game_generator import generate_tree, GameNode
from src.game.minimax.game_nfa import GameNFA
from src.game.minimax.preference_oracle import PreferenceOracle
from src.lstar_mcts.learner import run_lstar_mcts


def main():
    parser = argparse.ArgumentParser(
        description='Learn P2 strategy automaton for a minimax game via L* + MCTS.'
    )
    parser.add_argument('--depth',   type=int, default=4,  help='Game tree depth')
    parser.add_argument('--seed',    type=int, default=0,  help='RNG seed for tree generation')
    parser.add_argument('--depth-n', dest='depth_n', type=int, default=None,
                        help='MCTS search depth (default: game depth)')
    parser.add_argument('--K',       type=int, default=200,
                        help='MCTS rollout budget per equivalence query')
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--viz',     action='store_true',
                        help='Print enriched output: table B summary and automaton stats')
    args = parser.parse_args()

    depth_n = args.depth_n if args.depth_n is not None else args.depth

    root      = generate_tree(args.depth, seed=args.seed)
    nfa       = GameNFA(root)
    oracle    = PreferenceOracle(nfa)
    p1_inputs = list(root.children.keys())

    print(f'Minimax  depth={args.depth}  seed={args.seed}  depth_n={depth_n}  K={args.K}')
    print()

    model, sul, mcts, table_b = run_lstar_mcts(
        nfa       = nfa,
        oracle    = oracle,
        p1_inputs = p1_inputs,
        depth_n   = depth_n,
        K         = args.K,
        epsilon   = args.epsilon,
        verbose   = args.verbose,
    )

    scores = evaluate(model, root, nfa)

    print('Learned automaton:')
    print(f'  States       : {len(model.states)}')
    print(f'  Cache entries: {len(sul._cache)}')
    print(f'  Eq. queries  : {mcts.num_queries}')
    print()
    print('Score (mean over all P1 sequences):')
    print(f'  Optimal   : {scores["optimal_mean"]:.3f}')
    print(f'  Learned   : {scores["learned_mean"]:.3f}')
    print(f'  Random    : {scores["random_mean"]:.3f}')
    print(f'  Normalised: {scores["normalised"]:.3f}  (0=random  1=optimal)')

    if args.viz:
        print()
        print(table_b.summary())

    out_dir = Path(__file__).parents[2] / 'outputs'
    out_dir.mkdir(exist_ok=True)
    model.save(str(out_dir / 'learned_strategy_minimax'))
    print(f'\nSaved: {out_dir / "learned_strategy_minimax.dot"}')


def evaluate(model, root: GameNode, nfa: GameNFA) -> dict:
    """Normalised score vs optimal and random baselines."""
    rng = random.Random(0)

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
        node, trace, total, idx = root, [], root.value, 0
        while not node.is_terminal():
            if node.player == 'P1':
                if idx >= len(p1_sequence):
                    break
                action = p1_sequence[idx]; idx += 1
            else:
                action = p2_choice_fn(trace)
            if action not in node.children:
                break
            trace.append(action)
            node   = node.children[action]
            total += node.value
        return total

    p1_seqs = all_p1_sequences(root, [])
    optimal_scores, learned_scores, random_scores = [], [], []

    for p1_seq in p1_seqs:
        def opt_p2(trace):
            node = nfa.get_node(trace)
            return None if node is None else max(
                node.children, key=lambda a: optimal_score(node.children[a]))

        model.reset_to_initial()
        p2_outputs = [model.step(p) for p in p1_seq]
        it = iter(p2_outputs)

        def learned_p2(trace, _it=it):
            return next(_it, None)

        def rand_p2(trace):
            node = nfa.get_node(trace)
            return None if node is None else rng.choice(list(node.children.keys()))

        optimal_scores.append(score_for_strategy(p1_seq, opt_p2))
        learned_scores.append(score_for_strategy(p1_seq, learned_p2))
        random_scores.append(score_for_strategy(p1_seq, rand_p2))

    opt = sum(optimal_scores) / len(optimal_scores)
    lrn = sum(learned_scores) / len(learned_scores)
    rnd = sum(random_scores)  / len(random_scores)

    return {
        'optimal_mean': opt,
        'learned_mean': lrn,
        'random_mean':  rnd,
        'normalised':   1.0 if opt == rnd else (lrn - rnd) / (opt - rnd),
        'n_sequences':  len(p1_seqs),
    }


if __name__ == '__main__':
    main()
