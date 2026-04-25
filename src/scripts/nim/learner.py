"""
Learn P2's strategy automaton for Nim via L* + MCTS.

Usage:
    python -m src.scripts.nim.learner
    python -m src.scripts.nim.learner --piles 1 2 3 --depth-n 5 --K 200
    python -m src.scripts.nim.learner --oracle-depth 2 --verbose
"""

import argparse
import random
from pathlib import Path
from aalpy.learning_algs import run_Lstar

from src.game.nim.game_nfa import NimNFA
from src.game.nim.preference_oracle import NimOracle
from src.lstar_mcts.game_sul import GameSUL
from src.lstar_mcts.table_b import TableB
from src.lstar_mcts.mcts_oracle import MCTSEquivalenceOracle


MAX_ROUNDS = 10


def main():
    parser = argparse.ArgumentParser(
        description='Learn P2 strategy automaton for Nim via L* + MCTS.'
    )
    parser.add_argument('--piles',        type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--depth-n',      dest='depth_n',      type=int,   default=5)
    parser.add_argument('--K',            type=int,   default=200)
    parser.add_argument('--epsilon',      type=float, default=0.05)
    parser.add_argument('--oracle-depth', dest='oracle_depth', type=int,   default=None,
                        help='Minimax lookahead for oracle (default: None = optimal)')
    parser.add_argument('--verbose',      action='store_true')
    args = parser.parse_args()

    piles  = tuple(args.piles)
    nfa    = NimNFA(piles=piles)
    oracle = NimOracle(nfa, depth=args.oracle_depth)
    sul    = GameSUL(nfa, oracle)
    table_b = TableB()

    eq = MCTSEquivalenceOracle(
        sul=sul, nfa=nfa, oracle=oracle, table_b=table_b,
        depth_N=args.depth_n, K=args.K, epsilon=args.epsilon,
        verbose=args.verbose,
    )

    p1_inputs = nfa.alphabet

    print(f'Nim piles={list(piles)}  '
          f'oracle_depth={args.oracle_depth}  depth_n={args.depth_n}  K={args.K}')
    print()

    model = None
    for rnd in range(1, MAX_ROUNDS + 1):
        print(f'Round {rnd}', end='  ', flush=True)
        model = run_Lstar(
            alphabet=p1_inputs,
            sul=sul,
            eq_oracle=eq,
            automaton_type='mealy',
            print_level=2 if args.verbose else 0,
            cache_and_non_det_check=False,
        )

        remaining = {None: args.K}
        for _ in range(args.K):
            eq._rollout(model)

        improvement = eq._check_for_improvement(model)
        print(f'states={len(model.states)}', end='  ')

        if improvement is None:
            print('converged')
            break
        print('improvement found → re-learning')

    print()
    losses, draws, wins = evaluate_vs_random(model, nfa, n_games=200, seed=0)
    n = losses + draws + wins
    print(f'Evaluation vs random P1 (200 games):')
    print(f'  wins={wins}  draws={draws}  losses={losses}')
    print(f'  win rate={wins/n:.1%}  loss rate={losses/n:.1%}')

    out_dir = Path(__file__).parents[3] / 'outputs'
    out_dir.mkdir(exist_ok=True)
    out = out_dir / 'learned_strategy_nim.dot'
    model.save(str(out_dir / 'learned_strategy_nim'))
    print(f'Saved: {out}')


def evaluate_vs_random(model, nfa: NimNFA,
                       n_games: int, seed: int) -> tuple[int, int, int]:
    """Return (losses, draws, wins) for learned P2 vs random P1."""
    rng = random.Random(seed)
    losses = draws = wins = 0

    for _ in range(n_games):
        state = nfa.root
        model.reset_to_initial()

        while not state.is_terminal():
            p1_move = rng.choice(list(state.children.keys()))
            p2_move = model.step(p1_move)
            state   = state.children[p1_move]

            if state.is_terminal():
                break

            if p2_move not in state.children:
                p2_move = rng.choice(list(state.children.keys()))
            state = state.children[p2_move]

        w = state.winner()
        if w == 'P1':   losses += 1
        elif w == 'P2': wins   += 1
        else:           draws  += 1

    return losses, draws, wins


if __name__ == '__main__':
    main()
