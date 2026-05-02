"""
Learn P2's strategy automaton for Nim via L* + MCTS.

Usage:
    python -m src.scripts.learner_nim
    python -m src.scripts.learner_nim --piles 1 2 3 --K 200
    python -m src.scripts.learner_nim --oracle-depth 2 --viz
"""

import argparse
import random
from pathlib import Path

from src.game.nim.game_nfa import NimNFA
from src.game.nim.preference_oracle import NimOracle
from src.lstar_mcts.learner import run_lstar_mcts


def main():
    parser = argparse.ArgumentParser(
        description='Learn P2 strategy automaton for Nim via L* + MCTS.'
    )
    parser.add_argument('--piles',        type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--depth-n',      dest='depth_n',      type=int,   default=5)
    parser.add_argument('--K',            type=int,   default=200)
    parser.add_argument('--oracle-depth', dest='oracle_depth', type=int,   default=None,
                        help='Minimax lookahead for oracle (default: None = full search)')
    parser.add_argument('--verbose',      action='store_true')
    parser.add_argument('--viz',          action='store_true',
                        help='Print enriched output: table B summary')
    args = parser.parse_args()

    piles  = tuple(args.piles)
    nfa    = NimNFA(piles=piles)
    oracle = NimOracle(nfa, depth=args.oracle_depth)

    print(f'Nim  piles={list(piles)}  oracle_depth={args.oracle_depth}  '
          f'depth_n={args.depth_n}  K={args.K}')
    print()

    model, sul, mcts, table_b = run_lstar_mcts(
        nfa       = nfa,
        oracle    = oracle,
        p1_inputs = nfa.alphabet,
        depth_n   = args.depth_n,
        K         = args.K,
        verbose   = args.verbose,
    )

    losses, draws, wins = evaluate_vs_random(model, nfa, n_games=200, seed=0)
    n = losses + draws + wins

    print('Learned automaton:')
    print(f'  States       : {len(model.states)}')
    print(f'  Cache entries: {len(sul._cache)}')
    print(f'  Eq. queries  : {mcts.num_queries}')
    print()
    print('Evaluation vs random P1 (200 games):')
    print(f'  wins={wins}  draws={draws}  losses={losses}')
    print(f'  win rate={wins/n:.1%}  loss rate={losses/n:.1%}')

    if args.viz:
        print()
        print(table_b.summary())

    out_dir = Path(__file__).parents[2] / 'outputs'
    out_dir.mkdir(exist_ok=True)
    model.save(str(out_dir / 'learned_strategy_nim'))
    print(f'\nSaved: {out_dir / "learned_strategy_nim.dot"}')


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
