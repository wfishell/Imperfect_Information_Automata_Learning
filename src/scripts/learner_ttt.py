"""
Learn P2's (O) strategy automaton for Tic-Tac-Toe via L* + MCTS.

Usage:
    python -m src.scripts.learner_ttt
    python -m src.scripts.learner_ttt --K 200 --depth-n 5
    python -m src.scripts.learner_ttt --viz
"""

import argparse
import random
from pathlib import Path

from src.game.tic_tac_toe.game_nfa import TicTacToeNFA
from src.game.tic_tac_toe.preference_oracle import TicTacToeOracle
from src.lstar_mcts.learner import run_lstar_mcts


def main():
    parser = argparse.ArgumentParser(
        description='Learn O strategy automaton for Tic-Tac-Toe via L* + MCTS.'
    )
    parser.add_argument('--depth-n',      dest='depth_n',      type=int,   default=5)
    parser.add_argument('--K',            type=int,   default=200)
    parser.add_argument('--oracle-depth', dest='oracle_depth', type=int,   default=None,
                        help='Minimax lookahead for oracle (default: None = full search)')
    parser.add_argument('--verbose',      action='store_true')
    parser.add_argument('--viz',     action='store_true',
                        help='Print enriched output: table B summary')
    args = parser.parse_args()

    nfa       = TicTacToeNFA()
    oracle    = TicTacToeOracle(nfa, depth=args.oracle_depth)
    p1_inputs = list(nfa.root.children.keys())

    print(f'Tic-Tac-Toe  oracle_depth={args.oracle_depth}  depth_n={args.depth_n}  K={args.K}')
    print()

    model, sul, mcts, table_b = run_lstar_mcts(
        nfa       = nfa,
        oracle    = oracle,
        p1_inputs = p1_inputs,
        depth_n   = args.depth_n,
        K         = args.K,
        verbose   = args.verbose,
    )

    losses = evaluate_vs_random(model, nfa, n_games=500, seed=0)

    print('Learned automaton:')
    print(f'  States       : {len(model.states)}')
    print(f'  Cache entries: {len(sul._cache)}')
    print(f'  Eq. queries  : {mcts.num_queries}')
    print()
    print(f'Evaluation vs random X (500 games): losses={losses}')

    if args.viz:
        print()
        print(table_b.summary())

    out_dir = Path(__file__).parents[2] / 'outputs'
    out_dir.mkdir(exist_ok=True)
    model.save(str(out_dir / 'learned_strategy_ttt'))
    print(f'\nSaved: {out_dir / "learned_strategy_ttt.dot"}')


def evaluate_vs_random(model, nfa: TicTacToeNFA,
                       n_games: int = 500, seed: int = 0) -> int:
    """Return number of games where learned O lost to random X."""
    rng = random.Random(seed)
    losses = 0

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

        if state.winner() == 'P1':
            losses += 1

    return losses


if __name__ == '__main__':
    main()
