"""
Learn P2's (O) strategy automaton for Tic-Tac-Toe via L* + MCTS.

Usage:
    python -m src.scripts.learner_ttt
    python -m src.scripts.learner_ttt --depth-n 5 --K 200 --verbose
"""

import argparse
import random
from aalpy.learning_algs import run_Lstar

from src.game.tic_tac_toe.game_nfa import TicTacToeNFA
from src.game.tic_tac_toe.preference_oracle import TicTacToeOracle
from src.lstar_mcts.game_sul import GameSUL
from src.lstar_mcts.table_b import TableB
from src.lstar_mcts.mcts_oracle import MCTSEquivalenceOracle


def main():
    parser = argparse.ArgumentParser(
        description='Learn O strategy automaton for Tic-Tac-Toe via L* + MCTS.'
    )
    parser.add_argument('--depth-n', dest='depth_n', type=int, default=5,
                        help='MCTS search depth N (default: 5)')
    parser.add_argument('--K',       type=int, default=200,
                        help='MCTS rollout budget per equivalence query')
    parser.add_argument('--epsilon', type=float, default=0.05,
                        help='Min advantage to accept a counterexample')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    nfa     = TicTacToeNFA()
    oracle  = TicTacToeOracle(nfa)
    sul     = GameSUL(nfa, oracle)
    table_b = TableB()

    eq_oracle = MCTSEquivalenceOracle(
        sul     = sul,
        nfa     = nfa,
        oracle  = oracle,
        table_b = table_b,
        depth_N = args.depth_n,
        K       = args.K,
        epsilon = args.epsilon,
        verbose = args.verbose,
    )

    p1_inputs = list(nfa.root.children.keys())  # [0..8]

    print(f'Running L* on Tic-Tac-Toe  depth_N={args.depth_n}  K={args.K}')
    print()

    model = run_Lstar(
        alphabet                = p1_inputs,
        sul                     = sul,
        eq_oracle               = eq_oracle,
        automaton_type          = 'mealy',
        print_level             = 2 if args.verbose else 0,
        cache_and_non_det_check = False,
    )

    print()
    print('Learned automaton:')
    print(f'  States : {len(model.states)}')
    print(f'  Queries: membership={sul.num_queries}  '
          f'equivalence={eq_oracle.num_queries}')
    print()

    losses = evaluate_vs_random(model, nfa, n_games=500, seed=0)
    print(f'Evaluation vs random X (500 games): losses={losses}  '
          f'{"PASS" if losses == 0 else "FAIL"}')
    print()
    print(table_b.summary())

    model.save('learned_strategy_ttt')
    print('Saved: learned_strategy_ttt.dot')


def evaluate_vs_random(model, nfa: TicTacToeNFA, n_games: int = 500,
                       seed: int = 0) -> int:
    """Return number of games where the learned O lost to random X."""
    rng = random.Random(seed)
    losses = 0

    for _ in range(n_games):
        state = nfa.root
        model.reset_to_initial()

        while not state.is_terminal():
            if state.player == 'P1':
                move = rng.choice(list(state.children.keys()))
                model.step(move)
                state = state.children[move]
            else:
                o_move = model.step(list(state.children.keys())[0])
                if o_move not in state.children:
                    o_move = rng.choice(list(state.children.keys()))
                state = state.children[o_move]

        if state.winner() == 'P1':
            losses += 1

    return losses


if __name__ == '__main__':
    main()
