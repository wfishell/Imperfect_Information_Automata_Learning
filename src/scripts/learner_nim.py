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
from src.eval.nim import RandomP1, GreedyP1, OptimalP1


def main():
    parser = argparse.ArgumentParser(
        description='Learn P2 strategy automaton for Nim via L* + MCTS.'
    )
    parser.add_argument('--piles',        type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--depth-n',      dest='depth_n',      type=int,   default=5)
    parser.add_argument('--K',            type=int,   default=200)
    parser.add_argument('--oracle-depth', dest='oracle_depth', type=int,   default=None,
                        help='Minimax lookahead for oracle (default: None = full search)')
    parser.add_argument('--n-random-games', dest='n_random_games', type=int, default=200,
                        help='Number of evaluation games vs RandomP1 (default 200)')
    parser.add_argument('--n-other-games',  dest='n_other_games', type=int, default=50,
                        help='Number of evaluation games vs Greedy/Optimal P1 (default 50)')
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
    print('Evaluation vs random P1 (200 games, sanity):')
    print(f'  wins={wins}  draws={draws}  losses={losses}')
    print(f'  win rate={wins/n:.1%}  loss rate={losses/n:.1%}')

    print()
    print(f'Evaluation: learned P2 Mealy vs each P1 strategy')
    print(f'  (n_random={args.n_random_games}  n_greedy={args.n_other_games}  '
          f'n_optimal={args.n_other_games})')

    p1_results = evaluate_against_p1_players(
        model, nfa, piles=piles,
        n_random_games = args.n_random_games,
        n_other_games  = args.n_other_games,
    )
    _print_wdl_table(p1_results)

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


def evaluate_against_p1_players(model, nfa: NimNFA, piles: tuple,
                                  n_random_games: int = 200,
                                  n_other_games:  int = 50) -> dict:
    """Mealy P2 vs each P1 strategy. Returns dict opponent → (W, D, L)."""

    def play_one(p1) -> str:
        state = nfa.root
        model.reset_to_initial()
        while not state.is_terminal():
            p1_action = p1.pick(state)
            p2_action = model.step(p1_action)
            state = state.children[p1_action]
            if state.is_terminal():
                break
            if p2_action not in state.children:
                p2_action = next(iter(state.children))
            state = state.children[p2_action]
        return state.winner()

    def play_n(p1, n) -> tuple:
        wins = draws = losses = 0
        for s in range(n):
            p1.rng = random.Random(s)
            w = play_one(p1)
            if   w == 'P2':  wins   += 1
            elif w == 'P1':  losses += 1
            else:            draws  += 1
        return (wins, draws, losses)

    return {
        'vs_random':  play_n(RandomP1(seed=0),                   n_random_games),
        'vs_greedy':  play_n(GreedyP1(seed=0),                   n_other_games),
        'vs_optimal': play_n(OptimalP1(piles=piles, seed=0),     n_other_games),
    }


def _print_wdl_table(results: dict) -> None:
    print(f'  {"opponent":<10}  {"wins":>5}  {"draws":>5}  {"losses":>6}  '
          f'{"win%":>6}  {"loss%":>6}  {"n":>4}')
    print(f'  {"-"*10}  {"-"*5}  {"-"*5}  {"-"*6}  {"-"*6}  {"-"*6}  {"-"*4}')
    for name in ('vs_random', 'vs_greedy', 'vs_optimal'):
        w, d, l = results[name]
        n = w + d + l
        win_pct  = w / n if n else 0.0
        loss_pct = l / n if n else 0.0
        print(f'  {name:<10}  {w:>5}  {d:>5}  {l:>6}  '
              f'{win_pct:>5.1%}  {loss_pct:>5.1%}  {n:>4}')


if __name__ == '__main__':
    main()
