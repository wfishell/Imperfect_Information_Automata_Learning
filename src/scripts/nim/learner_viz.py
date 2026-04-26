"""
Visualized L* strategy learner for Nim.

Usage:
    python -m src.scripts.nim.learner_viz
    python -m src.scripts.nim.learner_viz --piles 1 2 3 --depth-n 5 --K 200
    python -m src.scripts.nim.learner_viz --oracle-depth 2
    python -m src.scripts.nim.learner_viz --play
    python -m src.scripts.nim.learner_viz --play my_strategy.dot --piles 1 2 3
"""

import argparse
import ast
import random
from pathlib import Path
from aalpy.learning_algs import run_Lstar
from aalpy.utils import load_automaton_from_file
from rich.table import Table

from src.game.nim.game_nfa import NimNFA
from src.game.nim.preference_oracle import NimOracle
from src.lstar_mcts.game_sul import GameSUL
from src.lstar_mcts.table_b import TableB
from src.lstar_mcts.mcts_oracle import MCTSEquivalenceOracle
from src.viz.visualizer import Visualizer


MAX_ROUNDS = 10


def main():
    parser = argparse.ArgumentParser(
        description='L* + MCTS strategy learner for Nim with Rich visualisation.'
    )
    parser.add_argument('--piles',        type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--depth-n',      dest='depth_n',      type=int,   default=5)
    parser.add_argument('--K',            type=int,   default=200)
    parser.add_argument('--epsilon',      type=float, default=0.05)
    parser.add_argument('--oracle-depth', dest='oracle_depth', type=int,   default=None,
                        help='Minimax lookahead for oracle (default: None = optimal)')
    parser.add_argument('--n-eval',       dest='n_eval', type=int, default=200,
                        help='Games to play when evaluating learned strategy')
    parser.add_argument('--play',         nargs='?', const='learned_strategy_nim.dot',
                        metavar='DOT_FILE',
                        help='Play against a saved .dot controller instead of learning.')
    args = parser.parse_args()

    if args.play is not None:
        play_against_model(args.play, piles=tuple(args.piles))
        return

    viz    = Visualizer()
    piles  = tuple(args.piles)
    nfa    = NimNFA(piles=piles)
    oracle = NimOracle(nfa, depth=args.oracle_depth)
    sul    = GameSUL(nfa, oracle)
    table_b = TableB()

    eq = MCTSEquivalenceOracle(
        sul=sul, nfa=nfa, oracle=oracle, table_b=table_b,
        depth_N=args.depth_n, K=args.K, epsilon=args.epsilon,
        verbose=False,
    )

    p1_inputs = nfa.alphabet

    viz.console.print(
        f'[bold]Nim[/bold]  piles={list(piles)}  '
        f'oracle_depth={args.oracle_depth}  depth_n={args.depth_n}  K={args.K}'
    )

    model = None
    for rnd in range(1, MAX_ROUNDS + 1):
        viz.show_round_header(rnd)

        model = run_Lstar(
            alphabet=p1_inputs,
            sul=sul,
            eq_oracle=eq,
            automaton_type='mealy',
            print_level=0,
            cache_and_non_det_check=False,
        )

        viz.show_hypothesis(model, p1_inputs)

        for _ in range(args.K):
            eq._rollout(model)

        viz.show_table_b(eq.table_b)
        viz.show_deviations(eq._deviation_leaves)

        improvement = eq._check_for_improvement(model)
        viz.show_improvement(improvement)

        if improvement is None:
            break

    losses, draws, wins = _eval_vs_random(model, nfa, n_games=args.n_eval, seed=0)
    _show_eval(viz, losses, draws, wins, n_games=args.n_eval, game='Nim')

    viz.show_final_summary(model, sul, eq, table_b)

    diagrams_dir = Path(__file__).parents[2] / 'viz' / 'diagrams'
    diagrams_dir.mkdir(parents=True, exist_ok=True)
    piles_str = '_'.join(str(p) for p in piles)
    save_path = diagrams_dir / f'learned_strategy_nim_{piles_str}'
    model.save(str(save_path))
    viz.console.print(f'[dim]Saved: {save_path}.dot[/dim]')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eval_vs_random(model, nfa: NimNFA,
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


def _show_eval(viz: Visualizer, losses: int, draws: int, wins: int,
               n_games: int, game: str) -> None:
    loss_pct = 100 * losses / n_games
    draw_pct = 100 * draws  / n_games
    win_pct  = 100 * wins   / n_games

    table = Table(
        title=f'{game} — P2 (learned) vs Random P1  ({n_games} games)',
        show_header=True, header_style='bold',
    )
    table.add_column('Outcome', style='bold')
    table.add_column('Count',   justify='right')
    table.add_column('%',       justify='right')

    table.add_row('[green]P2 wins[/green]',  str(wins),   f'{win_pct:.1f}%')
    table.add_row('[dim]Draws[/dim]',        str(draws),  f'{draw_pct:.1f}%')
    table.add_row('[red]P1 wins[/red]',      str(losses), f'{loss_pct:.1f}%')
    ok = losses == 0
    table.add_row(
        '[bold]Verdict[/bold]',
        '[green bold]PASS[/green bold]' if ok else '[red bold]FAIL[/red bold]',
        '',
    )
    viz.console.print(table)
    viz.console.print()


def play_against_model(dot_path: str, piles: tuple) -> None:
    """Interactive game: human plays P1, loaded .dot controller plays P2."""
    dot_path = Path(dot_path)
    if not dot_path.exists():
        print(f'Error: {dot_path} not found. Run without --play to learn a strategy first.')
        return

    model = load_automaton_from_file(str(dot_path), automaton_type='mealy')
    model.reset_to_initial()

    nfa   = NimNFA(piles=piles)
    state = nfa.root

    print(f'\nNim piles={list(piles)}  —  You are P1, model is P2')
    _print_state(state)

    while not state.is_terminal():
        legal = list(state.children.keys())
        print(f'\nLegal moves (pile, amount): {legal}')
        while True:
            raw = input('Your move as "pile amount" (e.g. "1 2"): ').strip()
            try:
                parts = raw.split()
                move  = (int(parts[0]), int(parts[1]))
                if move in legal:
                    break
                print(f'  {move} is not a legal move.')
            except (ValueError, IndexError):
                print('  Enter two integers: pile index and amount to remove.')

        p2_out = model.step(str(move))
        state  = state.children[move]
        _print_state(state)

        if state.is_terminal():
            break

        try:
            p2_move = ast.literal_eval(str(p2_out))
        except (ValueError, SyntaxError):
            p2_move = None

        if p2_move not in state.children:
            p2_move = random.choice(list(state.children.keys()))
            print(f'  P2 (model fallback) plays {p2_move}')
        else:
            print(f'  P2 (model) plays {p2_move}')
        state = state.children[p2_move]
        _print_state(state)

    winner = state.winner()
    print()
    if winner == 'P1':   print('You win!')
    elif winner == 'P2': print('P2 (model) wins!')
    else:                print("It's a draw!")


def _print_state(state) -> None:
    print(f'  Piles: {list(state.piles)}')


if __name__ == '__main__':
    main()
