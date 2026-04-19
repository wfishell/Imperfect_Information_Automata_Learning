"""
Visualized L* strategy learner for Dots and Boxes.

Usage:
    python -m src.scripts.dab.learner_viz
    python -m src.scripts.dab.learner_viz --rows 2 --cols 2 --depth-n 5 --K 200
    python -m src.scripts.dab.learner_viz --oracle-depth 2
    python -m src.scripts.dab.learner_viz --play
    python -m src.scripts.dab.learner_viz --play my_strategy.dot --rows 2 --cols 2
"""

import argparse
import random
from pathlib import Path
from aalpy.learning_algs import run_Lstar
from aalpy.utils import load_automaton_from_file
from rich.table import Table

from src.game.dots_and_boxes.game_nfa import DotsAndBoxesNFA, PASS
from src.game.dots_and_boxes.board import _h_edge, _v_edge
from src.game.dots_and_boxes.preference_oracle import DotsAndBoxesOracle
from src.game.dots_and_boxes.dab_sul import DotsAndBoxesSUL
from src.lstar_mcts.table_b import TableB
from src.lstar_mcts.mcts_oracle import MCTSEquivalenceOracle
from src.viz.visualizer import Visualizer


MAX_ROUNDS = 10


def _render_board(state) -> str:
    """Render the board showing drawn edges and undrawn edge indices."""
    rows, cols = state.rows, state.cols
    lines = []
    for r in range(rows + 1):
        h_row = '.'
        for c in range(cols):
            idx = _h_edge(r, c, cols)
            h_row += '---' if state.edges[idx] else f'{idx:^3}'
            h_row += '.'
        lines.append(h_row)
        if r < rows:
            v_row = ''
            for c in range(cols + 1):
                idx = _v_edge(r, c, rows, cols)
                v_row += '|' if state.edges[idx] else f'{idx:<2}'
                if c < cols:
                    v_row += '   '
            lines.append(v_row)
    lines.append(f'({state.player} to move  P1:{state.p1_boxes}  P2:{state.p2_boxes})')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='L* + MCTS strategy learner for Dots and Boxes with Rich visualisation.'
    )
    parser.add_argument('--rows',         type=int,   default=2)
    parser.add_argument('--cols',         type=int,   default=2)
    parser.add_argument('--depth-n',      dest='depth_n',      type=int,   default=5)
    parser.add_argument('--K',            type=int,   default=200)
    parser.add_argument('--epsilon',      type=float, default=0.05)
    parser.add_argument('--oracle-depth', dest='oracle_depth', type=int,   default=None,
                        help='Minimax lookahead for oracle (default: None = optimal)')
    parser.add_argument('--n-eval',       dest='n_eval', type=int, default=200,
                        help='Games to play when evaluating learned strategy')
    parser.add_argument('--play',         nargs='?', const='learned_strategy_dab.dot',
                        metavar='DOT_FILE',
                        help='Play against a saved .dot controller instead of learning.')
    args = parser.parse_args()

    if args.play is not None:
        play_against_model(args.play, rows=args.rows, cols=args.cols)
        return

    viz     = Visualizer()
    nfa     = DotsAndBoxesNFA(rows=args.rows, cols=args.cols)
    oracle  = DotsAndBoxesOracle(nfa, depth=args.oracle_depth)
    sul     = DotsAndBoxesSUL(nfa, oracle)
    table_b = TableB()

    eq = MCTSEquivalenceOracle(
        sul=sul, nfa=nfa, oracle=oracle, table_b=table_b,
        depth_N=args.depth_n, K=args.K, epsilon=args.epsilon,
        verbose=False,
    )

    p1_inputs = list(nfa.root.children.keys()) + [PASS]

    viz.console.print(
        f'[bold]Dots and Boxes[/bold]  {args.rows}×{args.cols}  '
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

        remaining = {None: args.K}
        for _ in range(args.K):
            eq._rollout(model, remaining)

        viz.show_table_b(eq.table_b)
        viz.show_deviations(eq._deviation_leaves)

        improvement = eq._check_for_improvement(model)
        viz.show_improvement(improvement)

        if improvement is None:
            break

    losses, draws, wins = _eval_vs_random(model, nfa, n_games=args.n_eval, seed=0)
    _show_eval(viz, losses, draws, wins, n_games=args.n_eval,
               label=f'Dots and Boxes {args.rows}×{args.cols}')

    viz.show_final_summary(model, sul, eq, table_b)

    diagrams_dir = Path(__file__).parents[2] / 'viz' / 'diagrams'
    diagrams_dir.mkdir(parents=True, exist_ok=True)
    save_path = diagrams_dir / f'learned_strategy_dab_{args.rows}x{args.cols}'
    model.save(str(save_path))
    viz.console.print(f'[dim]Saved: {save_path}.dot[/dim]')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eval_vs_random(model, nfa: DotsAndBoxesNFA,
                    n_games: int, seed: int) -> tuple[int, int, int]:
    """Return (losses, draws, wins) for learned P2 vs random P1."""
    rng = random.Random(seed)
    losses = draws = wins = 0

    for _ in range(n_games):
        state = nfa.root
        model.reset_to_initial()

        while not state.is_terminal():
            if state.player == 'P1':
                p1_move = rng.choice(list(state.children.keys()))
                p2_move = model.step(p1_move)
                state   = state.children[p1_move]

                if state.is_terminal():
                    break

                if state.player == 'P1':
                    continue

                if p2_move == PASS or p2_move not in state.children:
                    p2_move = rng.choice(list(state.children.keys()))
                state = state.children[p2_move]

            else:
                p2_move = model.step(PASS)
                if p2_move == PASS or p2_move not in state.children:
                    p2_move = rng.choice(list(state.children.keys()))
                state = state.children[p2_move]

        w = state.winner()
        if w == 'P1':   losses += 1
        elif w == 'P2': wins   += 1
        else:           draws  += 1

    return losses, draws, wins


def _show_eval(viz: Visualizer, losses: int, draws: int, wins: int,
               n_games: int, label: str) -> None:
    loss_pct = 100 * losses / n_games
    draw_pct = 100 * draws  / n_games
    win_pct  = 100 * wins   / n_games

    table = Table(
        title=f'{label} — P2 (learned) vs Random P1  ({n_games} games)',
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


def play_against_model(dot_path: str, rows: int, cols: int) -> None:
    """Interactive game: human plays P1, loaded .dot controller plays P2."""
    dot_path = Path(dot_path)
    if not dot_path.exists():
        print(f'Error: {dot_path} not found. Run without --play to learn a strategy first.')
        return

    model = load_automaton_from_file(str(dot_path), automaton_type='mealy')
    model.reset_to_initial()

    nfa   = DotsAndBoxesNFA(rows=rows, cols=cols)
    state = nfa.root

    print(f'\nDots and Boxes {rows}×{cols}  —  You are P1, model is P2')
    print('Undrawn edges are shown by their index number.\n')
    print(_render_board(state))

    while not state.is_terminal():
        if state.player == 'P1':
            legal = list(state.children.keys())
            print(f'\nLegal moves: {legal}')
            while True:
                try:
                    move = int(input('Your move (edge index): '))
                    if move in legal:
                        break
                    print(f'  {move} is not a legal move.')
                except (ValueError, EOFError):
                    print('  Enter an integer edge index.')

            p2_response = model.step(move)
            state = state.children[move]
            print()
            print(_render_board(state))

            if state.is_terminal():
                break

            if state.player == 'P1':
                print('  You completed a box — take another turn!')
                continue

            if p2_response == PASS or p2_response not in state.children:
                p2_response = random.choice(list(state.children.keys()))
                print(f'  P2 (model fallback) plays edge {p2_response}')
            else:
                print(f'  P2 (model) plays edge {p2_response}')
            state = state.children[p2_response]
            print()
            print(_render_board(state))

        else:
            p2_response = model.step(PASS)
            if p2_response == PASS or p2_response not in state.children:
                p2_response = random.choice(list(state.children.keys()))
                print(f'  P2 (model fallback extra turn) plays edge {p2_response}')
            else:
                print(f'  P2 (model, extra turn) plays edge {p2_response}')
            state = state.children[p2_response]
            print()
            print(_render_board(state))

    winner = state.winner()
    print()
    if winner == 'P1':   print('You win!')
    elif winner == 'P2': print('P2 (model) wins!')
    else:                print("It's a draw!")
    print(f'Final score — P1: {state.p1_boxes}  P2: {state.p2_boxes}')


if __name__ == '__main__':
    main()
