"""
Learn P2's strategy automaton for Dots and Boxes via L* + MCTS.

DotsAndBoxes uses DotsAndBoxesSUL rather than the generic GameSUL because
of the PASS mechanic (extra turns after box completions), so components are
built manually here instead of delegating to run_lstar_mcts().

Usage:
    python -m src.scripts.learner_dab
    python -m src.scripts.learner_dab --rows 2 --cols 2 --K 200
    python -m src.scripts.learner_dab --oracle-depth 2 --viz
    python -m src.scripts.learner_dab --play [learned_strategy_dab.dot]
"""

import argparse
import random
from pathlib import Path

from src.game.dots_and_boxes.game_nfa import DotsAndBoxesNFA, PASS
from src.game.dots_and_boxes.board import _h_edge, _v_edge
from src.game.dots_and_boxes.preference_oracle import DotsAndBoxesOracle
from src.game.dots_and_boxes.dab_sul import DotsAndBoxesSUL
from src.lstar_mcts.table_b import TableB
from src.lstar_mcts.mcts_oracle import MCTSEquivalenceOracle
from src.lstar_mcts.custom_lstar import MealyLStar


def main():
    parser = argparse.ArgumentParser(
        description='Learn P2 strategy automaton for Dots and Boxes via L* + MCTS.'
    )
    parser.add_argument('--rows',         type=int,   default=2)
    parser.add_argument('--cols',         type=int,   default=2)
    parser.add_argument('--depth-n',      dest='depth_n',      type=int,   default=5)
    parser.add_argument('--K',            type=int,   default=200)
    parser.add_argument('--epsilon',      type=float, default=0.05)
    parser.add_argument('--oracle-depth', dest='oracle_depth', type=int,   default=None,
                        help='Minimax lookahead for oracle (default: None = full search)')
    parser.add_argument('--verbose',      action='store_true')
    parser.add_argument('--viz',          action='store_true',
                        help='Print enriched output: table B summary and board render')
    parser.add_argument('--play',         nargs='?', const='learned_strategy_dab.dot',
                        metavar='DOT_FILE',
                        help='Play against a saved strategy instead of learning')
    args = parser.parse_args()

    if args.play is not None:
        play_against_model(args.play, rows=args.rows, cols=args.cols)
        return

    nfa     = DotsAndBoxesNFA(rows=args.rows, cols=args.cols)
    oracle  = DotsAndBoxesOracle(nfa, depth=args.oracle_depth)
    sul     = DotsAndBoxesSUL(nfa, oracle)
    table_b = TableB()

    eq = MCTSEquivalenceOracle(
        sul=sul, nfa=nfa, oracle=oracle, table_b=table_b,
        depth_N=args.depth_n, K=args.K, epsilon=args.epsilon,
        verbose=args.verbose,
    )

    # Alphabet = all edges + PASS (P1 must signal when P2 earns an extra turn)
    p1_inputs = list(nfa.root.children.keys()) + [PASS]

    lstar = MealyLStar(
        alphabet  = p1_inputs,
        sul       = sul,
        eq_oracle = eq,
        verbose   = args.verbose,
    )

    print(f'Dots and Boxes {args.rows}x{args.cols}  '
          f'oracle_depth={args.oracle_depth}  depth_n={args.depth_n}  K={args.K}')
    print()

    model = lstar.run()

    losses, draws, wins = evaluate_vs_random(model, nfa, n_games=200, seed=0)
    n = losses + draws + wins

    print('Learned automaton:')
    print(f'  States       : {len(model.states)}')
    print(f'  Cache entries: {len(sul._cache)}')
    print(f'  Eq. queries  : {eq.num_queries}')
    print()
    print('Evaluation vs random P1 (200 games):')
    print(f'  wins={wins}  draws={draws}  losses={losses}')
    print(f'  win rate={wins/n:.1%}  loss rate={losses/n:.1%}')

    if args.viz:
        print()
        print(table_b.summary())

    out_dir = Path(__file__).parents[2] / 'outputs'
    out_dir.mkdir(exist_ok=True)
    model.save(str(out_dir / 'learned_strategy_dab'))
    print(f'\nSaved: {out_dir / "learned_strategy_dab.dot"}')


def evaluate_vs_random(model, nfa: DotsAndBoxesNFA,
                       n_games: int, seed: int) -> tuple[int, int, int]:
    """
    Return (losses, draws, wins) for learned P2 vs random P1.
    PASS is fed to the model whenever P2 earns an extra turn.
    """
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
                    # P1 completed a box — model returned PASS, loop continues
                    continue

                if p2_move == PASS or p2_move not in state.children:
                    p2_move = rng.choice(list(state.children.keys()))
                state = state.children[p2_move]

            else:
                # P2 earned an extra turn — feed PASS as forced P1 input
                p2_move = model.step(PASS)
                if p2_move == PASS or p2_move not in state.children:
                    p2_move = rng.choice(list(state.children.keys()))
                state = state.children[p2_move]

        w = state.winner()
        if w == 'P1':   losses += 1
        elif w == 'P2': wins   += 1
        else:           draws  += 1

    return losses, draws, wins


def _render_board(state) -> str:
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


def play_against_model(dot_path: str, rows: int, cols: int) -> None:
    """Interactive game: human plays P1, loaded .dot controller plays P2."""
    from aalpy.utils import load_automaton_from_file
    dot_path = Path(dot_path)
    if not dot_path.exists():
        print(f'Error: {dot_path} not found. Run without --play to learn first.')
        return

    model = load_automaton_from_file(str(dot_path), automaton_type='mealy')
    model.reset_to_initial()

    nfa   = DotsAndBoxesNFA(rows=rows, cols=cols)
    state = nfa.root

    print(f'\nDots and Boxes {rows}x{cols}  —  You are P1, model is P2')
    print('Undrawn edges shown by index.\n')
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
                    print(f'  {move} is not legal.')
                except (ValueError, EOFError):
                    print('  Enter an integer.')

            p2_response = model.step(move)
            state = state.children[move]
            print(); print(_render_board(state))

            if state.is_terminal():
                break

            if state.player == 'P1':
                print('  You completed a box — take another turn!')
                continue

            if p2_response == PASS or p2_response not in state.children:
                p2_response = random.choice(list(state.children.keys()))
                print(f'  P2 (fallback) plays edge {p2_response}')
            else:
                print(f'  P2 plays edge {p2_response}')
            state = state.children[p2_response]
            print(); print(_render_board(state))

        else:
            p2_response = model.step(PASS)
            if p2_response == PASS or p2_response not in state.children:
                p2_response = random.choice(list(state.children.keys()))
                print(f'  P2 (fallback extra turn) plays edge {p2_response}')
            else:
                print(f'  P2 (extra turn) plays edge {p2_response}')
            state = state.children[p2_response]
            print(); print(_render_board(state))

    winner = state.winner()
    print()
    if winner == 'P1':   print('You win!')
    elif winner == 'P2': print('P2 (model) wins!')
    else:                print("Draw!")
    print(f'Final — P1: {state.p1_boxes}  P2: {state.p2_boxes}')


if __name__ == '__main__':
    main()
