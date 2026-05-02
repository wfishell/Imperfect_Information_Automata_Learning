"""
Learn P2's strategy automaton for Dots and Boxes via L* + MCTS.

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
from src.lstar_mcts.game_sul          import GameSUL
from src.lstar_mcts.table_b           import TableB
from src.lstar_mcts.mcts_oracle       import MCTSEquivalenceOracle
from src.lstar_mcts.pac_oracle        import PACEqOracle
from src.lstar_mcts.composite_oracle  import CompositeEqOracle
from src.lstar_mcts.counting_oracle   import CountingOracle
from src.lstar_mcts.custom_lstar      import MealyLStar


def main():
    parser = argparse.ArgumentParser(
        description='Learn P2 strategy automaton for Dots and Boxes via L* + MCTS.'
    )
    parser.add_argument('--rows',         type=int,   default=2)
    parser.add_argument('--cols',         type=int,   default=2)
    parser.add_argument('--depth-n',      dest='depth_n',      type=int,   default=5)
    parser.add_argument('--K',            type=int,   default=200)
    parser.add_argument('--oracle-depth', dest='oracle_depth', type=int,   default=None,
                        help='Minimax lookahead for oracle (default: None = full search)')
    parser.add_argument('--pac-eps',      dest='pac_eps',      type=float, default=0.05)
    parser.add_argument('--pac-delta',    dest='pac_delta',    type=float, default=0.05)
    parser.add_argument('--pac-max-walk', dest='pac_max_walk', type=int,   default=40,
                        help='Max P1 inputs per sampled NFA walk in PAC phase')
    parser.add_argument('--no-pac',       dest='use_pac',      action='store_false',
                        help='Disable PAC validation phase (use bare MCTS oracle)')
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

    nfa         = DotsAndBoxesNFA(rows=args.rows, cols=args.cols)
    inner       = DotsAndBoxesOracle(nfa, depth=args.oracle_depth)
    oracle      = CountingOracle(inner)
    table_b     = TableB()
    sul         = GameSUL(nfa=nfa, oracle=oracle, table_b=table_b)

    mcts = MCTSEquivalenceOracle(
        sul=sul, nfa=nfa, oracle=oracle, table_b=table_b,
        depth_N=args.depth_n, K=args.K,
        verbose=args.verbose,
    )

    if args.use_pac:
        pac = PACEqOracle(
            alphabet       = list(nfa.p1_alphabet),
            sul            = sul,
            nfa            = nfa,
            eps            = args.pac_eps,
            delta          = args.pac_delta,
            max_walk_depth = args.pac_max_walk,
        )
        eq = CompositeEqOracle(mcts, pac, verbose=args.verbose)
    else:
        eq = mcts

    lstar = MealyLStar(
        alphabet  = nfa.p1_alphabet,
        sul       = sul,
        eq_oracle = eq,
        verbose   = args.verbose,
    )

    print(f'Dots and Boxes {args.rows}x{args.cols}  '
          f'oracle_depth={args.oracle_depth}  depth_n={args.depth_n}  K={args.K}  '
          f'pac={"on" if args.use_pac else "off"}')
    print()

    model = lstar.run()

    losses, draws, wins = evaluate_vs_random(model, nfa, n_games=200, seed=0)
    n = losses + draws + wins

    print('Learned automaton:')
    print(f'  States       : {len(model.states)}')
    print(f'  Cache entries: {len(sul._cache)}')
    print(f'  Eq. queries  : {eq.num_queries}')
    if hasattr(eq, 'mcts') and hasattr(eq, 'pac'):
        print(f'    MCTS phase : {eq.mcts.num_queries}')
        print(f'    PAC  phase : {eq.pac.num_queries}')
    print()
    print('Preference-oracle calls (during learning):')
    print(f'  compare()        : {oracle.compare_calls}')
    print(f'  preferred_move() : {oracle.preferred_move_calls}')
    print(f'  total            : {oracle.total_queries}')
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

    The loop always begins at a P1 state (real or forced-pass).  Forced-pass
    states are handled transparently: P1 sends PASS, model returns P2's next
    move; P2's forced-pass states consume PASS internally.
    """
    rng = random.Random(seed)
    losses = draws = wins = 0

    for _ in range(n_games):
        state = nfa.root
        model.reset_to_initial()

        while not state.is_terminal():
            # --- P1's section ---
            p1_move = PASS if state.forced_pass else rng.choice(list(state.children.keys()))
            p2_output = model.step(p1_move)
            state = state.children[p1_move]

            if state.is_terminal():
                break

            # --- P2's section ---
            if state.forced_pass:
                # P1 completed a box; P2 is forced to pass
                state = state.children[PASS]
            elif p2_output == PASS or p2_output not in state.children:
                state = state.children[rng.choice(list(state.children.keys()))]
            else:
                state = state.children[p2_output]

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
    fp = '  FORCED PASS' if state.forced_pass else ''
    lines.append(f'({state.player} to move{fp}  P1:{state.p1_boxes}  P2:{state.p2_boxes})')
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
        # --- P1's section ---
        if state.forced_pass:
            print('\n  P2 completed a box — you must pass.')
            p1_move = PASS
        else:
            legal = list(state.children.keys())
            print(f'\nLegal moves: {legal}')
            while True:
                try:
                    p1_move = int(input('Your move (edge index): '))
                    if p1_move in legal:
                        break
                    print(f'  {p1_move} is not legal.')
                except (ValueError, EOFError):
                    print('  Enter an integer.')

        p2_response = model.step(p1_move)
        state = state.children[p1_move]
        print(); print(_render_board(state))

        if state.is_terminal():
            break

        # --- P2's section ---
        if state.forced_pass:
            print('  You completed a box — P2 must pass, you get another turn!')
            state = state.children[PASS]
        elif p2_response == PASS or p2_response not in state.children:
            p2_response = random.choice(list(state.children.keys()))
            print(f'  P2 (fallback) plays edge {p2_response}')
            state = state.children[p2_response]
        else:
            print(f'  P2 plays edge {p2_response}')
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
