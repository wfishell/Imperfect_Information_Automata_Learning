"""
Learn P2's (O) strategy automaton for Hex via L* + MCTS.

P1 (X) connects top row → bottom row.
P2 (O) connects left col → right col.

Usage:
    python -m src.scripts.learner_hex
    python -m src.scripts.learner_hex --size 3 --K 200
    python -m src.scripts.learner_hex --oracle-depth 2 --viz
    python -m src.scripts.learner_hex --play [learned_strategy_hex_3x3.dot]
"""

import argparse
import random
import re
from pathlib import Path

from src.game.hex.board import EMPTY, X, O
from src.game.hex.game_nfa import HexNFA
from src.game.hex.preference_oracle import HexOracle
from src.lstar_mcts.learner import run_lstar_mcts
from src.lstar_mcts.custom_lstar import MealyMachine, MealyState
from src.eval.hex import RandomP1, GreedyP1, OptimalP1


def main():
    parser = argparse.ArgumentParser(
        description='Learn P2 strategy automaton for Hex via L* + MCTS.'
    )
    parser.add_argument('--size',         type=int,   default=3,
                        help='Board side length (default: 3)')
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
    parser.add_argument('--play',         nargs='?', const='',
                        metavar='DOT_FILE',
                        help='Play against a saved strategy instead of learning')
    args = parser.parse_args()

    if args.play is not None:
        dot_path = args.play or f'learned_strategy_hex_{args.size}x{args.size}.dot'
        play_against_model(dot_path, size=args.size)
        return

    nfa    = HexNFA(size=args.size)
    oracle = HexOracle(nfa, depth=args.oracle_depth)

    print(f'Hex {args.size}x{args.size}  oracle_depth={args.oracle_depth}  '
          f'depth_n={args.depth_n}  K={args.K}')
    print()

    model, sul, mcts, table_b = run_lstar_mcts(
        nfa       = nfa,
        oracle    = oracle,
        p1_inputs = list(nfa.root.children.keys()),
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
        model, nfa, size=args.size,
        n_random_games = args.n_random_games,
        n_other_games  = args.n_other_games,
    )
    _print_wdl_table(p1_results)

    if args.viz:
        print()
        print(table_b.summary())

    out_name = f'learned_strategy_hex_{args.size}x{args.size}'
    out_dir  = Path(__file__).parents[2] / 'outputs'
    out_dir.mkdir(exist_ok=True)
    model.save(str(out_dir / out_name))
    print(f'\nSaved: {out_dir / (out_name + ".dot")}')


def evaluate_vs_random(model, nfa: HexNFA,
                       n_games: int, seed: int) -> tuple[int, int, int]:
    """Return (losses, draws, wins) for learned P2 vs random P1.

    Hex has no draws, so draws will always be 0.
    """
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


def _render_board(state) -> str:
    symbols = {EMPTY: '.', X: 'X', O: 'O'}
    lines = []
    for r in range(state.size):
        indent = ' ' * r
        row = []
        for c in range(state.size):
            idx = r * state.size + c
            token = state.board[idx]
            row.append(f'{idx:2d}' if token == EMPTY else f' {symbols[token]}')
        lines.append(indent + ' '.join(row))
    lines.append(f'({state.player} to move)')
    return '\n'.join(lines)


def _resolve_dot_path(dot_path: str) -> Path:
    path = Path(dot_path)
    if path.exists():
        return path
    if not path.is_absolute():
        candidate = Path(__file__).parents[2] / 'outputs' / path
        if candidate.exists():
            return candidate
    return path


def _coerce_dot_value(raw: str):
    raw = raw.strip()
    if raw == 'None':
        return None
    if re.fullmatch(r'-?\d+', raw):
        return int(raw)
    return raw


def _load_mealy_from_dot(dot_path: Path) -> MealyMachine:
    node_re = re.compile(r'^\s*([A-Za-z_]\w*)\s+\[shape=(doublecircle|circle)\];\s*$')
    edge_re = re.compile(r'^\s*([A-Za-z_]\w*)\s*->\s*([A-Za-z_]\w*)\s*\[label="([^"]+)"\];\s*$')

    states: dict[str, MealyState] = {}
    initial_state_id: str | None = None
    edges: list[tuple[str, str, str, str]] = []

    with dot_path.open('r', encoding='utf-8') as f:
        for line in f:
            node_m = node_re.match(line)
            if node_m:
                state_id, shape = node_m.groups()
                if state_id not in states:
                    states[state_id] = MealyState(state_id, ())
                if shape == 'doublecircle':
                    initial_state_id = state_id
                continue

            edge_m = edge_re.match(line)
            if edge_m:
                src, dst, label = edge_m.groups()
                if '/' not in label:
                    raise ValueError(f'Invalid edge label "{label}" in {dot_path}')
                inp, out = label.split('/', 1)
                edges.append((src, dst, inp, out))

    if not states:
        raise ValueError(f'No states found in {dot_path}')
    if initial_state_id is None:
        raise ValueError(f'No initial state found in {dot_path} (expected shape=doublecircle)')

    for src, dst, inp, out in edges:
        if src not in states or dst not in states:
            raise ValueError(f'Edge references unknown state: {src} -> {dst}')
        states[src].transitions[_coerce_dot_value(inp)] = (_coerce_dot_value(out), states[dst])

    return MealyMachine(list(states.values()), states[initial_state_id])


def play_against_model(dot_path: str, size: int) -> None:
    """Interactive game: human plays P1 (X), loaded .dot controller plays P2 (O)."""
    model_path = _resolve_dot_path(dot_path)
    if not model_path.exists():
        print(f'Error: {dot_path} not found. Run without --play to learn first.')
        return

    try:
        model = _load_mealy_from_dot(model_path)
    except (OSError, ValueError) as e:
        print(f'Error: could not load {model_path}: {e}')
        return
    model.reset_to_initial()

    nfa = HexNFA(size=size)
    state = nfa.root

    print(f'\nHex {size}x{size}  —  You are P1 (X), model is P2 (O)')
    print('Empty cells are shown by index.\n')
    print(_render_board(state))

    while not state.is_terminal():
        legal = sorted(state.children.keys())
        print(f'\nLegal moves: {legal}')
        while True:
            try:
                move = int(input('Your move (cell index): '))
                if move in state.children:
                    break
                print(f'  {move} is not legal.')
            except (ValueError, EOFError):
                print('  Enter an integer.')

        p2_response = model.step(move)
        state = state.children[move]
        print()
        print(_render_board(state))

        if state.is_terminal():
            break

        if p2_response not in state.children:
            p2_response = random.choice(list(state.children.keys()))
            print(f'  P2 (fallback) plays {p2_response}')
        else:
            print(f'  P2 plays {p2_response}')
        state = state.children[p2_response]
        print()
        print(_render_board(state))

    winner = state.winner()
    print()
    if winner == 'P1':
        print('You win!')
    elif winner == 'P2':
        print('P2 (model) wins!')
    else:
        print('Draw!')


def evaluate_against_p1_players(model, nfa: HexNFA, size: int,
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
        'vs_random':  play_n(RandomP1(seed=0),                n_random_games),
        'vs_greedy':  play_n(GreedyP1(seed=0),                n_other_games),
        'vs_optimal': play_n(OptimalP1(size=size, seed=0),    n_other_games),
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
