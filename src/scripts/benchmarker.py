"""
Cross-game L* + MCTS benchmark.

Consolidates all per-game benchmarks into one file.  Each game is registered
in GAME_REGISTRY with a build function and an evaluation function.  Adding a
new game is one dict entry.

Each experiment run is wrapped in a timeout so a slow run cannot block the
entire sweep.

Sweeps K and depth_n across all registered games and seeds, then prints a
formatted summary table and (optionally) saves matplotlib figures.

Usage:
    python -m src.scripts.benchmarker
    python -m src.scripts.benchmarker --games nim ttt
    python -m src.scripts.benchmarker --K 50 200 --depth-n 3 5 --seeds 0 1
    python -m src.scripts.benchmarker --timeout 120 --plot
    python -m src.scripts.benchmarker --no-dab-3x3   # skip slow 3x3 board
"""

from __future__ import annotations

import argparse
import itertools
import random
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

# -----------------------------------------------------------------------
# Game imports
# -----------------------------------------------------------------------

from src.game.minimax.game_generator import generate_tree, GameNode
from src.game.minimax.game_nfa import GameNFA
from src.game.minimax.preference_oracle import PreferenceOracle

from src.game.nim.game_nfa import NimNFA
from src.game.nim.preference_oracle import NimOracle

from src.game.tic_tac_toe.game_nfa import TicTacToeNFA
from src.game.tic_tac_toe.preference_oracle import TicTacToeOracle

from src.game.dots_and_boxes.game_nfa import DotsAndBoxesNFA, PASS
from src.game.dots_and_boxes.preference_oracle import DotsAndBoxesOracle
from src.game.dots_and_boxes.dab_sul import DotsAndBoxesSUL

from src.lstar_mcts.learner import run_lstar_mcts
from src.lstar_mcts.table_b import TableB
from src.lstar_mcts.mcts_oracle import MCTSEquivalenceOracle
from src.lstar_mcts.custom_lstar import MealyLStar


# -----------------------------------------------------------------------
# Default sweep parameters
# -----------------------------------------------------------------------

K_VALUES   = [50, 100, 200]
DEPTH_NS   = [3, 5]
SEEDS      = [0, 1, 2]
TIMEOUT_S  = 300        # seconds per experiment before it is marked timed-out
N_EVAL     = 200        # games per evaluation call
DIAGRAMS_DIR = Path(__file__).parents[2] / 'outputs' / 'benchmarks'


# -----------------------------------------------------------------------
# Build helpers — return (nfa, oracle, p1_inputs, sul_override)
# sul_override is None for most games; DotsAndBoxes provides its own SUL.
# -----------------------------------------------------------------------

def _build_minimax(depth: int, seed: int):
    root   = generate_tree(depth, seed=seed)
    nfa    = GameNFA(root)
    oracle = PreferenceOracle(nfa)
    return nfa, oracle, list(root.children.keys()), None, root


def _build_nim(piles: tuple, oracle_depth):
    nfa    = NimNFA(piles=piles)
    oracle = NimOracle(nfa, depth=oracle_depth)
    return nfa, oracle, nfa.alphabet, None, None


def _build_ttt():
    nfa    = TicTacToeNFA()
    oracle = TicTacToeOracle(nfa)
    return nfa, oracle, list(nfa.root.children.keys()), None, None


def _build_dab(rows: int, cols: int, oracle_depth):
    nfa    = DotsAndBoxesNFA(rows=rows, cols=cols)
    oracle = DotsAndBoxesOracle(nfa, depth=oracle_depth)
    sul    = DotsAndBoxesSUL(nfa, oracle)
    return nfa, oracle, list(nfa.root.children.keys()) + [PASS], sul, None


# -----------------------------------------------------------------------
# Evaluation helpers — return raw_score in [0, 1]  (1 = best for P2)
# -----------------------------------------------------------------------

def _eval_minimax(model, nfa: GameNFA, root: GameNode, seed: int) -> float:
    """Normalised score: 0 = random baseline, 1 = optimal."""
    rng = random.Random(seed)

    def _opt(node):
        if node.is_terminal():
            return node.value
        cs = {a: _opt(c) for a, c in node.children.items()}
        return node.value + (max(cs.values()) if node.player == 'P2'
                             else sum(cs.values()) / len(cs))

    def _all_p1(node, seq):
        if node.is_terminal():     return [list(seq)]
        if node.player == 'P1':
            return [s for a, c in node.children.items()
                    for s in _all_p1(c, seq + [a])]
        return [s for c in node.children.values() for s in _all_p1(c, seq)]

    def _play(p1_seq, p2_fn):
        node, trace, total, idx = root, [], root.value, 0
        while not node.is_terminal():
            if node.player == 'P1':
                if idx >= len(p1_seq): break
                action = p1_seq[idx]; idx += 1
            else:
                action = p2_fn(trace)
            if action not in node.children: break
            trace.append(action); node = node.children[action]; total += node.value
        return total

    seqs = _all_p1(root, [])
    opt_scores, lrn_scores, rnd_scores = [], [], []

    for seq in seqs:
        def opt_p2(trace):
            n = nfa.get_node(trace)
            return None if n is None else max(n.children, key=lambda a: _opt(n.children[a]))

        model.reset_to_initial()
        outputs = [model.step(p) for p in seq]
        it = iter(outputs)

        opt_scores.append(_play(seq, opt_p2))
        lrn_scores.append(_play(seq, lambda _t, _it=it: next(_it, None)))
        rnd_scores.append(_play(seq, lambda t: (
            lambda n: None if n is None else rng.choice(list(n.children.keys()))
        )(nfa.get_node(t))))

    opt = sum(opt_scores) / len(opt_scores)
    lrn = sum(lrn_scores) / len(lrn_scores)
    rnd = sum(rnd_scores)  / len(rnd_scores)
    return 1.0 if opt == rnd else max(0.0, min(1.0, (lrn - rnd) / (opt - rnd)))


def _eval_vs_random(model, nfa, seed: int, is_dab: bool = False) -> float:
    """Win rate (+ 0.5 * draw rate) for learned P2 vs random P1."""
    rng = random.Random(seed)
    wins = draws = 0

    for _ in range(N_EVAL):
        state = nfa.root
        model.reset_to_initial()

        while not state.is_terminal():
            if state.player == 'P1':
                p1_move = rng.choice(list(state.children.keys()))
                p2_move = model.step(p1_move)
                state   = state.children[p1_move]
                if state.is_terminal(): break

                if is_dab and state.player == 'P1':
                    continue    # P1 completed a box, loop back

                if p2_move is None or (is_dab and p2_move == PASS) \
                        or p2_move not in state.children:
                    p2_move = rng.choice(list(state.children.keys()))
                state = state.children[p2_move]

            else:
                # P2 earned an extra turn (DaB only)
                p2_move = model.step(PASS)
                if p2_move is None or p2_move == PASS or p2_move not in state.children:
                    p2_move = rng.choice(list(state.children.keys()))
                state = state.children[p2_move]

        w = state.winner()
        if w == 'P2':   wins  += 1
        elif w == 'draw': draws += 1

    return (wins + 0.5 * draws) / N_EVAL


# -----------------------------------------------------------------------
# Game registry
#
# Each entry:
#   label     : human-readable name for tables/figures
#   build     : callable(seed) → (nfa, oracle, p1_inputs, sul_override, aux)
#   eval      : callable(model, nfa, aux, seed) → float in [0, 1]
# -----------------------------------------------------------------------

def _make_registry(include_dab_3x3: bool = True) -> dict:
    reg = {
        'minimax_d4': {
            'label': 'Minimax (depth=4)',
            'build': lambda seed: _build_minimax(depth=4, seed=seed),
            'eval':  lambda model, nfa, aux, seed: _eval_minimax(model, nfa, aux, seed),
        },
        'nim_123': {
            'label': 'Nim [1,2,3]',
            'build': lambda seed: _build_nim(piles=(1, 2, 3), oracle_depth=None),
            'eval':  lambda model, nfa, aux, seed: _eval_vs_random(model, nfa, seed),
        },
        'ttt': {
            'label': 'Tic-Tac-Toe',
            'build': lambda seed: _build_ttt(),
            'eval':  lambda model, nfa, aux, seed: _eval_vs_random(model, nfa, seed),
        },
        'dab_2x2': {
            'label': 'Dots & Boxes 2x2',
            'build': lambda seed: _build_dab(rows=2, cols=2, oracle_depth=None),
            'eval':  lambda model, nfa, aux, seed: _eval_vs_random(model, nfa, seed, is_dab=True),
        },
    }
    if include_dab_3x3:
        reg['dab_3x3'] = {
            'label': 'Dots & Boxes 3x3',
            'build': lambda seed: _build_dab(rows=3, cols=3, oracle_depth=None),
            'eval':  lambda model, nfa, aux, seed: _eval_vs_random(model, nfa, seed, is_dab=True),
        }
    return reg


# -----------------------------------------------------------------------
# Core experiment — one run with timeout
# -----------------------------------------------------------------------

def _run_one(game_name: str, game_cfg: dict, depth_n: int, K: int,
             seed: int, timeout_s: float) -> dict:
    """
    Run a single experiment and return a result dict.  If it exceeds
    timeout_s it returns {'timed_out': True, ...} instead.
    """
    base = dict(game=game_name, depth_n=depth_n, K=K, seed=seed)

    def _work():
        nfa, oracle, p1_inputs, sul_override, aux = game_cfg['build'](seed)
        t0 = time.perf_counter()

        if sul_override is not None:
            # DotsAndBoxes: build components manually, sul not created by run_lstar_mcts
            table_b = TableB()
            eq = MCTSEquivalenceOracle(
                sul=sul_override, nfa=nfa, oracle=oracle, table_b=table_b,
                depth_N=depth_n, K=K, epsilon=0.05, verbose=False,
            )
            lstar = MealyLStar(alphabet=p1_inputs, sul=sul_override,
                               eq_oracle=eq, verbose=False)
            model = lstar.run()
            sul, mcts = sul_override, eq
        else:
            model, sul, mcts, table_b = run_lstar_mcts(
                nfa=nfa, oracle=oracle, p1_inputs=p1_inputs,
                depth_n=depth_n, K=K, epsilon=0.05, verbose=False,
            )

        elapsed = time.perf_counter() - t0
        score   = game_cfg['eval'](model, nfa, aux, seed)

        return {
            **base,
            'timed_out':    False,
            'states':       len(model.states),
            'cache_size':   len(sul._cache),
            'eq_queries':   mcts.num_queries,
            'score':        score,
            'elapsed_s':    elapsed,
        }

    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_work)
        try:
            return future.result(timeout=timeout_s)
        except FuturesTimeout:
            future.cancel()
            return {**base, 'timed_out': True, 'states': None, 'cache_size': None,
                    'eq_queries': None, 'score': None, 'elapsed_s': timeout_s}
        except Exception as exc:
            return {**base, 'timed_out': False, 'error': str(exc), 'states': None,
                    'cache_size': None, 'eq_queries': None, 'score': None,
                    'elapsed_s': None}


# -----------------------------------------------------------------------
# Full sweep
# -----------------------------------------------------------------------

def run_sweep(registry: dict, k_values: list, depth_ns: list,
              seeds: list, timeout_s: float) -> list[dict]:
    """
    Sweep all (game, depth_n, K, seed) combinations.
    Returns a flat list of result dicts.
    """
    combos = list(itertools.product(registry.keys(), depth_ns, k_values, seeds))
    total  = len(combos)
    print(f'Total experiments: {total}  (timeout={timeout_s}s each)\n')
    results = []

    for idx, (game, dn, K, seed) in enumerate(combos, 1):
        label = registry[game]['label']
        print(f'  [{idx:>{len(str(total))}}/{total}]  {label:<24}  '
              f'depth_n={dn}  K={K:>3}  seed={seed}', end='  ', flush=True)

        result = _run_one(game, registry[game], dn, K, seed, timeout_s)
        results.append(result)

        if result.get('timed_out'):
            print(f'TIMEOUT (>{timeout_s}s)')
        elif result.get('error'):
            print(f'ERROR: {result["error"][:60]}')
        else:
            print(f'score={result["score"]:.3f}  states={result["states"]}  '
                  f'{result["elapsed_s"]:.1f}s')

    return results


# -----------------------------------------------------------------------
# Summary table
# -----------------------------------------------------------------------

def print_table(results: list[dict], registry: dict) -> None:
    """Print a compact summary: mean score and states across seeds per config."""
    from collections import defaultdict
    import statistics

    groups = defaultdict(list)
    for r in results:
        if not r.get('timed_out') and not r.get('error') and r['score'] is not None:
            groups[(r['game'], r['depth_n'], r['K'])].append(r)

    print()
    print(f'{"Game":<24}  {"dn":>4}  {"K":>5}  '
          f'{"score (mean)":>13}  {"states (mean)":>14}  {"time (mean)":>12}')
    print('-' * 82)

    for game in registry:
        for dn in sorted({r['depth_n'] for r in results}):
            for K in sorted({r['K'] for r in results}):
                key = (game, dn, K)
                runs = groups.get(key, [])
                if not runs:
                    continue
                scores  = [r['score']    for r in runs]
                states  = [r['states']   for r in runs]
                elapsed = [r['elapsed_s'] for r in runs]
                label   = registry[game]['label']
                print(f'{label:<24}  {dn:>4}  {K:>5}  '
                      f'{statistics.mean(scores):>12.3f}  '
                      f'{statistics.mean(states):>13.1f}  '
                      f'{statistics.mean(elapsed):>10.1f}s')
        print()


# -----------------------------------------------------------------------
# Optional figures
# -----------------------------------------------------------------------

def make_figures(results: list[dict], registry: dict,
                 k_values: list, depth_ns: list, out_prefix: str) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed — skipping figures.')
        return

    DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)

    # Colour maps
    k_colors  = {k: c for k, c in zip(sorted(k_values),
                  ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3'])}
    dn_colors = {d: c for d, c in zip(sorted(depth_ns),
                  ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3'])}

    games = list(registry.keys())
    labels = [registry[g]['label'] for g in games]

    # Score by K (fix depth_n = max)
    fixed_dn = max(depth_ns)
    _bar_figure(results, games, labels, k_values, k_colors,
                param_key='K', param_label='K', fixed_key='depth_n',
                fixed_val=fixed_dn, metric='score',
                title=f'Mean score by K  (depth_n={fixed_dn})',
                path=DIAGRAMS_DIR / f'{out_prefix}_score_by_K.png')

    # Score by depth_n (fix K = max)
    fixed_k = max(k_values)
    _bar_figure(results, games, labels, depth_ns, dn_colors,
                param_key='depth_n', param_label='depth_n', fixed_key='K',
                fixed_val=fixed_k, metric='score',
                title=f'Mean score by depth_n  (K={fixed_k})',
                path=DIAGRAMS_DIR / f'{out_prefix}_score_by_dn.png')

    # States by K
    _bar_figure(results, games, labels, k_values, k_colors,
                param_key='K', param_label='K', fixed_key='depth_n',
                fixed_val=fixed_dn, metric='states',
                title=f'Mean automaton states by K  (depth_n={fixed_dn})',
                path=DIAGRAMS_DIR / f'{out_prefix}_states_by_K.png')

    print(f'Figures saved to {DIAGRAMS_DIR}')


def _bar_figure(results, games, labels, param_values, colors,
                param_key, param_label, fixed_key, fixed_val,
                metric, title, path) -> None:
    import matplotlib.pyplot as plt
    import statistics

    fig, ax = plt.subplots(figsize=(max(8, len(games) * 2), 5),
                            constrained_layout=True)
    n_games  = len(games)
    n_params = len(param_values)
    width    = 0.8 / n_params
    xs       = range(n_games)

    for pi, pv in enumerate(sorted(param_values)):
        means = []
        for game in games:
            runs = [r for r in results
                    if r['game'] == game
                    and r[param_key] == pv
                    and r[fixed_key] == fixed_val
                    and not r.get('timed_out') and not r.get('error')
                    and r[metric] is not None]
            means.append(statistics.mean(r[metric] for r in runs) if runs else 0)

        offsets = [x + (pi - n_params / 2 + 0.5) * width for x in xs]
        ax.bar(offsets, means, width=width * 0.9,
               color=colors[pv], label=f'{param_label}={pv}', alpha=0.85)

    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=8)
    ax.set_ylabel(metric, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    if metric == 'score':
        ax.set_ylim(0, 1.1)
        ax.axhline(1.0, color='black', linewidth=1, linestyle='--', alpha=0.5)

    fig.savefig(path, dpi=120)
    plt.close(fig)


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Cross-game L* + MCTS benchmark.'
    )
    parser.add_argument('--games',      nargs='+', default=None,
                        help='Games to run (default: all). '
                             'Choices: minimax_d4 nim_123 ttt dab_2x2 dab_3x3')
    parser.add_argument('--K',          type=int, nargs='+', default=K_VALUES)
    parser.add_argument('--depth-n',    dest='depth_n', type=int, nargs='+',
                        default=DEPTH_NS)
    parser.add_argument('--seeds',      type=int, nargs='+', default=SEEDS)
    parser.add_argument('--timeout',    type=float, default=TIMEOUT_S,
                        help='Seconds per experiment before timeout (default: 300)')
    parser.add_argument('--no-dab-3x3', dest='no_dab_3x3', action='store_true',
                        help='Skip 3x3 Dots and Boxes (much slower)')
    parser.add_argument('--plot',       action='store_true',
                        help='Generate matplotlib bar figures')
    parser.add_argument('--out',        default='bench',
                        help='Prefix for figure filenames (default: bench)')
    args = parser.parse_args()

    registry = _make_registry(include_dab_3x3=not args.no_dab_3x3)

    if args.games:
        unknown = [g for g in args.games if g not in registry]
        if unknown:
            parser.error(f'Unknown games: {unknown}. '
                         f'Available: {list(registry.keys())}')
        registry = {g: registry[g] for g in args.games}

    print('=== L* + MCTS cross-game benchmark ===')
    print(f'Games    : {list(registry.keys())}')
    print(f'K values : {args.K}')
    print(f'depth_n  : {args.depth_n}')
    print(f'Seeds    : {args.seeds}')
    print(f'Timeout  : {args.timeout}s per run')
    print()

    results = run_sweep(registry, args.K, args.depth_n, args.seeds, args.timeout)

    print_table(results, registry)

    if args.plot:
        make_figures(results, registry, args.K, args.depth_n, args.out)

    print('Done.')


if __name__ == '__main__':
    main()
