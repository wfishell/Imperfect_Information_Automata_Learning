"""
Cross-game L* + MCTS benchmark.

Sweeps K, depth_n, oracle_depth, and seeds across all registered games.
oracle_depth controls how many minimax levels the preference oracle looks
ahead — the core research variable: lower depth = more suboptimal oracle.
None = fully optimal (unbounded search).

Each experiment is wrapped in a configurable timeout so a slow run cannot
block the sweep.  Results are printed as a summary table and, optionally,
saved as matplotlib figures.

Adding a new game requires one entry in GAME_REGISTRY.

Usage:
    python -m src.scripts.benchmarker
    python -m src.scripts.benchmarker --games nim_123 ttt
    python -m src.scripts.benchmarker --K 50 200 --depth-n 3 5 --oracle-depth 0 1 None
    python -m src.scripts.benchmarker --timeout 120 --plot
    python -m src.scripts.benchmarker --no-dab-3x3     # skip slow 3x3 board
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

from src.game.hex.game_nfa import HexNFA
from src.game.hex.preference_oracle import HexOracle

from src.lstar_mcts.learner import run_lstar_mcts
from src.lstar_mcts.table_b import TableB
from src.lstar_mcts.mcts_oracle import MCTSEquivalenceOracle
from src.lstar_mcts.custom_lstar import MealyLStar


# -----------------------------------------------------------------------
# Default sweep parameters
# -----------------------------------------------------------------------

K_VALUES      = [50, 100, 200]
DEPTH_NS      = [3, 5]
# None = unbounded optimal; integers = bounded lookahead (suboptimal)
ORACLE_DEPTHS = [0, 1, 2, None]
SEEDS         = [0, 1, 2]
TIMEOUT_S     = 300          # seconds per experiment
N_EVAL        = 200          # evaluation games per run

DIAGRAMS_DIR = Path(__file__).parents[2] / 'outputs' / 'benchmarks'


# -----------------------------------------------------------------------
# Build helpers
# Each returns (nfa, oracle, p1_inputs, sul_override, aux)
#   sul_override : None for most games; DaB provides its own SUL type
#   aux          : extra data forwarded to eval (e.g. game root for minimax)
# -----------------------------------------------------------------------

def _build_minimax(depth: int, seed: int, oracle_depth):
    # minimax PreferenceOracle has no depth param — oracle_depth is ignored
    root   = generate_tree(depth, seed=seed)
    nfa    = GameNFA(root)
    oracle = PreferenceOracle(nfa)
    return nfa, oracle, list(root.children.keys()), None, root


def _build_nim(piles: tuple, oracle_depth, seed: int):
    nfa    = NimNFA(piles=piles)
    oracle = NimOracle(nfa, depth=oracle_depth)
    return nfa, oracle, nfa.alphabet, None, None


def _build_ttt(oracle_depth, seed: int):
    nfa    = TicTacToeNFA()
    oracle = TicTacToeOracle(nfa, depth=oracle_depth)
    return nfa, oracle, list(nfa.root.children.keys()), None, None


def _build_dab(rows: int, cols: int, oracle_depth, seed: int):
    nfa    = DotsAndBoxesNFA(rows=rows, cols=cols)
    oracle = DotsAndBoxesOracle(nfa, depth=oracle_depth)
    sul    = DotsAndBoxesSUL(nfa, oracle)
    return nfa, oracle, list(nfa.root.children.keys()) + [PASS], sul, None


def _build_hex(size: int, oracle_depth, seed: int):
    nfa    = HexNFA(size=size)
    oracle = HexOracle(nfa, depth=oracle_depth)
    return nfa, oracle, list(nfa.root.children.keys()), None, None


# -----------------------------------------------------------------------
# Evaluation helpers — return raw_score in [0, 1]  (higher = better for P2)
# -----------------------------------------------------------------------

def _eval_minimax(model, nfa: GameNFA, root: GameNode, seed: int) -> float:
    """Normalised score: 0 = random baseline, 1 = optimal P2."""
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
            trace.append(action)
            node   = node.children[action]
            total += node.value
        return total

    seqs = _all_p1(root, [])
    opt_scores, lrn_scores, rnd_scores = [], [], []

    for seq in seqs:
        def opt_p2(trace):
            n = nfa.get_node(trace)
            return None if n is None else max(n.children,
                                              key=lambda a: _opt(n.children[a]))

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
    """Win rate + 0.5 * draw rate for learned P2 vs random P1."""
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
                    continue    # P1 completed a box

                if (p2_move is None or p2_move not in state.children
                        or (is_dab and p2_move == PASS)):
                    p2_move = rng.choice(list(state.children.keys()))
                state = state.children[p2_move]

            else:
                # P2 earned an extra turn (DaB only)
                p2_move = model.step(PASS)
                if (p2_move is None or p2_move == PASS
                        or p2_move not in state.children):
                    p2_move = rng.choice(list(state.children.keys()))
                state = state.children[p2_move]

        w = state.winner()
        if w == 'P2':     wins  += 1
        elif w == 'draw': draws += 1

    return (wins + 0.5 * draws) / N_EVAL


# -----------------------------------------------------------------------
# Game registry
#
# Each entry:
#   label         : human-readable name
#   oracle_depths : oracle_depth values supported by this game's oracle.
#                   None in the list = unbounded optimal.
#                   Games whose oracle has no depth param list only [None].
#   build         : callable(seed, oracle_depth) → (nfa, oracle, p1_inputs,
#                                                   sul_override, aux)
#   eval          : callable(model, nfa, aux, seed) → float in [0, 1]
# -----------------------------------------------------------------------

def _make_registry(include_dab_3x3: bool = True) -> dict:
    reg = {
        'minimax_d4': {
            'label':         'Minimax (depth=4)',
            # PreferenceOracle has no depth param — always globally optimal
            'oracle_depths': [None],
            'build': lambda seed, od: _build_minimax(depth=4, seed=seed,
                                                     oracle_depth=od),
            'eval':  lambda model, nfa, aux, seed: _eval_minimax(model, nfa,
                                                                  aux, seed),
        },
        'nim_123': {
            'label':         'Nim [1,2,3]',
            'oracle_depths': [0, 1, 2, None],
            'build': lambda seed, od: _build_nim(piles=(1, 2, 3),
                                                 oracle_depth=od, seed=seed),
            'eval':  lambda model, nfa, aux, seed: _eval_vs_random(model, nfa,
                                                                    seed),
        },
        'ttt': {
            'label':         'Tic-Tac-Toe',
            'oracle_depths': [0, 1, 2, None],
            'build': lambda seed, od: _build_ttt(oracle_depth=od, seed=seed),
            'eval':  lambda model, nfa, aux, seed: _eval_vs_random(model, nfa,
                                                                    seed),
        },
        'dab_2x2': {
            'label':         'Dots & Boxes 2x2',
            'oracle_depths': [0, 1, 2, None],
            'build': lambda seed, od: _build_dab(rows=2, cols=2,
                                                 oracle_depth=od, seed=seed),
            'eval':  lambda model, nfa, aux, seed: _eval_vs_random(
                model, nfa, seed, is_dab=True),
        },
    }
    if include_dab_3x3:
        reg['dab_3x3'] = {
            'label':         'Dots & Boxes 3x3',
            'oracle_depths': [0, 1, 2, None],
            'build': lambda seed, od: _build_dab(rows=3, cols=3,
                                                 oracle_depth=od, seed=seed),
            'eval':  lambda model, nfa, aux, seed: _eval_vs_random(
                model, nfa, seed, is_dab=True),
        }
    reg['hex_3x3'] = {
        'label':         'Hex 3x3',
        'oracle_depths': [0, 1, 2, None],
        'build': lambda seed, od: _build_hex(size=3, oracle_depth=od, seed=seed),
        'eval':  lambda model, nfa, aux, seed: _eval_vs_random(model, nfa, seed),
    }
    return reg


# -----------------------------------------------------------------------
# Core experiment — one (game, oracle_depth, depth_n, K, seed) run
# -----------------------------------------------------------------------

def _run_one(game_name: str, game_cfg: dict, oracle_depth,
             depth_n: int, K: int, seed: int, timeout_s: float) -> dict:
    """Run one experiment with a timeout guard. Returns a result dict."""
    base = dict(game=game_name, oracle_depth=oracle_depth,
                depth_n=depth_n, K=K, seed=seed)

    def _work():
        nfa, oracle, p1_inputs, sul_override, aux = game_cfg['build'](seed, oracle_depth)
        t0 = time.perf_counter()

        if sul_override is not None:
            # DotsAndBoxes: must build components manually (specialised SUL type)
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
            'timed_out':  False,
            'states':     len(model.states),
            'cache_size': len(sul._cache),
            'eq_queries': mcts.num_queries,
            'score':      score,
            'elapsed_s':  elapsed,
        }

    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_work)
        try:
            return future.result(timeout=timeout_s)
        except FuturesTimeout:
            future.cancel()
            return {**base, 'timed_out': True,
                    'states': None, 'cache_size': None,
                    'eq_queries': None, 'score': None, 'elapsed_s': timeout_s}
        except Exception as exc:
            return {**base, 'timed_out': False, 'error': str(exc),
                    'states': None, 'cache_size': None,
                    'eq_queries': None, 'score': None, 'elapsed_s': None}


# -----------------------------------------------------------------------
# Full sweep
# -----------------------------------------------------------------------

def run_sweep(registry: dict, oracle_depths: list, k_values: list,
              depth_ns: list, seeds: list, timeout_s: float) -> list[dict]:
    """
    For each game, intersect the requested oracle_depths with those the game
    supports, then sweep all (oracle_depth, depth_n, K, seed) combinations.
    Returns a flat list of result dicts.
    """
    combos = []
    for game, cfg in registry.items():
        active_ods = [od for od in oracle_depths if od in cfg['oracle_depths']]
        for od, dn, K, seed in itertools.product(active_ods, depth_ns,
                                                  k_values, seeds):
            combos.append((game, od, dn, K, seed))

    total = len(combos)
    print(f'Total experiments: {total}  (timeout={timeout_s}s each)\n')
    results = []

    for idx, (game, od, dn, K, seed) in enumerate(combos, 1):
        label  = registry[game]['label']
        od_str = str(od) if od is not None else 'None'
        print(f'  [{idx:>{len(str(total))}}/{total}]  {label:<24}  '
              f'oracle_depth={od_str:<4}  depth_n={dn}  K={K:>3}  seed={seed}',
              end='  ', flush=True)

        result = _run_one(game, registry[game], od, dn, K, seed, timeout_s)
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
    """Mean score, states, and time grouped by (game, oracle_depth, depth_n, K)."""
    import statistics
    from collections import defaultdict

    groups: dict = defaultdict(list)
    for r in results:
        if not r.get('timed_out') and not r.get('error') and r['score'] is not None:
            groups[(r['game'], r['oracle_depth'], r['depth_n'], r['K'])].append(r)

    hdr = (f'{"Game":<24}  {"oracle_d":>8}  {"dn":>4}  {"K":>5}  '
           f'{"score":>7}  {"states":>7}  {"time":>8}')
    print()
    print(hdr)
    print('-' * len(hdr))

    for game in registry:
        label   = registry[game]['label']
        printed = False
        for od in [0, 1, 2, None]:
            for dn in sorted({r['depth_n'] for r in results}):
                for K in sorted({r['K'] for r in results}):
                    runs = groups.get((game, od, dn, K), [])
                    if not runs:
                        continue
                    od_str = str(od) if od is not None else 'None'
                    score  = statistics.mean(r['score']     for r in runs)
                    states = statistics.mean(r['states']    for r in runs)
                    etime  = statistics.mean(r['elapsed_s'] for r in runs)
                    gl     = label if not printed else ''
                    print(f'{gl:<24}  {od_str:>8}  {dn:>4}  {K:>5}  '
                          f'{score:>7.3f}  {states:>7.1f}  {etime:>7.1f}s')
                    printed = True
        if printed:
            print()


# -----------------------------------------------------------------------
# Optional figures
# -----------------------------------------------------------------------

def make_figures(results: list[dict], registry: dict,
                 k_values: list, depth_ns: list,
                 oracle_depths: list, out_prefix: str) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed — skipping figures.')
        return

    DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)

    od_colors = {0: '#e41a1c', 1: '#ff7f00', 2: '#4daf4a', None: '#377eb8'}
    k_colors  = {k: c for k, c in zip(sorted(k_values),
                  ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3'])}

    games  = list(registry.keys())
    labels = [registry[g]['label'] for g in games]
    fixed_dn = max(depth_ns)
    fixed_k  = max(k_values)

    # Score vs oracle_depth (fixed depth_n and K)
    _bar_figure(
        results, games, labels, oracle_depths, od_colors,
        param_key='oracle_depth', param_label='oracle_depth',
        fixed_key='depth_n', fixed_val=fixed_dn, metric='score',
        title=f'Mean P2 score vs oracle depth  (depth_n={fixed_dn}, K={fixed_k})',
        extra_filter=lambda r: r['K'] == fixed_k,
        path=DIAGRAMS_DIR / f'{out_prefix}_score_by_oracle_depth.png',
    )

    # Score vs K at optimal oracle (fixed depth_n)
    _bar_figure(
        results, games, labels, k_values, k_colors,
        param_key='K', param_label='K',
        fixed_key='depth_n', fixed_val=fixed_dn, metric='score',
        title=f'Mean P2 score vs K  (depth_n={fixed_dn}, oracle_depth=None)',
        extra_filter=lambda r: r['oracle_depth'] is None,
        path=DIAGRAMS_DIR / f'{out_prefix}_score_by_K.png',
    )

    # States vs oracle_depth (fixed depth_n and K)
    _bar_figure(
        results, games, labels, oracle_depths, od_colors,
        param_key='oracle_depth', param_label='oracle_depth',
        fixed_key='depth_n', fixed_val=fixed_dn, metric='states',
        title=f'Mean automaton states vs oracle depth  (depth_n={fixed_dn}, K={fixed_k})',
        extra_filter=lambda r: r['K'] == fixed_k,
        path=DIAGRAMS_DIR / f'{out_prefix}_states_by_oracle_depth.png',
    )

    print(f'Figures saved to {DIAGRAMS_DIR}')


def _bar_figure(results, games, labels, param_values, colors,
                param_key, param_label, fixed_key, fixed_val,
                metric, title, path, extra_filter=None) -> None:
    import statistics
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(8, len(games) * 2.2), 5),
                            constrained_layout=True)
    n_params = len(param_values)
    width    = 0.8 / n_params

    for pi, pv in enumerate(param_values):
        means = []
        for game in games:
            runs = [r for r in results
                    if r['game'] == game
                    and r[param_key] == pv
                    and r[fixed_key] == fixed_val
                    and not r.get('timed_out') and not r.get('error')
                    and r[metric] is not None
                    and (extra_filter is None or extra_filter(r))]
            means.append(statistics.mean(r[metric] for r in runs) if runs else 0)

        xs     = [i + (pi - n_params / 2 + 0.5) * width for i in range(len(games))]
        pv_str = str(pv) if pv is not None else 'None'
        ax.bar(xs, means, width=width * 0.9,
               color=colors.get(pv, '#999999'),
               label=f'{param_label}={pv_str}', alpha=0.85)

    ax.set_xticks(list(range(len(games))))
    ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=8)
    ax.set_ylabel(metric, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    if metric == 'score':
        ax.set_ylim(0, 1.12)
        ax.axhline(1.0, color='black', linewidth=1, linestyle='--', alpha=0.4)

    fig.savefig(path, dpi=120)
    plt.close(fig)


# -----------------------------------------------------------------------
# CLI helpers
# -----------------------------------------------------------------------

def _parse_oracle_depths(values: list[str]) -> list:
    """Convert CLI strings to int | None — e.g. ['0', '1', 'None'] → [0, 1, None]."""
    out = []
    for v in values:
        out.append(None if v.lower() == 'none' else int(v))
    return out


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Cross-game L* + MCTS benchmark.'
    )
    parser.add_argument('--games', nargs='+', default=None,
                        help='Subset of games. '
                             'Choices: minimax_d4 nim_123 ttt dab_2x2 dab_3x3 hex_3x3')
    parser.add_argument('--K',            type=int,   nargs='+', default=K_VALUES)
    parser.add_argument('--depth-n',      dest='depth_n', type=int, nargs='+',
                        default=DEPTH_NS)
    parser.add_argument('--oracle-depth', dest='oracle_depth', nargs='+',
                        default=[str(od) if od is not None else 'None'
                                 for od in ORACLE_DEPTHS],
                        help='Oracle depth values to sweep. Use "None" for optimal. '
                             'Example: --oracle-depth 0 1 None')
    parser.add_argument('--seeds',        type=int,   nargs='+', default=SEEDS)
    parser.add_argument('--timeout',      type=float, default=TIMEOUT_S,
                        help='Seconds per run before timeout (default: 300)')
    parser.add_argument('--no-dab-3x3',  dest='no_dab_3x3', action='store_true',
                        help='Skip 3x3 Dots and Boxes')
    parser.add_argument('--plot',         action='store_true',
                        help='Generate matplotlib figures in outputs/benchmarks/')
    parser.add_argument('--out',          default='bench',
                        help='Figure filename prefix (default: bench)')
    args = parser.parse_args()

    oracle_depths = _parse_oracle_depths(args.oracle_depth)
    registry      = _make_registry(include_dab_3x3=not args.no_dab_3x3)

    if args.games:
        unknown = [g for g in args.games if g not in registry]
        if unknown:
            parser.error(f'Unknown games: {unknown}. '
                         f'Available: {list(registry.keys())}')
        registry = {g: registry[g] for g in args.games}

    print('=== L* + MCTS cross-game benchmark ===')
    print(f'Games         : {list(registry.keys())}')
    print(f'oracle_depths : {oracle_depths}')
    print(f'K values      : {args.K}')
    print(f'depth_n       : {args.depth_n}')
    print(f'Seeds         : {args.seeds}')
    print(f'Timeout       : {args.timeout}s per run')
    print()

    results = run_sweep(registry, oracle_depths, args.K, args.depth_n,
                        args.seeds, args.timeout)

    print_table(results, registry)

    if args.plot:
        make_figures(results, registry, args.K, args.depth_n,
                     oracle_depths, args.out)

    print('Done.')


if __name__ == '__main__':
    main()
