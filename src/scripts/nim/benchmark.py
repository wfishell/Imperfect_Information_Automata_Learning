"""
benchmark_nim.py — convergence experiments for L* + MCTS on Nim.

Mirrors benchmark_dab.py with a pile-configuration sweep axis.

Sweeps over oracle lookahead depth, K, MCTS depth_n, pile_configs, and seeds.
Produces five figures in viz/diagrams/:

  nim_score_K.png        — normalised score vs round, varying K     (fix depth_n, [1,2,3])
  nim_states_K.png       — automaton states  vs round, varying K
  nim_score_dn.png       — normalised score  vs round, varying depth_n (fix K)
  nim_states_dn.png      — automaton states  vs round, varying depth_n
  nim_score_piles.png    — score vs oracle_depth, rows=pile_config (scaling plot)

Layout: rows = oracle_depths, columns = seeds.

Usage:
    python -m src.scripts.nim.benchmark
    python -m src.scripts.nim.benchmark --n-eval 100 --out nim_bench
    python -m src.scripts.nim.benchmark --small   # [1,2,3] only, fast
"""

import itertools
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from aalpy.learning_algs import run_Lstar

from src.game.nim.game_nfa import NimNFA
from src.game.nim.preference_oracle import NimOracle
from src.lstar_mcts.game_sul import GameSUL
from src.lstar_mcts.table_b import TableB
from src.lstar_mcts.mcts_oracle import MCTSEquivalenceOracle


# -----------------------------------------------------------------------
# Experiment parameters
# -----------------------------------------------------------------------

ORACLE_DEPTHS = [0, 1, 2, 3]      # oracle lookahead depth (0 = greedy)
K_VALUES      = [1, 10, 50, 100]
DEPTH_NS      = [2, 4, 6]
SEEDS         = [0, 1, 2]
PILE_CONFIGS  = [(1, 2, 3), (2, 3, 4), (3, 4, 5)]

N_EVAL        = 200
EVAL_EPSILON  = 0.2    # P1 plays randomly with this probability during evaluation
EPSILON       = 0.05
MAX_ROUNDS = 10

DIAGRAMS_DIR = Path(__file__).parents[1] / 'viz' / 'diagrams'

K_COLORS    = {1: '#e41a1c', 10: '#ff7f00', 50: '#4daf4a', 100: '#377eb8'}
DN_COLORS   = {2: '#e41a1c',  4: '#ff7f00',   6: '#377eb8'}
OD_COLORS   = {0: '#e41a1c',  1: '#ff7f00',   2: '#4daf4a', 3: '#377eb8'}
PILE_COLORS = {
    (1,2,3): '#4daf4a',
    (2,3,4): '#984ea3',
    (3,4,5): '#ff7f00',
}


# -----------------------------------------------------------------------
# Core experiment
# -----------------------------------------------------------------------

def run_experiment(piles: tuple, oracle_depth,
                   depth_n: int, K: int, seed: int) -> list[dict]:
    """
    Run one full L* + MCTS learning session with the outer improvement loop.

    Returns a list of per-round dicts:
        round       : int
        states      : int
        raw_score   : float  ((wins + 0.5*draws) / N_EVAL)
        elapsed_s   : float
    """
    nfa    = NimNFA(piles=piles)
    oracle = NimOracle(nfa, depth=oracle_depth)
    sul    = GameSUL(nfa, oracle)
    tb     = TableB()

    eq = MCTSEquivalenceOracle(
        sul=sul, nfa=nfa, oracle=oracle, table_b=tb,
        depth_N=depth_n, K=K, epsilon=EPSILON, verbose=False,
    )

    history: list[dict] = []

    for rnd in range(1, MAX_ROUNDS + 1):
        t0 = time.perf_counter()

        model = run_Lstar(
            alphabet=nfa.alphabet,
            sul=sul,
            eq_oracle=eq,
            automaton_type='mealy',
            print_level=0,
            cache_and_non_det_check=False,
        )

        for _ in range(K):
            eq._rollout(model)

        losses, draws, wins = _eval_vs_adversarial(model, nfa, N_EVAL, seed=seed)
        elapsed = time.perf_counter() - t0

        history.append({
            'round':     rnd,
            'states':    len(model.states),
            'raw_score': (wins + 0.5 * draws) / N_EVAL,
            'elapsed_s': elapsed,
        })

        improvement = eq._check_for_improvement(model)
        if improvement is None:
            break

    return history


def _eval_vs_adversarial(model, nfa: NimNFA,
                          n_games: int, seed: int) -> tuple[int, int, int]:
    """Return (losses, draws, wins) for learned P2 vs epsilon-greedy adversarial P1."""
    adversary = NimOracle(nfa, depth=None)
    rng = random.Random(seed)
    losses = draws = wins = 0

    for _ in range(n_games):
        state = nfa.root
        model.reset_to_initial()

        while not state.is_terminal():
            available = list(state.children.keys())
            if rng.random() < EVAL_EPSILON:
                p1_move = rng.choice(available)
            else:
                p1_move = min(
                    available,
                    key=lambda mv: adversary._minimax(state.children[mv], None),
                )
            p2_move = model.step(p1_move)
            state   = state.children[p1_move]

            if state.is_terminal():
                break

            if p2_move not in state.children:
                p2_move = next(iter(state.children))
            state = state.children[p2_move]

        w = state.winner()
        if w == 'P1':   losses += 1
        elif w == 'P2': wins   += 1
        else:           draws  += 1

    return losses, draws, wins


# -----------------------------------------------------------------------
# Full sweep
# -----------------------------------------------------------------------

def run_all(pile_configs, oracle_depths, k_values, depth_ns, seeds):
    """
    Sweep all (piles, oracle_depth, depth_n, K, seed) combinations.
    Returns dict keyed by (piles, oracle_depth, depth_n, K, seed) → history.
    """
    combos = list(itertools.product(pile_configs, oracle_depths, depth_ns, k_values, seeds))
    print(f'Total experiments: {len(combos)}')
    results = {}

    for idx, (piles, od, dn, K, seed) in enumerate(combos, 1):
        od_str = str(od) if od is not None else '∞'
        print(
            f'  [{idx}/{len(combos)}]  piles={list(piles)}  oracle_depth={od_str:>2}  '
            f'depth_n={dn}  K={K}  seed={seed}',
            end='', flush=True,
        )
        t0   = time.perf_counter()
        hist = run_experiment(piles, od, dn, K, seed)
        elapsed = time.perf_counter() - t0
        results[(piles, od, dn, K, seed)] = hist
        final = hist[-1]
        print(f'  → {len(hist)} rounds  score={final["raw_score"]:.3f}  '
              f'states={final["states"]}  {elapsed:.1f}s')

    return results


# -----------------------------------------------------------------------
# Normalisation
# -----------------------------------------------------------------------

def normalise(results, pile_configs, oracle_depths, k_values, depth_ns, seeds):
    """
    Add 'normalised' key: raw_score / reference_raw_score
    where reference = deepest oracle_depth for the same (piles, depth_n, K, seed).
    """
    ref_od = max(oracle_depths)
    for piles in pile_configs:
        for od in oracle_depths:
            for dn in depth_ns:
                for K in k_values:
                    for seed in seeds:
                        opt_hist  = results.get((piles, ref_od, dn, K, seed), [])
                        opt_score = opt_hist[-1]['raw_score'] if opt_hist else None

                        hist = results.get((piles, od, dn, K, seed), [])
                        for entry in hist:
                            if opt_score and opt_score > 0:
                                entry['normalised'] = min(
                                    1.0, entry['raw_score'] / opt_score
                                )
                            else:
                                entry['normalised'] = entry['raw_score']


# -----------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------

def _plot_sweep(results_slice, param_values, param_colors, param_label,
                row_values, row_title_fn, seeds, metric, ylabel, ax_grid):
    """
    Generic sweep plotter.
    results_slice : dict keyed by (row_value, param_value, seed)
    row_title_fn  : callable(row_value) -> str for subplot title
    """
    for ri, rv in enumerate(row_values):
        for ci, seed in enumerate(seeds):
            ax = ax_grid[ri][ci]
            ax.set_title(f'{row_title_fn(rv)}  seed={seed}', fontsize=8)
            ax.set_xlabel('MCTS round', fontsize=7)
            ax.set_ylabel(ylabel, fontsize=7)
            ax.tick_params(labelsize=6)

            if metric == 'normalised':
                ax.axhline(1.0, color='black', linewidth=1.2,
                           linestyle='--', label='optimal')
                ax.set_ylim(-0.05, 1.15)

            for pv in param_values:
                hist = results_slice.get((rv, pv, seed)) or []
                if not hist:
                    continue
                xs = [e['round'] for e in hist]
                ys = [e[metric]  for e in hist]
                ax.plot(xs, ys, marker='o', markersize=3,
                        color=param_colors[pv],
                        label=f'{param_label}={pv}')

            ax.legend(fontsize=5, loc='lower right')


def make_figures(results, pile_configs, oracle_depths, k_values, depth_ns, seeds,
                 fixed_k=100, fixed_dn=None, out_prefix='nim'):
    """
    Nine figures (piles=[1,2,3]):
      Rows = oracle_depths:
        {prefix}_score_K.png      — normalised score, vary K      (fix depth_n)
        {prefix}_states_K.png     — automaton states, vary K
        {prefix}_score_dn.png     — normalised score, vary depth_n (fix K)
        {prefix}_states_dn.png    — automaton states, vary depth_n

      Rows = depth_ns:
        {prefix}_score_K_dn.png   — normalised score, vary K      (fix oracle_depth)
        {prefix}_states_K_dn.png  — automaton states, vary K
        {prefix}_score_od_dn.png  — normalised score, vary oracle_depth (fix K)
        {prefix}_states_od_dn.png — automaton states, vary oracle_depth

      Scaling:
        {prefix}_score_piles.png  — score vs oracle_depth, rows=pile_config
    """
    DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)

    fdn        = fixed_dn if fixed_dn is not None else max(depth_ns)
    fod        = max(oracle_depths)
    base_piles = (1, 2, 3)

    od_title = lambda od: f'oracle_depth={od}'
    dn_title = lambda dn: f'depth_n={dn}'

    for metric, ylabel, suffix in [
        ('normalised', 'Normalised score  (1.0 = matches optimal oracle)', 'score'),
        ('states',     'Automaton states',                                  'states'),
    ]:
        n_od_rows = len(oracle_depths)
        n_dn_rows = len(depth_ns)
        n_cols    = len(seeds)
        od_figsize = (4 * n_cols, 3.5 * n_od_rows)
        dn_figsize = (4 * n_cols, 3.5 * n_dn_rows)

        # --- Rows=oracle_depths: vary K (fix depth_n=fdn) ---
        k_slice = {
            (od, K, seed): results.get((base_piles, od, fdn, K, seed), [])
            for od, K, seed in itertools.product(oracle_depths, k_values, seeds)
        }
        fig, axes = plt.subplots(n_od_rows, n_cols, figsize=od_figsize,
                                  squeeze=False, constrained_layout=True)
        fig.suptitle(f'{ylabel}  —  varying K  (depth_n={fdn}, piles={list(base_piles)})', fontsize=10)
        _plot_sweep(k_slice, k_values, K_COLORS, 'K',
                    oracle_depths, od_title, seeds, metric, ylabel, axes)
        path = DIAGRAMS_DIR / f'{out_prefix}_{suffix}_K.png'
        fig.savefig(path, dpi=120); plt.close(fig); print(f'Saved: {path}')

        # --- Rows=oracle_depths: vary depth_n (fix K=fixed_k) ---
        dn_slice = {
            (od, dn, seed): results.get((base_piles, od, dn, fixed_k, seed), [])
            for od, dn, seed in itertools.product(oracle_depths, depth_ns, seeds)
        }
        fig, axes = plt.subplots(n_od_rows, n_cols, figsize=od_figsize,
                                  squeeze=False, constrained_layout=True)
        fig.suptitle(f'{ylabel}  —  varying depth_n  (K={fixed_k}, piles={list(base_piles)})', fontsize=10)
        _plot_sweep(dn_slice, depth_ns, DN_COLORS, 'depth_n',
                    oracle_depths, od_title, seeds, metric, ylabel, axes)
        path = DIAGRAMS_DIR / f'{out_prefix}_{suffix}_dn.png'
        fig.savefig(path, dpi=120); plt.close(fig); print(f'Saved: {path}')

        # --- Rows=depth_ns: vary K (fix oracle_depth=fod) ---
        k_dn_slice = {
            (dn, K, seed): results.get((base_piles, fod, dn, K, seed), [])
            for dn, K, seed in itertools.product(depth_ns, k_values, seeds)
        }
        fig, axes = plt.subplots(n_dn_rows, n_cols, figsize=dn_figsize,
                                  squeeze=False, constrained_layout=True)
        fig.suptitle(f'{ylabel}  —  varying K  by depth_n  (oracle_depth={fod})', fontsize=10)
        _plot_sweep(k_dn_slice, k_values, K_COLORS, 'K',
                    depth_ns, dn_title, seeds, metric, ylabel, axes)
        path = DIAGRAMS_DIR / f'{out_prefix}_{suffix}_K_dn.png'
        fig.savefig(path, dpi=120); plt.close(fig); print(f'Saved: {path}')

        # --- Rows=depth_ns: vary oracle_depth (fix K=fixed_k) ---
        od_dn_slice = {
            (dn, od, seed): results.get((base_piles, od, dn, fixed_k, seed), [])
            for dn, od, seed in itertools.product(depth_ns, oracle_depths, seeds)
        }
        fig, axes = plt.subplots(n_dn_rows, n_cols, figsize=dn_figsize,
                                  squeeze=False, constrained_layout=True)
        fig.suptitle(f'{ylabel}  —  varying oracle_depth  by depth_n  (K={fixed_k})', fontsize=10)
        _plot_sweep(od_dn_slice, oracle_depths, OD_COLORS, 'oracle_depth',
                    depth_ns, dn_title, seeds, metric, ylabel, axes)
        path = DIAGRAMS_DIR / f'{out_prefix}_{suffix}_od_dn.png'
        fig.savefig(path, dpi=120); plt.close(fig); print(f'Saved: {path}')

    # --- Pile scaling: score vs oracle_depth, rows=pile_config ---
    n_piles = len(pile_configs)
    fig, axes = plt.subplots(n_piles, n_cols, figsize=(4 * n_cols, 3.5 * n_piles),
                              squeeze=False, constrained_layout=True)
    fig.suptitle(f'Normalised score vs oracle depth — pile config comparison  '
                 f'(K={fixed_k}, depth_n={fdn})', fontsize=10)

    depth_labels = [str(od) if od is not None else '∞' for od in oracle_depths]

    for ri, piles in enumerate(pile_configs):
        for ci, seed in enumerate(seeds):
            ax = axes[ri][ci]
            ax.set_title(f'piles={list(piles)}  seed={seed}', fontsize=8)
            ax.set_xlabel('Oracle depth', fontsize=7)
            ax.set_ylabel('Normalised score', fontsize=7)
            ax.tick_params(labelsize=6)
            ax.axhline(1.0, color='black', linewidth=1.2, linestyle='--', label='optimal')
            ax.set_ylim(-0.05, 1.15)

            scores = []
            for od in oracle_depths:
                hist = results.get((piles, od, fdn, fixed_k, seed), [])
                scores.append(hist[-1]['normalised'] if hist else np.nan)

            ax.plot(depth_labels, scores, marker='o', markersize=4,
                    color=PILE_COLORS.get(piles, '#333333'),
                    label=str(list(piles)))
            ax.legend(fontsize=6, loc='lower right')

    path = DIAGRAMS_DIR / f'{out_prefix}_score_piles.png'
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'Saved: {path}')


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Nim L* + MCTS benchmark sweep')
    parser.add_argument('--out',        default='nim',  help='Output file prefix')
    parser.add_argument('--n-eval',     type=int, default=N_EVAL)
    parser.add_argument('--fixed-k',    type=int, default=100)
    parser.add_argument('--max-rounds', type=int, default=MAX_ROUNDS)
    parser.add_argument('--small',      action='store_true',
                        help='Only run [1,2,3] piles (faster)')
    args = parser.parse_args()

    N_EVAL     = args.n_eval
    MAX_ROUNDS = args.max_rounds
    piles      = [(1, 2, 3)] if args.small else PILE_CONFIGS

    print('=== Nim L* + MCTS benchmark ===')
    print(f'Pile configs  : {piles}')
    print(f'Oracle depths : {ORACLE_DEPTHS}')
    print(f'K values      : {K_VALUES}')
    print(f'MCTS depth_n  : {DEPTH_NS}')
    print(f'Seeds         : {SEEDS}')
    print(f'Eval games/rnd: {N_EVAL}')
    print(f'Max rounds    : {MAX_ROUNDS}')
    print()

    results = run_all(piles, ORACLE_DEPTHS, K_VALUES, DEPTH_NS, SEEDS)

    print('\nNormalising scores relative to optimal oracle...')
    normalise(results, piles, ORACLE_DEPTHS, K_VALUES, DEPTH_NS, SEEDS)

    print('Generating figures...')
    make_figures(
        results, piles, ORACLE_DEPTHS, K_VALUES, DEPTH_NS, SEEDS,
        fixed_k=args.fixed_k,
        out_prefix=args.out,
    )
    print('Done.')
