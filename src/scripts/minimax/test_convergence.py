"""
test_convergence.py — convergence experiments for L* + MCTS strategy learner.

Sweeps over K, game depth, MCTS search depth, and seeds.
Produces two figures:
  1. Normalised strategy quality vs MCTS round  (roofline at 1.0 = optimal)
  2. Learned automaton size (states) vs MCTS round

Layout: rows = game depths, columns = seeds.
Each subplot shows one line per K value.
A separate pair of figures shows the effect of varying MCTS search depth (depth_n).
"""

import sys
import time
import itertools
import collections
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from aalpy.learning_algs import run_Lstar

from src.game.minimax.game_generator import generate_tree
from src.game.minimax.game_nfa import GameNFA
from src.game.minimax.preference_oracle import PreferenceOracle
from src.lstar_mcts.game_sul import GameSUL
from src.lstar_mcts.table_b import TableB
from src.lstar_mcts.mcts_oracle import MCTSEquivalenceOracle
from src.scripts.minimax.learner import evaluate


# -----------------------------------------------------------------------
# Experiment parameters
# -----------------------------------------------------------------------

GAME_DEPTHS = [8, 10, 12]      # game tree depth  (state system size)
K_VALUES    = [10, 25, 50, 100] # MCTS rollout budget per equivalence query
DEPTH_NS    = [1, 2, 4, 6]     # MCTS search depth (depth_n)
SEEDS       = [100, 101, 102]  # random seeds (= "3 games")

EPSILON     = 0.05
MAX_ROUNDS  = 20               # safety cap on outer loop iterations


# -----------------------------------------------------------------------
# Core experiment runner
# -----------------------------------------------------------------------

def run_experiment(game_depth: int, depth_n: int, K: int, seed: int) -> list[dict]:
    """
    Run one full L* + MCTS learning session.

    Returns a list of per-round dicts:
        round       : int   (1-indexed)
        states      : int   (automaton states after this round's L* convergence)
        normalised  : float (0 = random, 1 = optimal)
        elapsed_s   : float (wall-clock seconds for this round)
    """
    root   = generate_tree(game_depth, seed=seed)
    nfa    = GameNFA(root)
    oracle = PreferenceOracle(nfa)
    sul    = GameSUL(nfa, oracle)
    tb     = TableB()

    eq = MCTSEquivalenceOracle(
        sul=sul, nfa=nfa, oracle=oracle, table_b=tb,
        depth_N=depth_n, K=K, epsilon=EPSILON, verbose=False,
    )

    p1_inputs = list(root.children.keys())
    history: list[dict] = []

    for rnd in range(1, MAX_ROUNDS + 1):
        t0 = time.perf_counter()

        model = run_Lstar(
            alphabet=p1_inputs,
            sul=sul,
            eq_oracle=eq,
            automaton_type='mealy',
            print_level=0,
            cache_and_non_det_check=False,
        )

        for _ in range(K):
            eq._rollout(model)

        scores = evaluate(model, root)
        elapsed = time.perf_counter() - t0

        history.append({
            'round':     rnd,
            'states':    len(model.states),
            'normalised': max(0.0, min(1.0, scores['normalised'])),
            'elapsed_s': elapsed,
        })

        improvement = eq._check_for_improvement(model)
        if improvement is None:
            break   # converged

    return history


# -----------------------------------------------------------------------
# Run all experiments
# -----------------------------------------------------------------------

def run_all(game_depths, k_values, depth_ns, seeds):
    """
    Run the full parameter sweep over all combinations of
    (game_depth, depth_n, K, seed).  Skips depth_n > game_depth.

    Returns dict keyed by (game_depth, depth_n, K, seed) → history list.
    """
    results = {}
    combos  = [
        (gd, dn, K, seed)
        for gd, dn, K, seed in itertools.product(game_depths, depth_ns, k_values, seeds)
        if dn <= gd
    ]
    print(f'Total experiments: {len(combos)}')
    for idx, (gd, dn, K, seed) in enumerate(combos, 1):
        print(f'  [{idx}/{len(combos)}] game_depth={gd}  depth_n={dn}  K={K}  seed={seed}',
              end='', flush=True)
        t0   = time.perf_counter()
        hist = run_experiment(gd, dn, K, seed)
        elapsed = time.perf_counter() - t0
        results[(gd, dn, K, seed)] = hist
        print(f'  → {len(hist)} rounds  {elapsed:.1f}s')

    return results


# -----------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------

K_COLORS  = {10: '#e41a1c', 25: '#ff7f00', 50: '#4daf4a', 100: '#377eb8'}
DN_COLORS = {1: '#e41a1c',  2: '#ff7f00',  4: '#4daf4a',  6: '#377eb8'}


def _pad_history(histories: list[list[dict]], metric: str) -> np.ndarray:
    """
    Pad histories of different lengths with their last value and stack into
    a 2-D array of shape (n_runs, max_rounds).
    """
    if not histories:
        return np.array([[]])
    max_len = max(len(h) for h in histories if h)
    rows = []
    for h in histories:
        if not h:
            rows.append(np.full(max_len, np.nan))
            continue
        vals = [entry[metric] for entry in h]
        pad  = [vals[-1]] * (max_len - len(vals))   # repeat last value
        rows.append(np.array(vals + pad))
    return np.array(rows)


def _plot_sweep(results, param_values, param_colors, param_label,
                game_depths, seeds, metric, ylabel, title_prefix, ax_grid):
    """
    Fill an axis grid (rows=game_depths, cols=seeds) with one line per
    param value, showing mean ± std over repeated seeds (here each cell is
    one seed, so we just plot the raw trace).
    """
    for ri, gd in enumerate(game_depths):
        for ci, seed in enumerate(seeds):
            ax = ax_grid[ri][ci]
            ax.set_title(f'depth={gd}  seed={seed}', fontsize=8)
            ax.set_xlabel('MCTS round', fontsize=7)
            ax.set_ylabel(ylabel, fontsize=7)
            ax.tick_params(labelsize=6)

            if metric == 'normalised':
                ax.axhline(1.0, color='black', linewidth=1.2,
                           linestyle='--', label='optimal')
                ax.set_ylim(-0.05, 1.15)

            for pv in param_values:
                hist = results.get((gd, pv, seed)) or []
                if not hist:
                    continue
                xs = [e['round']  for e in hist]
                ys = [e[metric]   for e in hist]
                ax.plot(xs, ys, marker='o', markersize=3,
                        color=param_colors[pv],
                        label=f'{param_label}={pv}')

            ax.legend(fontsize=5, loc='lower right')


DIAGRAMS_DIR = Path(__file__).parents[1] / 'viz' / 'diagrams'


def make_figures(results, game_depths, k_values, depth_ns, seeds,
                 fixed_k=50, fixed_dn=None, out_prefix='convergence'):
    """
    Produce and save four figures:
      {out_prefix}_score_K.png     — normalised score, vary K   (one depth_n per subplot)
      {out_prefix}_states_K.png    — automaton states, vary K
      {out_prefix}_score_dn.png    — normalised score, vary depth_n  (one K per subplot)
      {out_prefix}_states_dn.png   — automaton states, vary depth_n

    Rows = game_depths, columns = seeds.
    """
    DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)

    # Pick a representative depth_n for the K-sweep plots
    fdn = fixed_dn if fixed_dn is not None else max(dn for dn in depth_ns)

    n_rows  = len(game_depths)
    n_cols  = len(seeds)
    figsize = (4 * n_cols, 3.5 * n_rows)

    for metric, ylabel, suffix in [
        ('normalised', 'Normalised score  (0=random, 1=optimal)', 'score'),
        ('states',     'Automaton states',                         'states'),
    ]:
        # --- Vary K (fix depth_n = fdn) ---
        k_slice = {(gd, K, seed): results.get((gd, fdn, K, seed), [])
                   for gd, K, seed in itertools.product(game_depths, k_values, seeds)}
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                                  squeeze=False, constrained_layout=True)
        fig.suptitle(f'{ylabel}  —  varying K  (depth_n={fdn})', fontsize=10)
        _plot_sweep(k_slice, k_values, K_COLORS, 'K',
                    game_depths, seeds, metric, ylabel, '', axes)
        path = DIAGRAMS_DIR / f'{out_prefix}_{suffix}_K.png'
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f'Saved: {path}')

        # --- Vary depth_n (fix K = fixed_k) ---
        dn_slice = {(gd, dn, seed): results.get((gd, dn, fixed_k, seed), [])
                    for gd, dn, seed in itertools.product(game_depths, depth_ns, seeds)}
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                                  squeeze=False, constrained_layout=True)
        fig.suptitle(f'{ylabel}  —  varying depth_n  (K={fixed_k})', fontsize=10)
        _plot_sweep(dn_slice, depth_ns, DN_COLORS, 'depth_n',
                    game_depths, seeds, metric, ylabel, '', axes)
        path = DIAGRAMS_DIR / f'{out_prefix}_{suffix}_dn.png'
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f'Saved: {path}')


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convergence experiment suite')
    parser.add_argument('--out',      default='convergence', help='Output file prefix')
    parser.add_argument('--fixed-k',  type=int, default=50,
                        help='K to use in the depth_n sweep (default 50)')
    parser.add_argument('--max-rounds', type=int, default=MAX_ROUNDS)
    args = parser.parse_args()

    MAX_ROUNDS = args.max_rounds

    print('=== L* + MCTS convergence experiments ===')
    print(f'Game depths : {GAME_DEPTHS}')
    print(f'K values    : {K_VALUES}')
    print(f'depth_n     : {DEPTH_NS}')
    print(f'Seeds       : {SEEDS}')
    print()

    results = run_all(GAME_DEPTHS, K_VALUES, DEPTH_NS, SEEDS)

    print('\nGenerating figures...')
    make_figures(
        results,
        GAME_DEPTHS, K_VALUES, DEPTH_NS, SEEDS,
        fixed_k=args.fixed_k,
        out_prefix=args.out,
    )
    print('Done.')
