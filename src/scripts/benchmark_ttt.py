"""
benchmark_ttt.py — parameter sweep for L* + MCTS on Tic-Tac-Toe.

Sweeps oracle lookahead depth, MCTS rollout budget (K), MCTS search depth
(depth_n), and random seeds.  Produces four figures saved to viz/diagrams/:

  ttt_score_oracle_depth.png  — normalised score vs oracle lookahead
  ttt_states_oracle_depth.png — automaton states  vs oracle lookahead
  ttt_score_K.png             — normalised score  vs K
  ttt_states_K.png            — automaton states  vs K

Layout: rows = depth_n values, columns = seeds.
Each subplot draws one line per varied parameter.

Normalised score = (wins + 0.5 * draws) / n_eval
  0.0 → loses every game
  1.0 → wins or draws every game (optimal)

Usage:
    python -m src.scripts.benchmark_ttt
    python -m src.scripts.benchmark_ttt --n-eval 200 --out ttt_bench
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

from src.game.tic_tac_toe.game_nfa import TicTacToeNFA
from src.game.tic_tac_toe.preference_oracle import TicTacToeOracle
from src.lstar_mcts.game_sul import GameSUL
from src.lstar_mcts.table_b import TableB
from src.lstar_mcts.mcts_oracle import MCTSEquivalenceOracle


# -----------------------------------------------------------------------
# Experiment parameters
# -----------------------------------------------------------------------

ORACLE_DEPTHS = [1, 2, 3, None]   # None = unbounded (globally optimal)
K_VALUES      = [50, 100, 200]
DEPTH_NS      = [2, 4, 6]
SEEDS         = [0, 1, 2]

N_EVAL   = 500
EPSILON  = 0.05

DIAGRAMS_DIR = Path(__file__).parents[1] / 'viz' / 'diagrams'

# Colour palettes
K_COLORS = {50: '#e41a1c', 100: '#ff7f00', 200: '#377eb8'}
OD_COLORS = {1: '#e41a1c', 2: '#ff7f00', 3: '#4daf4a', None: '#377eb8'}
OD_LABELS = {1: 'depth=1', 2: 'depth=2', 3: 'depth=3', None: 'depth=∞ (optimal)'}


# -----------------------------------------------------------------------
# Core experiment
# -----------------------------------------------------------------------

def run_experiment(oracle_depth, depth_n: int, K: int, seed: int) -> dict:
    """
    Run one L* + MCTS session and return a result dict:
        states      : int   — automaton states
        normalised  : float — (wins + 0.5*draws) / N_EVAL
        win_rate    : float
        draw_rate   : float
        loss_rate   : float
        elapsed_s   : float
    """
    nfa    = TicTacToeNFA()
    oracle = TicTacToeOracle(nfa, depth=oracle_depth)
    sul    = GameSUL(nfa, oracle)
    tb     = TableB()

    eq = MCTSEquivalenceOracle(
        sul=sul, nfa=nfa, oracle=oracle, table_b=tb,
        depth_N=depth_n, K=K, epsilon=EPSILON, verbose=False,
    )

    p1_inputs = list(nfa.root.children.keys())

    t0 = time.perf_counter()
    model = run_Lstar(
        alphabet=p1_inputs,
        sul=sul,
        eq_oracle=eq,
        automaton_type='mealy',
        print_level=0,
        cache_and_non_det_check=False,
    )
    elapsed = time.perf_counter() - t0

    losses, draws, wins = _eval_vs_random(model, nfa, N_EVAL, seed=seed)
    normalised = (wins + 0.5 * draws) / N_EVAL

    return {
        'states':     len(model.states),
        'normalised': normalised,
        'win_rate':   wins  / N_EVAL,
        'draw_rate':  draws / N_EVAL,
        'loss_rate':  losses / N_EVAL,
        'elapsed_s':  elapsed,
    }


def _eval_vs_random(model, nfa: TicTacToeNFA, n_games: int,
                    seed: int) -> tuple[int, int, int]:
    """Return (losses, draws, wins) for learned O vs random X."""
    rng = random.Random(seed)
    losses = draws = wins = 0

    for _ in range(n_games):
        state = nfa.root
        model.reset_to_initial()

        while not state.is_terminal():
            p1_move = rng.choice(list(state.children.keys()))
            o_move  = model.step(p1_move)
            state   = state.children[p1_move]
            if state.is_terminal():
                break
            if o_move not in state.children:
                o_move = rng.choice(list(state.children.keys()))
            state = state.children[o_move]

        w = state.winner()
        if w == 'P1':   losses += 1
        elif w == 'P2': wins   += 1
        else:           draws  += 1

    return losses, draws, wins


# -----------------------------------------------------------------------
# Full sweep
# -----------------------------------------------------------------------

def run_all(oracle_depths, k_values, depth_ns, seeds):
    """
    Sweep all (oracle_depth, depth_n, K, seed) combinations.
    Returns dict keyed by (oracle_depth, depth_n, K, seed) → result dict.
    """
    combos = list(itertools.product(oracle_depths, depth_ns, k_values, seeds))
    print(f'Total experiments: {len(combos)}')
    results = {}

    for idx, (od, dn, K, seed) in enumerate(combos, 1):
        od_str = str(od) if od is not None else '∞'
        print(
            f'  [{idx}/{len(combos)}]  oracle_depth={od_str:>2}  '
            f'depth_n={dn}  K={K}  seed={seed}',
            end='', flush=True,
        )
        r = run_experiment(od, dn, K, seed)
        results[(od, dn, K, seed)] = r
        print(f'  → states={r["states"]}  score={r["normalised"]:.3f}  {r["elapsed_s"]:.1f}s')

    return results


# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------

def _x_ticks(oracle_depths):
    """Numeric x positions and string labels for oracle_depth axis."""
    xs     = list(range(len(oracle_depths)))
    labels = [str(d) if d is not None else '∞' for d in oracle_depths]
    return xs, labels


def _make_oracle_depth_figure(results, oracle_depths, k_values, depth_ns,
                               seeds, metric, ylabel, out_path):
    """
    Rows = depth_n, cols = seeds.
    X-axis = oracle_depth.  One line per K value.
    """
    n_rows, n_cols = len(depth_ns), len(seeds)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4 * n_cols, 3.5 * n_rows),
                              squeeze=False, constrained_layout=True)
    fig.suptitle(f'{ylabel}  —  varying oracle depth', fontsize=10)

    xs, xlabels = _x_ticks(oracle_depths)

    for ri, dn in enumerate(depth_ns):
        for ci, seed in enumerate(seeds):
            ax = axes[ri][ci]
            ax.set_title(f'depth_n={dn}  seed={seed}', fontsize=8)
            ax.set_xticks(xs)
            ax.set_xticklabels(xlabels, fontsize=7)
            ax.set_xlabel('Oracle lookahead depth', fontsize=7)
            ax.set_ylabel(ylabel, fontsize=7)
            ax.tick_params(labelsize=6)

            if metric == 'normalised':
                ax.axhline(1.0, color='black', linewidth=1.0,
                           linestyle='--', label='optimal')
                ax.set_ylim(-0.05, 1.15)

            for K in k_values:
                ys = [
                    results.get((od, dn, K, seed), {}).get(metric, np.nan)
                    for od in oracle_depths
                ]
                ax.plot(xs, ys, marker='o', markersize=4,
                        color=K_COLORS[K], label=f'K={K}')

            ax.legend(fontsize=5, loc='lower right')

    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'Saved: {out_path}')


def _make_K_figure(results, oracle_depths, k_values, depth_ns,
                   seeds, metric, ylabel, out_path):
    """
    Rows = depth_n, cols = seeds.
    X-axis = K.  One line per oracle_depth.
    """
    n_rows, n_cols = len(depth_ns), len(seeds)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4 * n_cols, 3.5 * n_rows),
                              squeeze=False, constrained_layout=True)
    fig.suptitle(f'{ylabel}  —  varying K', fontsize=10)

    for ri, dn in enumerate(depth_ns):
        for ci, seed in enumerate(seeds):
            ax = axes[ri][ci]
            ax.set_title(f'depth_n={dn}  seed={seed}', fontsize=8)
            ax.set_xlabel('K (rollout budget)', fontsize=7)
            ax.set_ylabel(ylabel, fontsize=7)
            ax.tick_params(labelsize=6)

            if metric == 'normalised':
                ax.axhline(1.0, color='black', linewidth=1.0,
                           linestyle='--', label='optimal')
                ax.set_ylim(-0.05, 1.15)

            for od in oracle_depths:
                ys = [
                    results.get((od, dn, K, seed), {}).get(metric, np.nan)
                    for K in k_values
                ]
                ax.plot(k_values, ys, marker='o', markersize=4,
                        color=OD_COLORS[od], label=OD_LABELS[od])

            ax.legend(fontsize=5, loc='lower right')

    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'Saved: {out_path}')


def make_figures(results, oracle_depths, k_values, depth_ns, seeds,
                 out_prefix='ttt'):
    DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)

    for metric, ylabel, suffix in [
        ('normalised', 'Normalised score  (0=always lose, 1=never lose)', 'score'),
        ('states',     'Automaton states',                                  'states'),
    ]:
        _make_oracle_depth_figure(
            results, oracle_depths, k_values, depth_ns, seeds,
            metric, ylabel,
            DIAGRAMS_DIR / f'{out_prefix}_{suffix}_oracle_depth.png',
        )
        _make_K_figure(
            results, oracle_depths, k_values, depth_ns, seeds,
            metric, ylabel,
            DIAGRAMS_DIR / f'{out_prefix}_{suffix}_K.png',
        )


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='TTT benchmark sweep')
    parser.add_argument('--out',    default='ttt',  help='Output file prefix')
    parser.add_argument('--n-eval', type=int, default=N_EVAL,
                        help=f'Games for evaluation (default {N_EVAL})')
    args = parser.parse_args()

    N_EVAL = args.n_eval

    print('=== Tic-Tac-Toe L* + MCTS benchmark ===')
    print(f'Oracle depths : {ORACLE_DEPTHS}')
    print(f'K values      : {K_VALUES}')
    print(f'MCTS depth_n  : {DEPTH_NS}')
    print(f'Seeds         : {SEEDS}')
    print(f'Eval games    : {N_EVAL}')
    print()

    results = run_all(ORACLE_DEPTHS, K_VALUES, DEPTH_NS, SEEDS)

    print('\nGenerating figures...')
    make_figures(results, ORACLE_DEPTHS, K_VALUES, DEPTH_NS, SEEDS,
                 out_prefix=args.out)
    print('Done.')
