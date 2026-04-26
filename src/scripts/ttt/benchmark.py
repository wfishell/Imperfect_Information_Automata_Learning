"""
benchmark_ttt.py — convergence experiments for L* + MCTS on Tic-Tac-Toe.

Mirrors test_convergence.py structure:
  - Outer improvement loop: run_Lstar → K rollouts → _check_for_improvement → repeat
  - Per-round history: states and normalised score at each round
  - Normalised score relative to optimal oracle (oracle_depth=None), not 1.0

Sweeps over oracle lookahead depth, K, MCTS depth_n, and seeds.
Produces four figures in viz/diagrams/:

  ttt_score_K.png      — normalised score vs round, varying K     (fix depth_n)
  ttt_states_K.png     — automaton states  vs round, varying K
  ttt_score_dn.png     — normalised score  vs round, varying depth_n (fix K)
  ttt_states_dn.png    — automaton states  vs round, varying depth_n

Layout: rows = oracle_depths, columns = seeds.

Usage:
    python -m src.scripts.ttt.benchmark
    python -m src.scripts.ttt.benchmark --n-eval 200 --out ttt_bench
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

ORACLE_DEPTHS = [0, 1, 2, 3]      # oracle lookahead depth (0 = greedy)
K_VALUES      = [1, 10, 50, 100]
DEPTH_NS      = [2, 4, 6]
SEEDS         = [0, 1, 2]

N_EVAL        = 200
EVAL_EPSILON  = 0.2    # P1 plays randomly with this probability during evaluation
EPSILON       = 0.05
MAX_ROUNDS = 10               # safety cap on outer loop iterations

DIAGRAMS_DIR = Path(__file__).parents[1] / 'viz' / 'diagrams'

# Colour palettes
K_COLORS  = {1: '#e41a1c', 10: '#ff7f00', 50: '#4daf4a', 100: '#377eb8'}
DN_COLORS = {2: '#e41a1c',  4: '#ff7f00',   6: '#377eb8'}
OD_COLORS = {0: '#e41a1c',  1: '#ff7f00',   2: '#4daf4a', 3: '#377eb8'}


# -----------------------------------------------------------------------
# Core experiment — mirrors run_experiment in test_convergence.py
# -----------------------------------------------------------------------

def run_experiment(oracle_depth, depth_n: int, K: int, seed: int) -> list[dict]:
    """
    Run one full L* + MCTS learning session with the outer improvement loop.

    Returns a list of per-round dicts:
        round       : int   (1-indexed)
        states      : int   (automaton states after this round's L* convergence)
        raw_score   : float ((wins + 0.5*draws) / N_EVAL)
        elapsed_s   : float (wall-clock seconds for this round)
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


def _eval_vs_adversarial(model, nfa: TicTacToeNFA,
                          n_games: int, seed: int) -> tuple[int, int, int]:
    """
    Return (losses, draws, wins) for learned O vs epsilon-greedy adversarial X.

    P1 (X) plays minimax-optimal with probability (1 - EVAL_EPSILON), and
    uniformly random with probability EVAL_EPSILON.  Running n_games games
    makes the score discriminate between strong and weak learned strategies:
    a better P2 wins more of the games where P1 blunders.
    """
    from src.game.tic_tac_toe.preference_oracle import TicTacToeOracle
    adversary = TicTacToeOracle(nfa, depth=None)
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
                    key=lambda sq: adversary._minimax(state.children[sq], None),
                )
            o_move = model.step(p1_move)
            state  = state.children[p1_move]
            if state.is_terminal():
                break
            if o_move not in state.children:
                o_move = next(iter(state.children))
            state = state.children[o_move]

        w = state.winner()
        if w == 'P1':   losses += 1
        elif w == 'P2': wins   += 1
        else:           draws  += 1

    return losses, draws, wins


# -----------------------------------------------------------------------
# Full sweep — mirrors run_all in test_convergence.py
# -----------------------------------------------------------------------

def run_all(oracle_depths, k_values, depth_ns, seeds):
    """
    Sweep all (oracle_depth, depth_n, K, seed) combinations.
    Returns dict keyed by (oracle_depth, depth_n, K, seed) → history list.
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
        t0   = time.perf_counter()
        hist = run_experiment(od, dn, K, seed)
        elapsed = time.perf_counter() - t0
        results[(od, dn, K, seed)] = hist
        final = hist[-1]
        print(f'  → {len(hist)} rounds  score={final["raw_score"]:.3f}  '
              f'states={final["states"]}  {elapsed:.1f}s')

    return results


# -----------------------------------------------------------------------
# Normalisation — relative to optimal oracle per (depth_n, K, seed)
# -----------------------------------------------------------------------

def normalise(results, oracle_depths, k_values, depth_ns, seeds):
    """
    Add a 'normalised' key to every history entry.
    normalised = raw_score / reference_raw_score
    where reference = deepest oracle_depth for the same (depth_n, K, seed).
    Falls back to raw_score if the reference run is missing.
    """
    ref_od = max(od for od in oracle_depths)
    for od in oracle_depths:
        for dn in depth_ns:
            for K in k_values:
                for seed in seeds:
                    opt_hist  = results.get((ref_od, dn, K, seed), [])
                    opt_score = opt_hist[-1]['raw_score'] if opt_hist else None

                    hist = results.get((od, dn, K, seed), [])
                    for entry in hist:
                        if opt_score and opt_score > 0:
                            entry['normalised'] = min(
                                1.0, entry['raw_score'] / opt_score
                            )
                        else:
                            entry['normalised'] = entry['raw_score']


# -----------------------------------------------------------------------
# Plotting helpers — mirrors test_convergence.py
# -----------------------------------------------------------------------

def _pad_history(histories, metric):
    if not histories:
        return np.array([[]])
    max_len = max(len(h) for h in histories if h)
    rows = []
    for h in histories:
        if not h:
            rows.append(np.full(max_len, np.nan))
            continue
        vals = [e[metric] for e in h]
        pad  = [vals[-1]] * (max_len - len(vals))
        rows.append(np.array(vals + pad))
    return np.array(rows)


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
                xs = [e['round']  for e in hist]
                ys = [e[metric]   for e in hist]
                ax.plot(xs, ys, marker='o', markersize=3,
                        color=param_colors[pv],
                        label=f'{param_label}={pv}')

            ax.legend(fontsize=5, loc='lower right')


def make_figures(results, oracle_depths, k_values, depth_ns, seeds,
                 fixed_k=100, fixed_dn=None, out_prefix='ttt'):
    """
    Eight figures:
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
    """
    DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)

    fdn     = fixed_dn if fixed_dn is not None else max(depth_ns)
    fod     = oracle_depths[-1]   # deepest oracle as the fixed oracle_depth

    od_title = lambda od, _=None: f'oracle_depth={od}'
    dn_title = lambda dn, _=None: f'depth_n={dn}'

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
            (od, K, seed): results.get((od, fdn, K, seed), [])
            for od, K, seed in itertools.product(oracle_depths, k_values, seeds)
        }
        fig, axes = plt.subplots(n_od_rows, n_cols, figsize=od_figsize,
                                  squeeze=False, constrained_layout=True)
        fig.suptitle(f'{ylabel}  —  varying K  (depth_n={fdn})', fontsize=10)
        _plot_sweep(k_slice, k_values, K_COLORS, 'K',
                    oracle_depths, od_title, seeds, metric, ylabel, axes)
        path = DIAGRAMS_DIR / f'{out_prefix}_{suffix}_K.png'
        fig.savefig(path, dpi=120); plt.close(fig); print(f'Saved: {path}')

        # --- Rows=oracle_depths: vary depth_n (fix K=fixed_k) ---
        dn_slice = {
            (od, dn, seed): results.get((od, dn, fixed_k, seed), [])
            for od, dn, seed in itertools.product(oracle_depths, depth_ns, seeds)
        }
        fig, axes = plt.subplots(n_od_rows, n_cols, figsize=od_figsize,
                                  squeeze=False, constrained_layout=True)
        fig.suptitle(f'{ylabel}  —  varying depth_n  (K={fixed_k})', fontsize=10)
        _plot_sweep(dn_slice, depth_ns, DN_COLORS, 'depth_n',
                    oracle_depths, od_title, seeds, metric, ylabel, axes)
        path = DIAGRAMS_DIR / f'{out_prefix}_{suffix}_dn.png'
        fig.savefig(path, dpi=120); plt.close(fig); print(f'Saved: {path}')

        # --- Rows=depth_ns: vary K (fix oracle_depth=fod) ---
        k_dn_slice = {
            (dn, K, seed): results.get((fod, dn, K, seed), [])
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
            (dn, od, seed): results.get((od, dn, fixed_k, seed), [])
            for dn, od, seed in itertools.product(depth_ns, oracle_depths, seeds)
        }
        fig, axes = plt.subplots(n_dn_rows, n_cols, figsize=dn_figsize,
                                  squeeze=False, constrained_layout=True)
        fig.suptitle(f'{ylabel}  —  varying oracle_depth  by depth_n  (K={fixed_k})', fontsize=10)
        _plot_sweep(od_dn_slice, oracle_depths, OD_COLORS, 'oracle_depth',
                    depth_ns, dn_title, seeds, metric, ylabel, axes)
        path = DIAGRAMS_DIR / f'{out_prefix}_{suffix}_od_dn.png'
        fig.savefig(path, dpi=120); plt.close(fig); print(f'Saved: {path}')


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='TTT L* + MCTS benchmark sweep')
    parser.add_argument('--out',        default='ttt',  help='Output file prefix')
    parser.add_argument('--n-eval',     type=int, default=N_EVAL,
                        help=f'Games for evaluation per round (default {N_EVAL})')
    parser.add_argument('--fixed-k',    type=int, default=100,
                        help='K to hold fixed in the depth_n sweep (default 100)')
    parser.add_argument('--max-rounds', type=int, default=MAX_ROUNDS)
    args = parser.parse_args()

    N_EVAL     = args.n_eval
    MAX_ROUNDS = args.max_rounds

    print('=== Tic-Tac-Toe L* + MCTS benchmark ===')
    print(f'Oracle depths : {ORACLE_DEPTHS}')
    print(f'K values      : {K_VALUES}')
    print(f'MCTS depth_n  : {DEPTH_NS}')
    print(f'Seeds         : {SEEDS}')
    print(f'Eval games/rnd: {N_EVAL}')
    print(f'Max rounds    : {MAX_ROUNDS}')
    print()

    results = run_all(ORACLE_DEPTHS, K_VALUES, DEPTH_NS, SEEDS)

    print('\nNormalising scores relative to optimal oracle...')
    normalise(results, ORACLE_DEPTHS, K_VALUES, DEPTH_NS, SEEDS)

    print('Generating figures...')
    make_figures(
        results,
        ORACLE_DEPTHS, K_VALUES, DEPTH_NS, SEEDS,
        fixed_k=args.fixed_k,
        out_prefix=args.out,
    )
    print('Done.')
