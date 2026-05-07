"""
End-to-end evaluation grid for the minimax game.

Sweeps (depth × seed) and inside each cell evaluates every P2 strategy
on 200 games against {RandomP1, GreedyP1, OptimalP1}. Outputs a CSV
(written incrementally after each cell) and a pretty-printed summary
table to stdout.

P2 strategies evaluated per cell:
    Trivial     : RandomP2, GreedyP2, OptimalP2
    UCT (× K)   : P2_UCT_pref, P2_UCT_terminal, P2_UCT_terminal_negamax
    Q-Learning  : P2_QLearn (terminal reward) × episodes
    L* MCTS     : (depth_n × K) — our Mealy

Usage:
    python -m src.scripts.eval_minimax
    python -m src.scripts.eval_minimax --depths 4 6 --seeds 0 1 --n-games 200
    python -m src.scripts.eval_minimax --lstar-Ks 50 100 --lstar-depth-ns 1 2

Smart timeout (per (depth, seed) cell):
    If L* with (depth_n=d, K=k) times out, skip all (d' >= d, k' >= k)
    in the same cell. Timeout is per single L* run (default 10 min).
    Other baselines have no timeout (they're cheap).
"""

from __future__ import annotations
import argparse
import csv
import contextlib
import io
import os
import random
import signal
import statistics
import sys
import time
from pathlib import Path

from src.game.minimax.game_generator    import generate_tree
from src.game.minimax.game_nfa          import GameNFA
from src.game.minimax.preference_oracle import PreferenceOracle
from src.lstar_mcts.learner             import run_lstar_mcts

from src.eval.minimax     import RandomP1, GreedyP1, OptimalP1
from src.baselines.minimax import (
    RandomP2, GreedyP2, OptimalP2,
    P2_UCT_pref, P2_UCT_terminal, P2_UCT_terminal_negamax,
    P2_QLearn,
)


# ----------------------------------------------------------------------
# Game playing — unified for both Mealy (.step) and baselines (.pick)
# ----------------------------------------------------------------------

def play_one(p2, p1, root, mealy: bool = False) -> float:
    """Play one full game with given P1 and P2; return cumulative trace value."""
    if mealy:
        p2.reset_to_initial()
    node, trace, total = root, [], root.value

    while not node.is_terminal():
        assert node.player == 'P1', f'Expected P1 turn, got {node.player}'
        p1_action = p1.pick(node)

        # Mealy emits its P2 output on the P1 input it just consumed.
        if mealy:
            p2_predicted = p2.step(p1_action)

        node   = node.children[p1_action]
        trace.append(p1_action)
        total += node.value
        if node.is_terminal():
            break

        assert node.player == 'P2'
        if mealy:
            p2_action = (p2_predicted if p2_predicted in node.children
                         else next(iter(node.children)))
        else:
            p2_action = p2.pick(node, trace)

        node   = node.children[p2_action]
        trace.append(p2_action)
        total += node.value

    return total


def evaluate(p2, root, p1_makers, n_games: int, mealy: bool = False) -> dict:
    """For each P1 maker, play n_games and return mean score per opponent."""
    out = {}
    for name, make_p1 in p1_makers.items():
        scores = []
        for s in range(n_games):
            p1 = make_p1(s)
            if hasattr(p2, 'rng'):
                p2.rng = random.Random(s)
            scores.append(play_one(p2, p1, root, mealy=mealy))
        out[name] = statistics.mean(scores)
    return out


# ----------------------------------------------------------------------
# Timeout helper — POSIX signal-based, per L* run
# ----------------------------------------------------------------------

@contextlib.contextmanager
def time_limit(seconds: int):
    def handler(signum, frame):
        raise TimeoutError()
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


@contextlib.contextmanager
def silence_stdout():
    """Redirect stdout to /dev/null while inside the block.
    The L* learner emits rollout-by-rollout debug prints that we don't
    want flooding the eval output."""
    saved = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = saved


# ----------------------------------------------------------------------
# CSV / table output
# ----------------------------------------------------------------------

CSV_FIELDS = [
    'depth', 'seed', 'strategy', 'params',
    'vs_random', 'vs_greedy', 'vs_optimal',
    'states', 'time_s', 'timeout',
]


def save_csv(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)


def print_summary(rows: list) -> None:
    if not rows:
        return
    print()
    print(f'{"depth":>5}  {"seed":>4}  {"strategy":<25}  {"params":<22}  '
          f'{"vs_rand":>8}  {"vs_gre":>8}  {"vs_opt":>8}  {"states":>6}  {"t(s)":>6}')
    print('-' * 110)
    for r in rows:
        vr = (f'{r["vs_random"]:>8.2f}'  if r.get('vs_random')  is not None
              else f'{"TIMEOUT":>8}')
        vg = (f'{r["vs_greedy"]:>8.2f}'  if r.get('vs_greedy')  is not None else f'{"":>8}')
        vo = (f'{r["vs_optimal"]:>8.2f}' if r.get('vs_optimal') is not None else f'{"":>8}')
        st = (f'{r["states"]:>6}'        if r.get('states')     is not None else f'{"":>6}')
        t  = r.get('time_s', 0.0)
        print(f'{r["depth"]:>5}  {r["seed"]:>4}  {r["strategy"]:<25}  '
              f'{r["params"]:<22}  {vr}  {vg}  {vo}  {st}  {t:>6.1f}')


# ----------------------------------------------------------------------
# Main grid
# ----------------------------------------------------------------------

def run_grid(depths, seeds, n_games,
             lstar_Ks, lstar_depth_ns,
             uct_Ks, ql_episodes,
             lstar_timeout, out_path) -> list:
    rows: list = []

    for depth in depths:
        for seed in seeds:
            print(f'\n=== depth={depth}  seed={seed} ===')

            t0 = time.time()
            root      = generate_tree(depth=depth, seed=seed)
            nfa       = GameNFA(root)
            oracle    = PreferenceOracle(nfa)
            p1_inputs = list(root.children.keys())
            print(f'  tree built in {time.time() - t0:.1f}s')

            # Pre-build OptimalP1 once per cell (caches minimax solve).
            optimal_p1 = OptimalP1(root)

            p1_makers = {
                'vs_random':  lambda s: RandomP1(seed=s),
                'vs_greedy':  lambda s, _gp=GreedyP1(seed=0): _gp_with_seed(_gp, s),
                'vs_optimal': lambda s, _op=optimal_p1:        _gp_with_seed(_op, s),
            }

            # ---- Trivial baselines ----
            for name, make_p2 in [('Random',  lambda: RandomP2(seed=0)),
                                  ('Greedy',  lambda: GreedyP2(seed=0)),
                                  ('Optimal', lambda: OptimalP2(root, seed=0))]:
                p2 = make_p2()
                t0 = time.time()
                scores = evaluate(p2, root, p1_makers, n_games)
                rows.append({
                    'depth': depth, 'seed': seed,
                    'strategy': name, 'params': '',
                    **scores, 'time_s': time.time() - t0,
                })
                _print_row(rows[-1])

            # ---- UCT baselines × K ----
            for K in uct_Ks:
                for cls_name, make_p2 in [
                    ('UCT_pref',             lambda K=K: P2_UCT_pref(oracle, K=K, c=1.4, seed=0)),
                    ('UCT_terminal',         lambda K=K: P2_UCT_terminal(root, K=K, c=1.4, seed=0)),
                    ('UCT_terminal_negamax', lambda K=K: P2_UCT_terminal_negamax(root, K=K, c=1.4, seed=0)),
                ]:
                    p2 = make_p2()
                    t0 = time.time()
                    scores = evaluate(p2, root, p1_makers, n_games)
                    rows.append({
                        'depth': depth, 'seed': seed,
                        'strategy': cls_name, 'params': f'K={K}',
                        **scores, 'time_s': time.time() - t0,
                    })
                    _print_row(rows[-1])

            # ---- Q-Learning × episodes (terminal reward) ----
            for ep in ql_episodes:
                t0 = time.time()
                p2 = P2_QLearn(root, n_episodes=ep, reward_shape='terminal', seed=0)
                scores = evaluate(p2, root, p1_makers, n_games)
                rows.append({
                    'depth': depth, 'seed': seed,
                    'strategy': 'QLearn', 'params': f'episodes={ep}',
                    **scores, 'time_s': time.time() - t0,
                })
                _print_row(rows[-1])

            # ---- L* MCTS × (depth_n, K) — with smart timeout-skip ----
            timeouts: list[tuple] = []   # (depth_n, K) that already timed out

            # Sort ascending by (depth_n * K) so smaller params run first;
            # that way we can skip strict-superset combinations on timeout.
            combos = sorted(
                [(dn, K) for K in lstar_Ks for dn in lstar_depth_ns if dn <= depth],
                key=lambda c: (c[0] * c[1], c[0], c[1])
            )

            for (dn, K) in combos:
                # Skip if dominated by a timed-out combo.
                if any(dn >= dn_to and K >= K_to for (dn_to, K_to) in timeouts):
                    rows.append({
                        'depth': depth, 'seed': seed,
                        'strategy': 'L*', 'params': f'depth_n={dn} K={K}',
                        'vs_random': None, 'vs_greedy': None, 'vs_optimal': None,
                        'states': None, 'time_s': 0.0, 'timeout': True,
                    })
                    print(f'  {"L*":<25}  depth_n={dn} K={K:<3d}  SKIPPED (dominated)')
                    continue

                print(f'  {"L*":<25}  depth_n={dn} K={K:<3d}  running...', end=' ', flush=True)
                t0 = time.time()
                try:
                    with time_limit(lstar_timeout), silence_stdout():
                        model, _, _, _ = run_lstar_mcts(
                            nfa=nfa, oracle=oracle, p1_inputs=p1_inputs,
                            depth_n=dn, K=K, verbose=False,
                        )
                    scores = evaluate(model, root, p1_makers, n_games, mealy=True)
                    rows.append({
                        'depth': depth, 'seed': seed,
                        'strategy': 'L*', 'params': f'depth_n={dn} K={K}',
                        **scores, 'states': len(model.states),
                        'time_s': time.time() - t0,
                    })
                    print(f'states={len(model.states)} '
                          f'rand={scores["vs_random"]:+.2f} gre={scores["vs_greedy"]:+.2f} '
                          f'opt={scores["vs_optimal"]:+.2f} ({time.time()-t0:.1f}s)')
                except TimeoutError:
                    timeouts.append((dn, K))
                    rows.append({
                        'depth': depth, 'seed': seed,
                        'strategy': 'L*', 'params': f'depth_n={dn} K={K}',
                        'vs_random': None, 'vs_greedy': None, 'vs_optimal': None,
                        'states': None, 'time_s': time.time() - t0, 'timeout': True,
                    })
                    print(f'TIMEOUT after {time.time()-t0:.1f}s')

            # Save CSV after each (depth, seed) cell.
            save_csv(rows, out_path)

    return rows


def _gp_with_seed(player, seed):
    """Set the player's RNG to a fresh seed (for tiebreak variation) and return it."""
    player.rng = random.Random(seed)
    return player


def _print_row(r: dict) -> None:
    vr = (f'{r["vs_random"]:+.2f}'  if r.get('vs_random')  is not None else 'TIMEOUT')
    vg = (f'{r["vs_greedy"]:+.2f}'  if r.get('vs_greedy')  is not None else '-')
    vo = (f'{r["vs_optimal"]:+.2f}' if r.get('vs_optimal') is not None else '-')
    print(f'  {r["strategy"]:<25}  {r["params"]:<20}  '
          f'rand={vr} gre={vg} opt={vo} ({r["time_s"]:.1f}s)')


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Eval grid for minimax game.')
    parser.add_argument('--depths',         nargs='+', type=int, default=[10, 12, 14])
    parser.add_argument('--seeds',          nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--n-games',        type=int, default=200)
    parser.add_argument('--lstar-Ks',       nargs='+', type=int, default=[10, 50, 100, 200])
    parser.add_argument('--lstar-depth-ns', nargs='+', type=int, default=[ 2, 4, 6])
    parser.add_argument('--uct-Ks',         nargs='+', type=int, default=[10, 50, 100, 200])
    parser.add_argument('--ql-episodes',    nargs='+', type=int, default=[1_000, 10_000, 50_000])
    parser.add_argument('--lstar-timeout',  type=int, default=600,
                        help='Seconds per L* run before timeout (default 600 = 10 min)')
    parser.add_argument('--out',            type=str,
                        default='outputs/eval_minimax.csv')
    args = parser.parse_args()

    out_path = Path(args.out)
    print(f'Eval grid for minimax')
    print(f'  depths={args.depths}  seeds={args.seeds}  n_games={args.n_games}')
    print(f'  L* sweep: depth_ns={args.lstar_depth_ns}  Ks={args.lstar_Ks}  '
          f'(timeout={args.lstar_timeout}s)')
    print(f'  UCT sweep: Ks={args.uct_Ks}')
    print(f'  QL sweep:  episodes={args.ql_episodes}')
    print(f'  CSV out:   {out_path}')

    rows = run_grid(
        depths         = args.depths,
        seeds          = args.seeds,
        n_games        = args.n_games,
        lstar_Ks       = args.lstar_Ks,
        lstar_depth_ns = args.lstar_depth_ns,
        uct_Ks         = args.uct_Ks,
        ql_episodes    = args.ql_episodes,
        lstar_timeout  = args.lstar_timeout,
        out_path       = out_path,
    )

    print(f'\nDone. {len(rows)} rows written to {out_path}.\n')
    print_summary(rows)


if __name__ == '__main__':
    main()
