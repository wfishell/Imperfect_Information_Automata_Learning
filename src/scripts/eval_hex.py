"""
End-to-end evaluation grid for Hex.

Sweeps size × oracle_depth × strategies. Mirrors eval_minimax.py.

NOTE: size=4 (4×4 board) is impractical — full minimax solve produces
hundreds-of-thousands-to-millions of states. Default is [3]. Pass
`--sizes 3 4` only if you have time/compute for it.

Usage:
    python -m src.scripts.eval_hex
    python -m src.scripts.eval_hex --sizes 3 --oracle-depths None 0 1 2
"""

from __future__ import annotations
import argparse
import csv
import contextlib
import os
import random
import signal
import statistics
import sys
import time
from pathlib import Path

from src.game.hex.game_nfa          import HexNFA
from src.game.hex.preference_oracle import HexOracle
from src.lstar_mcts.learner         import run_lstar_mcts

from src.eval.hex     import RandomP1, GreedyP1, OptimalP1
from src.baselines.hex import (
    RandomP2, GreedyP2, OptimalP2,
    P2_UCT_pref, P2_UCT_terminal, P2_UCT_terminal_negamax,
    P2_QLearn,
)


def play_one(p2, p1, root, mealy: bool = False) -> float:
    if mealy:
        p2.reset_to_initial()
    state, trace = root, []
    while not state.is_terminal():
        assert state.player == 'P1'
        p1_action = p1.pick(state)
        if mealy:
            p2_predicted = p2.step(p1_action)
        state = state.children[p1_action]
        trace.append(p1_action)
        if state.is_terminal():
            break
        assert state.player == 'P2'
        if mealy:
            p2_action = (p2_predicted if p2_predicted in state.children
                         else next(iter(state.children)))
        else:
            p2_action = p2.pick(state, trace)
            if p2_action not in state.children:
                p2_action = next(iter(state.children))
        state = state.children[p2_action]
        trace.append(p2_action)
    return float(state.value)


def evaluate(p2, root, p1_makers, n_games: int, mealy: bool = False) -> dict:
    out = {}
    for name, make_p1 in p1_makers.items():
        scores = []
        for s in range(n_games):
            p1 = make_p1(s)
            if hasattr(p2, 'rng'):
                p2.rng = random.Random(s)
            scores.append(play_one(p2, p1, root, mealy=mealy))
        out[name]            = statistics.mean(scores)
        out[f'{name}_w']     = sum(1 for s in scores if s ==  1)
        out[f'{name}_d']     = sum(1 for s in scores if s ==  0)
        out[f'{name}_l']     = sum(1 for s in scores if s == -1)
    return out


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
    saved = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = saved


CSV_FIELDS = [
    'shape', 'oracle_depth', 'strategy', 'params',
    'vs_random',  'vs_random_w',  'vs_random_d',  'vs_random_l',
    'vs_greedy',  'vs_greedy_w',  'vs_greedy_d',  'vs_greedy_l',
    'vs_optimal', 'vs_optimal_w', 'vs_optimal_d', 'vs_optimal_l',
    'states', 'time_s', 'timeout',
]


def save_csv(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _wdl_str(r: dict, name: str) -> str:
    if r.get(name) is None:
        return f'{"TIMEOUT":>13}'
    w, d, l = r.get(f'{name}_w', 0), r.get(f'{name}_d', 0), r.get(f'{name}_l', 0)
    return f'{w:>3}/{d:>3}/{l:>3}'


def print_summary(rows: list) -> None:
    if not rows:
        return
    print()
    print(f'{"shape":>8}  {"od":>4}  {"strategy":<25}  {"params":<22}  '
          f'{"rand W/D/L":>13}  {"gre W/D/L":>13}  {"opt W/D/L":>13}  '
          f'{"states":>6}  {"t(s)":>6}')
    print('-' * 130)
    for r in rows:
        vr = _wdl_str(r, 'vs_random')
        vg = _wdl_str(r, 'vs_greedy')
        vo = _wdl_str(r, 'vs_optimal')
        st = (f'{r["states"]:>6}'        if r.get('states')     is not None else f'{"":>6}')
        t  = r.get('time_s', 0.0)
        od = r.get('oracle_depth', '*')
        print(f'{r["shape"]:>8}  {str(od):>4}  {r["strategy"]:<25}  {r["params"]:<22}  '
              f'{vr}  {vg}  {vo}  {st}  {t:>6.1f}')


def _print_row(r: dict) -> None:
    vr = _wdl_str(r, 'vs_random')
    vg = _wdl_str(r, 'vs_greedy')
    vo = _wdl_str(r, 'vs_optimal')
    od = r.get('oracle_depth', '*')
    print(f'  od={str(od):>4}  {r["strategy"]:<25}  {r["params"]:<20}  '
          f'rand={vr}  gre={vg}  opt={vo}  ({r["time_s"]:.1f}s)')


def _set_seed(player, seed):
    player.rng = random.Random(seed)
    return player


# ----------------------------------------------------------------------
# Main grid
# ----------------------------------------------------------------------

def run_grid(sizes, oracle_depths, n_games,
             lstar_Ks, lstar_depth_ns,
             uct_Ks, ql_episodes,
             lstar_timeout, out_path) -> list:
    rows: list = []

    for size in sizes:
        shape_str = f'{size}x{size}'
        print(f'\n=== Hex {shape_str} ===')

        nfa       = HexNFA(size=size)
        root      = nfa.root
        p1_inputs = list(root.children.keys())

        print(f'  building OptimalP1 (full minimax solve)...')
        optimal_p1 = OptimalP1(size=size, seed=0)
        print(f'  OptimalP1 ready (cache: {len(optimal_p1._cache)} states)')

        p1_makers = {
            'vs_random':  lambda s: RandomP1(seed=s),
            'vs_greedy':  lambda s: _set_seed(GreedyP1(seed=0), s),
            'vs_optimal': lambda s, _op=optimal_p1: _set_seed(_op, s),
        }

        # --- Oracle-independent baselines ---
        print('  [oracle-independent baselines]')

        for name, make_p2 in [('Random',  lambda: RandomP2(seed=0)),
                              ('Greedy',  lambda: GreedyP2(seed=0)),
                              ('Optimal', lambda sz=size: OptimalP2(size=sz, seed=0))]:
            p2 = make_p2()
            t0 = time.time()
            scores = evaluate(p2, root, p1_makers, n_games)
            rows.append({'shape': shape_str, 'oracle_depth': '*',
                         'strategy': name, 'params': '',
                         **scores, 'time_s': time.time() - t0})
            _print_row(rows[-1])

        for K in uct_Ks:
            for cls_name, P2cls in [('UCT_terminal',         P2_UCT_terminal),
                                    ('UCT_terminal_negamax', P2_UCT_terminal_negamax)]:
                p2 = P2cls(K=K, c=1.4, seed=0)
                t0 = time.time()
                scores = evaluate(p2, root, p1_makers, n_games)
                rows.append({'shape': shape_str, 'oracle_depth': '*',
                             'strategy': cls_name, 'params': f'K={K}',
                             **scores, 'time_s': time.time() - t0})
                _print_row(rows[-1])

        for ep in ql_episodes:
            t0 = time.time()
            p2 = P2_QLearn(size=size, n_episodes=ep, seed=0)
            scores = evaluate(p2, root, p1_makers, n_games)
            rows.append({'shape': shape_str, 'oracle_depth': '*',
                         'strategy': 'QLearn', 'params': f'episodes={ep}',
                         **scores, 'time_s': time.time() - t0})
            _print_row(rows[-1])

        save_csv(rows, out_path)

        # --- Oracle-dependent baselines ---

        for od in oracle_depths:
            print(f'\n  [oracle_depth={od}]')
            oracle = HexOracle(nfa, depth=od)

            for K in uct_Ks:
                t0 = time.time()
                p2 = P2_UCT_pref(oracle, K=K, c=1.4, seed=0)
                scores = evaluate(p2, root, p1_makers, n_games)
                rows.append({'shape': shape_str, 'oracle_depth': od,
                             'strategy': 'UCT_pref', 'params': f'K={K}',
                             **scores, 'time_s': time.time() - t0})
                _print_row(rows[-1])

            timeouts: list[tuple] = []
            combos = sorted(
                [(dn, K) for K in lstar_Ks for dn in lstar_depth_ns],
                key=lambda c: (c[0] * c[1], c[0], c[1])
            )
            for (dn, K) in combos:
                if any(dn >= dn_to and K >= K_to for (dn_to, K_to) in timeouts):
                    rows.append({'shape': shape_str, 'oracle_depth': od,
                                 'strategy': 'L*', 'params': f'depth_n={dn} K={K}',
                                 'vs_random': None, 'vs_greedy': None, 'vs_optimal': None,
                                 'states': None, 'time_s': 0.0, 'timeout': True})
                    print(f'  L*  depth_n={dn} K={K}  SKIPPED (dominated)')
                    continue
                print(f'  L*  depth_n={dn} K={K}  running...', end=' ', flush=True)
                t0 = time.time()
                try:
                    with time_limit(lstar_timeout), silence_stdout():
                        model, _, _, _ = run_lstar_mcts(
                            nfa=nfa, oracle=oracle, p1_inputs=p1_inputs,
                            depth_n=dn, K=K, verbose=False,
                        )
                    scores = evaluate(model, root, p1_makers, n_games, mealy=True)
                    rows.append({'shape': shape_str, 'oracle_depth': od,
                                 'strategy': 'L*', 'params': f'depth_n={dn} K={K}',
                                 **scores, 'states': len(model.states),
                                 'time_s': time.time() - t0})
                    print(f'states={len(model.states)}  '
                          f'rand={_wdl_str(scores, "vs_random")}  '
                          f'gre={_wdl_str(scores, "vs_greedy")}  '
                          f'opt={_wdl_str(scores, "vs_optimal")}  '
                          f'({time.time()-t0:.1f}s)')
                except TimeoutError:
                    timeouts.append((dn, K))
                    rows.append({'shape': shape_str, 'oracle_depth': od,
                                 'strategy': 'L*', 'params': f'depth_n={dn} K={K}',
                                 'vs_random': None, 'vs_greedy': None, 'vs_optimal': None,
                                 'states': None, 'time_s': time.time() - t0, 'timeout': True})
                    print(f'TIMEOUT after {time.time()-t0:.1f}s')

            save_csv(rows, out_path)

    return rows


def _od_arg(v: str):
    return None if v.lower() in ('none', 'null', '') else int(v)


def main():
    parser = argparse.ArgumentParser(description='Eval grid for Hex.')
    parser.add_argument('--sizes', nargs='+', type=int, default=[3],
                        help='Hex board size(s). Default [3]. size=4 takes a long time.')
    parser.add_argument('--oracle-depths', nargs='+', type=_od_arg,
                        default=[None, 0, 1, 2])
    parser.add_argument('--n-games',        type=int, default=200)
    parser.add_argument('--lstar-Ks',       nargs='+', type=int, default=[10, 50, 100, 200])
    parser.add_argument('--lstar-depth-ns', nargs='+', type=int, default=[2, 3, 4])
    parser.add_argument('--uct-Ks',         nargs='+', type=int, default=[10, 50, 100, 200])
    parser.add_argument('--ql-episodes',    nargs='+', type=int, default=[1_000, 10_000, 50_000])
    parser.add_argument('--lstar-timeout',  type=int, default=600)
    parser.add_argument('--out',            type=str, default='outputs/eval_hex.csv')
    args = parser.parse_args()

    out_path = Path(args.out)
    print(f'Eval grid for Hex')
    print(f'  sizes={args.sizes}  oracle_depths={args.oracle_depths}  n_games={args.n_games}')
    print(f'  L*: depth_ns={args.lstar_depth_ns}  Ks={args.lstar_Ks}  timeout={args.lstar_timeout}s')
    print(f'  UCT Ks={args.uct_Ks}   QL episodes={args.ql_episodes}')
    print(f'  CSV: {out_path}')

    rows = run_grid(
        sizes          = args.sizes,
        oracle_depths  = args.oracle_depths,
        n_games        = args.n_games,
        lstar_Ks       = args.lstar_Ks,
        lstar_depth_ns = args.lstar_depth_ns,
        uct_Ks         = args.uct_Ks,
        ql_episodes    = args.ql_episodes,
        lstar_timeout  = args.lstar_timeout,
        out_path       = out_path,
    )
    print(f'\nDone. {len(rows)} rows -> {out_path}\n')
    print_summary(rows)


if __name__ == '__main__':
    main()
