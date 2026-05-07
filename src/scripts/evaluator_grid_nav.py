"""
Evaluator for grid-nav reactive controllers.

Given a learned Mealy machine, evaluates it on:
    1. The curated trap suite (every layout where greedy is known to fail)
    2. A held-out pool of random K-obstacle layouts (generalisation test)

Compares against the Manhattan-greedy baseline on the same layouts.
Reports success rate, average gas-on-success, bump count.

The Mealy machine is loaded by re-running the learner on a chosen
training layout (single-layout regime). For the multi-board
generalisation regime you'd train against a scenario pool and pass
the resulting model in here.

Usage:
    python -m src.scripts.evaluator_grid_nav --train-trap u_trap
    python -m src.scripts.evaluator_grid_nav --train-trap late_detour --held-out 50
"""

import argparse
import statistics
from typing import Optional

from src.game.grid_nav             import GridNavNFA, GridNavOracle
from src.game.grid_nav.board       import make_observation, ACTIONS, DELTAS
from src.game.grid_nav.scenarios   import (
    trap, CURATED_TRAPS, held_out_pool, ScenarioPool,
    CuratedPool, curated_pool, mixed_pool,
)
from src.lstar_mcts.learner        import run_lstar_mcts


# ----------------------------------------------------------------------
# Rollout helpers
# ----------------------------------------------------------------------

def run_mealy(model, nfa: GridNavNFA) -> dict:
    """Roll out a learned Mealy controller on `nfa`. Returns a dict of metrics."""
    from src.game.grid_nav.preference_oracle import _greedy_action_from_obs

    model.reset_to_initial()
    car_pos    = nfa.start
    obstacles  = nfa.obstacles
    n          = nfa.grid_size
    goal       = nfa.goal
    move_count = 0
    bumps      = 0
    fallbacks  = 0

    while car_pos != goal and move_count < nfa.max_moves:
        obs    = make_observation(car_pos, obstacles, n, goal)
        action = model.step(obs)
        # If Mealy returns an unrecognised symbol (e.g., None for an
        # observation it never saw during training), fall back to the
        # layout-free greedy default — same prior the SUL used.
        if action not in ACTIONS:
            action = _greedy_action_from_obs(obs)
            fallbacks += 1
        ddx, ddy = DELTAS[action]
        nx, ny   = car_pos[0] + ddx, car_pos[1] + ddy
        bumped   = (nx < 0 or nx >= n or ny < 0 or ny >= n
                    or (nx, ny) in obstacles)
        if not bumped:
            car_pos = (nx, ny)
        else:
            bumps += 1
        move_count += 1

    return {
        'success':   car_pos == goal,
        'gas':       move_count,
        'bumps':     bumps,
        'fallbacks': fallbacks,
    }


def run_greedy(nfa: GridNavNFA) -> dict:
    """Roll out the Manhattan-greedy preference oracle on `nfa`."""
    oracle     = GridNavOracle(nfa)
    car_pos    = nfa.start
    obstacles  = nfa.obstacles
    n          = nfa.grid_size
    goal       = nfa.goal
    move_count = 0
    bumps      = 0
    trace: list = []

    while car_pos != goal and move_count < nfa.max_moves:
        # Build the trace: alternating observation / action
        obs = make_observation(car_pos, obstacles, n, goal)
        trace.append(obs)
        action = oracle.preferred_move(trace)
        if action is None:
            break
        trace.append(action)
        ddx, ddy = DELTAS[action]
        nx, ny   = car_pos[0] + ddx, car_pos[1] + ddy
        bumped   = (nx < 0 or nx >= n or ny < 0 or ny >= n
                    or (nx, ny) in obstacles)
        if not bumped:
            car_pos = (nx, ny)
        else:
            bumps += 1
        move_count += 1

    return {
        'success': car_pos == goal,
        'gas':     move_count,
        'bumps':   bumps,
    }


# ----------------------------------------------------------------------
# Aggregation helpers
# ----------------------------------------------------------------------

def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {'n': 0, 'success_rate': 0.0,
                'avg_gas_on_success': 0.0, 'avg_bumps': 0.0}
    succs   = [r for r in rows if r['success']]
    return {
        'n':                  len(rows),
        'success_rate':       len(succs) / len(rows),
        'avg_gas_on_success': statistics.mean(r['gas']   for r in succs) if succs else float('nan'),
        'avg_bumps':          statistics.mean(r['bumps'] for r in rows),
    }


def _print_table(title: str, rows: dict) -> None:
    print(f'\n=== {title} ===')
    print(f'{"layout":<22}  {"method":<8}  {"success":>7}  {"gas":>5}  {"bumps":>5}')
    print('-' * 60)
    for layout_name, methods in rows.items():
        for method_name, m in methods.items():
            ok = 'YES' if m['success'] else 'no'
            print(f'{layout_name:<22}  {method_name:<8}  {ok:>7}  '
                  f'{m["gas"]:>5}  {m["bumps"]:>5}')
        print('-' * 60)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Evaluate a learned Mealy controller on the trap '
                    'suite + held-out pool, vs Manhattan-greedy baseline.'
    )
    parser.add_argument('--train-trap', dest='train_trap',
                        choices=list(CURATED_TRAPS), default='u_trap',
                        help='trap layout used as the seed scenario for MCTS '
                             '(only matters when --pool-size 0)')
    parser.add_argument('--pool-mode',  dest='pool_mode', type=str,
                        choices=['none', 'random', 'curated', 'mixed'],
                        default='none',
                        help='none = single layout (--train-trap); '
                             'random = N random layouts; '
                             'curated = sample from CURATED_TRAPS; '
                             'mixed = curated + random')
    parser.add_argument('--pool-size',  dest='pool_size', type=int, default=50,
                        help='size hint for random/mixed pools '
                             '(curated uses len(CURATED_TRAPS))')
    parser.add_argument('--pool-seed',  dest='pool_seed', type=int, default=0)
    parser.add_argument('--K',          type=int, default=50)
    parser.add_argument('--depth-n',    dest='depth_n', type=int, default=5)
    parser.add_argument('--max-moves',  dest='max_moves', type=int, default=30)
    parser.add_argument('--held-out',   dest='held_out_n', type=int, default=50,
                        help='number of held-out random layouts to evaluate on')
    parser.add_argument('--no-pac',     dest='use_pac', action='store_false')
    args = parser.parse_args()

    # ---- Train ----
    nfa_train = trap(args.train_trap, max_moves=args.max_moves)

    if args.pool_mode == 'random':
        pool = ScenarioPool(k=3, max_moves=args.max_moves,
                             base_seed=args.pool_seed)
        oracle_factory = GridNavOracle
        p1_inputs      = pool.union_alphabet(args.pool_size)
        pool_label     = f'random (size={args.pool_size}, seed={args.pool_seed})'
    elif args.pool_mode == 'curated':
        pool = curated_pool(max_moves=args.max_moves)
        oracle_factory = GridNavOracle
        p1_inputs      = pool.union_alphabet()
        pool_label     = f'curated (n={len(pool.nfas)} traps)'
    elif args.pool_mode == 'mixed':
        pool = mixed_pool(n_random=args.pool_size, max_moves=args.max_moves,
                          base_seed=args.pool_seed)
        oracle_factory = GridNavOracle
        p1_inputs      = pool.union_alphabet()
        pool_label     = f'mixed (curated + {args.pool_size} random)'
    else:
        pool = None
        oracle_factory = None
        p1_inputs      = nfa_train.p1_alphabet
        pool_label     = f'single layout ({args.train_trap})'

    print(f'Training Mealy on {pool_label}  '
          f'alphabet={len(p1_inputs)}  K={args.K}  depth_n={args.depth_n}  '
          f'pac={"on" if args.use_pac else "off"}')

    oracle = GridNavOracle(nfa_train)
    model, sul, eq, table_b = run_lstar_mcts(
        nfa            = nfa_train,
        oracle         = oracle,
        p1_inputs      = p1_inputs,
        depth_n        = args.depth_n,
        K              = args.K,
        use_pac        = args.use_pac,
        scenario_pool  = pool,
        oracle_factory = oracle_factory,
    )
    print(f'  trained Mealy: {len(model.states)} states  '
          f'({eq.num_queries} EQ rounds)')

    # ---- Curated trap suite ----
    print('\nEvaluating on curated trap suite...')
    suite_results: dict = {}
    for name in CURATED_TRAPS:
        nfa = trap(name, max_moves=args.max_moves)
        suite_results[name] = {
            'mealy':  run_mealy(model, nfa),
            'greedy': run_greedy(nfa),
        }
    _print_table('Curated traps (per-layout)', suite_results)

    # ---- Held-out random pool ----
    print(f'\nEvaluating on {args.held_out_n} held-out random K=3 layouts...')
    mealy_rows, greedy_rows = [], []
    for nfa_eval in held_out_pool(n=args.held_out_n, k=3,
                                   max_moves=args.max_moves):
        mealy_rows.append(run_mealy(model, nfa_eval))
        greedy_rows.append(run_greedy(nfa_eval))

    print(f'\n=== Held-out pool aggregate (n={args.held_out_n}) ===')
    print(f'{"method":<10}  {"success_rate":>12}  {"avg_gas":>9}  {"avg_bumps":>10}')
    print('-' * 50)
    for label, rows in [('mealy', mealy_rows), ('greedy', greedy_rows)]:
        agg = aggregate(rows)
        print(f'{label:<10}  {agg["success_rate"]:>12.1%}  '
              f'{agg["avg_gas_on_success"]:>9.2f}  {agg["avg_bumps"]:>10.2f}')

    # ---- Trap suite aggregate ----
    print(f'\n=== Trap suite aggregate (n={len(CURATED_TRAPS)}) ===')
    print(f'{"method":<10}  {"success_rate":>12}  {"avg_gas":>9}  {"avg_bumps":>10}')
    print('-' * 50)
    for label, key in [('mealy', 'mealy'), ('greedy', 'greedy')]:
        rows = [suite_results[name][key] for name in CURATED_TRAPS]
        agg  = aggregate(rows)
        print(f'{label:<10}  {agg["success_rate"]:>12.1%}  '
              f'{agg["avg_gas_on_success"]:>9.2f}  {agg["avg_bumps"]:>10.2f}')


if __name__ == '__main__':
    main()
