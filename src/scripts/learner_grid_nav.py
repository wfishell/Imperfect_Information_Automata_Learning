"""
Learn a P2 reactive controller (Mealy machine) for the gridworld
navigation task via L*+MCTS.

Input alphabet  : observation symbols  "{direction}|{NESW-blocked-bits}"
Output alphabet : actions               {N, E, S, W}

The locally-greedy preference oracle picks an action from the LAST
observation alone (layout-free greedy rule). MCTS rollouts use the
trajectory-level preference (gas-cost compare) to find observation
prefixes where some other action beats greedy on whole-trace cost.
Counterexamples are folded in as overrides; the Mealy gains states
exactly where memory is needed.

Usage:
    python -m src.scripts.learner_grid_nav --trap u_trap
    python -m src.scripts.learner_grid_nav --random --seed 42 --k 3
    python -m src.scripts.learner_grid_nav --trap late_detour --K 300 --depth-n 10
"""

import argparse
from pathlib import Path

from src.game.grid_nav             import GridNavNFA, GridNavOracle
from src.game.grid_nav.scenarios   import trap, CURATED_TRAPS, ScenarioPool
from src.lstar_mcts.learner        import run_lstar_mcts


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Learn a P2 reactive controller for gridworld navigation.'
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--trap', choices=list(CURATED_TRAPS),
                     help='use one of the curated trap layouts')
    src.add_argument('--random', action='store_true',
                     help='use a random K-obstacle layout (with --seed and --k)')

    parser.add_argument('--grid-size',  dest='grid_size', type=int, default=5)
    parser.add_argument('--seed',       type=int,   default=42,
                        help='RNG seed (only used with --random)')
    parser.add_argument('--k',          type=int,   default=3,
                        help='number of obstacles (only with --random)')
    parser.add_argument('--max-moves',  dest='max_moves', type=int, default=30)

    parser.add_argument('--depth-n',    dest='depth_n', type=int, default=8,
                        help='MCTS rollout depth')
    parser.add_argument('--K',          type=int,   default=200,
                        help='MCTS rollout budget per equivalence query')
    parser.add_argument('--no-pac',     dest='use_pac', action='store_false',
                        help='disable PAC validation phase')
    parser.add_argument('--pac-eps',    dest='pac_eps',   type=float, default=0.05)
    parser.add_argument('--pac-delta',  dest='pac_delta', type=float, default=0.05)
    parser.add_argument('--pac-max-walk', dest='pac_max_walk', type=int, default=20)
    parser.add_argument('--pool-size',  dest='pool_size', type=int, default=0,
                        help='if >0, train on a ScenarioPool of this many '
                             'random K-obstacle layouts (multi-board mode); '
                             'default 0 = single-layout mode using --trap or --random')
    parser.add_argument('--pool-seed',  dest='pool_seed', type=int, default=0,
                        help='base RNG seed for the scenario pool')
    parser.add_argument('--verbose',    action='store_true')
    parser.add_argument('--viz',        action='store_true',
                        help='print Table B summary after training')
    args = parser.parse_args()

    if args.trap:
        nfa = trap(args.trap, grid_size=args.grid_size, max_moves=args.max_moves)
        scenario_label = f'trap={args.trap}'
        scenario_tag   = args.trap
    else:
        nfa = GridNavNFA(grid_size=args.grid_size, k=args.k,
                         seed=args.seed, max_moves=args.max_moves)
        scenario_label = f'random k={args.k} seed={args.seed}'
        scenario_tag   = f'random_k{args.k}_s{args.seed}'

    oracle    = GridNavOracle(nfa)
    p1_inputs = nfa.p1_alphabet

    print(f'Gridworld {args.grid_size}×{args.grid_size}   {scenario_label}')
    print(nfa.render())
    print(f'Obstacles: {sorted(nfa.obstacles)}')
    print(f'P1 alphabet: {len(p1_inputs)} observation symbols   '
          f'P2 alphabet: {nfa.p2_alphabet}')
    print(f'Max moves (gas budget): {args.max_moves}')
    print()

    # Multi-board mode: build a ScenarioPool sampled per MCTS rollout.
    # L*'s alphabet becomes the UNION of reachable observations across
    # pool samples, so the observation table covers every symbol any
    # pool layout can emit.
    pool           = None
    oracle_factory = None
    if args.pool_size > 0:
        pool = ScenarioPool(k=args.k, grid_size=args.grid_size,
                             max_moves=args.max_moves, base_seed=args.pool_seed)
        oracle_factory = GridNavOracle
        p1_inputs = pool.union_alphabet(args.pool_size)
        print(f'Multi-board training: pool size={args.pool_size}, '
              f'base seed={args.pool_seed}, union alphabet={len(p1_inputs)}')
        scenario_tag = f'pool{args.pool_size}_s{args.pool_seed}'
    else:
        print(f'Single-layout training on {scenario_label}')

    print(f'Running L*+MCTS  depth_n={args.depth_n}  K={args.K}  '
          f'pac={"on" if args.use_pac else "off"}')
    print()

    model, sul, eq_oracle, table_b = run_lstar_mcts(
        nfa            = nfa,
        oracle         = oracle,
        p1_inputs      = p1_inputs,
        depth_n        = args.depth_n,
        K              = args.K,
        verbose        = args.verbose,
        use_pac        = args.use_pac,
        pac_eps        = args.pac_eps,
        pac_delta      = args.pac_delta,
        pac_max_walk   = args.pac_max_walk,
        scenario_pool  = pool,
        oracle_factory = oracle_factory,
    )

    success, gas, bumps, action_seq = evaluate(model, nfa)

    print()
    print('Learned Mealy machine:')
    print(f'  States       : {len(model.states)}')
    print(f'  MQ cache     : {len(sul._cache)}')
    print(f'  Eq queries   : {eq_oracle.num_queries}')
    if hasattr(eq_oracle, 'mcts') and hasattr(eq_oracle, 'pac'):
        print(f'    MCTS phase : {eq_oracle.mcts.num_queries}')
        print(f'    PAC phase  : {eq_oracle.pac.num_queries}')
    print()

    print('Evaluation on training layout:')
    print(f'  Reached goal: {success}')
    print(f'  Gas spent  : {gas}')
    print(f'  Bumps      : {bumps}')
    print(f'  Action seq : {" ".join(action_seq)}')

    if args.viz:
        print()
        print(table_b.summary())

    out_dir = Path(__file__).parents[2] / 'outputs'
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f'learned_grid_nav_{scenario_tag}'
    model.save(str(out_file))
    print(f'\nSaved: {out_file}.dot')


def evaluate(model, nfa: GridNavNFA) -> tuple[bool, int, int, list]:
    """
    Roll out the learned Mealy on the training layout. At each step:
      P1 emits the REAL observation for the current world state;
      Mealy.step(obs) returns the action;
      world transitions accordingly.
    Returns (reached_goal, gas_used, bumps, action_sequence).
    """
    from src.game.grid_nav.board import make_observation, ACTIONS, DELTAS

    model.reset_to_initial()
    car_pos = nfa.start
    obstacles = nfa.obstacles
    n = nfa.grid_size
    goal = nfa.goal
    move_count = 0
    bumps = 0
    actions_taken: list = []

    while car_pos != goal and move_count < nfa.max_moves:
        obs = make_observation(car_pos, obstacles, n, goal)
        action = model.step(obs)
        if action not in ACTIONS:
            # Mealy emitted an unexpected symbol; treat as a bump
            actions_taken.append('?')
            bumps += 1
            move_count += 1
            continue

        ddx, ddy = DELTAS[action]
        nx, ny = car_pos[0] + ddx, car_pos[1] + ddy
        bumped = (nx < 0 or nx >= n or ny < 0 or ny >= n
                  or (nx, ny) in obstacles)
        if not bumped:
            car_pos = (nx, ny)
        else:
            bumps += 1
        actions_taken.append(action)
        move_count += 1

    return (car_pos == goal), move_count, bumps, actions_taken


if __name__ == '__main__':
    main()
