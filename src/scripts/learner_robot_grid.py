"""
Learn a P2 controller for the gas-grid robot via L* + MCTS + PAC + safety.

Three-stage equivalence oracle:
    MCTS    : strategy refinement using preference rollouts
    PAC     : behavioural validation on iid sampled walks
    Safety  : G(gas > 0) model-check; on violation, patches the SUL with
              spec-derived overrides

Wall-clock timeout via --lstar-timeout. Quick iteration recipe:
    python -m src.scripts.learner_robot_grid \\
        --no-safety --no-pac \\
        --task-cells "(2,2)" \\
        --K 10 --depth-n 8 --lstar-timeout 60

That collapses the alphabet (one task cell instead of 14), drops K, and
wraps a 60s wall-clock cap around L*.run() so you always get back to
the prompt with whatever Mealy was learnt.

Usage:
    python -m src.scripts.learner_robot_grid
    python -m src.scripts.learner_robot_grid --no-safety --no-pac --task-cells "(2,2)"
    python -m src.scripts.learner_robot_grid --K 100 --depth-n 12 --verbose
"""

from __future__ import annotations
import argparse
import contextlib
import random
import signal
import time
from pathlib import Path

from src.control_systems.RobotGrid                    import (
    RobotGridNFA, RobotGridOracle, RobotGridState,
    N, S, E, W, PICKUP, DROP, REFUEL,
)
from src.control_systems.RobotGrid.safety_oracle      import SafetyEqOracle
from src.lstar_mcts.game_sul                           import GameSUL
from src.lstar_mcts.table_b                            import TableB
from src.lstar_mcts.mcts_oracle                        import MCTSEquivalenceOracle
from src.lstar_mcts.pac_oracle                         import PACEqOracle
from src.lstar_mcts.composite_oracle                   import CompositeEqOracle
from src.lstar_mcts.custom_lstar                       import MealyLStar


# ----------------------------------------------------------------------
# Wall-clock timeout for lstar.run()
# ----------------------------------------------------------------------

@contextlib.contextmanager
def time_limit(seconds: int):
    """Raise TimeoutError after `seconds` of wall-clock time inside the block."""
    if seconds <= 0:
        yield
        return
    def handler(signum, frame):
        raise TimeoutError(f'L* timeout after {seconds}s')
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ----------------------------------------------------------------------
# Eval helpers — run the learned Mealy on a sequence of user-prompted tasks
# ----------------------------------------------------------------------

def play_episode(model, nfa: RobotGridNFA, task_sequence: list,
                 verbose: bool = False) -> dict:
    """Step the Mealy through an episode where P1 inputs are: a task
    arrival from `task_sequence` whenever the env is idle, otherwise
    the deterministic observation digest."""
    state = nfa.root
    model.reset_to_initial()
    task_iter = iter(task_sequence)

    while not state.is_terminal():
        if state.player == 'P1':
            if state.task_loc is None:
                try:
                    loc = next(task_iter)
                except StopIteration:
                    break
                p1_input = ('TASK', loc)
            else:
                p1_input = state.observation
            p2_output = model.step(p1_input)
            state = state.children[p1_input]
            if state.is_terminal():
                break
            if p2_output not in state.children:
                # Mealy emitted illegal action — fall back to first legal.
                p2_output = next(iter(state.children))
            if verbose:
                print(f'  obs={p1_input}  →  {p2_output}  '
                      f'(now pos={state.children[p2_output].pos} '
                      f'gas={state.children[p2_output].gas})')
            state = state.children[p2_output]
        else:
            break

    return {
        'delivered':   state.delivered_count,
        'gas_left':    state.gas,
        'steps':       state.step_count,
        'outcome':     state.winner() or 'in_progress',
        'final_state': state,
    }


def evaluate_model(model, nfa: RobotGridNFA, task_sequences: list[list],
                   verbose: bool = False) -> None:
    print()
    print(f'  {"#":>2}  {"task sequence":<32}  {"delivered":>9}  '
          f'{"gas_left":>8}  {"steps":>5}  {"outcome":<10}')
    print(f'  {"-"*2}  {"-"*32}  {"-"*9}  {"-"*8}  {"-"*5}  {"-"*10}')
    for i, seq in enumerate(task_sequences):
        seq_str = ' '.join(str(t) for t in seq)
        if len(seq_str) > 30:
            seq_str = seq_str[:29] + '…'
        result = play_episode(model, nfa, seq, verbose=verbose)
        print(f'  {i+1:>2}  {seq_str:<32}  {result["delivered"]:>9}  '
              f'{result["gas_left"]:>8}  {result["steps"]:>5}  {result["outcome"]:<10}')


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _parse_task(s: str) -> tuple[int, int]:
    s = s.strip().lstrip('(').rstrip(')')
    r, c = (int(x.strip()) for x in s.split(','))
    return (r, c)


def main():
    parser = argparse.ArgumentParser(
        description='Learn a P2 controller for the gas-grid robot via L*+MCTS+PAC+safety.'
    )
    # Env shape
    parser.add_argument('--rows',      type=int, default=4)
    parser.add_argument('--cols',      type=int, default=4)
    parser.add_argument('--gas-max',   type=int, default=10)
    parser.add_argument('--max-steps', type=int, default=50)
    parser.add_argument('--home',      type=_parse_task, default=(0, 0),
                        help='Home cell (default: (0,0))')
    parser.add_argument('--refuel',    type=_parse_task, default=None,
                        help='Refuel cell (default: top-right corner)')
    parser.add_argument('--dropoff',   type=_parse_task, default=None,
                        help='Dropoff cell (default: bottom-right corner)')
    parser.add_argument('--task-cells', nargs='+', type=_parse_task,
                        default=None,
                        help='Restrict eligible task cells (default: every '
                             'non-special cell). Smaller = far fewer L* MQs. '
                             'E.g. --task-cells "(1,1)"')
    # MCTS / L* knobs
    parser.add_argument('--depth-n',   dest='depth_n', type=int, default=8,
                        help='MCTS rollout depth. Must be long enough for '
                             'rollouts to actually deliver a package, otherwise '
                             'the score-based comparison cannot discriminate.')
    parser.add_argument('--K',         type=int, default=30,
                        help='MCTS rollout budget per equivalence query')
    parser.add_argument('--lstar-timeout', dest='lstar_timeout', type=int,
                        default=120,
                        help='Wall-clock seconds before L*.run() is killed and '
                             'the partial Mealy is returned. Set 0 to disable.')
    # Pipeline stages
    parser.add_argument('--pac-eps',   dest='pac_eps',   type=float, default=0.05)
    parser.add_argument('--pac-delta', dest='pac_delta', type=float, default=0.05)
    parser.add_argument('--no-pac',    dest='use_pac',   action='store_false',
                        help='Disable PAC validation phase')
    parser.add_argument('--no-safety', dest='use_safety', action='store_false',
                        help='Disable safety stage (run raw L*+MCTS [+ PAC])')
    parser.add_argument('--safety-max-cex', type=int, default=None,
                        help='Cap on safety violations collected per round '
                             '(default: collect ALL reachable gas=0 traces)')
    # Eval
    parser.add_argument('--tasks',     nargs='*', type=_parse_task,
                        default=[(2, 2), (3, 0), (1, 2)],
                        help='Sequence of task locations to evaluate the learned Mealy on')
    parser.add_argument('--verbose',   action='store_true')
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Build the pipeline
    # ------------------------------------------------------------------

    eligible_cells = tuple(args.task_cells) if args.task_cells else None
    nfa     = RobotGridNFA(rows=args.rows, cols=args.cols,
                            home=args.home, refuel=args.refuel, dropoff=args.dropoff,
                            gas_max=args.gas_max, max_steps=args.max_steps,
                            eligible_cells=eligible_cells)
    oracle  = RobotGridOracle(nfa)
    table_b = TableB()
    sul     = GameSUL(nfa, oracle, table_b)

    mcts = MCTSEquivalenceOracle(
        sul=sul, nfa=nfa, oracle=oracle, table_b=table_b,
        depth_N=args.depth_n, K=args.K,
        verbose=args.verbose,
    )

    stages = [mcts]
    pac    = None
    safety = None
    if args.use_pac:
        pac = PACEqOracle(
            alphabet       = nfa.p1_alphabet,
            sul            = sul,
            nfa            = nfa,
            eps            = args.pac_eps,
            delta          = args.pac_delta,
            max_walk_depth = args.max_steps,
        )
        stages.append(pac)
    if args.use_safety:
        safety = SafetyEqOracle(sul=sul, nfa=nfa, oracle=oracle,
                                 verbose=args.verbose,
                                 max_violations_per_round=args.safety_max_cex)
        stages.append(safety)

    eq    = CompositeEqOracle(*stages, verbose=args.verbose)
    lstar = MealyLStar(alphabet=nfa.p1_alphabet, sul=sul, eq_oracle=eq,
                       verbose=args.verbose)

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    print(f'RobotGrid {args.rows}x{args.cols}  gas_max={args.gas_max}  '
          f'max_steps={args.max_steps}')
    print(f'  refuel={nfa.refuel}  dropoff={nfa.dropoff}  home={nfa.home}')
    print(f'  eligible task cells (n={len(nfa.eligible_cells)}): {list(nfa.eligible_cells)[:6]}'
          f'{"..." if len(nfa.eligible_cells) > 6 else ""}')
    print(f'  |p1_alphabet|={len(nfa.p1_alphabet)}  '
          f'|p2_alphabet|={len(nfa.p2_alphabet)}')
    print(f'  pipeline: {" → ".join(s.__class__.__name__ for s in stages)}')
    print(f'  K={args.K}  depth_n={args.depth_n}  '
          f'timeout={args.lstar_timeout}s')

    # ------------------------------------------------------------------
    # Run L* (with wall-clock timeout)
    # ------------------------------------------------------------------

    print()
    print('Running L*...')
    t0 = time.time()
    timed_out = False
    model = None
    try:
        with time_limit(args.lstar_timeout):
            model = lstar.run()
    except TimeoutError as e:
        timed_out = True
        print(f'\n[!] {e}')
        # MealyLStar should still have a current hypothesis; try to recover it.
        model = getattr(lstar, 'hypothesis', None) or getattr(lstar, 'model', None)
    elapsed = time.time() - t0

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    print()
    print(f'L* finished in {elapsed:.1f}s' + ('  (TIMED OUT)' if timed_out else ''))
    if model is None:
        print('No hypothesis produced; aborting eval.')
        return

    print(f'Learned automaton:')
    print(f'  States       : {len(model.states)}')
    print(f'  Cache entries: {len(sul._cache)}')
    print(f'  Spec-locked  : {len(getattr(sul, "_spec_locked", set()))}')
    print(f'  Eq. queries  : {eq.num_queries}')
    print(f'    MCTS phase   : {mcts.num_queries}')
    if pac    is not None: print(f'    PAC  phase   : {pac.num_queries}')
    if safety is not None: print(f'    Safety phase : {safety.num_queries}'
                                  f'  (patches: {safety.n_patches_total})')

    # ------------------------------------------------------------------
    # Eval — multi-task episodes
    # ------------------------------------------------------------------

    print()
    print(f'Evaluation on user-prompted task sequences:')

    # Filter eval tasks to those actually in eligible_cells (so we don't
    # try to feed an off-alphabet symbol to the Mealy).
    eligible = set(nfa.eligible_cells)
    def filter_seq(seq):
        return [t for t in seq if t in eligible]

    sequences = [filter_seq(s) for s in [
        args.tasks,                                  # the user's main run
        [(2, 2)],                                    # short single task
        [(3, 0), (1, 3), (3, 2)],                    # three sequential tasks
        [(3, 1), (3, 2), (3, 0), (1, 1)],             # heavier sequence
    ]]
    sequences = [s for s in sequences if s]
    evaluate_model(model, nfa, sequences, verbose=False)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    out_dir = Path(__file__).parents[2] / 'outputs'
    out_dir.mkdir(exist_ok=True)
    model.save(str(out_dir / 'learned_strategy_robot_grid'))
    print(f'\nSaved: {out_dir / "learned_strategy_robot_grid.dot"}')


if __name__ == '__main__':
    main()
