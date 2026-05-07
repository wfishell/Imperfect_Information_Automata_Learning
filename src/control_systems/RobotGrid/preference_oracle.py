"""
RobotGridOracle — the naive preference oracle for the gas-grid robot.

Naive logic (the planted blind spot):

    preferred_move(prefix) =
        PICKUP                                if at task cell, not carrying
        DROP                                  if at dropoff,  carrying
        manhattan-distance step toward target otherwise
            where target = task_loc  if not carrying
                           dropoff   if carrying

Notably absent: any awareness of gas. The oracle never returns REFUEL,
never routes through the refuel cell — even when running out is one step
away. That is the reward-hacking surface the safety stage will catch via
G(gas > 0) and patch.

`compare(t1, t2)` is preference over completed traces:
    delivered + fewer steps  >  delivered + more steps  >  not delivered
"""

from __future__ import annotations
import random

from src.control_systems.RobotGrid.board import (
    RobotGridState, manhattan,
    N, S, E, W, PICKUP, DROP, REFUEL,
)
from src.control_systems.RobotGrid.game_nfa import RobotGridNFA


# ----------------------------------------------------------------------
# Greedy step toward a target cell
# ----------------------------------------------------------------------

def _best_move_toward(pos: tuple[int, int],
                      target: tuple[int, int],
                      rows: int, cols: int,
                      rng: random.Random) -> str:
    """
    Return one of {N, S, E, W} that strictly reduces manhattan(pos, target).
    If both axes have a non-zero delta, choose the larger; ties broken by rng.
    Caller is responsible for ensuring pos != target.
    """
    dr = target[0] - pos[0]
    dc = target[1] - pos[1]
    candidates = []
    if dr > 0 and pos[0] + 1 < rows:    candidates.append((abs(dr), S))
    if dr < 0 and pos[0] - 1 >= 0:      candidates.append((abs(dr), N))
    if dc > 0 and pos[1] + 1 < cols:    candidates.append((abs(dc), E))
    if dc < 0 and pos[1] - 1 >= 0:      candidates.append((abs(dc), W))
    if not candidates:
        return N  # should not happen if pos != target and grid sane
    max_delta = max(d for d, _ in candidates)
    best = [a for d, a in candidates if d == max_delta]
    return rng.choice(best)


# ----------------------------------------------------------------------
# Oracle
# ----------------------------------------------------------------------

class RobotGridOracle:

    def __init__(self,
                 nfa: RobotGridNFA,
                 seed: int | None = 0) -> None:
        self.nfa = nfa
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preferred_move(self, prefix: list) -> str | None:
        """Naive controller-action recommendation at the state reached by
        prefix. Returns None if it isn't a controller-decision point.

        Manhattan-distance heuristic. NO gas awareness. Will never
        recommend REFUEL — that's the blind spot the safety stage targets.
        """
        state = self.nfa.get_node(prefix)
        if state is None or state.player != 'P2' or state.is_terminal():
            return None
        return self._naive_action(state)

    def compare(self, trace1: list, trace2: list) -> str:
        """Compare two traces from the controller's perspective.
        Returns 't1', 't2', or 'equal'.
        """
        v1 = self._trace_score(trace1)
        v2 = self._trace_score(trace2)
        if v1 > v2: return 't1'
        if v2 > v1: return 't2'
        return 'equal'

    # ------------------------------------------------------------------
    # Naive action picker (the heuristic, gas-blind)
    # ------------------------------------------------------------------

    def _naive_action(self, state: RobotGridState) -> str:
        # In going-to-refuel mode the env auto-routes — there's exactly
        # one legal action (the next step toward refuel, or REFUEL at
        # the cell). Just return it.
        if state.mode == 'going_to_refuel':
            kids = list(state.children.keys())
            return kids[0] if kids else N

        # If we're carrying and standing on dropoff: deliver.
        if state.carrying and state.pos == state.dropoff:
            return DROP

        # If task is here and we're not carrying: pick it up.
        if (state.task_loc is not None
                and state.pos == state.task_loc
                and not state.carrying):
            return PICKUP

        # Otherwise: move toward the implicit target.
        if state.carrying:
            target = state.dropoff
        elif state.task_loc is not None:
            target = state.task_loc
        else:
            legal = list(state.children.keys())
            return legal[0] if legal else N

        return _best_move_toward(state.pos, target,
                                  state.rows, state.cols, self.rng)

    # ------------------------------------------------------------------
    # Trace scoring (for `compare`)
    # ------------------------------------------------------------------

    def _trace_score(self, trace: list) -> float:
        """Higher is better. More deliveries dominate; among traces with
        equal delivery counts, fewer-step traces win. Step count enters
        as a small tiebreaker so it never overrides delivery count.
        """
        state = self.nfa.get_node(trace)
        if state is None:
            return -1.0
        return state.delivered_count - 0.01 * state.step_count


# ----------------------------------------------------------------------
# Quick demo
# ----------------------------------------------------------------------

if __name__ == '__main__':
    nfa    = RobotGridNFA()
    oracle = RobotGridOracle(nfa)

    # Multi-task walk: idle → user-prompted task → naive controller → drop → idle.
    # Three sequential tasks, all with the naive (gas-blind) oracle.
    user_tasks = [(2, 2), (3, 0), (1, 2)]
    task_iter = iter(user_tasks)

    state, trace = nfa.root, []
    print('Walking the naive oracle through 3 sequential tasks (gas-blind):')
    for _ in range(60):
        if state.is_terminal():
            print(f'  TERMINAL: {state.winner()} '
                  f'(gas={state.gas} delivered={state.delivered_count} '
                  f'steps={state.step_count})')
            break
        if state.player == 'P1':
            if state.task_loc is None:
                # Idle: prompt the next user task (or end if no more).
                try:
                    loc = next(task_iter)
                except StopIteration:
                    print(f'  No more tasks. delivered={state.delivered_count} '
                          f'gas={state.gas} steps={state.step_count}')
                    break
                p1_input = ('TASK', loc)
                print(f'  TASK arrives at {loc}')
            else:
                p1_input = state.observation
            trace.append(p1_input)
            state = state.children[p1_input]
        else:
            action = oracle.preferred_move(trace)
            trace.append(action)
            state = state.children[action]
            print(f'  {action:<7}  → {state}')

    print()
    print(f'Final delivered_count: {state.delivered_count}')
    print(f'Final gas: {state.gas}  (gas_max = {nfa.gas_max})')
