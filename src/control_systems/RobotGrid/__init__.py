"""
RobotGrid — a 4×4 gas-constrained delivery robot.

Same shape as the game-theory benchmarks (board / game_nfa / preference_oracle)
but the "P1" is the environment producing observations and the "P2" is the
controller emitting actions. Strict alternation P1 → P2 → P1 → ... so the
existing L*+MCTS pipeline runs unchanged.

Multi-task episode model: each successful DROP returns the robot to idle,
where P1 may prompt a new task. The episode ends only on:
    gas == 0                — safety failure (the spec target G(gas > 0))
    step_count >= max_steps — timeout

Controller actions:
    N, S, E, W   — move (cost 1 gas)
    PICKUP       — only legal at the active task cell, when not carrying
    DROP         — only legal at the dropoff cell, when carrying
    REFUEL       — only legal at the refuel cell  (must be CHOSEN, not auto)

Observation the Mealy reads at active P1 turns (fully observable):
    (pos, gas_band, task_loc, carrying)
At idle P1 turns the input is a task-arrival symbol ('TASK', (r, c))
chosen from `eligible_cells`.

Trace value (oracle's score):
    delivered_count − 0.01 × step_count

The naive preference oracle picks the manhattan-distance-minimising move
toward the implicit goal (task if not carrying, dropoff if carrying). It
does NOT consider gas — that's the planted blind spot the safety stage
catches via G(gas > 0).
"""

from src.control_systems.RobotGrid.board             import (
    RobotGridState,
    N, S, E, W, PICKUP, DROP, REFUEL, GO_TO_REFUEL, NOOP,
)
from src.control_systems.RobotGrid.game_nfa          import RobotGridNFA
from src.control_systems.RobotGrid.preference_oracle import RobotGridOracle

__all__ = [
    'RobotGridState', 'RobotGridNFA', 'RobotGridOracle',
    'N', 'S', 'E', 'W', 'PICKUP', 'DROP', 'REFUEL', 'GO_TO_REFUEL', 'NOOP',
]
