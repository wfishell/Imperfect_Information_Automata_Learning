"""
RobotGridState — single state of the 4×4 gas-grid robot.

Multi-task episode model: after each successful DROP the robot returns
to idle, awaiting the next task. The episode itself only ends on:
    gas == 0                — safety failure (the spec target)
    step_count >= max_steps — timeout
The number of tasks served per episode is open-ended; the oracle scores
traces by `delivered_count` with `step_count` as a tiebreaker.

State alternates P1 → P2 → P1 → ...

P1 turn (idle, task_loc is None):
    children is keyed by task-arrival events ('TASK', (r, c)) — one per
    cell in `eligible_cells`. P1 picks which task arrives; the resulting
    state has task_loc set.

P1 turn (active task, task_loc is set):
    children is a single-key dict {observation_tuple: next_state}, where
    observation_tuple is what the controller's Mealy sees:
        (pos, gas_band, task_loc, carrying)
    This is fully-observable: every observation uniquely identifies the
    underlying env state (modulo the gas-band quantisation), so the
    minimal Mealy is genuinely a lookup table over inputs.

P2 turn:
    children is keyed by the legal action, mapping to the resulting env
    state with player flipped back to 'P1'. DROP returns to idle
    (task_loc=None, carrying=False, delivered_count++).

`value` is `delivered_count` — the running tally of completed deliveries.
"""

from __future__ import annotations
from functools import cached_property


# ----------------------------------------------------------------------
# Action tokens
# ----------------------------------------------------------------------

N            = 'N'
S            = 'S'
E            = 'E'
W            = 'W'
PICKUP       = 'PICKUP'
DROP         = 'DROP'
REFUEL       = 'REFUEL'
GO_TO_REFUEL = 'GO_TO_REFUEL'   # mode-switch output: ask env to route to refuel
NOOP         = 'NOOP'           # only used at terminal states (no action)


# ----------------------------------------------------------------------
# Observation-digest helpers
# ----------------------------------------------------------------------

GAS_BANDS = ('full', 'mid', 'low', 'critical')


def gas_band(gas: int, gas_max: int) -> str:
    """4-band quantization of the gas level."""
    if gas <= 0:           return 'critical'
    if gas == gas_max:     return 'full'
    if gas >= gas_max / 2: return 'mid'
    if gas >= 2:           return 'low'
    return 'critical'


def manhattan(p: tuple[int, int], q: tuple[int, int]) -> int:
    return abs(p[0] - q[0]) + abs(p[1] - q[1])


def direction(frm: tuple[int, int], to: tuple[int, int]) -> str:
    """One of {'N', 'S', 'E', 'W', 'here'} — row-axis priority on ties."""
    if frm == to:
        return 'here'
    dr = to[0] - frm[0]
    dc = to[1] - frm[1]
    if abs(dr) >= abs(dc) and dr != 0:
        return 'S' if dr > 0 else 'N'
    return 'E' if dc > 0 else 'W'


# ----------------------------------------------------------------------
# State
# ----------------------------------------------------------------------

class RobotGridState:
    """A single state in the gas-grid robot's two-player NFA."""

    def __init__(self,
                 rows: int = 4,
                 cols: int = 4,
                 refuel:  tuple[int, int] | None = None,
                 dropoff: tuple[int, int] | None = None,
                 gas_max:   int = 10,
                 max_steps: int = 50,
                 eligible_cells: tuple | None = None,
                 # Per-state varying fields
                 pos:        tuple[int, int]      = (0, 0),
                 gas:        int | None           = None,
                 task_loc:   tuple[int, int] | None = None,
                 carrying:   bool                 = False,
                 delivered_count: int             = 0,
                 mode:       str                  = 'normal',
                 player:     str                  = 'P1',
                 step_count: int                  = 0) -> None:
        # Env config (constants for a given NFA instance)
        self.rows      = rows
        self.cols      = cols
        # Default refuel = top-right, dropoff = bottom-right corner of grid.
        if refuel is None:
            refuel = (0, cols - 1)
        if dropoff is None:
            dropoff = (rows - 1, cols - 1)
        self.refuel    = refuel
        self.dropoff   = dropoff
        self.gas_max   = gas_max
        self.max_steps = max_steps
        # Cells where tasks may spawn (default: all non-special cells)
        if eligible_cells is None:
            eligible_cells = tuple(
                (r, c) for r in range(rows) for c in range(cols)
                if (r, c) != refuel and (r, c) != dropoff
            )
        self.eligible_cells = eligible_cells
        # Per-state varying
        self.pos             = pos
        self.gas             = gas if gas is not None else gas_max
        self.task_loc        = task_loc
        self.carrying        = carrying
        self.delivered_count = delivered_count
        # Two modes: 'normal' (controller chooses next action) or
        # 'going_to_refuel' (env auto-routes to refuel cell, refuels,
        # returns to 'normal'). Mode is set to 'going_to_refuel' when
        # the controller emits GO_TO_REFUEL in normal mode.
        self.mode            = mode
        self.player          = player
        self.step_count      = step_count

    # ------------------------------------------------------------------
    # Observation digest (the P1-input symbol at this state)
    # ------------------------------------------------------------------

    @property
    def target(self) -> tuple[int, int] | None:
        """Where the controller's implicit goal points right now."""
        if self.carrying:
            return self.dropoff
        return self.task_loc

    @property
    def observation(self) -> tuple:
        """Fully-observable digest the Mealy reads at active P1 turns:
            (pos, gas_band, task_loc, carrying)

        `mode` is NOT in the observation. Two prefixes that end at the
        same (pos, gas_band, task_loc, carrying) but in different modes
        produce the same observation symbol — but L*'s state-merging
        splits them anyway because their MQ-response signatures differ
        (normal-mode prefixes emit task-bound actions or GO_TO_REFUEL;
        refuel-mode prefixes emit auto-route actions). Keeping mode out
        of the alphabet halves closure cost without losing expressivity.
        """
        return (
            self.pos,
            gas_band(self.gas, self.gas_max),
            self.task_loc,
            self.carrying,
        )

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def is_terminal(self) -> bool:
        # Episode only ends on safety failure or step exhaustion. DROP
        # does NOT terminate — it just returns the robot to idle so the
        # next task can be served.
        return self.gas <= 0 or self.step_count >= self.max_steps

    def winner(self) -> str | None:
        """'failed' / 'timeout' / None (non-terminal). No 'delivered'
        terminal — successful drops just increment delivered_count."""
        if not self.is_terminal():
            return None
        if self.gas <= 0: return 'failed'
        return 'timeout'

    @property
    def value(self) -> int:
        """Controller's score = number of completed deliveries so far."""
        return self.delivered_count

    # ------------------------------------------------------------------
    # Children
    # ------------------------------------------------------------------

    @cached_property
    def children(self) -> dict:
        if self.is_terminal():
            return {}

        if self.player == 'P1':
            if self.task_loc is None:
                # Idle: P1 picks which task arrives. One child per
                # eligible cell, keyed by the ('TASK', loc) symbol.
                return {
                    ('TASK', loc): self._copy(task_loc=loc, player='P2')
                    for loc in self.eligible_cells
                }
            # Active task: deterministic observation digest.
            obs = self.observation
            return {obs: self._copy(player='P2')}

        # ---- P2 turn ----

        # Going-to-refuel mode: env auto-routes. EXACTLY ONE legal
        # action — Mealy's emission is effectively ignored (the SUL's
        # fallback uses the only legal child). At the refuel cell the
        # auto-action is REFUEL and we exit the mode.
        if self.mode == 'going_to_refuel':
            if self.pos == self.refuel:
                return {REFUEL: self._copy(
                    gas=self.gas_max,
                    mode='normal',
                    player='P1', step_count=self.step_count + 1,
                )}
            # Step toward refuel. Pick the row-axis-priority direction.
            dr = self.refuel[0] - self.pos[0]
            dc = self.refuel[1] - self.pos[1]
            if abs(dr) >= abs(dc) and dr != 0:
                step_action = S if dr > 0 else N
                new_pos = (self.pos[0] + (1 if dr > 0 else -1), self.pos[1])
            else:
                step_action = E if dc > 0 else W
                new_pos = (self.pos[0], self.pos[1] + (1 if dc > 0 else -1))
            return {step_action: self._copy(
                pos=new_pos, gas=self.gas - 1,
                player='P1', step_count=self.step_count + 1,
            )}

        # ---- Normal mode P2 ----

        result: dict = {}
        r, c = self.pos

        # Movement actions cost 1 gas; only legal if in-bounds.
        for action, (dr, dc) in (
            (N, (-1, 0)), (S, (1, 0)),
            (W, (0, -1)), (E, (0, 1)),
        ):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                result[action] = self._copy(
                    pos=(nr, nc), gas=self.gas - 1,
                    player='P1', step_count=self.step_count + 1,
                )

        # PICKUP: at the task cell, not carrying.
        if (self.task_loc is not None and self.pos == self.task_loc
                and not self.carrying):
            result[PICKUP] = self._copy(
                carrying=True,
                player='P1', step_count=self.step_count + 1,
            )

        # DROP: at the dropoff cell, carrying. Returns to idle.
        if self.pos == self.dropoff and self.carrying:
            result[DROP] = self._copy(
                task_loc=None,
                carrying=False,
                delivered_count=self.delivered_count + 1,
                player='P1', step_count=self.step_count + 1,
            )

        # REFUEL: only at the refuel cell. Manual variant — the
        # controller can also reach REFUEL via GO_TO_REFUEL macro.
        if self.pos == self.refuel:
            result[REFUEL] = self._copy(
                gas=self.gas_max,
                player='P1', step_count=self.step_count + 1,
            )

        # GO_TO_REFUEL: mode switch. Env auto-routes from next P2 turn
        # onward. Cost: one step (no gas; mode change only).
        result[GO_TO_REFUEL] = self._copy(
            mode='going_to_refuel',
            player='P1', step_count=self.step_count + 1,
        )

        return result

    # ------------------------------------------------------------------
    # Internal: copy with overrides
    # ------------------------------------------------------------------

    def _copy(self, **overrides) -> 'RobotGridState':
        kwargs = dict(
            rows=self.rows, cols=self.cols,
            refuel=self.refuel, dropoff=self.dropoff,
            gas_max=self.gas_max, max_steps=self.max_steps,
            eligible_cells=self.eligible_cells,
            pos=self.pos, gas=self.gas,
            task_loc=self.task_loc, carrying=self.carrying,
            delivered_count=self.delivered_count,
            mode=self.mode,
            player=self.player, step_count=self.step_count,
        )
        kwargs.update(overrides)
        return RobotGridState(**kwargs)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        flags = []
        if self.carrying:               flags.append('CARRYING')
        if self.task_loc is None:       flags.append('IDLE')
        if self.mode != 'normal':       flags.append(self.mode.upper())
        return (f'RobotGridState(pos={self.pos} gas={self.gas} '
                f'task={self.task_loc} {" ".join(flags)} '
                f'delivered={self.delivered_count} '
                f'player={self.player} step={self.step_count})')


# ----------------------------------------------------------------------
# Quick demo
# ----------------------------------------------------------------------

if __name__ == '__main__':
    s = RobotGridState()
    print('initial (idle):', s)
    print('  legal P1 inputs (task arrival events):',
          len(s.children), 'cells —', list(s.children.keys())[:3], '...')

    # Pick a task arrival: package at (2,2)
    s2 = s.children[('TASK', (2, 2))]
    print()
    print('after task arrival at (2,2):', s2)
    print('  legal P2 actions:', list(s2.children.keys()))
