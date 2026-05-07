"""
Preference oracle for the gridworld navigation game.

The oracle is *locally-optimal* and *gas-aware*: it ranks traces by
estimated total gas cost to reach the goal, where

    estimated_total_gas(trace)
        = gas_spent_so_far  +  admissible_lower_bound_on_remaining_gas
        = move_count        +  manhattan_distance_from_car_to_goal

Each move costs 1 unit of gas (whether or not it bumps), so move_count
literally equals gas spent. Manhattan distance is the smallest amount of
gas that COULD still be needed (admissible — straight-line minimum).

Higher score (= less negative gas estimate) is better. Reached-goal
traces score by gas actually spent; timeouts get a heavy penalty.

This is the locally-greedy heuristic: at every state it picks the
action whose successor minimises `move_count + manhattan` *one step
ahead*. It cannot detour. When detours cost less gas overall, MCTS
lookahead in L*'s equivalence oracle discovers them, and the learned
Mealy machine encodes the strictly-improved strategy.

API mirrors the other game oracles (NimOracle, TicTacToeOracle, ...):
    compare(trace_a, trace_b) -> 't1' | 't2' | 'equal'
    preferred_move(prefix)    -> action chosen greedily at this state
"""

from __future__ import annotations

from src.game.grid_nav.board    import ACTIONS
from src.game.grid_nav.game_nfa import GridNavNFA


# Per direction-bucket, ordered list of preferred actions to try.
# The first unblocked action in the list is chosen.
_DIRECTION_PREFERENCES: dict[str, tuple[str, ...]] = {
    'AT': ('N', 'E', 'S', 'W'),    # at goal — direction is irrelevant
    'N':  ('N', 'E', 'W', 'S'),
    'NE': ('N', 'E', 'W', 'S'),
    'E':  ('E', 'N', 'S', 'W'),
    'SE': ('E', 'S', 'N', 'W'),
    'S':  ('S', 'E', 'W', 'N'),
    'SW': ('S', 'W', 'E', 'N'),
    'W':  ('W', 'N', 'S', 'E'),
    'NW': ('N', 'W', 'E', 'S'),
}


def _greedy_action_from_obs(obs: str) -> str:
    """
    Pure observation-to-action greedy rule. Layout-free, deterministic.
    `obs` is a string like 'NE|1011' (direction-bucket | NESW-blocked-bits).
    """
    direction, blocked = obs.split('|')
    blocked_map = {a: (b == '1') for a, b in zip(ACTIONS, blocked)}
    for action in _DIRECTION_PREFERENCES.get(direction, ACTIONS):
        if not blocked_map[action]:
            return action
    # All cardinals blocked: bump in goal direction (will stay in place).
    return _DIRECTION_PREFERENCES.get(direction, ACTIONS)[0]


class GridNavOracle:

    def __init__(self, nfa: GridNavNFA, depth: int | None = None) -> None:
        self.nfa   = nfa
        # `depth` is accepted for API compatibility with other oracles; the
        # gridworld oracle is a pure greedy heuristic — depth=None = full
        # locally-greedy descent, integer values are not yet used.
        self.depth = depth

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare(self, trace_a: list, trace_b: list) -> str:
        sa = self._score(trace_a)
        sb = self._score(trace_b)
        if sa > sb: return 't1'
        if sb > sa: return 't2'
        return 'equal'

    def preferred_move(self, prefix: list):
        """
        Locally-greedy from the LAST observation symbol alone — fully
        layout-agnostic. Implements: prefer the action that points
        toward the goal direction; if blocked, try perpendicular
        unblocked directions; only retreat as a last resort.

        This is what makes the SUL deterministic on observation
        sequences regardless of layout: same observation → same default
        action, every time.
        """
        if not prefix:
            return None
        last_obs = prefix[-1]
        if not isinstance(last_obs, str) or '|' not in last_obs:
            # Trace ended on an action symbol — caller used the wrong prefix.
            return None
        return _greedy_action_from_obs(last_obs)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score(self, trace: list) -> float:
        """
        Score = − estimated total gas cost to reach the goal.

        At goal       : score = − move_count   (actual gas spent — the truth).
        Timed out     : large negative penalty (10× the budget).
        In progress   : score = −(move_count + manhattan)   (gas-so-far
                        plus admissible lower bound on remaining gas).
        Invalid trace : −∞.

        Higher score = lower estimated gas = preferred trace.
        """
        node = self.nfa.get_node(trace)
        if node is None:
            return -1e9

        if node.car_pos == self.nfa.goal:
            return -float(node.move_count)

        if node.move_count >= self.nfa.max_moves:
            return -10.0 * self.nfa.max_moves

        gx, gy = self.nfa.goal
        cx, cy = node.car_pos
        manhattan = abs(gx - cx) + abs(gy - cy)
        return -float(node.move_count + manhattan)
