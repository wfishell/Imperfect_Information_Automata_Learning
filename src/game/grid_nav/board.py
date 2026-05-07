"""
Gridworld navigation state.

The car starts at (0, 0) on an n x n grid and must reach the goal cell.
K static obstacles block specific cells. Each step the car emits an
observation (P1) and the controller picks an action (P2). The trace
alternates observation/action symbols, matching the L* infrastructure's
expectation of P1-input / P2-output alphabets.

Coordinate convention:
    (x, y) with x = column (east-positive), y = row (north-positive)
    start = (0, 0) at bottom-left, goal at (grid_size-1, grid_size-1).

Observation alphabet (P1):
    "{direction-to-goal-bucket}|{4-cardinal-blocked-vector}"
    e.g.  "NE|0010"   = goal NE of car; S neighbour blocked, others free.

Action alphabet (P2):  N, E, S, W

Bumping into an obstacle or wall keeps the car in place but still
increments move_count. Termination: car at goal (P2 wins) OR move_count
reaches max_moves (P1 wins, timeout).
"""

from __future__ import annotations
from functools import cached_property


from itertools import product as _product


ACTIONS: tuple[str, ...] = ('N', 'E', 'S', 'W')
DELTAS: dict[str, tuple[int, int]] = {
    'N': ( 0,  1),
    'E': ( 1,  0),
    'S': ( 0, -1),
    'W': (-1,  0),
}

_DIR_BUCKETS: tuple[str, ...] = ('AT', 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW')

# Full observation alphabet: 9 direction buckets × 16 blocked patterns = 144.
# The NFA is permissive at P1 nodes: every observation symbol is accepted.
# The world state evolves only through P2 actions (observations are
# labels). At deployment only "real" observations occur; L* learns a
# Mealy whose behaviour on any unreachable observation is irrelevant.
ALL_OBSERVATIONS: tuple[str, ...] = tuple(
    f'{d}|{"".join(b)}'
    for d in _DIR_BUCKETS
    for b in _product('01', repeat=4)
)


def _direction_bucket(car_pos: tuple[int, int],
                      goal:    tuple[int, int]) -> str:
    cx, cy = car_pos
    gx, gy = goal
    dx, dy = gx - cx, gy - cy
    if dx == 0 and dy == 0: return 'AT'
    if dx == 0 and dy >  0: return 'N'
    if dx == 0 and dy <  0: return 'S'
    if dx >  0 and dy == 0: return 'E'
    if dx <  0 and dy == 0: return 'W'
    if dx >  0 and dy >  0: return 'NE'
    if dx >  0 and dy <  0: return 'SE'
    if dx <  0 and dy >  0: return 'NW'
    return 'SW'


def _blocked_vector(car_pos:   tuple[int, int],
                    obstacles: frozenset,
                    grid_size: int) -> str:
    cx, cy = car_pos
    bits = []
    for action in ACTIONS:
        ddx, ddy = DELTAS[action]
        nx, ny = cx + ddx, cy + ddy
        is_blocked = (
            nx < 0 or nx >= grid_size or
            ny < 0 or ny >= grid_size or
            (nx, ny) in obstacles
        )
        bits.append('1' if is_blocked else '0')
    return ''.join(bits)


def make_observation(car_pos:   tuple[int, int],
                     obstacles: frozenset,
                     grid_size: int,
                     goal:      tuple[int, int]) -> str:
    return f"{_direction_bucket(car_pos, goal)}|{_blocked_vector(car_pos, obstacles, grid_size)}"


class GridNavState:
    """One alternating-player state in the gridworld trajectory NFA.

    Memoized at class level so equivalent (car_pos, move_count, player,
    last_obs, obstacles, ...) configurations share the same instance.
    """

    _cache: dict = {}

    def __new__(cls,
                car_pos:    tuple[int, int],
                obstacles:  frozenset,
                grid_size:  int,
                goal:       tuple[int, int],
                max_moves:  int,
                move_count: int = 0,
                player:     str = 'P1',
                last_obs:   str | None = None) -> 'GridNavState':
        obstacles = obstacles if isinstance(obstacles, frozenset) else frozenset(obstacles)
        key = (car_pos, obstacles, grid_size, goal, max_moves,
               move_count, player, last_obs)
        if key in cls._cache:
            return cls._cache[key]
        inst = super().__new__(cls)
        cls._cache[key] = inst
        return inst

    def __init__(self,
                 car_pos:    tuple[int, int],
                 obstacles:  frozenset,
                 grid_size:  int,
                 goal:       tuple[int, int],
                 max_moves:  int,
                 move_count: int = 0,
                 player:     str = 'P1',
                 last_obs:   str | None = None) -> None:
        if getattr(self, '_initialized', False):
            return
        self._initialized = True
        self.car_pos    = car_pos
        self.obstacles  = obstacles if isinstance(obstacles, frozenset) else frozenset(obstacles)
        self.grid_size  = grid_size
        self.goal       = goal
        self.max_moves  = max_moves
        self.move_count = move_count
        self.player     = player
        self.last_obs   = last_obs

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def is_terminal(self) -> bool:
        return self.car_pos == self.goal or self.move_count >= self.max_moves

    def winner(self) -> str | None:
        if self.car_pos == self.goal:           return 'P2'
        if self.move_count >= self.max_moves:   return 'P1'
        return None

    @property
    def value(self) -> int:
        w = self.winner()
        if w == 'P2': return  1
        if w == 'P1': return -1
        return 0

    # ------------------------------------------------------------------
    # Children — alternating P1 (emit observation) / P2 (pick action)
    # ------------------------------------------------------------------

    @cached_property
    def children(self) -> dict:
        if self.is_terminal():
            return {}

        if self.player == 'P1':
            # Deterministic: the environment emits exactly one
            # observation here, computed from the world state.
            obs = make_observation(self.car_pos, self.obstacles,
                                   self.grid_size, self.goal)
            return {
                obs: GridNavState(
                    car_pos    = self.car_pos,
                    obstacles  = self.obstacles,
                    grid_size  = self.grid_size,
                    goal       = self.goal,
                    max_moves  = self.max_moves,
                    move_count = self.move_count,
                    player     = 'P2',
                    last_obs   = obs,
                )
            }

        # P2 — apply each action, bump if illegal
        result: dict = {}
        cx, cy = self.car_pos
        for action in ACTIONS:
            ddx, ddy = DELTAS[action]
            nx, ny = cx + ddx, cy + ddy
            illegal = (
                nx < 0 or nx >= self.grid_size or
                ny < 0 or ny >= self.grid_size or
                (nx, ny) in self.obstacles
            )
            new_pos = self.car_pos if illegal else (nx, ny)
            result[action] = GridNavState(
                car_pos    = new_pos,
                obstacles  = self.obstacles,
                grid_size  = self.grid_size,
                goal       = self.goal,
                max_moves  = self.max_moves,
                move_count = self.move_count + 1,
                player     = 'P1',
                last_obs   = None,
            )
        return result

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (f'GridNavState(car={self.car_pos}, move={self.move_count}, '
                f'player={self.player}, obs={self.last_obs})')
