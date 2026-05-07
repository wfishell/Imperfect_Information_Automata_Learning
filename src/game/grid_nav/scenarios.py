"""
Curated trap suite + held-out random layout pool + scenario sampler.

CURATED_TRAPS
    Hand-designed K=3 obstacle layouts where Manhattan-greedy
    demonstrably fails (gets stuck, oscillates, or requires multi-step
    detours).  Each trap exercises a different failure mode.  These
    serve as a fixed evaluation set for any controller.

held_out_pool(...)
    Generator yielding deterministically-seeded random K-obstacle
    layouts the controller has never seen.  Used to measure
    generalisation: the trained Mealy should still solve these.

ScenarioPool
    Sampler used inside the equivalence-query MCTS — each rollout
    draws a fresh layout via .sample(), so the Mealy is trained on
    the layout *distribution* rather than a single fixed layout.
"""

from __future__ import annotations

from src.game.grid_nav.game_nfa import GridNavNFA


# ----------------------------------------------------------------------
# Curated trap suite — each defeats a one-step Manhattan-greedy oracle.
# ----------------------------------------------------------------------
#
# Coordinates: (x, y) with x = column (east-positive), y = row
# (north-positive).  Start (0,0) at bottom-left, goal (4,4) at top-right
# on a 5x5 grid.  Every trap is reachability-checked at NFA construction.
#
# Greedy's preferred path on this game (tie-break N over E) goes up
# the west edge from (0,0) to (0,4), then across the top to (4,4).
# Every trap below blocks greedy AT a specific cell along that path:
# at the trap cell, N and E are both blocked, so greedy bumps forever
# (any S/W move retreats and scores worse than a bump under the
# gas+manhattan score). The goal remains reachable in every layout via
# detours that only multi-step lookahead can find.
#
# Trap cell along greedy's path (early -> late):
#   u_trap           : trap at (0,1)   — earliest, basic U-trap
#   u_trap_mid       : trap at (0,2)   — same idea, one cell up
#   u_trap_high      : trap at (0,3)   — even later in the climb
#   reverse_u_trap   : trap at (0,4)   — corner, top of west edge
#   late_detour      : trap at (1,4)   — first cell after corner
#   mid_top_trap     : trap at (2,4)   — deep into the top corridor

CURATED_TRAPS: dict[str, list[tuple[int, int]]] = {
    'u_trap':         [(0, 2), (1, 1)],
    'u_trap_mid':     [(0, 3), (1, 2)],
    'u_trap_high':    [(0, 4), (1, 3), (3, 3)],
    'reverse_u_trap': [(1, 4), (2, 3)],
    'late_detour':    [(2, 4), (3, 4), (4, 2)],
    'mid_top_trap':   [(3, 4), (4, 2), (1, 1)],
}


def trap(name: str, *,
         grid_size: int = 5,
         max_moves: int = 30) -> GridNavNFA:
    """Build a GridNavNFA from one of the curated trap names."""
    if name not in CURATED_TRAPS:
        raise ValueError(
            f'unknown trap {name!r}; available: {list(CURATED_TRAPS)}'
        )
    return GridNavNFA(
        grid_size = grid_size,
        obstacles = CURATED_TRAPS[name],
        max_moves = max_moves,
    )


# ----------------------------------------------------------------------
# Held-out random pool — for generalisation evaluation
# ----------------------------------------------------------------------

def held_out_pool(n: int                      = 200,
                  k: int                      = 3,
                  grid_size: int              = 5,
                  max_moves: int              = 30,
                  base_seed: int              = 10_000):
    """
    Yield n distinct random K-obstacle layouts, deterministically seeded
    so the test set is reproducible. Skips seeds that produce
    unreachable goals.

    Disjoint from any seed pool used for training (start at base_seed
    well above ScenarioPool defaults).
    """
    yielded = 0
    seed    = base_seed
    while yielded < n:
        try:
            yield GridNavNFA(
                grid_size = grid_size,
                k         = k,
                seed      = seed,
                max_moves = max_moves,
            )
            yielded += 1
        except RuntimeError:
            pass   # rejected: goal unreachable from start
        seed += 1


# ----------------------------------------------------------------------
# Scenario pool — sampler used during EQ-phase MCTS rollouts
# ----------------------------------------------------------------------

class ScenarioPool:
    """
    Layout sampler. Each .sample() call returns a fresh GridNavNFA
    drawn from the configured K-obstacle distribution. Skips seeds
    whose layout has unreachable goal.

    Used by MCTSEquivalenceOracle during EQ rollouts so the hypothesis
    Mealy is exposed to the layout distribution (not a fixed layout).
    Aggregated visit counts and preference values land in the shared
    Table B keyed by observation-trace, giving cross-layout learning.
    """

    def __init__(self,
                 k: int          = 3,
                 grid_size: int  = 5,
                 max_moves: int  = 30,
                 base_seed: int  = 0) -> None:
        self.k         = k
        self.grid_size = grid_size
        self.max_moves = max_moves
        self._seed     = base_seed
        self._sampled  = 0

    def sample(self) -> GridNavNFA:
        while True:
            try:
                nfa = GridNavNFA(
                    grid_size = self.grid_size,
                    k         = self.k,
                    seed      = self._seed,
                    max_moves = self.max_moves,
                )
                self._seed   += 1
                self._sampled += 1
                return nfa
            except RuntimeError:
                self._seed += 1

    @property
    def total_sampled(self) -> int:
        return self._sampled

    def union_alphabet(self, n_samples: int) -> list[str]:
        """
        Return the union of reachable observation alphabets across
        n_samples layouts drawn from the pool. Used as L*'s input
        alphabet for multi-board training so the table covers every
        observation symbol that any pool layout can produce. Resets
        the sampler's internal seed counter when done so training
        rollouts draw the SAME layouts (deterministic).
        """
        saved_seed   = self._seed
        saved_count  = self._sampled
        union: set   = set()
        for _ in range(n_samples):
            nfa = self.sample()
            union.update(nfa.p1_alphabet)
        self._seed    = saved_seed   # rewind so training reuses same draws
        self._sampled = saved_count
        return sorted(union)


# ----------------------------------------------------------------------
# Curated pool — sample from a fixed list of layouts (e.g., trap suite)
# ----------------------------------------------------------------------

class CuratedPool:
    """
    Layout sampler over a fixed, pre-built list of GridNavNFA instances.

    Useful for biased training: by sampling uniformly over the curated
    trap suite, every MCTS rollout is on a layout where greedy is known
    to fail. The Mealy learns to handle those failure modes; the
    cross-rollout vote is then over a distribution where the deviation
    *does* win majority (because every layout is a trap).

    Constructor takes any iterable of GridNavNFA. The helpers below
    build common pools.
    """

    def __init__(self, nfas, rng=None) -> None:
        import random as _random
        self.nfas = list(nfas)
        if not self.nfas:
            raise ValueError('CuratedPool requires at least one layout')
        self._rng     = rng if rng is not None else _random.Random(0)
        self._sampled = 0

    def sample(self):
        self._sampled += 1
        return self._rng.choice(self.nfas)

    @property
    def total_sampled(self) -> int:
        return self._sampled

    def union_alphabet(self, n_samples: int = 0) -> list[str]:
        """Union of reachable observation alphabets across all curated
        layouts. Argument is ignored (we union over the full fixed list)."""
        union: set = set()
        for nfa in self.nfas:
            union.update(nfa.p1_alphabet)
        return sorted(union)


def curated_pool(max_moves: int = 30, grid_size: int = 5,
                  rng=None) -> CuratedPool:
    """Pool over every layout in CURATED_TRAPS."""
    return CuratedPool(
        [trap(name, grid_size=grid_size, max_moves=max_moves)
         for name in CURATED_TRAPS],
        rng=rng,
    )


def mixed_pool(n_random: int     = 10,
               max_moves: int    = 30,
               grid_size: int    = 5,
               k: int            = 3,
               base_seed: int    = 0,
               rng=None) -> CuratedPool:
    """Curated traps + n_random random layouts. Heavily trap-biased."""
    nfas = [trap(name, grid_size=grid_size, max_moves=max_moves)
            for name in CURATED_TRAPS]
    rand_pool = ScenarioPool(k=k, grid_size=grid_size,
                              max_moves=max_moves, base_seed=base_seed)
    nfas += [rand_pool.sample() for _ in range(n_random)]
    return CuratedPool(nfas, rng=rng)


# ----------------------------------------------------------------------
# Quick demo
# ----------------------------------------------------------------------

if __name__ == '__main__':
    print('=== Curated trap suite ===')
    for name in CURATED_TRAPS:
        try:
            nfa = trap(name)
            print(f'\n{name}:  obstacles={CURATED_TRAPS[name]}')
            print(nfa.render())
        except ValueError as exc:
            print(f'\n{name}: ERROR — {exc}')

    print('\n=== Held-out pool sample (first 3) ===')
    for i, nfa in enumerate(held_out_pool(n=3)):
        print(f'\nlayout {i}:  obstacles={sorted(nfa.obstacles)}')
        print(nfa.render())

    print('\n=== Scenario pool ===')
    pool = ScenarioPool(k=3, base_seed=0)
    for i in range(3):
        nfa = pool.sample()
        print(f'sample {i}: obstacles={sorted(nfa.obstacles)}')
