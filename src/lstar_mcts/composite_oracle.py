"""
N-stage equivalence oracle: chains together one or more sub-oracles
(MCTS strategy refinement, PAC validation, safety verification, ...).

On every find_cex call, each stage runs in order:
    stage 1 (e.g., MCTS):   may return a counterexample (often after
                            mutating the SUL via update_strategy).
    stage 2 (e.g., PAC):    if stage 1 returned None, samples iid walks
                            and tests hypothesis(x) == sul(x) directly.
    stage 3 (e.g., Safety): if previous stages returned None, runs a
                            spec-driven model-check on the hypothesis;
                            on violation, may patch the SUL and return
                            the violating trace.

A find_cex call returns the first non-None counterexample. If every
stage passes, the call returns None — joint convergence.

The class is variadic to keep the existing 2-stage usage
(`CompositeEqOracle(mcts, pac)`) working unchanged while supporting
arbitrarily many additional stages.

Per-round PAC guarantee on the terminating call (when PAC is in the
chain): with probability >= 1 - delta_pac, the final hypothesis
disagrees with the final SUL on at most eps-fraction of D-distributed
inputs.
"""

from __future__ import annotations
from aalpy.base import Oracle


class CompositeEqOracle(Oracle):

    def __init__(self, *stages, verbose: bool = False) -> None:
        if not stages:
            raise ValueError('CompositeEqOracle requires at least one stage')
        first = stages[0]
        super().__init__(first.alphabet, first.sul)
        self.stages  = stages
        self.verbose = verbose

    def find_cex(self, hypothesis):
        import time as _time
        self.num_queries += 1

        for i, stage in enumerate(self.stages, start=1):
            name = stage.__class__.__name__
            print(f'  [eq round {self.num_queries}] {name} ...',
                  end=' ', flush=True)
            t0 = _time.time()
            cex = stage.find_cex(hypothesis)
            dt = _time.time() - t0
            if cex is not None:
                print(f'CEX (len={len(cex)})  [{dt:.1f}s]')
                return cex
            print(f'no CEX  [{dt:.1f}s]')

        print(f'  [eq round {self.num_queries}] joint convergence — done')
        return None

    # ------------------------------------------------------------------
    # Backwards-compat aliases — old code accesses .mcts and .pac directly.
    # ------------------------------------------------------------------

    @property
    def mcts(self):
        return self.stages[0] if len(self.stages) >= 1 else None

    @property
    def pac(self):
        return self.stages[1] if len(self.stages) >= 2 else None
