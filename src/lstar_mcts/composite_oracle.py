"""
Two-phase equivalence oracle: MCTS strategy refinement + PAC validation.

On every find_cex call:
  Phase 1 (MCTS): the wrapped MCTSEquivalenceOracle runs K rollouts and
    may return a preference-majority counterexample (mutating the SUL
    via update_strategy in the process).
  Phase 2 (PAC):  if MCTS returned None, the wrapped PACEqOracle samples
    iid NFA-legal walks and tests hypothesis(x) == sul(x) directly.

Termination occurs when, in a single find_cex call, MCTS finds no
strategy improvement AND PAC finds no behavioral disagreement on its
m_k samples — joint convergence. Both phases run on every call: a PAC
counterexample triggers L* refinement, which produces a richer
hypothesis on which MCTS may discover further strategy improvements.

Per-round PAC guarantee on the terminating call: with probability
>= 1 - delta_pac, the final hypothesis disagrees with the final SUL on
at most eps-fraction of D-distributed inputs.
"""

from __future__ import annotations
from aalpy.base import Oracle


class CompositeEqOracle(Oracle):

    def __init__(self, mcts_oracle, pac_oracle, verbose: bool = False) -> None:
        super().__init__(mcts_oracle.alphabet, mcts_oracle.sul)
        self.mcts    = mcts_oracle
        self.pac     = pac_oracle
        self.verbose = verbose

    def find_cex(self, hypothesis):
        self.num_queries += 1

        if self.verbose:
            print(f'[composite] query {self.num_queries}: phase 1 (MCTS)...')
        cex = self.mcts.find_cex(hypothesis)
        if cex is not None:
            if self.verbose:
                print(f'[composite]   MCTS returned cex (len={len(cex)})')
            return cex

        if self.verbose:
            print(f'[composite]   MCTS returned None; phase 2 (PAC)...')
        cex = self.pac.find_cex(hypothesis)
        if cex is not None:
            if self.verbose:
                print(f'[composite]   PAC returned cex (len={len(cex)})')
            return cex

        if self.verbose:
            print(f'[composite]   joint convergence — both phases passed')
        return None
