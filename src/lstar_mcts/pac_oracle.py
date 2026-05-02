"""
PAC equivalence oracle (Kearns-Vazirani sample schedule).

Samples iid from a fixed distribution D over NFA-legal P1 walks. On the
i-th equivalence query, draws m_i = ceil((1/eps)(ln(1/delta) + i*ln 2))
samples and tests hypothesis(x) == sul(x) step-by-step. Returns the
shortest prefix on which they disagree, or None if all samples agree.

Intended to be composed AFTER MCTS in CompositeEqOracle: MCTS drives
strategy convergence (mutating the SUL), then PAC validates that the
final hypothesis behaviorally matches the (now stable) SUL on D.

If a PAC round returns None, the standard per-round PAC bound applies:
with probability >= 1 - delta on that round, the hypothesis disagrees
with the SUL on at most eps-fraction of D-distributed inputs.
"""

from __future__ import annotations
import math
import random
from aalpy.base import Oracle


class PACEqOracle(Oracle):

    def __init__(
        self,
        alphabet: list,
        sul,
        nfa,
        eps:            float = 0.05,
        delta:          float = 0.05,
        max_walk_depth: int   = 20,
    ) -> None:
        super().__init__(alphabet, sul)
        self.nfa            = nfa
        self.eps            = eps
        self.delta          = delta
        self.max_walk_depth = max_walk_depth

    # ------------------------------------------------------------------
    # AALpy oracle interface
    # ------------------------------------------------------------------

    def find_cex(self, hypothesis):
        self.num_queries += 1
        i = self.num_queries
        m = math.ceil((1 / self.eps) * (math.log(1 / self.delta) + i * math.log(2)))

        for _ in range(m):
            p1_seq = self._sample_legal_walk()
            if not p1_seq:
                continue
            cex = self._first_disagreement(hypothesis, p1_seq)
            if cex is not None:
                return cex
        return None

    # ------------------------------------------------------------------
    # Sampling: random NFA walk, collecting P1 inputs only
    # ------------------------------------------------------------------

    def _sample_legal_walk(self) -> list:
        node   = self.nfa.root
        p1_seq = []
        while (node is not None
               and not node.is_terminal()
               and len(p1_seq) < self.max_walk_depth):
            actions = list(node.children.keys())
            if not actions:
                break
            action = random.choice(actions)
            if node.player == 'P1':
                p1_seq.append(action)
            node = node.children[action]
        return p1_seq

    # ------------------------------------------------------------------
    # Comparison: feed P1 sequence to both hypothesis and SUL, find diff
    # ------------------------------------------------------------------

    def _first_disagreement(self, hypothesis, p1_seq: list):
        hypothesis.reset_to_initial()
        self.sul.pre()
        try:
            for k, p1 in enumerate(p1_seq):
                h_out = hypothesis.step(p1)
                s_out = self.sul.step(p1)
                self.num_steps += 1
                if h_out != s_out:
                    return list(p1_seq[:k + 1])
            return None
        finally:
            self.sul.post()
