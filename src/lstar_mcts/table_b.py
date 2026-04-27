"""
Table B — the MCTS exploration tree.

Stores all visited (trace_prefix, action) pairs across all equivalence
queries.  Never reset between queries — it accumulates continuously.

Each entry tracks:
  - visit count
  - current SMT-derived value  (0.0–1.0, higher = more preferred)
  - zero-probability flag       (pruned from future search)

The trace keys are full interleaved traces: [P1, P2, P1, P2, ...]
Actions are the next move from that state (P1 input or P2 response).
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math


HIGH_PRIOR = 0.75   # UCB score for completely unexplored actions


@dataclass
class ActionStats:
    visits:    int   = 0
    value:     float = 0.5   # neutral prior until SMT assigns a real value
    zero_prob: bool  = False


class TableB:

    def __init__(self, exploration_c: float = 1.4, depth_alpha: float = 1.2) -> None:
        """
        Parameters
        ----------
        exploration_c : UCB exploration constant
        depth_alpha   : depth discount factor (shallower nodes get more bonus)
                        UCB bonus is multiplied by alpha^(-depth)
        """
        self.c     = exploration_c
        self.alpha = depth_alpha

        # {tuple(trace_prefix): {action: ActionStats}}
        self._nodes: dict[tuple, dict[str, ActionStats]] = {}

    # ------------------------------------------------------------------
    # Node access
    # ------------------------------------------------------------------

    def _get(self, trace: list[str] | tuple, action: str) -> ActionStats:
        key = tuple(trace)
        if key not in self._nodes:
            self._nodes[key] = {}
        if action not in self._nodes[key]:
            self._nodes[key][action] = ActionStats()
        return self._nodes[key][action]

    def actions_at(self, trace: list[str]) -> dict[str, ActionStats]:
        """All actions recorded at this trace prefix."""
        return self._nodes.get(tuple(trace), {})

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    def record_visit(self, trace: list[str], action: str) -> None:
        stats = self._get(trace, action)
        if not stats.zero_prob:
            stats.visits += 1

    def update_value(self, trace: list[str], action: str, smt_value: float) -> None:
        stats = self._get(trace, action)
        stats.value = smt_value

    def set_zero_prob(self, trace: list[str], action: str) -> None:
        self._get(trace, action).zero_prob = True

    # ------------------------------------------------------------------
    # UCB selection
    # ------------------------------------------------------------------

    def ucb_score(self, trace: list[str], action: str,
                  available_actions: list[str]) -> float:
        """
        UCB score for (trace, action).

        Unexplored actions return HIGH_PRIOR.
        Zero-probability actions return -inf (never selected).
        """
        stats = self._get(trace, action)

        if stats.zero_prob:
            return float('-inf')

        if stats.visits == 0:
            return HIGH_PRIOR

        total_visits = sum(
            self._get(trace, a).visits
            for a in available_actions
            if not self._get(trace, a).zero_prob
        )
        if total_visits == 0:
            return HIGH_PRIOR

        depth        = len(trace)
        depth_disc   = self.alpha ** (-depth)
        exploration  = self.c * math.sqrt(math.log(total_visits) / stats.visits)

        return stats.value + exploration * depth_disc

    def best_action(self, trace: list[str], available_actions: list[str]) -> str | None:
        """Return the action with the highest UCB score."""
        live = [a for a in available_actions
                if not self._get(trace, a).zero_prob]
        if not live:
            return None
        return max(live, key=lambda a: self.ucb_score(trace, a, available_actions))

    # ------------------------------------------------------------------
    # Probabilistic P2 selection (softmax + high prior for unexplored)
    # ------------------------------------------------------------------

    def sample_p2_action(self, trace: list[str], available_actions: list[str],
                         temperature: float = 1.0) -> str | None:
        """
        Sample a P2 action proportional to exploration probability.
        Unexplored actions get a high prior; explored actions use
        softmax over their SMT values.
        """
        import random, math

        live = [a for a in available_actions
                if not self._get(trace, a).zero_prob]
        if not live:
            return None

        weights = []
        for a in live:
            stats = self._get(trace, a)
            if stats.visits == 0:
                weights.append(10.0)           # high prior for unexplored
            else:
                weights.append(math.exp(self.ucb_score(trace, a, live) / temperature))

        total = sum(weights)
        probs = [w / total for w in weights]

        return random.choices(live, weights=probs, k=1)[0]

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def prune_below_median(self, traces_at_depth: list[tuple[list[str], str]]) -> int:
        """
        Among the (trace, action) pairs in traces_at_depth, assign zero
        probability to any whose value is below the median.
        Returns the number of nodes pruned.
        """
        if not traces_at_depth:
            return 0

        vals = [self._get(t, a).value for t, a in traces_at_depth]
        median = sorted(vals)[len(vals) // 2]

        pruned = 0
        for trace, action in traces_at_depth:
            if self._get(trace, action).value < median:
                self.set_zero_prob(trace, action)
                pruned += 1
        return pruned

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def summary(self) -> str:
        total_nodes  = len(self._nodes)
        total_edges  = sum(len(v) for v in self._nodes.values())
        pruned_edges = sum(
            1 for v in self._nodes.values()
            for s in v.values() if s.zero_prob
        )
        return (f'TableB: {total_nodes} states, {total_edges} edges, '
                f'{pruned_edges} pruned')
