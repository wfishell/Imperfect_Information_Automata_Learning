"""
MCTS-based equivalence oracle for AALpy.

At each equivalence query the oracle runs K MCTS rollouts to depth N.
Each rollout:
  - At P1 nodes  : selects via UCB (coverage across all P1 inputs)
  - At P2 nodes  : samples probabilistically (high prior for unexplored,
                   softmax over SMT values for explored)
  - Deviations from the current hypothesis are tracked as candidate
    counterexamples

After K rollouts the oracle:
  1. Collects all Table A leaves (all P1 paths, hypothesis P2 choices, depth N)
  2. Collects all Table B deviation leaves (depth N)
  3. Feeds all n(n-1)/2 pairwise preferences across A ∪ B into the SMT solver
  4. Updates Table B values from the SMT solution
  5. Prunes depth-N Table B leaves below the median
  6. Compares mean(Table B values) vs mean(Table A values)
  7. If Table B mean > Table A mean + epsilon:
       - updates the SUL strategy override at the best deviation point
       - returns the P1 input sequence as a counterexample to AALpy

Table B persists across all equivalence queries.
"""

from __future__ import annotations
import random
from aalpy.base import Oracle

from src.game.minimax.game_nfa import GameNFA
from src.lstar_mcts.preference_oracle import PreferenceOracle
from src.lstar_mcts.game_sul import GameSUL
from src.lstar_mcts.table_b import TableB


class MCTSEquivalenceOracle(Oracle):

    def __init__(
        self,
        sul: GameSUL,
        nfa: GameNFA,
        oracle: PreferenceOracle,
        table_b: TableB,
        depth_N: int,
        K: int             = 200,
        epsilon: float     = 0.05,
        temperature: float = 1.0,
        budget_threshold: float = 0.05,
        verbose: bool      = False,
    ) -> None:
        alphabet = list(nfa.root.children.keys())   # P1's top-level inputs
        super().__init__(alphabet, sul)
        self.sul              = sul
        self.nfa              = nfa
        self.oracle           = oracle
        self.table_b          = table_b
        self.N                = depth_N
        self.K                = K
        self.epsilon          = epsilon
        self.temperature      = temperature
        self.budget_threshold = budget_threshold
        self.verbose          = verbose

        # {deviation_point (tuple): [trace_at_depth_N (list), ...]}
        self._deviation_leaves: dict[tuple, list[list[str]]] = {}

    # ------------------------------------------------------------------
    # AALpy Oracle interface
    # ------------------------------------------------------------------

    def _check_for_improvement(self, hypothesis) -> list[str] | None:
        """
        SMT-based CEX check over deviation leaves vs their Table A shadows.

        For each deviation point, collect its depth-N leaves and their shadows.
        Build a fresh SMT instance over just those leaves (not all of Table B),
        solve for consistent numeric values, then compare means.
        If mean(dev) > mean(shadow) + epsilon, accept as counterexample.
        """
        from src.lstar_mcts.smt_solver import SMTValueAssigner

        b_leaves = [l for leaves in self._deviation_leaves.values() for l in leaves]
        if not b_leaves:
            return None
        
        # DEBUG 
        print(f'Checking for improvement: {len(self._deviation_leaves)} deviations, '
              f'{len(b_leaves)} deviation leaves')
        
        print('  Deviation points:')
        for deviation in self._deviation_leaves:
            print(f'    {deviation}')

        print('  Sample deviation leaves:')
        for dev, leaves in self._deviation_leaves.items():
            print(f'    Deviation {dev}:')
            for leaf in leaves[:3]:
                print(f'      {leaf}')

        # --- Collect all (deviation, leaf, shadow) triples ---
        triples: list[tuple[tuple, list, list]] = []
        for deviation, dev_leaves in self._deviation_leaves.items():
            for leaf in dev_leaves:
                shadow = self._shadow_trace(hypothesis, leaf)
                if shadow is None or shadow == leaf:
                    continue
                triples.append((deviation, leaf, shadow))

        if not triples:
            return None

        # --- Build fresh SMT over only these depth-N leaves ---
        unique_traces = list({
            tuple(t)
            for _, leaf, shadow in triples
            for t in (leaf, shadow)
        })

        smt = SMTValueAssigner()
        for i, t1 in enumerate(unique_traces):
            for t2 in unique_traces[i + 1:]:
                pref = self.oracle.compare(list(t1), list(t2))
                smt.add(list(t1), list(t2), pref)

        values = smt.solve()
        if values is None:
            if self.verbose:
                print('  SMT unsatisfiable — skipping improvement check')
            return None

        # --- Feed SMT values back into Table B ---
        # Only update deviation leaves (not shadows — those are Table A traces).
        # Explored deviations that scored poorly get lower softmax weight;
        # those below the median are pruned from future rollouts entirely.
        dev_leaf_keys = {
            tuple(leaf)
            for dev_leaves in self._deviation_leaves.values()
            for leaf in dev_leaves
        }
        dev_smt_vals = [values[k] for k in dev_leaf_keys if k in values]
        if dev_smt_vals:
            median = sorted(dev_smt_vals)[len(dev_smt_vals) // 2]
            for trace_key in dev_leaf_keys:
                if trace_key not in values:
                    continue
                trace   = list(trace_key)
                smt_val = values[trace_key]
                self.table_b.update_value(trace[:-1], trace[-1], smt_val)
                if smt_val < median:
                    self.table_b.set_zero_prob(trace[:-1], trace[-1])

        # --- Compare means per deviation ---
        best_dev = None
        best_adv = self.epsilon

        for deviation, dev_leaves in self._deviation_leaves.items():
            dev_vals    = []
            shadow_vals = []
            for leaf in dev_leaves:
                shadow = self._shadow_trace(hypothesis, leaf)
                if shadow is None or shadow == leaf:
                    continue
                dv = values.get(tuple(leaf))
                sv = values.get(tuple(shadow))
                if dv is not None and sv is not None:
                    dev_vals.append(dv)
                    shadow_vals.append(sv)

            if not dev_vals:
                continue

            mean_dev    = sum(dev_vals)    / len(dev_vals)
            mean_shadow = sum(shadow_vals) / len(shadow_vals)
            adv = mean_dev - mean_shadow   # values already in [0,1] from SMT

            if self.verbose:
                print(f'  deviation={list(deviation)}  '
                      f'mean_dev={mean_dev:.3f}  mean_shadow={mean_shadow:.3f}  '
                      f'adv={adv:.3f}  n={len(dev_vals)}')
            if adv > best_adv:
                best_adv = adv
                best_dev = deviation

        if best_dev is None:
            return None

        best_leaf = max(
            self._deviation_leaves[best_dev],
            key=lambda l: values.get(tuple(l), 0.0),
        )
        if len(best_leaf) <= len(best_dev):
            return None

        new_p2_action = best_leaf[len(best_dev)]
        deviation_trace = list(best_dev)
        self.sul.update_strategy(deviation_trace, new_p2_action)
        if self.verbose:
            print(f'  Improvement: deviation at {deviation_trace}, new P2={new_p2_action}')
        return self.sul.p1_inputs_from_trace(deviation_trace + [new_p2_action])

    def find_cex(self, hypothesis) -> list[str] | None:
        """
        Exhaustive equivalence check: hypothesis vs current SUL.
        Tries all P1 input sequences up to depth N and returns the first
        discrepancy as a counterexample, or None if fully consistent.
        MCTS runs separately in the outer loop after convergence.
        """
        self.num_queries += 1
        from itertools import product
        p1_alphabet = list(self.nfa.root.children.keys())

        for length in range(1, self.N + 1):
            for p1_seq in product(p1_alphabet, repeat=length):
                hypothesis.reset_to_initial()
                hyp_out = [hypothesis.step(inp) for inp in p1_seq]

                self.sul.pre()
                sul_out = [self.sul.step(inp) for inp in p1_seq]
                self.sul.post()

                if hyp_out != sul_out:
                    return list(p1_seq)

        return None

    # ------------------------------------------------------------------
    # Single rollout
    # ------------------------------------------------------------------

    def _rollout(self, hypothesis, remaining: dict) -> None:
        trace: list[str] = []
        active_deviation: tuple | None = None

        while len(trace) < self.N * 2:
            node = self.nfa.get_node(trace)
            if node is None or node.is_terminal():
                break

            available = list(node.children.keys())

            if node.player == 'P1':
                action = self.table_b.best_action(trace, available)
                if action is None:
                    action = random.choice(available)

            else:
                hyp_output = self._hypothesis_output(hypothesis, trace)

                if active_deviation is None:
                    # No deviation yet — explore freely
                    action = self.table_b.sample_p2_action(
                        trace, available, self.temperature
                    )
                    if action is None:
                        action = random.choice(available)
                    # Detect first deviation
                    if hyp_output and action != hyp_output:
                        active_deviation = tuple(trace)
                else:
                    # Already deviated — follow hypothesis for all later P2 nodes
                    # so the leaf differs from its shadow only at the one deviation point
                    action = hyp_output if hyp_output in available else available[0]

            self.table_b.record_visit(trace, action)
            trace = trace + [action]

            # Mid-exploration pruning
            if active_deviation is not None:
                if not self._budget_check(active_deviation, remaining):
                    break

        # Store deviation leaf
        if active_deviation is not None and trace:
            self._deviation_leaves.setdefault(active_deviation, []).append(list(trace))

    # ------------------------------------------------------------------
    # Mid-exploration pruning
    # ------------------------------------------------------------------

    def _budget_check(self, deviation: tuple, remaining: dict) -> bool:
        """
        Reduce budget for a deviation subtree based on Table B values from
        the previous SMT round.  Return False to abandon this rollout.
        """
        leaf_list = self._deviation_leaves.get(deviation, [])
        if len(leaf_list) < 4:
            return True

        # Use stored Table B values (from previous SMT round) to assess promise
        vals = []
        for leaf in leaf_list:
            if len(leaf) >= 1:
                stats = self.table_b.actions_at(leaf[:-1]).get(leaf[-1])
                if stats is not None:
                    vals.append(stats.value)

        if not vals:
            return True

        mean_val = sum(vals) / len(vals)
        if mean_val < 0.4:   # below neutral — subtree looks unpromising
            budget = remaining.get(deviation, self.K)
            budget *= 0.7
            remaining[deviation] = budget
            if budget < self.budget_threshold * self.K:
                return False

        return True

    # ------------------------------------------------------------------
    # Table A leaf enumeration
    # ------------------------------------------------------------------

    def _shadow_trace(self, hypothesis, deviation_leaf: list[str]) -> list[str] | None:
        """
        Given a deviation leaf trace, return the shadow trace that uses the
        same P1 inputs but follows the hypothesis P2 choices at every step.
        """

        # DEBUG 
        print(f"  Finding shadow for deviation leaf: {deviation_leaf}: \n Hypothesis: {hypothesis}")

        shadow: list[str] = []
        node = self.nfa.root
        p1_inputs = deviation_leaf[::2]   # every other element starting at 0
        for p1_inp in p1_inputs:
            if node is None or node.is_terminal():
                break
            if p1_inp not in node.children:
                return None
            shadow.append(p1_inp)
            node = node.children[p1_inp]
            if node is None or node.is_terminal():
                break
            hyp_action = self._hypothesis_output(hypothesis, shadow)
            available = list(node.children.keys())
            action = hyp_action if hyp_action in available else available[0]
            shadow.append(action)
            node = node.children[action]
        return shadow if shadow else None

    def _collect_table_a_leaves(self, hypothesis) -> list[list[str]]:
        """
        All complete traces of length N*2, enumerating all P1 choices and
        following the current hypothesis for P2 choices.
        """
        results: list[list[str]] = []
        target = self.N * 2

        def _dfs(node, trace: list[str]) -> None:
            if len(trace) >= target or node is None or node.is_terminal():
                if trace:
                    results.append(list(trace))
                return
            if node.player == 'P1':
                for action, child in node.children.items():
                    trace.append(action)
                    _dfs(child, trace)
                    trace.pop()
            else:
                hyp_action = self._hypothesis_output(hypothesis, trace)
                available = list(node.children.keys())
                action = hyp_action if hyp_action in available else available[0]
                trace.append(action)
                _dfs(node.children[action], trace)
                trace.pop()

        _dfs(self.nfa.root, [])
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _hypothesis_output(self, hypothesis, trace: list[str]) -> str | None:
        """What does the current hypothesis say P2 should do at this trace?"""
        try:
            p1_inputs = self.sul.p1_inputs_from_trace(trace)
            hypothesis.reset_to_initial()
            output = None
            for inp in p1_inputs:
                output = hypothesis.step(inp)
            return output
        except Exception:
            return None

    # Appears to never be called? Dead Code?
    # def _collect_depth_n_leaves(self) -> list[tuple[list[str], str]]:
    #     """Collect all (trace, action) pairs at depth N*2 for pruning."""
    #     results = []
    #     target = self.N * 2 - 1
    #     for key in self.table_b._nodes:
    #         if len(key) == target:
    #             for action in self.table_b._nodes[key]:
    #                 results.append((list(key), action))

    #     # DEBUG
    #     print(f'Collected {len(results)} depth-N leaves for pruning:'
    #           f' {results[:5]}{"..." if len(results) > 5 else ""}')
    #     return results

    def _prune_and_update_table_b(
        self,
        depth_n_leaves: list[tuple[list[str], str]],
        a_leaves: list[list[str]],
    ) -> None:
        """
        Update Table B values using normalised oracle scores (no SMT).
        Prune depth-N leaves below the median score.
        """
        if not depth_n_leaves:
            return

        # Score each leaf via the oracle's internal _score (ordinal-only proxy)
        scored = []
        for trace, action in depth_n_leaves:
            leaf = list(trace) + [action]
            scored.append((trace, action, self.oracle._score(leaf)))

        if not scored:
            return

        lo = min(s for _, _, s in scored)
        hi = max(s for _, _, s in scored)
        span = hi - lo if hi != lo else 1

        for trace, action, raw in scored:
            v = (raw - lo) / span
            self.table_b.update_value(trace, action, v)

        # Prune below median
        vals = [s for _, _, s in scored]
        median = sorted(vals)[len(vals) // 2]
        for trace, action, raw in scored:
            if raw < median:
                self.table_b.set_zero_prob(trace, action)
