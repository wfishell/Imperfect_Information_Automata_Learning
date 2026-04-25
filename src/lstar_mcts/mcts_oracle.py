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
  1. Collects all deviation leaves (depth N) and their Table A shadows
  2. Feeds all pairwise preferences into the SMT solver
  3. Updates Table B softmax probabilities from the SMT solution
  4. Prunes depth-N Table B leaves below the median SMT value
  5. For each deviation point, counts oracle.compare(dev_leaf, shadow) == 't1'
  6. If majority (>50%) of pairs prefer the deviation:
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
        self.verbose          = verbose

        # {deviation_point (tuple): [trace_at_depth_N (list), ...]}
        self._deviation_leaves: dict[tuple, list[list[str]]] = {}

    # ------------------------------------------------------------------
    # AALpy Oracle interface
    # ------------------------------------------------------------------

    def _check_for_improvement(self, hypothesis) -> list[str] | None:
        """
        Majority-vote CEX check over deviation leaves vs their Table A shadows.

        For each deviation point, compare each depth-N deviation leaf directly
        against its shadow using oracle.compare().  Accept as a counterexample
        if a strict majority (>50%) of pairs prefer the deviation leaf.

        SMT is still run over all leaves to update Table B softmax probabilities,
        but the accept/reject decision is purely ordinal (no Z3 values involved).
        """
        from src.lstar_mcts.smt_solver import SMTValueAssigner

        b_leaves = [l for leaves in self._deviation_leaves.values() for l in leaves]
        if not b_leaves:
            return None
        
        # # DEBUG 
        # print(f'Checking for improvement: {len(self._deviation_leaves)} deviations, '
        #       f'{len(b_leaves)} deviation leaves')
        
        # print('  Deviation points:')
        # for deviation in self._deviation_leaves:
        #     print(f'    {deviation}')

        # print('  Sample deviation leaves:')
        # for dev, leaves in self._deviation_leaves.items():
        #     print(f'    Deviation {dev}:')
        #     for leaf in leaves[:3]:
        #         print(f'      {leaf}')

        # --- Collect all (deviation, full leaf, shadow, cmp_leaf) tuples ---
        # cmp_leaf is leaf truncated to shadow depth when the hypothesis
        # terminates early; otherwise it equals the full leaf.
        triples: list[tuple[tuple, list, list, list]] = []
        for deviation, dev_leaves in self._deviation_leaves.items():
            for leaf in dev_leaves:
                shadow = self._shadow_trace(hypothesis, leaf)
                if shadow is None:
                    continue
                cmp_leaf = leaf[:len(shadow)] if len(shadow) < len(leaf) else leaf
                if cmp_leaf == shadow:
                    continue
                triples.append((deviation, leaf, shadow, cmp_leaf))

        if not triples:
            return None

        # --- Build fresh SMT over cmp_leaf/shadow pairs (for Table B only) ---
        unique_traces = list({
            tuple(t)
            for _, _leaf, shadow, cmp_leaf in triples
            for t in (cmp_leaf, shadow)
        })

        smt = SMTValueAssigner()
        for i, t1 in enumerate(unique_traces):
            for t2 in unique_traces[i + 1:]:
                pref = self.oracle.compare(list(t1), list(t2))
                smt.add(list(t1), list(t2), pref)

        values = smt.solve()

        # --- Feed SMT values back into Table B ---
        # Use cmp_leaf (the trace actually compared) to look up the SMT value,
        # then write it to the Table B entry for that trace's last action.
        if values is not None:
            cmp_keys = {tuple(cmp_leaf) for _, _leaf, _shadow, cmp_leaf in triples}
            cmp_smt_vals = [values[k] for k in cmp_keys if k in values]
            if cmp_smt_vals:
                median = sorted(cmp_smt_vals)[len(cmp_smt_vals) // 2]
                for cmp_key in cmp_keys:
                    if cmp_key not in values:
                        continue
                    trace   = list(cmp_key)
                    smt_val = values[cmp_key]
                    self.table_b.update_value(trace[:-1], trace[-1], smt_val)
                    if smt_val < median:
                        self.table_b.set_zero_prob(trace[:-1], trace[-1])

        # --- Majority vote per deviation ---
        # Accept the deviation with the highest fraction of oracle-preferred leaves,
        # provided that fraction strictly exceeds 0.5.
        best_dev      = None
        best_majority = 0.5   # must beat this to be accepted

        for deviation, dev_leaves in self._deviation_leaves.items():
            n_total = 0
            n_prefer_dev = 0
            for leaf in dev_leaves:
                shadow = self._shadow_trace(hypothesis, leaf)
                if shadow is None:
                    continue
                cmp_leaf = leaf[:len(shadow)] if len(shadow) < len(leaf) else leaf
                if cmp_leaf == shadow:
                    continue
                pref = self.oracle.compare(cmp_leaf, shadow)
                n_total += 1
                if pref == 't1':
                    n_prefer_dev += 1

            if n_total == 0:
                continue

            majority = n_prefer_dev / n_total
            if self.verbose:
                print(f'  deviation={list(deviation)}  '
                      f'prefer_dev={n_prefer_dev}/{n_total}  '
                      f'majority={majority:.3f}')
            if majority > best_majority:
                best_majority = majority
                best_dev = deviation

        if best_dev is None:
            return None

        # Rank by SMT value of cmp_leaf (truncated to shadow depth if needed).
        def _leaf_smt_value(l: list) -> float:
            shadow = self._shadow_trace(hypothesis, l)
            if shadow is None or values is None:
                return 0.0
            cmp = l[:len(shadow)] if len(shadow) < len(l) else l
            return values.get(tuple(cmp), 0.0)

        best_leaf = max(self._deviation_leaves[best_dev], key=_leaf_smt_value)
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

    def _rollout(self, hypothesis) -> None:
        """
        Branch on ALL P1 (environment) inputs; sample one P2 (system) action
        per node from the Table B distribution.  A single call produces one
        leaf per P1 input sequence combination up to depth N.

        Deviation tracking is per-path: when P2's sampled action first differs
        from the hypothesis the deviation point is recorded, and subsequent P2
        nodes on that path continue with free probabilistic exploration rather
        than snapping back to the hypothesis.
        """
        def _recurse(trace: list[str], active_deviation: tuple | None) -> None:
            if len(trace) >= self.N * 2:
                if active_deviation is not None:
                    self._deviation_leaves.setdefault(active_deviation, []).append(list(trace))
                return

            node = self.nfa.get_node(trace)
            if node is None or node.is_terminal():
                if active_deviation is not None and trace:
                    self._deviation_leaves.setdefault(active_deviation, []).append(list(trace))
                return

            available = list(node.children.keys())

            if node.player == 'P1':
                for action in available:
                    self.table_b.record_visit(trace, action)
                    _recurse(trace + [action], active_deviation)
            else:
                hyp_output = self._hypothesis_output(hypothesis, trace)
                action = self.table_b.sample_p2_action(trace, available, self.temperature)
                if action is None:
                    action = random.choice(available)

                self.table_b.record_visit(trace, action)

                new_deviation = active_deviation
                if active_deviation is None and hyp_output is not None and action != hyp_output:
                    new_deviation = tuple(trace)

                _recurse(trace + [action], new_deviation)

        _recurse([], None)

    # ------------------------------------------------------------------
    # Table A leaf enumeration
    # ------------------------------------------------------------------

    def _shadow_trace(self, hypothesis, deviation_leaf: list[str]) -> list[str] | None:
        """
        Given a deviation leaf trace, return the shadow trace that uses the
        same P1 inputs but follows the hypothesis P2 choices at every step.
        """

        # DEBUG 
        # print(f"  Finding shadow for deviation leaf: {deviation_leaf}: \n Hypothesis: {hypothesis}")

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
    # TODO: Reincorporate?
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

    # Appears to never be called? Dead Code?
    # TODO: Reincorporate?
    # def _prune_and_update_table_b(
    #     self,
    #     depth_n_leaves: list[tuple[list[str], str]],
    #     a_leaves: list[list[str]],
    # ) -> None:
    #     """
    #     Update Table B values using normalized oracle scores (no SMT).
    #     Prune depth-N leaves below the median score.
    #     """

    #     # DEBUG
    #     print(f"Looking at leaves: {depth_n_leaves} and A leaves: {a_leaves}")

    #     if not depth_n_leaves:
    #         return

    #     # Score each leaf via the oracle's internal _score (ordinal-only proxy)
    #     scored = []
    #     for trace, action in depth_n_leaves:
    #         leaf = list(trace) + [action]
    #         scored.append((trace, action, self.oracle._score(leaf)))

    #     if not scored:
    #         return

    #     lo = min(s for _, _, s in scored)
    #     hi = max(s for _, _, s in scored)
    #     span = hi - lo if hi != lo else 1

    #     for trace, action, raw in scored:
    #         v = (raw - lo) / span
    #         self.table_b.update_value(trace, action, v)

    #     # Prune below median
    #     vals = [s for _, _, s in scored]
    #     median = sorted(vals)[len(vals) // 2]
    #     for trace, action, raw in scored:
    #         if raw < median:
    #             self.table_b.set_zero_prob(trace, action)
