"""
Safety stage for the gas-grid robot, structured as three independent layers:

    1. SPEC                a predicate `is_bad(env_state) -> bool`.
                           For G(gas > 0): `lambda s: s.gas <= 0`.
                           This is the ONLY domain-specific input.

    2. SPEC CHECKER        `SafetyChecker.find_all_violations(mealy)` does
                           a reachability BFS over (Mealy ⊗ env). Returns
                           every reachable env state where `is_bad` holds,
                           with the trace that led there. Pure reachability
                           — no arithmetic, no heuristics.

    3. STRATEGY SYNTHESIS  `solve_safety_game(nfa, is_bad)` runs a backward
                           fixpoint over the env's reachable abstract state
                           space and returns
                               (W, strategy)
                           where W is the maximal winning set (states from
                           which the spec can be satisfied) and `strategy`
                           is a dict mapping P2 state-keys to a safe action.
                           The "head to refuel / REFUEL when there" rule is
                           DERIVED here, not declared.

    4. PATCHING            `compute_patches(cex_trace, nfa, strategy)` walks
                           the CEX and at each P2 state looks up the strategy.
                           No thresholds, no arithmetic — just a lookup.

`SafetyEqOracle` plugs these together as a stage of `CompositeEqOracle`.
"""

from __future__ import annotations
from typing import Callable
from aalpy.base import Oracle

from src.control_systems.RobotGrid.board    import (
    RobotGridState,
    N, S, E, W, PICKUP, DROP, REFUEL, GO_TO_REFUEL,
)
from src.control_systems.RobotGrid.game_nfa import RobotGridNFA


# ----------------------------------------------------------------------
# Deterministic manhattan-greedy. Used as prefer_action inside the
# safety stage so that solve_safety_game and the boundary-state
# enumeration agree on what manhattan would pick at any state — the
# oracle's `_naive_action` uses rng.choice for diagonal ties, which
# would produce spurious "boundary" entries at every tied state.
# ----------------------------------------------------------------------

def deterministic_manhattan(state: RobotGridState) -> str | None:
    if state.is_terminal():
        return None
    if state.mode == 'going_to_refuel':
        kids = list(state.children.keys())
        return kids[0] if kids else None
    if state.carrying and state.pos == state.dropoff:
        return DROP
    if (state.task_loc is not None
            and state.pos == state.task_loc and not state.carrying):
        return PICKUP
    if state.carrying:
        target = state.dropoff
    elif state.task_loc is not None:
        target = state.task_loc
    else:
        kids = list(state.children.keys())
        return kids[0] if kids else None
    dr = target[0] - state.pos[0]
    dc = target[1] - state.pos[1]
    if abs(dr) >= abs(dc) and dr != 0:
        return S if dr > 0 else N
    if dc != 0:
        return E if dc > 0 else W
    return None


# ----------------------------------------------------------------------
# Default spec for this domain.
# ----------------------------------------------------------------------

def gas_positive_spec(env_state: RobotGridState) -> bool:
    """Default `is_bad` for the gas-grid: G(gas > 0) violated iff gas <= 0."""
    return env_state.gas <= 0


# ----------------------------------------------------------------------
# Abstract state key — quotients out fields the spec G(gas>0) doesn't see
# (step_count, delivered_count). Keeping these in the key would blow the
# state space up unnecessarily; the spec doesn't depend on either.
# ----------------------------------------------------------------------

def safety_state_key(env: RobotGridState) -> tuple:
    return (env.pos, env.gas, env.task_loc, env.carrying, env.mode, env.player)


# ----------------------------------------------------------------------
# Strategy synthesis — backward fixpoint over the env's reachable states.
# ----------------------------------------------------------------------

def solve_safety_game(
    nfa:           RobotGridNFA,
    is_bad:        Callable[[RobotGridState], bool],
    prefer_action: Callable[[RobotGridState], str] | None = None,
) -> tuple[set, dict, dict]:
    """
    Solve a 2-player safety game on the env's reachable abstract states.

    Player 1 (env) is treated adversarially — at idle states it may
    choose ANY task arrival, and the controller must stay safe under
    all such choices. (Concretely: state s_P1 is winning iff every
    legal P1 input leads to a winning successor.)

    Player 2 (controller) is cooperative — at P2 states the controller
    can pick any legal action. State s_P2 is winning iff SOME action
    leads to a winning successor.

    Parameters
    ----------
    nfa            : env NFA. We BFS from `nfa.root` to enumerate the
                     reachable abstract state space.
    is_bad         : spec violation predicate over env states.
    prefer_action  : optional tie-breaker; when multiple winning actions
                     exist at a P2 state, return the action this function
                     suggests (e.g. the manhattan-greedy oracle's choice).
                     If it returns a non-winning action or raises, fall
                     back to the first winning action.

    Returns
    -------
    (W, strategy, states)
        W         : set of state-keys in the winning set.
        strategy  : dict {p2_state_key: action} — one safe action per
                    winning P2 state.
        states    : dict {state_key: example RobotGridState instance}
                    so callers can recover state info (e.g. to compute
                    manhattan-greedy at boundary states).
    """
    # ----- phase 1: enumerate reachable abstract states -----------------
    states:      dict[tuple, RobotGridState]              = {}
    transitions: dict[tuple, dict[object, tuple]]         = {}

    queue: list[RobotGridState] = [nfa.root]
    while queue:
        s   = queue.pop(0)                          # BFS — visit at smallest depth
        key = safety_state_key(s)
        if key in states:
            continue
        states[key] = s

        if s.is_terminal():
            transitions[key] = {}
            continue

        edges: dict[object, tuple] = {}
        for action, child in s.children.items():
            child_key       = safety_state_key(child)
            edges[action]   = child_key
            queue.append(child)
        transitions[key] = edges

    # ----- phase 2: backward fixpoint -----------------------------------
    bad_keys = {k for k, s in states.items() if is_bad(s)}
    W        = set(states.keys()) - bad_keys

    while True:
        prev_size = len(W)
        for key in list(W):
            s = states[key]
            if s.is_terminal():
                continue                            # stays in W (no successors)
            if s.player == 'P1':
                if not all(t in W for t in transitions[key].values()):
                    W.discard(key)
            else:                                   # P2
                if not any(t in W for t in transitions[key].values()):
                    W.discard(key)
        if len(W) == prev_size:
            break

    # ----- phase 3: extract strategy at P2 states in W ------------------
    # Tie-breaking order:
    #   1. prefer_action (manhattan-greedy) if it's in `winning`
    #         → no behavioural divergence at safe states
    #   2. GO_TO_REFUEL macro if it's in `winning`
    #         → preferred FALLBACK when manhattan is unsafe; one mode-change
    #           output instead of installing many directional patches
    #   3. first winning action (deterministic for reproducibility)
    strategy: dict[tuple, str] = {}
    for key in W:
        s = states[key]
        if s.player != 'P2' or s.is_terminal():
            continue
        winning = [a for a, t in transitions[key].items() if t in W]
        if not winning:
            continue                                # shouldn't happen given W defn

        if prefer_action is not None:
            try:
                pref = prefer_action(s)
            except Exception:
                pref = None
            if pref is not None and pref in winning:
                strategy[key] = pref
                continue

        if GO_TO_REFUEL in winning:
            strategy[key] = GO_TO_REFUEL
            continue

        strategy[key] = winning[0]

    return W, strategy, states


# ----------------------------------------------------------------------
# Spec checker — reachability of `is_bad` over (Mealy ⊗ env).
# ----------------------------------------------------------------------

class SafetyChecker:
    """Reachability check over the product of the Mealy hypothesis and
    the env. Returns input traces that reach a state satisfying `is_bad`."""

    def __init__(self,
                 nfa:    RobotGridNFA,
                 is_bad: Callable[[RobotGridState], bool] = gas_positive_spec) -> None:
        self.nfa    = nfa
        self.is_bad = is_bad

    def find_violation(self, mealy) -> list | None:
        all_violations = self.find_all_violations(mealy, max_violations=1)
        return all_violations[0] if all_violations else None

    def find_all_violations(self, mealy, max_violations: int | None = None
                              ) -> list[list]:
        violations: list[list] = []
        visited:    set        = set()
        frontier:   list[list] = [[]]

        while frontier:
            trace = frontier.pop(0)
            env   = self.nfa.get_node(trace)
            if env is None:
                continue

            # Spec violation — the predicate fired on the previous step.
            if self.is_bad(env):
                violations.append(trace)
                if max_violations is not None and len(violations) >= max_violations:
                    return violations
                continue

            if env.is_terminal():
                continue

            # Dedupe by (mealy_state_id, env_state_key).
            m_state, _ = self._replay(mealy, trace)
            key = (id(m_state), self._env_key(env))
            if key in visited:
                continue
            visited.add(key)

            if env.player != 'P1':
                continue

            for p1_input in env.children:
                _, p2_output = self._replay(mealy, trace + [p1_input])
                env_after_p1 = env.children[p1_input]

                if env_after_p1.is_terminal():
                    if self.is_bad(env_after_p1):
                        violations.append(trace + [p1_input])
                        if (max_violations is not None
                                and len(violations) >= max_violations):
                            return violations
                    continue

                # Fallback to first legal action when Mealy's emission is
                # illegal at this state — matches deployment behaviour.
                if p2_output not in env_after_p1.children:
                    p2_output = next(iter(env_after_p1.children))

                env_after_p2 = env_after_p1.children[p2_output]
                frontier.append(trace + [p1_input, p2_output])

        return violations

    @staticmethod
    def _env_key(env: RobotGridState) -> tuple:
        return (env.pos, env.gas, env.task_loc, env.carrying,
                env.delivered_count, env.player, env.step_count)

    @staticmethod
    def _replay(mealy, trace: list) -> tuple:
        mealy.reset_to_initial()
        out = None
        for sym in trace:
            out = mealy.step(sym)
        return mealy.current_state, out


# ----------------------------------------------------------------------
# One-sweep boundary patch enumeration.
#
# Walk the env from root, following the strategy at P2 states and
# branching at idle-P1 states. At every P2 state where the strategy
# diverges from prefer_action, record a patch at the canonical prefix
# that got there. This visits every distinct env state once and returns
# the FINITE set of "inconsistencies" — exactly the few settings where
# the spec says something different from the preference oracle.
#
# Used by SafetyEqOracle to install all patches in ONE batch on the
# first find_cex call, instead of trickling them in via CEX traces.
# ----------------------------------------------------------------------

def find_all_boundary_patches(
    nfa:           RobotGridNFA,
    strategy:      dict[tuple, str],
    prefer_action: Callable[[RobotGridState], str] | None = None,
) -> list[tuple[list, str]]:
    """Enumerate every (prefix, safe_action) where the strategy
    recommends a different action than manhattan-greedy. One canonical
    prefix per env state-key (no duplicate patches for the same state)."""
    visited: set                          = set()
    patches: list[tuple[list, str]]       = []
    queue:   list[tuple[RobotGridState, list]] = [(nfa.root, [])]

    while queue:
        env, prefix = queue.pop(0)
        if env is None or env.is_terminal():
            continue

        key = safety_state_key(env)
        if key in visited:
            continue
        visited.add(key)

        if env.player == 'P1':
            for p1, child in env.children.items():
                queue.append((child, prefix + [p1]))
            continue

        # P2 state.
        safe = strategy.get(key)
        if safe is None or safe not in env.children:
            continue

        if prefer_action is not None:
            try:
                pref = prefer_action(env)
            except Exception:
                pref = None
            divergent = (pref != safe)
        else:
            divergent = True

        if divergent:
            patches.append((list(prefix), safe))

        # Continue exploration via the strategy's action.
        queue.append((env.children[safe], prefix + [safe]))

    return patches


# ----------------------------------------------------------------------
# Patching — walk the CEX, look up the strategy at each P2 state.
# (Kept for the post-sweep validation rounds; first round uses the
# one-sweep enumeration above.)
# ----------------------------------------------------------------------

def compute_patches(cex_trace: list,
                    nfa:       RobotGridNFA,
                    strategy:  dict[tuple, str],
                    prefer_action: Callable[[RobotGridState], str] | None = None
                    ) -> list[tuple[list, str]]:
    """Walk the CEX forward; for each P2 state visited, look up the
    safe action in `strategy`. Queue a patch ONLY if the strategy
    recommends an action different from what `prefer_action` (e.g. the
    manhattan-greedy oracle) would say. Those divergences are exactly
    the moments the spec demands a fallback — heading to refuel /
    emitting REFUEL — that the naive policy wouldn't pick on its own.

    If `prefer_action` is None, falls back to patching at every P2
    state with a strategy entry (less surgical, more verbose).
    """
    patches: list[tuple[list, str]] = []
    env = nfa.root

    for i, sym in enumerate(cex_trace):
        if env is None or env.is_terminal():
            break
        if sym not in env.children:
            break
        env = env.children[sym]

        if env.player != 'P2' or env.is_terminal():
            continue

        key = safety_state_key(env)
        safe_action = strategy.get(key)
        if safe_action is None or safe_action not in env.children:
            continue

        # Only patch when the strategy diverges from the naive policy.
        # Same-action patches are correctness-preserving but redundant —
        # they bloat _spec_locked and add no behavioural change.
        if prefer_action is not None:
            try:
                pref = prefer_action(env)
            except Exception:
                pref = None
            if pref == safe_action:
                continue

        prefix = list(cex_trace[: i + 1])
        patches.append((prefix, safe_action))

    return patches


# ----------------------------------------------------------------------
# Equivalence-oracle stage.
# ----------------------------------------------------------------------

class SafetyEqOracle(Oracle):
    """Third stage of `CompositeEqOracle`. On `find_cex`:
        - run the spec CHECKER (reachability of `is_bad`);
        - on violation, look up the strategy and queue patches;
        - apply patches via `sul.patch`;
        - return the shortest violating trace so L* refines.
    """

    def __init__(self,
                 sul,
                 nfa:                       RobotGridNFA,
                 oracle                                  = None,
                 is_bad:                    Callable[[RobotGridState], bool] = gas_positive_spec,
                 verbose:                   bool                            = False,
                 max_violations_per_round:  int  | None                     = None) -> None:
        super().__init__(nfa.p1_alphabet, sul)
        self.sul     = sul
        self.nfa     = nfa
        self.verbose = verbose
        self.is_bad  = is_bad
        self.max_violations_per_round = max_violations_per_round

        self.checker = SafetyChecker(nfa, is_bad=is_bad)

        # Always use deterministic manhattan inside the safety stage —
        # the oracle's _naive_action has rng tiebreaks, which would
        # produce spurious "boundary" entries at every tied state.
        self._prefer_action = deterministic_manhattan
        self.W, self.strategy, self._states = solve_safety_game(
            nfa, is_bad, prefer_action=self._prefer_action,
        )

        # Boundary states = P2 states where strategy ≠ manhattan.
        # These are the FINITE set of "few settings" where the
        # spec actively disagrees with the preference oracle.
        self._boundary_state_patches: list[tuple] = []
        for state_key, safe_action in self.strategy.items():
            state = self._states.get(state_key)
            if state is None:
                continue
            pref = None
            if self._prefer_action is not None:
                try:
                    pref = self._prefer_action(state)
                except Exception:
                    pref = None
            if pref != safe_action:
                self._boundary_state_patches.append((state_key, safe_action))

        # Wire SUL to use state-keyed overrides at the safety state-key
        # granularity. ANY input sequence reaching one of the patched
        # env states will get the patched answer — no per-prefix
        # bookkeeping needed.
        self.sul.state_key_fn = safety_state_key

        self._sweep_done = False

        # Returned-CEX dedup. Each entry is the tuple of an input trace
        # we've already handed back to L*. If VERIFY keeps producing the
        # same trace, L* can't make progress and we need to escalate
        # (untrimmed trace) or declare stuck.
        self._returned_cexes: set[tuple] = set()
        self._stuck_announced = False

        if verbose:
            print(f'[safety] strategy synthesised: '
                  f'|W|={len(self.W)} winning states, '
                  f'|strategy|={len(self.strategy)} P2-state→action entries, '
                  f'|boundary states|={len(self._boundary_state_patches)}.')

        self.n_patches_total    = 0
        self.n_violations_total = 0

    def find_cex(self, hypothesis):
        self.num_queries += 1

        # ---- First call: install all state-keyed patches in batch ----
        if not self._sweep_done:
            self._sweep_done = True

            for state_key, action in self._boundary_state_patches:
                self.sul.patch_state(state_key, action)
            self.n_patches_total += len(self._boundary_state_patches)

            n_patches = len(self._boundary_state_patches)
            actions   = [a for _, a in self._boundary_state_patches]
            unique_actions = sorted(set(actions))

            # Find a violation trace under the CURRENT Mealy (which has
            # not absorbed the patches yet). This becomes the CEX so
            # L* has something to refine on.
            violations = self.checker.find_all_violations(
                hypothesis, max_violations=1,
            )
            if not violations:
                print(f'      ↳ safety [SWEEP]: '
                      f'{n_patches} state-keyed patch(es) installed; '
                      f'unique actions: {unique_actions}; '
                      f'no violation trace under current Mealy.')
                return None

            cex_full          = violations[0]
            cex_short, _tail  = self._trim_to_first_boundary(cex_full)
            self._returned_cexes.add(tuple(cex_short))
            print(f'      ↳ safety [SWEEP]: '
                  f'{n_patches} state-keyed patch(es) installed in batch.  '
                  f'unique actions: {unique_actions}.  '
                  f'CEX returned to L* (P1-only) len={len(cex_short)} '
                  f'(full interleaved violation len={len(cex_full)}).')
            return cex_short

        # ---- Subsequent calls: spec check only ----
        # The sweep installed every boundary patch; if L* has absorbed
        # them this returns None and the run terminates.
        # Pull several candidate violations so we can pick one L* hasn't
        # already been given. Same-CEX-every-round means L* can't refine
        # further from this prefix and we need to escalate.
        violations = self.checker.find_all_violations(
            hypothesis, max_violations=8,
        )
        if not violations:
            return None

        # Pass 1: try TRIMMED violations; return first one that's new.
        # Trimmed = P1-inputs-only prefix up to the first patched boundary
        # state — exactly the format L*'s SUL.step loop can consume.
        for cex_full in violations:
            cex_short, env_tail = self._trim_to_first_boundary(cex_full)
            key = tuple(cex_short)
            if key not in self._returned_cexes:
                self._returned_cexes.add(key)
                self.n_violations_total += 1
                self._announce_verify_cex(cex_short, env_tail, mode='trimmed')
                return cex_short

        # Pass 2: every trimmed CEX is a repeat — escalate to a FULL
        # P1-only trace. Walking deeper P1 inputs past the boundary
        # gives L* additional rows whose MQ-outputs reflect SUL's
        # post-boundary trajectory; this can surface discrepancies
        # the trimmed prefix didn't.
        for cex_full in violations:
            cex_full_p1 = self._p1_inputs_only(cex_full)
            key = tuple(cex_full_p1)
            if key not in self._returned_cexes:
                self._returned_cexes.add(key)
                self.n_violations_total += 1
                self._announce_verify_cex(cex_full_p1, None, mode='full-escalated')
                return cex_full_p1

        # Pass 3: even the full traces are repeats — L* genuinely cannot
        # split the relevant equivalence classes from prefix-only CEXes.
        # Declare stuck and let the run terminate; the Mealy is the best
        # L* can do without a richer CEX (suffix injection, additional E
        # columns) which is out of scope here.
        if not self._stuck_announced:
            self._stuck_announced = True
            print(f'      ↳ safety [VERIFY]: STUCK — '
                  f'{len(violations)} candidate violation(s), all duplicates '
                  f'of previously-returned CEXes.  L* cannot refine further '
                  f'from prefix CEXes.  Returning None to terminate.')
        return None

    def _announce_verify_cex(self, cex: list, env_tail, mode: str) -> None:
        """Print VERIFY-mode CEX diagnostic. `mode` ∈ {'trimmed','full-escalated'}.
        `env_tail` (optional) is the env state at the CEX tail — supplied
        by `_trim_to_first_boundary` since a P1-only CEX cannot be re-walked
        through the env directly. Pass None for full-escalated traces (the
        tail is past the boundary; the boundary info is what matters and
        was logged on the trimmed-pass attempt that preceded escalation)."""
        boundary_keys = {sk for sk, _ in self._boundary_state_patches}
        if env_tail is not None:
            env_key = safety_state_key(env_tail)
            in_overrides   = env_key in getattr(self.sul, '_state_overrides', {})
            is_boundary    = env_key in boundary_keys
            patched_action = (self.sul._state_overrides.get(env_key)
                              if in_overrides else None)
            tail_info = (f'tail boundary={is_boundary} patched={in_overrides} '
                         f'action={patched_action}')
        else:
            tail_info = 'tail beyond boundary (P1-only escalation)'
        print(f'      ↳ safety [VERIFY,{mode}]: residual violation found.  '
              f'CEX returned to L* (P1-only) len={len(cex)}.  {tail_info}.')

    def _trim_to_first_boundary(self, cex_trace: list):
        """Walk the trace; return (p1_only_prefix, env_at_tail).

        IMPORTANT — output is P1-INPUTS ONLY, not the interleaved env trace.
        L* (custom_lstar.MealyLStar._query) feeds each CEX symbol to
        SUL.step as a P1 input. Interleaved P2 actions in the CEX would
        be misinterpreted as P1 inputs, fail env's deterministic-obs
        check, and produce None-filled junk rows that don't refine the
        observation table.

        Returns the P1-input prefix up to (and including) the P1 input
        that lands the env at a patched boundary state. `env_at_tail` is
        the corresponding env state (P2-turn boundary, or whatever the
        trace ends at if no boundary is hit) so callers don't need to
        re-walk the trace through the env afterwards."""
        boundary_keys = {sk for sk, _ in self._boundary_state_patches}
        env = self.nfa.root
        for i, sym in enumerate(cex_trace):
            if env is None or env.is_terminal():
                break
            if sym not in env.children:
                break
            env = env.children[sym]
            if env.player == 'P2' and safety_state_key(env) in boundary_keys:
                return self._p1_inputs_only(cex_trace[: i + 1]), env
        return self._p1_inputs_only(cex_trace), env

    @staticmethod
    def _p1_inputs_only(interleaved: list) -> list:
        """Strip P2 outputs from an interleaved [P1,P2,P1,P2,...] trace.
        P1 inputs sit at even indices."""
        return [interleaved[i] for i in range(0, len(interleaved), 2)]
