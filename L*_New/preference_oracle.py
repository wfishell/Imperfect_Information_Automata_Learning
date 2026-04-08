"""
Preference oracle for Dots and Boxes 2x2.

A preference query takes two complete game traces and returns:
   +1  if trace1 is preferred (better for P2)
   -1  if trace2 is preferred (better for P2)
    0  if indifferent (equal outcome)

Two oracles are provided:

  OutcomeOracle   -- compares final P2 scores only (win/draw/loss)
  MinimaxOracle   -- pre-computes optimal play via minimax, then compares
                     traces on (1) final outcome, then (2) how closely P2
                     followed optimal play at each decision point
"""

from dots_and_boxes import (
    DotsAndBoxesNFA, INITIAL_STATE, EDGE_NAMES, EDGE_INDEX,
    legal_moves, transition, is_terminal, winner, NUM_EDGES,
)


def spot_to_trace(spot_str):
    """
    Convert a spot semantics string back to a raw edge index list.

    e.g. "p1_V00,p2_V01;p1_PASS,p2_H01" -> [6, 7, 1]

    PASS entries are skipped — they carry no game action.
    """
    trace = []
    for step in spot_str.split(";"):
        inp, out = step.strip().split(",")
        inp = inp.strip()
        out = out.strip()
        if inp != "p1_PASS":
            edge_name = inp[len("p1_"):]
            trace.append(EDGE_INDEX[edge_name])
        if out != "p2_PASS":
            edge_name = out[len("p2_"):]
            trace.append(EDGE_INDEX[edge_name])
    return trace


# ---------------------------------------------------------------------------
# Outcome-based oracle
# ---------------------------------------------------------------------------

class OutcomeOracle:
    """
    Ground-truth preference based solely on P2's final box count.

      P2 score in trace1 > trace2  ->  +1
      P2 score in trace1 < trace2  ->  -1
      equal                        ->   0
    """

    def __init__(self, nfa):
        self.nfa = nfa

    def query(self, trace1, trace2):
        s1 = self.nfa.state_after(trace1)
        s2 = self.nfa.state_after(trace2)
        assert s1 is not None and s2 is not None, "traces must be complete valid games"
        p2_score1 = s1.score[1]
        p2_score2 = s2.score[1]
        if p2_score1 > p2_score2:
            return 1
        elif p2_score1 < p2_score2:
            return -1
        return 0

    def query_spot(self, spot1, spot2):
        """Accept spot semantics strings directly."""
        return self.query(spot_to_trace(spot1), spot_to_trace(spot2))


# ---------------------------------------------------------------------------
# Minimax oracle
# ---------------------------------------------------------------------------

class MinimaxOracle:
    """
    Pre-computes the minimax value (optimal P1 box count with both players
    playing perfectly) for every reachable game state.

    Preference between two traces:
      Primary   : P1's final score  (higher is better)
      Secondary : number of P1 moves that were minimax-optimal
                  (tie-breaks traces with the same outcome — rewards better play)
    """

    def __init__(self, nfa):
        self.nfa = nfa
        self._value = {}        # GameState -> int (minimax P1 score)
        self._optimal = {}      # GameState -> frozenset of optimal edge indices
        print("MinimaxOracle: pre-computing minimax values...", end=" ", flush=True)
        self._compute(INITIAL_STATE)
        print("done.")

    # -- Minimax recursion ---------------------------------------------------

    def _compute(self, state):
        if state in self._value:
            return self._value[state]

        if is_terminal(state):
            self._value[state]   = state.score[1]
            self._optimal[state] = frozenset()
            return state.score[1]

        moves = legal_moves(state)
        child_values = {m: self._compute(transition(state, m)) for m in moves}

        if state.player == 1:           # P2 maximises
            best = max(child_values.values())
        else:                           # P1 minimises P2's score
            best = min(child_values.values())

        self._value[state]   = best
        self._optimal[state] = frozenset(m for m, v in child_values.items() if v == best)
        return best

    # -- Public interface ----------------------------------------------------

    def optimal_value(self, state=None):
        """Minimax value (optimal P1 score) from a given state."""
        return self._value[state or INITIAL_STATE]

    def optimal_moves(self, state):
        """Set of edge indices that are minimax-optimal from state."""
        return self._optimal.get(state, frozenset())

    def is_p1_optimal(self, state, edge):
        """True if P1 choosing this edge is minimax-optimal."""
        return edge in self._optimal.get(state, frozenset())

    def query(self, trace1, trace2):
        """
        Compare two complete game traces from P1's perspective.
        Returns +1, -1, or 0.
        """
        s1 = self.nfa.state_after(trace1)
        s2 = self.nfa.state_after(trace2)
        assert s1 is not None and s2 is not None, "traces must be complete valid games"

        # Primary: final outcome
        p2_1, p2_2 = s1.score[1], s2.score[1]
        if p2_1 != p2_2:
            return 1 if p2_1 > p2_2 else -1

        # Secondary: how many of P2's moves were minimax-optimal
        opt1 = self._p2_optimal_move_count(trace1)
        opt2 = self._p2_optimal_move_count(trace2)
        if opt1 != opt2:
            return 1 if opt1 > opt2 else -1

        return 0

    def query_spot(self, spot1, spot2):
        """Accept spot semantics strings directly."""
        return self.query(spot_to_trace(spot1), spot_to_trace(spot2))

    def _p2_optimal_move_count(self, trace):
        state = INITIAL_STATE
        count = 0
        for edge in trace:
            if state.player == 1 and self.is_p1_optimal(state, edge):
                count += 1
            state = transition(state, edge)
        return count

    def explain(self, trace1, trace2):
        """Human-readable explanation of why one trace is preferred."""
        s1 = self.nfa.state_after(trace1)
        s2 = self.nfa.state_after(trace2)
        p = self.query(trace1, trace2)

        print(f"Trace 1: {[EDGE_NAMES[e] for e in trace1]}")
        print(f"  P1 score: {s1.score[0]}  P2 score: {s1.score[1]}")
        print(f"  P2 optimal moves: {self._p2_optimal_move_count(trace1)}")
        print(f"  Minimax value from start (P2): {self.optimal_value()}")
        print()
        print(f"Trace 2: {[EDGE_NAMES[e] for e in trace2]}")
        print(f"  P1 score: {s2.score[0]}  P2 score: {s2.score[1]}")
        print(f"  P2 optimal moves: {self._p2_optimal_move_count(trace2)}")
        print()
        result = {1: "Trace 1 preferred (+1)", -1: "Trace 2 preferred (-1)", 0: "Indifferent (0)"}
        print(f"Preference: {result[p]}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    nfa = DotsAndBoxesNFA()

    outcome_oracle  = OutcomeOracle(nfa)
    minimax_oracle  = MinimaxOracle(nfa)

    print(f"\nMinimax optimal P2 score from start: {minimax_oracle.optimal_value()}")
    print(f"Optimal first moves for P2 (responses to P1's opener): computed per state")

    # Generate some random games to compare
    def random_game(seed):
        rng = random.Random(seed)
        state = INITIAL_STATE
        trace = []
        while not is_terminal(state):
            edge = rng.choice(legal_moves(state))
            trace.append(edge)
            state = transition(state, edge)
        return trace

    print("\n" + "="*60)
    print("Sample preference queries (OutcomeOracle vs MinimaxOracle)")
    print("="*60)

    pairs = [(0, 1), (2, 3), (4, 5)]
    for seed_a, seed_b in pairs:
        t1 = random_game(seed_a)
        t2 = random_game(seed_b)
        o  = outcome_oracle.query(t1, t2)
        m  = minimax_oracle.query(t1, t2)
        s1 = nfa.state_after(t1).score
        s2 = nfa.state_after(t2).score
        print(f"\nTrace A (seed {seed_a}): score {s1}  "
              f"opt_moves={minimax_oracle._p1_optimal_move_count(t1)}")
        print(f"Trace B (seed {seed_b}): score {s2}  "
              f"opt_moves={minimax_oracle._p1_optimal_move_count(t2)}")
        print(f"  OutcomeOracle : {o:+d}")
        print(f"  MinimaxOracle : {m:+d}")

    # Full explain on one pair
    print("\n" + "="*60)
    print("Detailed explanation (MinimaxOracle)")
    print("="*60 + "\n")
    minimax_oracle.explain(random_game(0), random_game(1))
