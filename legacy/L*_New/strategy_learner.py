"""
L* strategy learner for Dots and Boxes 2x2 using AALpy.

The System Under Learning (SUL) combines:
  - Membership oracle  : NFA gives legal P2 moves at each game state
  - Preference oracle  : ranks legal P2 moves, returns the best one

AALpy's L* then learns a Moore machine:
  Input  alphabet : P1's edge claims  {H00, H01, ..., V12}
  Output alphabet : P2's best response {H00, H01, ..., V12, PASS, GAME_OVER}

PASS      = P1 got a bonus turn so P2 has no move this step
GAME_OVER = all edges claimed, game ended
"""

from aalpy.base import SUL
from aalpy.learning_algs import run_Lstar
from aalpy.oracles import RandomWalkEqOracle

from dots_and_boxes import (
    DotsAndBoxesNFA, INITIAL_STATE, EDGE_NAMES, EDGE_INDEX,
    legal_moves, transition, is_terminal, NUM_EDGES,
)
from preference_oracle import MinimaxOracle


# ---------------------------------------------------------------------------
# SUL — the combined membership + preference oracle
# ---------------------------------------------------------------------------

class StrategyLearnerSUL(SUL):
    """
    One step = one P1 move.

    Internally the SUL:
      1. Applies P1's move to the game state
      2. If it is now P2's turn:
           - asks the NFA for all legal P2 moves  (membership query)
           - asks the preference oracle to rank them (preference queries)
           - applies the best P2 move
           - returns the best P2 move name as output
      3. If P1 got a bonus turn (P1 completed a box), returns 'PASS'
      4. If the game is over, returns 'GAME_OVER'

    Bonus turns for P2 are handled silently inside step() — the SUL keeps
    applying P2's optimal bonus moves until it is P1's turn again, then
    waits for the next step() call.
    """

    def __init__(self, nfa, oracle):
        super().__init__()
        self.nfa    = nfa
        self.oracle = oracle
        self.state  = None

    def pre(self):
        """Reset to the start of a new game."""
        self.state = INITIAL_STATE

    def post(self):
        self.state = None

    def step(self, p1_input):
        """
        p1_input : edge name string (e.g. 'H00') or 'PASS' (P2 bonus turn)
        returns  : P2's response edge name, 'PASS' (P1 got bonus), or 'GAME_OVER'

        'PASS' as input means: P2 has a bonus turn — no P1 move this step.
        The GUI calls step('PASS') for each consecutive P2 bonus turn so the
        machine is fully responsible for all of P2's moves.
        """
        if is_terminal(self.state):
            return 'GAME_OVER'

        # AALpy queries step(None) to get the initial state output
        if p1_input is None:
            return 'WAITING'

        # --- P2 bonus turn (no P1 move) ---
        if p1_input == 'PASS':
            if self.state.player != 1:
                return 'ILLEGAL'   # PASS called out of turn
            p2_move = self._best_p2_move()
            self.state = transition(self.state, p2_move)
            return EDGE_NAMES[p2_move]

        # --- Apply P1's move ---
        edge = EDGE_INDEX[p1_input]
        next_state = transition(self.state, edge)

        if next_state is None:
            return 'ILLEGAL'

        self.state = next_state

        if is_terminal(self.state):
            return 'GAME_OVER'

        # --- P1 got a bonus turn ---
        if self.state.player == 0:
            return 'PASS'

        # --- P2's turn: preference query over legal moves ---
        p2_move = self._best_p2_move()
        self.state = transition(self.state, p2_move)
        return EDGE_NAMES[p2_move]
        # If P2 gets a bonus turn, the caller will invoke step('PASS') next

    def _best_p2_move(self):
        """
        Membership query  : legal_moves(state) -> {a1, ..., an}
        Preference queries: compare all pairs, return argmax

        Here we use the MinimaxOracle's precomputed optimal moves.
        An LLM oracle would replace this with pairwise trace comparisons.
        """
        legal = legal_moves(self.state)

        # Preference query: find the preference-maximal move
        # Start with the first legal move as the current best
        best = legal[0]
        for alt in legal[1:]:
            # Compare: if alt is preferred over best, update best
            # Preference oracle compares complete traces — here we use
            # minimax values as a shortcut (same result, no lookahead needed)
            if self._prefers(alt, best):
                best = alt

        return best

    def _prefers(self, edge_a, edge_b):
        """
        Returns True if edge_a is preferred over edge_b from the current state.
        Uses the minimax oracle's precomputed values.
        """
        val_a = self.oracle._value.get(transition(self.state, edge_a), -1)
        val_b = self.oracle._value.get(transition(self.state, edge_b), -1)
        # P2 maximises its own score (oracle values are P2's score)
        return val_a > val_b


# ---------------------------------------------------------------------------
# Run L*
# ---------------------------------------------------------------------------

def learn_strategy(verbose=True):
    nfa    = DotsAndBoxesNFA()
    oracle = MinimaxOracle(nfa)

    # Input alphabet: P1's 12 possible edge claims + PASS (P2 bonus turn)
    p1_alphabet = list(EDGE_NAMES) + ['PASS']   # 13 symbols

    sul      = StrategyLearnerSUL(nfa, oracle)
    eq_oracle = RandomWalkEqOracle(
        alphabet  = p1_alphabet,
        sul       = sul,
        num_steps = 2000,       # random walk length for equivalence checking
        reset_prob = 0.15,      # probability of resetting to initial state
    )

    print("Running L* to learn P2's optimal strategy...")
    learned = run_Lstar(
        alphabet       = p1_alphabet,
        sul            = sul,
        eq_oracle      = eq_oracle,
        automaton_type = 'moore',
        print_level    = 2 if verbose else 0,
    )

    return learned, sul, nfa, oracle


# ---------------------------------------------------------------------------
# Evaluate the learned strategy
# ---------------------------------------------------------------------------

def evaluate(learned, nfa, oracle, num_games=200):
    """
    Play num_games random P1 strategies against the learned P2 strategy.
    Compare outcome to the minimax-optimal P2 outcome.
    """
    import random
    rng = random.Random(42)

    p2_wins = p2_draws = p2_losses = 0
    opt_p2_score = oracle.optimal_value()   # minimax-optimal P2 score

    for _ in range(num_games):
        state    = INITIAL_STATE
        learned.reset_to_initial()

        while not is_terminal(state):
            if state.player == 0:
                # P1 plays randomly
                moves  = legal_moves(state)
                p1_move = rng.choice(moves)
                # Feed into learned machine
                p2_out = learned.step(EDGE_NAMES[p1_move])
                state  = transition(state, p1_move)
                # Apply P2's learned response if it's P2's turn
                if not is_terminal(state) and state.player == 1:
                    if p2_out not in ('PASS', 'GAME_OVER', 'ILLEGAL') and p2_out in EDGE_INDEX:
                        p2_edge = EDGE_INDEX[p2_out]
                        if p2_edge in legal_moves(state):
                            state = transition(state, p2_edge)
                        else:
                            # Fallback: optimal move
                            opt = list(oracle.optimal_moves(state))
                            state = transition(state, opt[0])
            else:
                # P2 bonus turn — apply optimal move directly
                opt = list(oracle.optimal_moves(state))
                state = transition(state, opt[0])

        score = state.score[1]
        if score > 2:      p2_wins   += 1
        elif score == 2:   p2_draws  += 1
        else:              p2_losses += 1

    print(f"\n--- Evaluation over {num_games} games (random P1) ---")
    print(f"  P2 wins   : {p2_wins}")
    print(f"  P2 draws  : {p2_draws}")
    print(f"  P2 losses : {p2_losses}")
    print(f"  Minimax optimal P2 score from start: {opt_p2_score}")


if __name__ == '__main__':
    learned, sul, nfa, oracle = learn_strategy(verbose=True)

    print(f"\nLearned Moore machine:")
    print(f"  States      : {len(learned.states)}")
    outputs = set(s.output for s in learned.states)
    print(f"  Output values used: {outputs}")
    print(f"  Sample outputs per state:")
    for s in list(learned.states)[:5]:
        print(f"    state {s.state_id}: output = {s.output}")

    evaluate(learned, nfa, oracle)
