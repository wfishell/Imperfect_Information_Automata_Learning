"""
Generate and display sample game traces for Dots and Boxes 2x2.
"""
import random
from dots_and_boxes import DotsAndBoxesNFA, EDGE_NAMES, winner, is_terminal


def random_game(nfa, seed=None):
    """Play a random game, returning the sequence of edges claimed."""
    rng = random.Random(seed)
    state = nfa.initial_state
    trace = []
    while not is_terminal(state):
        moves = nfa.legal_moves_at(state)
        edge = rng.choice(moves)
        trace.append(edge)
        state = nfa.delta(state, edge)
    return trace, state


def display_trace(nfa, trace):
    """Print a step-by-step walkthrough of a game trace."""
    state = nfa.initial_state
    print(f"  Start — P{state.player+1} to move")
    print()
    nfa.display_state(state)
    print()

    for step, edge in enumerate(trace):
        player = state.player
        state = nfa.delta(state, edge)
        boxes_before = sum(state.score)
        print(f"  Step {step+1}: P{player+1} claims {EDGE_NAMES[edge]}")
        nfa.display_state(state)
        print()

    w = winner(state)
    score = state.score
    if w is None:
        result = f"Draw  ({score[0]}-{score[1]})"
    else:
        result = f"P{w+1} wins  ({score[0]}-{score[1]})"
    print(f"  Result: {result}")
    print(f"  Trace:  {[EDGE_NAMES[e] for e in trace]}")


if __name__ == "__main__":
    nfa = DotsAndBoxesNFA()

    NUM_SAMPLES = 3
    print("=" * 50)
    print(f"  Dots and Boxes 2x2 — {NUM_SAMPLES} sample games")
    print("=" * 50)

    for i in range(NUM_SAMPLES):
        print(f"\n{'='*50}")
        print(f"  Game {i+1}")
        print(f"{'='*50}\n")
        trace, final_state = random_game(nfa, seed=i)
        display_trace(nfa, trace)

    # Also verify all traces are accepted by the NFA
    print(f"\n{'='*50}")
    print("  Verification: all traces accepted by NFA")
    print(f"{'='*50}")
    for i in range(NUM_SAMPLES):
        trace, _ = random_game(nfa, seed=i)
        accepted = nfa.accepts(trace)
        print(f"  Game {i+1}: {[EDGE_NAMES[e] for e in trace]} -> accepted={accepted}")
