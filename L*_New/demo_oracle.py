"""
Demo: two game traces fed into the preference oracle.
"""
import random
from dots_and_boxes import (
    DotsAndBoxesNFA, EDGE_NAMES, INITIAL_STATE,
    transition, legal_moves, is_terminal, winner
)
from preference_oracle import OutcomeOracle, MinimaxOracle


def play_game(nfa, seed):
    rng = random.Random(seed)
    state = INITIAL_STATE
    trace = []
    while not is_terminal(state):
        edge = rng.choice(legal_moves(state))
        trace.append(edge)
        state = transition(state, edge)
    return trace


def trace_to_spot(trace):
    """
    Convert a raw edge trace to spot semantics:
        input1,output1;input2,output2;...

    Input  alphabet: p1_<edge> or p1_PASS  (P1's move or P1 skips on P2 bonus)
    Output alphabet: p2_<edge> or p2_PASS  (P2's move or P2 skips on P1 bonus)

    PASS means: the other player completed a box and gets another turn,
                so this player has no move this step.
    """
    state         = INITIAL_STATE
    steps         = []
    current_input = None   # p1_* symbol for the current step

    for edge in trace:
        name = EDGE_NAMES[edge]
        if state.player == 0:
            # P1 moves
            if current_input is not None:
                # P1 moved consecutively — P1 bonus, P2 didn't get a turn
                steps.append(f"{current_input},p2_PASS")
            current_input = f"p1_{name}"
        else:
            # P2 moves
            if current_input is None:
                # P2 bonus — P1 didn't move this step
                current_input = "p1_PASS"
            steps.append(f"{current_input},p2_{name}")
            current_input = None
        state = transition(state, edge)

    # Final P1 move with no P2 response (game ended on P1 bonus)
    if current_input is not None:
        steps.append(f"{current_input},p2_PASS")

    return ";".join(steps), state


def show_trace(nfa, trace, label):
    spot, final_state = trace_to_spot(trace)
    w      = winner(final_state)
    result = f"P{w+1} wins {final_state.score[0]}-{final_state.score[1]}" if w is not None else f"Draw {final_state.score[0]}-{final_state.score[1]}"
    print(f"  {label}  [{result}]")
    print(f"  {spot}")
    print()


if __name__ == "__main__":
    nfa = DotsAndBoxesNFA()
    oracle = MinimaxOracle(nfa)

    trace1 = play_game(nfa, seed=0)
    trace2 = play_game(nfa, seed=1)

    show_trace(nfa, trace1, "Trace 1 (seed 0)")
    show_trace(nfa, trace2, "Trace 2 (seed 1)")

    print(f"{'='*50}")
    print(f"  Preference Query")
    print(f"{'='*50}")
    print(f"  Input  : (trace1, trace2)")
    print(f"  Trace 1 P2 score : {nfa.state_after(trace1).score[1]}")
    print(f"  Trace 2 P2 score : {nfa.state_after(trace2).score[1]}")
    result = oracle.query(trace1, trace2)
    label  = {1: "Trace 1 preferred", -1: "Trace 2 preferred", 0: "Indifferent"}
    print(f"  Output : {result:+d}  ({label[result]})")
