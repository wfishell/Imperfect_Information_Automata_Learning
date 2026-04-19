"""
Explore the Dots and Boxes 2x2 regular language:
  - Build the full reachable DFA via BFS
  - Print state/transition counts
  - Export a DOT file for visualisation
  - Enumerate a sample of accepted words
"""
from collections import deque
from dots_and_boxes import (
    DotsAndBoxesNFA, EDGE_NAMES, NUM_EDGES, is_terminal, winner
)


def build_dfa(nfa):
    """
    BFS over reachable states from q0.
    Returns:
        states      : list of reachable GameState objects
        transitions : dict  state -> {symbol -> next_state}
        accepting   : set of accepting state indices
    """
    q0 = nfa.initial_state

    state_to_id = {q0: 0}
    id_to_state  = [q0]
    transitions  = {}   # state_id -> {symbol -> state_id}
    accepting    = set()
    queue        = deque([q0])

    while queue:
        state = queue.popleft()
        sid   = state_to_id[state]
        transitions[sid] = {}

        if is_terminal(state):
            accepting.add(sid)
            continue                  # no outgoing edges from terminal states

        for symbol in nfa.alphabet:
            next_state = nfa.delta(state, symbol)
            if next_state is None:
                continue              # illegal move — dead state, skip
            if next_state not in state_to_id:
                nid = len(id_to_state)
                state_to_id[next_state] = nid
                id_to_state.append(next_state)
                queue.append(next_state)
            transitions[sid][symbol] = state_to_id[next_state]

    return id_to_state, transitions, accepting


def enumerate_words(nfa, max_words=10):
    """
    Enumerate accepted words (valid complete games) via DFS.
    Yields at most max_words words.
    """
    found  = 0
    stack  = [(nfa.initial_state, [])]
    while stack and found < max_words:
        state, path = stack.pop()
        if is_terminal(state):
            yield path
            found += 1
            continue
        for symbol in nfa.legal_moves_at(state):
            next_state = nfa.delta(state, symbol)
            stack.append((next_state, path + [symbol]))


def export_dot(id_to_state, transitions, accepting, path="dfa.dot", max_states=80):
    """
    Write a Graphviz DOT file for the DFA.
    Limits output to max_states nodes so the graph stays readable.
    """
    with open(path, "w") as f:
        f.write("digraph DotsAndBoxesDFA {\n")
        f.write("  rankdir=LR;\n")
        f.write('  node [shape=circle fontsize=9];\n')

        num_states = min(len(id_to_state), max_states)

        for sid in range(num_states):
            state = id_to_state[sid]
            shape = "doublecircle" if sid in accepting else "circle"
            score = state.score
            label = f"s{sid}\\nP{state.player+1}\\n{score[0]}-{score[1]}"
            f.write(f'  s{sid} [shape={shape} label="{label}"];\n')

        # Mark initial state
        f.write('  start [shape=point];\n')
        f.write('  start -> s0;\n')

        for sid, edges in transitions.items():
            if sid >= max_states:
                continue
            for symbol, nid in edges.items():
                if nid >= max_states:
                    continue
                label = EDGE_NAMES[symbol]
                f.write(f'  s{sid} -> s{nid} [label="{label}"];\n')

        f.write("}\n")
    print(f"  DOT file written to {path}")
    print(f"  (showing first {num_states} of {len(id_to_state)} states)")
    print(f"  Render with:  dot -Tpng dfa.dot -o dfa.png")


if __name__ == "__main__":
    nfa = DotsAndBoxesNFA()

    # -----------------------------------------------------------------------
    print("Building full reachable DFA via BFS...")
    id_to_state, transitions, accepting = build_dfa(nfa)

    total_states      = len(id_to_state)
    total_transitions = sum(len(v) for v in transitions.values())

    print(f"\n--- DFA summary ---")
    print(f"  Reachable states      : {total_states}")
    print(f"  Accepting states      : {len(accepting)}  (complete games)")
    print(f"  Total transitions     : {total_transitions}")
    print(f"  Alphabet size         : {NUM_EDGES} symbols")
    print(f"  Initial state         : s0")

    # -----------------------------------------------------------------------
    print(f"\n--- State breakdown by number of claimed edges ---")
    depth_counts = {}
    for state in id_to_state:
        d = len(state.claimed)
        depth_counts[d] = depth_counts.get(d, 0) + 1
    for d in sorted(depth_counts):
        tag = " <- accepting" if d == NUM_EDGES else ""
        print(f"  depth {d:2d} : {depth_counts[d]:6d} states{tag}")

    # -----------------------------------------------------------------------
    print(f"\n--- Sample accepted words (valid complete games) ---")
    for word in enumerate_words(nfa, max_words=5):
        names  = [EDGE_NAMES[e] for e in word]
        state  = nfa.state_after(word)
        w      = winner(state)
        result = f"P{w+1} wins {state.score[0]}-{state.score[1]}" if w is not None else f"Draw {state.score[0]}-{state.score[1]}"
        print(f"  {names}")
        print(f"    -> {result}")

    # -----------------------------------------------------------------------
    print(f"\n--- Outcome distribution (over all {len(accepting)} complete games) ---")
    p1_wins = p2_wins = draws = 0
    for sid in accepting:
        state = id_to_state[sid]
        w = winner(state)
        if w == 0:   p1_wins += 1
        elif w == 1: p2_wins += 1
        else:        draws   += 1
    print(f"  P1 wins : {p1_wins}")
    print(f"  P2 wins : {p2_wins}")
    print(f"  Draws   : {draws}")

    # -----------------------------------------------------------------------
    export_dot(id_to_state, transitions, accepting)
