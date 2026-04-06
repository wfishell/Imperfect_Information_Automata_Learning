"""
Preference Power Set Pipeline

Loads traces from src/data/Kuhn_Poker/kuhn_traces.txt and builds a corpus
graph using the same graph model as test/pipeline/corpus/build_graph.py.

Trace file format:
    - One trace per line.
    - Steps within a trace are separated by semicolons (;).
    - Each step is a Boolean formula string (e.g. "a0&!a1&...").

Graph semantics:
    START --> step_0 --> step_1 --> ... --> step_n   (one trace = one path)

Two traces that share a step string share a single vertex in the graph.
Edge weight counts how many traces traverse that transition.
"""

import os

import networkx as nx

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE       = os.path.dirname(os.path.abspath(__file__))          # src/pipeline/
_SRC        = os.path.dirname(_HERE)                              # src/
TRACES_PATH = os.path.join(_SRC, "data", "Kuhn_Poker", "kuhn_traces.txt")

START = "__START__"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_traces(path: str = TRACES_PATH) -> list:
    """
    Return a list of traces from kuhn_traces.txt.

    Each trace is a list of step strings.  Lines are traces; steps within a
    line are separated by semicolons.  Blank lines are skipped.
    """
    traces = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            steps = [s.strip() for s in line.split(";") if s.strip()]
            traces.append(steps)
    return traces


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_graph(traces: list) -> nx.DiGraph:
    """
    Build a directed graph from a list of traces.

    Each unique step string becomes a node; directed edges connect consecutive
    steps within a trace.  The virtual START node links to the first step of
    every trace so the graph has a single root.

    Node attributes:
        step_count   -- total appearances of this step across all traces
        trace_ids    -- list of trace indices that pass through this node
        step_indices -- positions within those traces (parallel to trace_ids)

    Edge attributes:
        weight       -- number of traces using this (u, v) transition
        trace_ids    -- list of trace indices using this edge
    """
    G = nx.DiGraph()
    G.add_node(START, step_count=len(traces),
               trace_ids=list(range(len(traces))),
               step_indices=[-1] * len(traces))

    for trace_idx, trace in enumerate(traces):
        prev = START
        for step_idx, step in enumerate(trace):
            # ---- node ----
            if step not in G:
                G.add_node(step, step_count=0, trace_ids=[], step_indices=[])
            G.nodes[step]["step_count"] += 1
            G.nodes[step]["trace_ids"].append(trace_idx)
            G.nodes[step]["step_indices"].append(step_idx)

            # ---- edge ----
            if G.has_edge(prev, step):
                G[prev][step]["weight"] += 1
                G[prev][step]["trace_ids"].append(trace_idx)
            else:
                G.add_edge(prev, step, weight=1, trace_ids=[trace_idx])

            prev = step

    return G


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    traces = load_traces()
    G = build_graph(traces)

    n_real = G.number_of_nodes() - 1  # exclude START
    print(f"Traces : {len(traces)}")
    print(f"Nodes  : {n_real}  (+ 1 virtual START)")
    print(f"Edges  : {G.number_of_edges()}")
