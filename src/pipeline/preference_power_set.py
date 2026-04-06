"""
Preference Power Set Pipeline

Loads traces from src/data/Kuhn_Poker/kuhn_traces.txt and pairwise trace
preferences from src/data/Kuhn_Poker/kuhn_prefs_clean.json, then computes
a total preference ordering over every prefix (subtrace) of the corpus.

Steps
-----
1. load_traces        -- parse kuhn_traces.txt  (one trace per line, ;-separated steps)
2. load_prefs         -- parse kuhn_prefs_clean.json  (pairwise trace preferences)
3. build_graph        -- corpus DAG  (START + one node per unique step string)
4. build_trace_ranks  -- Borda-count score → integer rank for every trace
5. enumerate_prefixes -- all unique prefixes (START, step_0, ..., step_k)
6. futures_of_prefix  -- reachable trace indices for a given prefix
7. expected_rank      -- mean trace rank reachable from a prefix (lower = better)
8. compute_prefix_preferences -- pairwise pref for every pair of prefixes
9. write output JSON  -- src/data/Kuhn_Poker/prefix_preferences.json

Output format (mirrors the input pref format)
---------------------------------------------
[
  {
    "i":       <int>,          # index of prefix i in the enumerated prefix list
    "j":       <int>,          # index of prefix j
    "pref":    1 | -1 | 0,    # 1 = i preferred, -1 = j preferred, 0 = indifferent
    "subtrace_i": <str>,       # full subtrace as ;-separated steps
    "subtrace_j": <str>,
    "er_i":    <float>,        # expected rank of prefix i (lower = better)
    "er_j":    <float>
  },
  ...
]
"""

import json
import os

import networkx as nx

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE        = os.path.dirname(os.path.abspath(__file__))          # src/pipeline/
_SRC         = os.path.dirname(_HERE)                              # src/
_DATA        = os.path.join(_SRC, "data", "Kuhn_Poker")

TRACES_PATH  = os.path.join(_DATA, "kuhn_traces.txt")
PREFS_PATH   = os.path.join(_DATA, "kuhn_prefs_clean.json")
OUTPUT_PATH  = os.path.join(_DATA, "prefix_preferences.json")

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


def load_prefs(path: str = PREFS_PATH) -> list:
    """
    Return the raw pairwise preference list from kuhn_prefs_clean.json.

    Each entry is {"i": int, "j": int, "pref": 1 | -1 | 0}.
      pref =  1  -> trace i preferred over trace j
      pref = -1  -> trace j preferred over trace i
      pref =  0  -> indifferent
    """
    with open(path) as f:
        return json.load(f)


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
            if step not in G:
                G.add_node(step, step_count=0, trace_ids=[], step_indices=[])
            G.nodes[step]["step_count"] += 1
            G.nodes[step]["trace_ids"].append(trace_idx)
            G.nodes[step]["step_indices"].append(step_idx)

            if G.has_edge(prev, step):
                G[prev][step]["weight"] += 1
                G[prev][step]["trace_ids"].append(trace_idx)
            else:
                G.add_edge(prev, step, weight=1, trace_ids=[trace_idx])

            prev = step

    return G


# ---------------------------------------------------------------------------
# Trace ranking from pairwise preferences
# ---------------------------------------------------------------------------

def build_trace_ranks(prefs: list, n_traces: int) -> dict:
    """
    Derive integer ranks from a consistent set of pairwise preferences via
    topological sort.

    A directed graph is built where an edge i -> j means trace i is strictly
    preferred over trace j (pref=1).  Indifferent pairs (pref=0) are treated
    as equivalent and collapsed to the same rank.  Because the preferences are
    guaranteed consistent (no cycles), the graph is a DAG and a unique
    topological order exists.

    Rank is assigned by layer (longest path from a source node):
        layer 0  -> rank 1  (most preferred, no trace beats it)
        layer 1  -> rank 2
        ...

    Traces within the same layer are tied and share a rank.

    Returns:
        dict mapping trace index -> integer rank (1 = best).
    """
    # Build preference DAG (strict edges only; skip pref=0)
    dag = nx.DiGraph()
    dag.add_nodes_from(range(n_traces))
    for entry in prefs:
        i, j, p = entry["i"], entry["j"], entry["pref"]
        if p == 1:
            dag.add_edge(i, j)
        elif p == -1:
            dag.add_edge(j, i)

    # Assign each node to its layer = length of the longest path from any source
    layer: dict = {t: 0 for t in range(n_traces)}
    for t in nx.topological_sort(dag):
        for successor in dag.successors(t):
            if layer[successor] < layer[t] + 1:
                layer[successor] = layer[t] + 1

    # Layer 0 = rank 1 (most preferred)
    return {t: layer[t] + 1 for t in range(n_traces)}


# ---------------------------------------------------------------------------
# Prefix enumeration
# ---------------------------------------------------------------------------

def enumerate_prefixes(traces: list) -> list:
    """
    Return every unique prefix across all traces as a tuple of step strings.

    A prefix is a tuple rooted at START:
        (START,)
        (START, step_0)
        (START, step_0, step_1)
        ...
        (START, step_0, ..., step_n)   <- full trace included

    Prefixes are deduplicated; order of first appearance is preserved.
    """
    seen: set = set()
    prefixes = []

    root = (START,)
    seen.add(root)
    prefixes.append(root)

    for trace in traces:
        for length in range(1, len(trace) + 1):
            prefix = (START,) + tuple(trace[:length])
            if prefix not in seen:
                seen.add(prefix)
                prefixes.append(prefix)

    return prefixes


# ---------------------------------------------------------------------------
# Futures of a prefix
# ---------------------------------------------------------------------------

def futures_of_prefix(G: nx.DiGraph, prefix: tuple) -> frozenset:
    """
    Return the set of trace indices whose path begins with this prefix.

    Computed by intersecting the trace_ids on each consecutive edge of the
    prefix.  Returns all traces for the lone-START prefix.
    """
    if len(prefix) < 2:
        return frozenset(G.nodes[START]["trace_ids"])

    result = None
    for i in range(len(prefix) - 1):
        u, v = prefix[i], prefix[i + 1]
        if not G.has_edge(u, v):
            return frozenset()
        edge_traces = frozenset(G[u][v]["trace_ids"])
        result = edge_traces if result is None else result & edge_traces

    return result if result is not None else frozenset()


# ---------------------------------------------------------------------------
# Expected rank
# ---------------------------------------------------------------------------

def expected_rank(future_ranks: frozenset) -> float:
    """
    Mean rank of the reachable trace set.

    Lower mean = better prefix (rank 1 = most preferred).
    Returns float('inf') for an empty future set.
    """
    if not future_ranks:
        return float("inf")
    return sum(future_ranks) / len(future_ranks)


# ---------------------------------------------------------------------------
# Prefix label
# ---------------------------------------------------------------------------

def _prefix_label(prefix: tuple) -> str:
    """Full subtrace as semicolon-separated steps (START node excluded)."""
    return ";".join(prefix[1:])


# ---------------------------------------------------------------------------
# Compute pairwise prefix preferences
# ---------------------------------------------------------------------------

def compute_prefix_preferences(
    G: nx.DiGraph,
    traces: list,
    trace_ranks: dict,
) -> list:
    """
    Compute a pairwise preference for every pair of prefixes using expected rank.

    For prefix pair (P_i, P_j):
        pref =  1  if E[rank(P_i)] < E[rank(P_j)]   (P_i preferred)
        pref = -1  if E[rank(P_i)] > E[rank(P_j)]   (P_j preferred)
        pref =  0  if E[rank(P_i)] == E[rank(P_j)]  (indifferent / tied)

    Returns a list of dicts matching the output schema described at the top.
    """
    prefixes = enumerate_prefixes(traces)

    # Pre-compute expected rank for every prefix
    er: dict = {}
    label: dict = {}
    for p in prefixes:
        future_trace_ids = futures_of_prefix(G, p)
        future_ranks     = frozenset(trace_ranks[t] for t in future_trace_ids)
        er[p]    = expected_rank(future_ranks)
        label[p] = _prefix_label(p)

    # Only compare real game prefixes (depth >= 1); exclude the virtual START node
    real_prefixes = [(idx, p) for idx, p in enumerate(prefixes) if len(p) > 1]

    # Enumerate all pairs
    results = []
    for a in range(len(real_prefixes)):
        for b in range(a + 1, len(real_prefixes)):
            i, pi = real_prefixes[a]
            j, pj = real_prefixes[b]
            ei, ej = er[pi], er[pj]

            if ei < ej:
                pref = 1
            elif ei > ej:
                pref = -1
            else:
                pref = 0

            results.append({
                "i":          i,
                "j":          j,
                "pref":       pref,
                "subtrace_i": label[pi],
                "subtrace_j": label[pj],
                "er_i":       ei if ei != float("inf") else None,
                "er_j":       ej if ej != float("inf") else None,
            })

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Load data
    traces = load_traces()
    prefs  = load_prefs()
    print(f"Traces loaded : {len(traces)}")
    print(f"Pref entries  : {len(prefs)}")

    # 2. Build corpus graph
    G = build_graph(traces)
    n_real = G.number_of_nodes() - 1
    print(f"Graph nodes   : {n_real}  (+ 1 virtual START)")
    print(f"Graph edges   : {G.number_of_edges()}")

    # 3. Derive trace ranks from pairwise preferences
    trace_ranks = build_trace_ranks(prefs, len(traces))
    print(f"\nTrace ranks (topological sort, 1 = most preferred):")
    for t in range(len(traces)):
        print(f"  trace {t:>2}  rank {trace_ranks[t]}")

    # 4. Compute prefix preferences
    prefix_prefs = compute_prefix_preferences(G, traces, trace_ranks)
    n_prefixes   = len(set(r["i"] for r in prefix_prefs) | set(r["j"] for r in prefix_prefs)) + 1
    print(f"\nPrefixes enumerated : {n_prefixes}")
    print(f"Prefix pref pairs   : {len(prefix_prefs)}")

    # 5. Write output
    with open(OUTPUT_PATH, "w") as f:
        json.dump(prefix_prefs, f, indent=2)
    print(f"\nOutput written -> {OUTPUT_PATH}")
