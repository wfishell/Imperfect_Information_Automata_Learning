"""
Corpus Graph Builder

Converts corpus.json (src/data/Kuhn_Poker/corpus.json) into a NetworkX
directed graph where:

  - Each unique step string ("input_formula/output_formula") is a node.
  - Directed edges connect consecutive steps within a trace.
  - Nodes shared across traces represent the same game state (path merging).
  - A virtual START node connects to the first step of each trace.

Graph semantics (matches the path-graph model from notes):

    START --> step_0 --> step_1 --> ... --> step_n   (one trace = one path)

Two traces that share a step string share a single vertex in the graph.
Edge weight counts how many traces traverse that transition.

Node attributes:
    step_count   -- total appearances of this step across all traces
    trace_ids    -- list of trace indices that pass through this node
    step_indices -- positions within those traces (parallel to trace_ids)

Edge attributes:
    weight       -- number of traces using this (u, v) transition
    trace_ids    -- list of trace indices using this edge

Usage:
    python test/pipeline/corpus/build_graph.py
"""

import json
import os

import networkx as nx

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))               # test/pipeline/corpus/
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE))) # project root
_SRC  = os.path.join(_ROOT, "src")

CORPUS_PATH = os.path.join(_SRC, "data", "Kuhn_Poker", "corpus.json")

# Virtual source node that every trace path originates from
START = "__START__"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_corpus(path: str = CORPUS_PATH) -> list:
    """Return list of traces; each trace is a list of step strings."""
    with open(path) as f:
        data = json.load(f)
    return data["traces"]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_graph(traces: list) -> nx.DiGraph:
    """
    Build a directed graph from the trace corpus.

    Each unique step string becomes a node; directed edges connect
    consecutive steps within a trace.  The virtual START node links
    to the first step of every trace so the graph has a single root.
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
# Analysis helpers
# ---------------------------------------------------------------------------

def shared_nodes(G: nx.DiGraph) -> list:
    """Return nodes that appear in more than one trace, sorted by coverage."""
    result = [
        (n, d)
        for n, d in G.nodes(data=True)
        if n != START and len(set(d["trace_ids"])) > 1
    ]
    result.sort(key=lambda x: len(set(x[1]["trace_ids"])), reverse=True)
    return result


def trace_paths(G: nx.DiGraph, traces: list) -> list:
    """
    Return each trace as an ordered list of node IDs (path through the graph),
    beginning with START.
    """
    paths = []
    for trace in traces:
        paths.append([START] + list(trace))
    return paths


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(G: nx.DiGraph, traces: list) -> None:
    print(f"\n{'='*64}")
    print(f"  CORPUS TRACE GRAPH")
    print(f"{'='*64}")
    print(f"  Corpus path : {CORPUS_PATH}")
    print(f"  Traces      : {len(traces)}")
    print(f"  Nodes       : {G.number_of_nodes() - 1}  (+ 1 virtual START)")
    print(f"  Edges       : {G.number_of_edges()}")

    s_nodes = shared_nodes(G)
    print(f"  Shared nodes (appear in >= 2 traces): {len(s_nodes)}")

    if s_nodes:
        print(f"\n  Top shared nodes:")
        for node, data in s_nodes[:5]:
            n_traces = len(set(data["trace_ids"]))
            preview = node[:72] + "..." if len(node) > 72 else node
            print(f"    [{n_traces:>2} traces] {preview}")

    print(f"\n  Trace lengths (steps per trace):")
    for i, trace in enumerate(traces):
        print(f"    [{i:>2}]  {len(trace)} steps")

    wcc = list(nx.weakly_connected_components(G))
    print(f"\n  Weakly connected components: {len(wcc)}")
    print(f"{'='*64}\n")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_graphml(G: nx.DiGraph, path: str) -> None:
    """
    Export the graph to GraphML.

    NetworkX requires all node/edge attributes to be primitive types for
    GraphML export, so list attributes are serialised as JSON strings.
    """
    H = nx.DiGraph()
    for node, data in G.nodes(data=True):
        H.add_node(node, **{
            k: json.dumps(v) if isinstance(v, list) else v
            for k, v in data.items()
        })
    for u, v, data in G.edges(data=True):
        H.add_edge(u, v, **{
            k: json.dumps(v_) if isinstance(v_, list) else v_
            for k, v_ in data.items()
        })
    nx.write_graphml(H, path)
    print(f"[graph] Exported -> {path}")


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
        (START, step_0, ..., step_n)  <- full trace

    Prefixes are deduplicated: two traces that share the same opening
    k steps yield a single prefix tuple.  Enumeration is from the raw
    trace list rather than graph traversal to guarantee every prefix
    corresponds to at least one real trace (no spurious graph paths).

    Returns:
        list of unique prefix tuples, ordered by first appearance.
    """
    seen: set = set()
    prefixes  = []

    # START alone is always the universal prefix
    seen.add((START,))
    prefixes.append((START,))

    for trace in traces:
        for length in range(1, len(trace) + 1):
            prefix = (START,) + tuple(trace[:length])
            if prefix not in seen:
                seen.add(prefix)
                prefixes.append(prefix)

    return prefixes


def futures_of_prefix(G: nx.DiGraph, prefix: tuple) -> frozenset:
    """
    Return the set of trace indices whose path begins with this prefix.

    Computed as the intersection of trace_ids on each consecutive edge
    in the prefix.  Because edges are only created between steps that are
    consecutive within a real trace, this intersection is exact.

    Args:
        G:      The corpus graph from build_graph().
        prefix: A tuple (START, v0, v1, ..., vk).

    Returns:
        frozenset of trace indices that extend this prefix.
    """
    if len(prefix) < 2:
        # Only START -- every trace is a future.
        return frozenset(range(len(G.nodes[START]["trace_ids"])))

    result = None
    for i in range(len(prefix) - 1):
        u, v = prefix[i], prefix[i + 1]
        edge_traces = frozenset(G[u][v]["trace_ids"])
        result = edge_traces if result is None else result & edge_traces

    return result if result is not None else frozenset()


# ---------------------------------------------------------------------------
# Stochastic dominance
# ---------------------------------------------------------------------------

_DOMINATES    = "dominates"
_DOMINATED    = "dominated"
_EQUIVALENT   = "equivalent"
_INCOMPARABLE = "incomparable"


def _cdf(ranks: frozenset, k: int, total: int) -> float:
    """Fraction of futures with rank <= k  (lower rank = more preferred)."""
    return sum(1 for r in ranks if r <= k) / total


def stochastic_dominance(
    ranks_p1: frozenset,
    ranks_p2: frozenset,
    n_traces: int,
) -> str:
    """
    First-order stochastic dominance comparison between two prefix future sets.

    A prefix P1 dominates P2 when, at every rank threshold k, the fraction
    of P1's futures that are rank <= k is >= the same fraction for P2
    (with at least one strict inequality).  Higher CDF at every threshold
    means P1 is more likely to reach a preferred outcome.

    Args:
        ranks_p1:  Set of trace ranks reachable from prefix P1.
        ranks_p2:  Set of trace ranks reachable from prefix P2.
        n_traces:  Total number of traces (rank ceiling).

    Returns:
        One of: "dominates", "dominated", "equivalent", "incomparable".
    """
    if not ranks_p1 or not ranks_p2:
        return _INCOMPARABLE

    n1, n2 = len(ranks_p1), len(ranks_p2)

    p1_ge_p2 = True  # CDF_P1(k) >= CDF_P2(k) for all k so far
    p2_ge_p1 = True  # CDF_P2(k) >= CDF_P1(k) for all k so far

    for k in range(1, n_traces + 1):
        c1 = _cdf(ranks_p1, k, n1)
        c2 = _cdf(ranks_p2, k, n2)
        if c1 < c2:
            p1_ge_p2 = False
        if c2 < c1:
            p2_ge_p1 = False
        if not p1_ge_p2 and not p2_ge_p1:
            break  # CDFs crossed in both directions -- already incomparable

    if p1_ge_p2 and p2_ge_p1:
        return _EQUIVALENT
    if p1_ge_p2:
        return _DOMINATES
    if p2_ge_p1:
        return _DOMINATED
    return _INCOMPARABLE


# ---------------------------------------------------------------------------
# Partial order computation
# ---------------------------------------------------------------------------

def compute_partial_order(
    G: nx.DiGraph,
    traces: list,
    trace_ranks: dict,
) -> nx.DiGraph:
    """
    Derive a partial order over all prefixes using first-order stochastic
    dominance.

    The total order over complete traces is given by trace_ranks:
        trace_ranks[i] = rank of trace i   (rank 1 = most preferred)

    For each prefix P, futures(P) is the set of complete trace ranks
    reachable from P.  Two prefixes are compared by checking whether one's
    future-rank distribution stochastically dominates the other's.

    The result is a DiGraph (the dominance DAG) where an edge (P1, P2)
    means P1 stochastically dominates P2.

    Node attributes:
        futures   -- frozenset of trace ranks reachable from this prefix
        trace_ids -- frozenset of trace indices reachable from this prefix
        label     -- short human-readable path summary

    Edge attributes:
        relation  -- always "dominates"

    Incomparable and equivalent pairs are stored on the graph object as:
        po.graph["incomparable"]  -- list of (P1, P2) tuples
        po.graph["equivalent"]    -- list of (P1, P2) tuples

    Args:
        G:           Corpus graph from build_graph().
        traces:      Raw trace list.
        trace_ranks: dict mapping trace index -> integer rank.

    Returns:
        nx.DiGraph representing the strict dominance partial order.
    """
    prefixes = enumerate_prefixes(traces)
    n        = len(traces)

    # Map each prefix tuple -> its set of future ranks
    prefix_futures = {}
    for p in prefixes:
        trace_ids = futures_of_prefix(G, p)
        prefix_futures[p] = frozenset(trace_ranks[t] for t in trace_ids)

    # Build the dominance DAG
    po = nx.DiGraph()
    for p in prefixes:
        po.add_node(p,
                    futures=prefix_futures[p],
                    trace_ids=futures_of_prefix(G, p),
                    label=_prefix_label(p))

    incomparable_pairs = []
    equivalent_pairs   = []

    for i in range(len(prefixes)):
        for j in range(i + 1, len(prefixes)):
            p1, p2 = prefixes[i], prefixes[j]
            rel = stochastic_dominance(prefix_futures[p1], prefix_futures[p2], n)

            if rel == _DOMINATES:
                po.add_edge(p1, p2, relation=_DOMINATES)
            elif rel == _DOMINATED:
                po.add_edge(p2, p1, relation=_DOMINATES)
            elif rel == _EQUIVALENT:
                equivalent_pairs.append((p1, p2))
            else:
                incomparable_pairs.append((p1, p2))

    po.graph["incomparable"] = incomparable_pairs
    po.graph["equivalent"]   = equivalent_pairs

    return po


def _prefix_label(prefix: tuple) -> str:
    """Short human-readable label for a prefix tuple."""
    if len(prefix) == 1:
        return "START"
    depth = len(prefix) - 1
    last  = prefix[-1]
    short = last[:40] + "..." if len(last) > 40 else last
    return f"depth={depth} | ...{short}"


# ---------------------------------------------------------------------------
# Partial order summary
# ---------------------------------------------------------------------------

def print_partial_order_summary(po: nx.DiGraph) -> None:
    """Print a structured summary of the stochastic-dominance partial order."""
    n_prefixes     = po.number_of_nodes()
    n_dominance    = po.number_of_edges()
    n_incomparable = len(po.graph["incomparable"])
    n_equivalent   = len(po.graph["equivalent"])
    n_pairs        = n_prefixes * (n_prefixes - 1) // 2

    print(f"\n{'='*64}")
    print(f"  STOCHASTIC DOMINANCE PARTIAL ORDER")
    print(f"{'='*64}")
    print(f"  Prefixes           : {n_prefixes}")
    print(f"  Ordered pairs      : {n_pairs}")
    print(f"  Dominance edges    : {n_dominance}  ({100*n_dominance/max(n_pairs,1):.1f}%)")
    print(f"  Equivalent pairs   : {n_equivalent}  ({100*n_equivalent/max(n_pairs,1):.1f}%)")
    print(f"  Incomparable pairs : {n_incomparable}  ({100*n_incomparable/max(n_pairs,1):.1f}%)")

    # Prefixes with the most outgoing dominance edges
    by_out = sorted(po.nodes(), key=lambda n: po.out_degree(n), reverse=True)
    print(f"\n  Most dominant prefixes (by out-degree in dominance DAG):")
    for p in by_out[:5]:
        data  = po.nodes[p]
        ranks = sorted(data["futures"])
        print(f"    [{po.out_degree(p):>3} dominated] futures={ranks}  {data['label']}")

    # Sample incomparable pairs
    if po.graph["incomparable"]:
        print(f"\n  Sample incomparable pairs (CDFs cross -- genuine ambiguity):")
        for p1, p2 in po.graph["incomparable"][:3]:
            r1 = sorted(po.nodes[p1]["futures"])
            r2 = sorted(po.nodes[p2]["futures"])
            print(f"    {_prefix_label(p1)}  futures={r1}")
            print(f"    {_prefix_label(p2)}  futures={r2}")
            print(f"    --> INCOMPARABLE")
            print()

    print(f"{'='*64}\n")


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def export_partial_order_json(
    po: nx.DiGraph,
    traces: list,
    trace_ranks: dict,
    path: str,
) -> None:
    """
    Write traces (with ranks) and the full prefix partial order to a JSON file.

    Schema
    ------
    {
      "_schema": { ... },
      "traces": [
        { "trace_id": 0, "rank": 1, "steps": ["inp/out", ...] },
        ...
      ],
      "prefixes": [
        {
          "prefix_id":       0,
          "depth":           3,
          "steps":           ["inp/out", "inp/out", "inp/out"],
          "future_ranks":    [1, 4],
          "future_trace_ids": [0, 3]
        },
        ...
      ],
      "partial_order": {
        "dominance_edges":   [[p1_id, p2_id], ...],
        "equivalent_pairs":  [[p1_id, p2_id], ...],
        "incomparable_pairs": [[p1_id, p2_id], ...]
      }
    }

    prefix_id 0 is always START (depth 0, no steps).
    dominance_edges[i] = [a, b] means prefix a stochastically dominates prefix b.
    """
    # Build a stable prefix_id mapping (insertion order from compute_partial_order)
    prefix_list = list(po.nodes())
    prefix_to_id = {p: i for i, p in enumerate(prefix_list)}

    prefixes_out = []
    for p in prefix_list:
        data = po.nodes[p]
        steps = list(p[1:])  # strip START
        prefixes_out.append({
            "prefix_id":        prefix_to_id[p],
            "depth":            len(steps),
            "steps":            steps,
            "future_ranks":     sorted(data["futures"]),
            "future_trace_ids": sorted(data["trace_ids"]),
        })

    traces_out = [
        {
            "trace_id": i,
            "rank":     trace_ranks[i],
            "steps":    list(trace),
        }
        for i, trace in enumerate(traces)
    ]

    partial_order_out = {
        "dominance_edges": [
            [prefix_to_id[u], prefix_to_id[v]]
            for u, v in po.edges()
        ],
        "equivalent_pairs": [
            [prefix_to_id[p1], prefix_to_id[p2]]
            for p1, p2 in po.graph["equivalent"]
        ],
        "incomparable_pairs": [
            [prefix_to_id[p1], prefix_to_id[p2]]
            for p1, p2 in po.graph["incomparable"]
        ],
    }

    payload = {
        "_schema": {
            "traces":        "list of traces with their assigned rank (rank 1 = most preferred)",
            "prefixes":      "list of all unique prefixes; depth=0 is START; steps exclude START",
            "partial_order": {
                "dominance_edges":    "[p1_id, p2_id] means prefix p1 stochastically dominates p2",
                "equivalent_pairs":   "[p1_id, p2_id] means identical future-rank CDF",
                "incomparable_pairs": "[p1_id, p2_id] means CDFs cross -- neither dominates",
            },
        },
        "traces":        traces_out,
        "prefixes":      prefixes_out,
        "partial_order": partial_order_out,
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[partial order] Exported -> {path}")


# ---------------------------------------------------------------------------
# Entry point  /  unit test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    traces = load_corpus()
    G      = build_graph(traces)
    print_summary(G, traces)

    # -- Export corpus graph --
    export_path = os.path.join(_HERE, "corpus_graph.graphml")
    export_graphml(G, export_path)

    # -----------------------------------------------------------------------
    # Unit test: stochastic dominance partial order
    #
    # Provide a total ordering over the corpus traces.
    # Rank 1 = most preferred, rank N = least preferred.
    # Natural index order is used here as a reproducible baseline;
    # swap values to experiment with different orderings.
    # -----------------------------------------------------------------------
    trace_ranks = {i: i + 1 for i in range(len(traces))}
    # trace_ranks = {0: 3, 1: 1, 2: 2, ...}  # custom ordering example

    print(f"  Unit-test trace ranking (rank 1 = most preferred):")
    for idx, rank in trace_ranks.items():
        print(f"    trace[{idx}] -> rank {rank}")
    print()

    po = compute_partial_order(G, traces, trace_ranks)
    print_partial_order_summary(po)

    po_path = os.path.join(_SRC, "data", "Kuhn_Poker", "partial_order.json")
    export_partial_order_json(po, traces, trace_ranks, po_path)
