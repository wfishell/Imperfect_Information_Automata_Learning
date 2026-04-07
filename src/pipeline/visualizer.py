#!/usr/bin/env python3
"""
Visualizer

Rendering utilities for pipeline artifacts.
"""

from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from pipeline.consistency_checker import (
    build_graph,
    expand_symmetric,
    get_cycle_edges,
    min_feedback_arc_set,
    tarjan_sccs,
)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _longest_path_ranks(dag: nx.DiGraph) -> dict:
    """
    Rank each node by the length of the longest path from any source.
    Sources get rank 0; each node sits one level below its deepest predecessor.
    """
    ranks = {}
    for node in nx.topological_sort(dag):
        preds = list(dag.predecessors(node))
        ranks[node] = max((ranks[p] for p in preds), default=-1) + 1
    return ranks


def _barycenter_order(layers: dict, dag: nx.DiGraph) -> dict:
    """
    One downward + one upward barycenter pass to reduce edge crossings.
    Returns ordered layers (rank -> list of nodes).
    """
    max_rank = max(layers)
    ordered = {0: sorted(layers[0])}
    pos_idx = {n: i for i, n in enumerate(ordered[0])}

    for r in range(1, max_rank + 1):
        layer = sorted(layers[r],
                       key=lambda v: (sum(pos_idx[p] for p in dag.predecessors(v)
                                         if p in pos_idx)
                                      / max(sum(1 for p in dag.predecessors(v)
                                               if p in pos_idx), 1)))
        ordered[r] = layer
        for i, n in enumerate(layer):
            pos_idx[n] = i

    for r in range(max_rank - 1, -1, -1):
        layer = sorted(ordered[r],
                       key=lambda v: (sum(pos_idx[s] for s in dag.successors(v)
                                         if s in pos_idx)
                                      / max(sum(1 for s in dag.successors(v)
                                               if s in pos_idx), 1)))
        ordered[r] = layer
        for i, n in enumerate(layer):
            pos_idx[n] = i

    return ordered


def _lattice_layout(dag: nx.DiGraph) -> dict:
    """
    Compute (x, y) positions for a lattice-style top-to-bottom layout.

    - Each node occupies one unit of horizontal space, centred within its rank.
    - Vertical spacing is uniform (one unit per rank step).
    - Edge crossings are reduced by a barycenter pass.
    """
    if not dag.nodes:
        return {}

    ranks = _longest_path_ranks(dag)
    max_rank = max(ranks.values(), default=0)

    layers = defaultdict(list)
    for node, r in ranks.items():
        layers[r].append(node)

    ordered = _barycenter_order(layers, dag)

    pos = {}
    for r, layer in ordered.items():
        n = len(layer)
        for col, node in enumerate(layer):
            pos[node] = (col - (n - 1) / 2.0, -float(r))
    return pos


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_graph(prefs: list, output_path: str, enricher=None, traces: list = None):
    """
    Render the preference graph as a clean lattice-style PNG.

    - Nodes arranged in horizontal bands by preference rank (top = most preferred).
    - Only direct edges are shown (transitive reduction / Hasse diagram).
    - Blue solid arrows : acyclic preference edges (the clean ordering).
    - Red dashed arrows : cycle-breaking (MFAS) edges that violate the ordering.
    """
    expanded = expand_symmetric(prefs)
    graph = build_graph(expanded)
    sccs = tarjan_sccs(graph)
    cycle_edges = set(get_cycle_edges(graph, sccs))
    mfas_edges = set(map(tuple, min_feedback_arc_set(graph)))
    bad_edges = mfas_edges | cycle_edges

    G = nx.DiGraph()
    for p in expanded:
        if p["pref"] == 1:
            G.add_edge(p["i"], p["j"])

    # DAG backbone: strip cycle edges, then take transitive reduction so only
    # direct (Hasse) edges remain — removes redundant A->C when A->B->C exists.
    dag = nx.DiGraph(e for e in G.edges() if e not in bad_edges)
    dag.add_nodes_from(G.nodes())
    hasse = nx.transitive_reduction(dag)

    pos = _lattice_layout(dag)

    # Centre the axes around the graph
    if pos:
        xs = [x for x, _ in pos.values()]
        ys = [y for _, y in pos.values()]
        x_pad, y_pad = max(1.0, (max(xs) - min(xs)) * 0.15), 0.6
        ax_xlim = (min(xs) - x_pad, max(xs) + x_pad)
        ax_ylim = (min(ys) - y_pad, max(ys) + y_pad)
    else:
        ax_xlim = ax_ylim = (-1, 1)

    labels = {}
    for n in G.nodes():
        if traces and enricher:
            labels[n] = f"T{n}\n{enricher.summarize_trace(traces[n])}"
        else:
            labels[n] = f"T{n}"

    hasse_edge_list = list(hasse.edges())
    bad_edge_list   = [(u, v) for u, v in G.edges() if (u, v) in bad_edges]

    n_nodes   = len(G.nodes())
    node_size = max(500, 1400 - n_nodes * 30)
    font_size = max(6, 9 - n_nodes // 10)

    # Figure size scales with graph dimensions
    n_ranks = int(-min(y for _, y in pos.values())) + 1 if pos else 2
    max_width = max(len([v for v, (_, y) in pos.items() if y == yr])
                    for yr in set(y for _, y in pos.values())) if pos else 1
    fig_w = max(10, max_width * 2.2)
    fig_h = max(7,  n_ranks  * 1.8)

    _, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(*ax_xlim)
    ax.set_ylim(*ax_ylim)

    nx.draw_networkx_nodes(hasse, pos, ax=ax,
                           node_size=node_size, node_color="#ddeeff",
                           edgecolors="#334466", linewidths=1.4)
    nx.draw_networkx_labels(hasse, pos, labels=labels, ax=ax, font_size=font_size)
    nx.draw_networkx_edges(hasse, pos, edgelist=hasse_edge_list, ax=ax,
                           edge_color="#2255bb", arrows=True, arrowsize=16,
                           width=1.6, connectionstyle="arc3,rad=0.05",
                           min_source_margin=20, min_target_margin=20)
    if bad_edge_list:
        nx.draw_networkx_edges(G, pos, edgelist=bad_edge_list, ax=ax,
                               edge_color="#cc2222", style="dashed",
                               arrows=True, arrowsize=14, width=1.4,
                               connectionstyle="arc3,rad=0.3",
                               min_source_margin=20, min_target_margin=20)

    n_cycles = len(sccs)
    title = "Preference Ordering — CONSISTENT" if n_cycles == 0 else \
            f"Preference Ordering — {n_cycles} cycle(s), {len(mfas_edges)} conflicting edge(s) shown in red"
    ax.set_title(title, fontsize=13, pad=16)

    ax.annotate("most preferred", xy=(0.5, 1.0), xycoords="axes fraction",
                fontsize=8, color="#666666", ha="center", va="bottom")
    ax.annotate("least preferred", xy=(0.5, 0.0), xycoords="axes fraction",
                fontsize=8, color="#666666", ha="center", va="top")

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[visualizer] Graph saved to {output_path}")
