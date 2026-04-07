#!/usr/bin/env python3
"""
Visualizer

Rendering utilities for pipeline artifacts.
"""

from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
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
# Shared rendering core
# ---------------------------------------------------------------------------

def _render_lattice(prefs: list, labels: dict, title: str, output_path: str):
    """
    Build and render a lattice-style preference ordering graph.

    prefs    : list of {i, j, pref} dicts (same format for traces and prefixes)
    labels   : node-id -> display string
    title    : figure title
    output_path : PNG save path
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

    # Strip cycle edges then take transitive reduction (Hasse diagram) to
    # remove redundant A->C edges implied by A->B->C.
    dag = nx.DiGraph(e for e in G.edges() if e not in bad_edges)
    dag.add_nodes_from(G.nodes())
    hasse = nx.transitive_reduction(dag)

    pos = _lattice_layout(dag)

    if pos:
        xs = [x for x, _ in pos.values()]
        ys = [y for _, y in pos.values()]
        x_pad = max(1.0, (max(xs) - min(xs)) * 0.15)
        ax_xlim = (min(xs) - x_pad, max(xs) + x_pad)
        ax_ylim = (min(ys) - 0.6,   max(ys) + 0.6)
    else:
        ax_xlim = ax_ylim = (-1, 1)

    hasse_edges = list(hasse.edges())
    bad_edge_list = [(u, v) for u, v in G.edges() if (u, v) in bad_edges]

    n_nodes   = len(G.nodes())
    node_size = max(400, 1400 - n_nodes * 12)
    font_size = max(5, 9 - n_nodes // 15)

    n_ranks   = int(-min(y for _, y in pos.values())) + 1 if pos else 2
    max_width = max(
        len([v for v, (_, y) in pos.items() if y == yr])
        for yr in set(y for _, y in pos.values())
    ) if pos else 1
    fig_w = max(10, max_width * 1.8)
    fig_h = max(7,  n_ranks  * 1.6)

    # Heatmap: normalise each node's y-position to [0, 1] where 1 = most
    # preferred (top, y=0) and 0 = least preferred (bottom, most negative y).
    colormap = cm.RdYlGn
    node_list = list(hasse.nodes())
    ys_all = [pos[n][1] for n in node_list]
    y_min, y_max = min(ys_all), max(ys_all)
    y_range = y_max - y_min if y_max != y_min else 1.0
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    node_colors = [colormap((pos[n][1] - y_min) / y_range) for n in node_list]

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(*ax_xlim)
    ax.set_ylim(*ax_ylim)

    nx.draw_networkx_nodes(hasse, pos, ax=ax, nodelist=node_list,
                           node_size=node_size, node_color=node_colors,
                           edgecolors="#333333", linewidths=1.2)
    nx.draw_networkx_labels(hasse, pos, labels=labels, ax=ax, font_size=font_size)
    nx.draw_networkx_edges(hasse, pos, edgelist=hasse_edges, ax=ax,
                           edge_color="#333333", arrows=True, arrowsize=14,
                           width=1.5, connectionstyle="arc3,rad=0.05",
                           min_source_margin=18, min_target_margin=18)
    if bad_edge_list:
        nx.draw_networkx_edges(G, pos, edgelist=bad_edge_list, ax=ax,
                               edge_color="#cc2222", style="dashed",
                               arrows=True, arrowsize=12, width=1.3,
                               connectionstyle="arc3,rad=0.3",
                               min_source_margin=18, min_target_margin=18)

    # Colorbar legend
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                        fraction=0.03, pad=0.02, shrink=0.6)
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels(["least preferred", "most preferred"])
    cbar.ax.tick_params(labelsize=8)

    n_cycles = len(sccs)
    full_title = title if n_cycles == 0 else \
        f"{title}\n{n_cycles} cycle(s) — {len(mfas_edges)} conflicting edge(s) shown in red"
    ax.set_title(full_title, fontsize=12, pad=14)

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[visualizer] Graph saved to {output_path}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_graph(prefs: list, output_path: str, enricher=None, traces: list = None):
    """
    Render the full-trace preference ordering as a lattice-style PNG.
    Nodes are labelled T0..TN with an optional one-line enricher summary.
    """
    labels = {}
    # Collect node ids from prefs
    node_ids = {p["i"] for p in prefs} | {p["j"] for p in prefs}
    for n in node_ids:
        if traces and enricher:
            labels[n] = f"T{n}\n{enricher.summarize_trace(traces[n])}"
        else:
            labels[n] = f"T{n}"

    _render_lattice(prefs, labels, "Trace Preference Ordering", output_path)


def plot_prefix_graph(prefix_prefs: dict, output_path: str, enricher=None):
    """
    Render the prefix preference ordering as a lattice-style PNG.

    prefix_prefs : dict with keys 'prefixes' (list of prefix strings)
                   and 'pairs' (list of {i, j, pref, ...} dicts).
    Nodes are labelled P0..PN; hovering over the saved PNG won't show the
    full formula, but the index lets you cross-reference prefix_prefs['prefixes'].
    """
    prefixes = prefix_prefs["prefixes"]
    pairs    = prefix_prefs["pairs"]

    labels = {}
    node_ids = {p["i"] for p in pairs} | {p["j"] for p in pairs}
    for n in node_ids:
        if enricher and n < len(prefixes):
            summary = enricher.summarize_trace(prefixes[n])
            labels[n] = f"P{n}\n{summary}"
        else:
            labels[n] = f"P{n}"

    _render_lattice(pairs, labels, "Prefix Preference Ordering", output_path)
