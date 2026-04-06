#!/usr/bin/env python3
"""
Consistency Checker

Loads a preference JSON file, builds a directed preference graph,
detects cycles via SCCs, finds the Minimum Feedback Arc Set (MFAS) --
the smallest set of preference edges to remove/re-query to make the
graph cycle-free -- then re-prompts the LLM for those specific pairs
with full context (all traces + all current preferences).

Re-query loop:
  1. Find cycles, compute MFAS
  2. Re-ask LLM for only the MFAS pairs, with all traces + all current
     preferences as prior context
  3. Update preferences, re-check
  4. Repeat up to --max-rounds times
  5. If still inconsistent, drop the remaining MFAS edges entirely
"""

import argparse
import json
import sys
import time
import os
from itertools import combinations

import anthropic
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api_keys import load_api_key


# ---------------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------------

def build_graph(prefs: list) -> dict:
    """
    Build adjacency list. Edge i->j means i preferred over j.
    pref=0 entries are skipped (no edge, no cycle risk).
    """
    graph = {}
    for p in prefs:
        i, j, pref = p["i"], p["j"], p["pref"]
        graph.setdefault(i, set())
        graph.setdefault(j, set())
        if pref == 1:
            graph[i].add(j)
        elif pref == -1:
            graph[j].add(i)
    return graph


def tarjan_sccs(graph: dict) -> list:
    """Returns SCCs with >1 node (actual cycles)."""
    index_counter = [0]
    stack = []
    lowlink = {}
    index = {}
    on_stack = {}
    sccs = []

    def strongconnect(v):
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True
        for w in graph.get(v, []):
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack.get(w, False):
                lowlink[v] = min(lowlink[v], index[w])
        if lowlink[v] == index[v]:
            scc = set()
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc.add(w)
                if w == v:
                    break
            if len(scc) > 1:
                sccs.append(scc)

    for v in graph:
        if v not in index:
            strongconnect(v)
    return sccs


def has_cycle(graph: dict) -> bool:
    visited = set()
    rec_stack = set()

    def dfs(v):
        visited.add(v)
        rec_stack.add(v)
        for w in graph.get(v, []):
            if w not in visited:
                if dfs(w):
                    return True
            elif w in rec_stack:
                return True
        rec_stack.discard(v)
        return False

    for v in graph:
        if v not in visited:
            if dfs(v):
                return True
    return False


def get_cycle_edges(graph: dict, sccs: list) -> list:
    """Return all edges (i,j) that lie within a cyclic SCC."""
    cycle_edges = []
    for scc in sccs:
        for v in scc:
            for w in graph.get(v, []):
                if w in scc:
                    cycle_edges.append((v, w))
    return cycle_edges


def remove_edges(graph: dict, edges: list) -> dict:
    edge_set = set(edges)
    return {v: {w for w in nbrs if (v, w) not in edge_set}
            for v, nbrs in graph.items()}


# ---------------------------------------------------------------------------
# Minimum Feedback Arc Set (MFAS)
# ---------------------------------------------------------------------------

def min_feedback_arc_set(graph: dict) -> list:
    """
    Find minimum set of edges to remove to make graph acyclic.
    Uses iterative deepening over cycle edges only (not all edges).
    Feasible for small graphs.
    """
    if not has_cycle(graph):
        return []

    sccs = tarjan_sccs(graph)
    candidates = get_cycle_edges(graph, sccs)

    for k in range(1, len(candidates) + 1):
        for subset in combinations(candidates, k):
            reduced = remove_edges(graph, list(subset))
            if not has_cycle(reduced):
                return list(subset)

    return candidates


# ---------------------------------------------------------------------------
# Prompt helpers (mirrors preference_elicitor.py with full context)
# ---------------------------------------------------------------------------

def _format_traces_block(traces: list, pairs: list, enricher=None) -> str:
    """Only include traces referenced in the pairs being re-queried + all prior."""
    referenced = sorted({idx for pair in pairs for idx in pair})
    lines = ["TRACES:"]
    for idx in referenced:
        t = traces[idx]
        if enricher is not None:
            summary = enricher.summarize_trace(t)
            lines.append(f"  T{idx}: {summary}")
        else:
            lines.append(f"  T{idx}: {t}")
    return "\n".join(lines)


def _format_prior_block(prefs: list) -> str:
    if not prefs:
        return ""
    lines = ["ALL CURRENT PREFERENCES (for consistency context):"]
    for p in prefs:
        i, j, pref = p["i"], p["j"], p["pref"]
        if pref == 1:
            lines.append(f"  T{i} > T{j}")
        elif pref == -1:
            lines.append(f"  T{j} > T{i}")
        else:
            lines.append(f"  T{i} = T{j}")
    return "\n".join(lines)


def _format_pairs_block(pairs: list) -> str:
    lines = ["PAIRS TO RE-EVALUATE (these created cycles — reassign carefully):"]
    for i, j in pairs:
        lines.append(f"  (T{i}, T{j})")
    return "\n".join(lines)


def _format_output_spec() -> str:
    return (
        "INSTRUCTIONS:\n"
        "The pairs listed above created preference cycles. Re-evaluate each one.\n"
        "For each pair (Ti, Tj) assign:\n"
        "  1   if Ti is preferred over Tj\n"
        " -1   if Tj is preferred over Ti\n"
        "  0   if they are equal / indifferent\n\n"
        "Reply with ONLY a JSON array with exactly one entry per listed pair.\n"
        "Do NOT include the reverse pair. Keys: \"i\", \"j\", \"pref\". Example:\n"
        '[{"i": 2, "j": 5, "pref": 0}]\n\n'
        "Do not include any explanation, only the JSON array."
    )


def build_requery_prompt(system_prompt: str, traces: list, pairs: list,
                         all_prefs: list, enricher=None) -> str:
    # All current preferences as full context
    prior_block = _format_prior_block(all_prefs)
    parts = [
        system_prompt.strip(),
        "",
        _format_traces_block(traces, pairs, enricher=enricher),
        "",
    ]
    if prior_block:
        parts += [prior_block, ""]
    parts += [
        _format_pairs_block(pairs),
        "",
        _format_output_spec(),
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_llm(client, model: str, prompt: str, retries: int = 4) -> str:
    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=1024,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"[checker] LLM error ({e}), retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)


def _parse_response(raw: str) -> dict:
    """Returns dict (i,j) -> pref for the queried pairs."""
    text = raw
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        entries = json.loads(text.strip())
    except json.JSONDecodeError as e:
        print(f"[checker] JSON parse error: {e}", file=sys.stderr)
        return {}
    return {(e["i"], e["j"]): e["pref"] for e in entries
            if "i" in e and "j" in e and "pref" in e}


# ---------------------------------------------------------------------------
# Preference update helpers
# ---------------------------------------------------------------------------

def expand_symmetric(prefs: list) -> list:
    """Add reverse entries so graph construction sees both directions."""
    expanded = list(prefs)
    seen = {(p["i"], p["j"]) for p in prefs}
    for p in prefs:
        rev = (p["j"], p["i"])
        if rev not in seen:
            expanded.append({"i": p["j"], "j": p["i"], "pref": -p["pref"]})
            seen.add(rev)
    return expanded


def update_prefs(prefs: list, updates: dict) -> list:
    """Apply LLM re-query results to the preference list."""
    updated = []
    update_set = set(updates.keys())
    reverse_set = {(j, i) for i, j in update_set}

    for p in prefs:
        key = (p["i"], p["j"])
        if key in update_set:
            new_pref = updates[key]
            if new_pref not in (1, -1, 0):
                new_pref = 0
            updated.append({**p, "pref": new_pref})
        elif key in reverse_set:
            # Update the reverse entry too
            fwd = (p["j"], p["i"])
            new_pref = -updates[fwd]
            updated.append({**p, "pref": new_pref})
        else:
            updated.append(p)
    return updated


# ---------------------------------------------------------------------------
# Main consistency check + re-query loop
# ---------------------------------------------------------------------------

def run(prefs: list, traces: list, system_prompt: str, model: str,
        max_rounds: int, enricher=None, verbose: bool = False):
    """
    Returns cleaned preference list (consistent, or best effort after max_rounds).
    """
    client = anthropic.Anthropic(api_key=load_api_key(model))
    current_prefs = expand_symmetric(prefs)

    for round_idx in range(max_rounds + 1):
        graph = build_graph(current_prefs)
        sccs = tarjan_sccs(graph)

        if not sccs:
            if verbose:
                print(f"[checker] Consistent after round {round_idx}.")
            break

        mfas_edges = min_feedback_arc_set(graph)

        if verbose:
            print(f"[checker] Round {round_idx}: {len(sccs)} cycle(s), "
                  f"MFAS = {len(mfas_edges)} edge(s): {mfas_edges}")

        if round_idx == max_rounds:
            if verbose:
                print(f"[checker] Max rounds reached — dropping {len(mfas_edges)} edge(s).")
            # Drop the remaining inconsistent edges
            drop = set(mfas_edges) | {(j, i) for i, j in mfas_edges}
            current_prefs = [p for p in current_prefs
                             if (p["i"], p["j"]) not in drop]
            break

        # Re-query LLM for just the MFAS pairs (canonical i<j direction)
        requery_pairs = [(min(i, j), max(i, j)) for i, j in mfas_edges]
        requery_pairs = list(dict.fromkeys(requery_pairs))  # deduplicate

        if verbose:
            print(f"[checker] Re-querying LLM for {len(requery_pairs)} pair(s)...")

        prompt = build_requery_prompt(
            system_prompt, traces, requery_pairs,
            [p for p in current_prefs if p["i"] < p["j"]],
            enricher=enricher,
        )
        raw = _call_llm(client, model, prompt)
        updates = _parse_response(raw)

        if verbose:
            for (i, j), p in updates.items():
                label = f"T{i}>T{j}" if p == 1 else (f"T{j}>T{i}" if p == -1 else f"T{i}=T{j}")
                print(f"  LLM reassigned ({i},{j}) -> {label}")

        current_prefs = update_prefs(current_prefs, updates)

    # Return canonical (i<j) only
    return [p for p in current_prefs if p["i"] < p["j"]]


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_graph(prefs: list, output_path: str, enricher=None, traces: list = None):
    """
    Render the preference graph as a PNG.
    - Nodes labeled T0..TN with optional one-line trace summary
    - Green edges: strict preference
    - Red edges: edges in cycles (MFAS)
    """
    expanded = expand_symmetric(prefs)
    graph = build_graph(expanded)
    sccs = tarjan_sccs(graph)
    cycle_edges = set(get_cycle_edges(graph, sccs))
    mfas = set(map(tuple, min_feedback_arc_set(graph)))

    G = nx.DiGraph()
    for p in expanded:
        if p["pref"] == 1:
            G.add_edge(p["i"], p["j"])

    pos = nx.circular_layout(G)

    # Node labels
    labels = {}
    for n in G.nodes():
        if traces and enricher:
            summary = enricher.summarize_trace(traces[n])
            labels[n] = f"T{n}\n{summary}"
        else:
            labels[n] = f"T{n}"

    edge_colors = ["red" if (u, v) in mfas or (u, v) in cycle_edges else "#4488cc"
                   for u, v in G.edges()]

    fig, ax = plt.subplots(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=800, node_color="#eeeeee", edgecolors="black")
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=7)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, arrows=True,
                           arrowsize=15, width=1.5, connectionstyle="arc3,rad=0.1")

    cycle_count = len(sccs)
    title = "Preference Graph — CONSISTENT" if cycle_count == 0 else \
            f"Preference Graph — {cycle_count} cycle(s), {len(mfas)} MFAS edge(s) in red"
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[checker] Graph saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Check preference consistency and resolve cycles via LLM re-querying."
    )
    parser.add_argument("prefs", help="Path to preferences JSON file.")
    parser.add_argument("traces", help="Path to traces .txt file.")
    parser.add_argument("--prompt", required=True,
                        help="System prompt string or path to .txt file.")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--max-rounds", type=int, default=3,
                        help="Max LLM re-query rounds before dropping edges (default: 3).")
    parser.add_argument("--enrich", choices=["kuhn_poker"], default=None)
    parser.add_argument("--output", help="Save cleaned preferences to this JSON file.")
    parser.add_argument("--plot", help="Save preference graph as PNG to this path.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.prefs) as f:
        prefs = json.load(f)

    with open(args.traces) as f:
        traces = [l.rstrip("\n") for l in f if l.strip()]

    if os.path.isfile(args.prompt):
        with open(args.prompt) as f:
            system_prompt = f.read()
    else:
        system_prompt = args.prompt

    enricher = None
    if args.enrich == "kuhn_poker":
        from semantics.kuhn_poker import KuhnPokerEnricher
        enricher = KuhnPokerEnricher()

    clean = run(prefs, traces, system_prompt, args.model,
                max_rounds=args.max_rounds, enricher=enricher,
                verbose=args.verbose)

    if args.plot:
        plot_graph(prefs, args.plot, enricher=enricher, traces=traces)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(clean, f, indent=2)
        print(f"[checker] Saved {len(clean)} cleaned preferences to {args.output}")
    else:
        print(json.dumps(clean, indent=2))


if __name__ == "__main__":
    main()
