#!/usr/bin/env python3
"""
Preference Elicitor

Given a text file of traces (one per line), a system prompt, an LLM model,
and a chunk size K, queries the LLM for pairwise preferences over all N^2-N
ordered pairs.

Chunked sequential strategy:
  - Chunk 1 : [all traces] + [chunk_1 pairs]
  - Chunk k : [all traces] + [chunk_k pairs] + [all prior preferences]

Output: JSON list of {"i": int, "j": int, "pref": 1|-1|0}
  pref =  1  trace i preferred over trace j
  pref = -1  trace j preferred over trace i
  pref =  0  equal / indifferent
"""

import argparse
import itertools
import json
import os
import sys
import time

import anthropic

from api_keys import load_api_key


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_traces(path: str) -> list:
    """Load traces from a text file, one trace per line. Skips blank lines."""
    with open(path) as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def save_preferences(prefs: list, path: str, traces: list = None) -> None:
    # Only save canonical pairs (i < j) — reverse is always inferrable
    canonical = [p for p in prefs if p["i"] < p["j"]]
    if traces:
        for p in canonical:
            p["trace_i"] = traces[p["i"]]
            p["trace_j"] = traces[p["j"]]
    with open(path, "w") as f:
        json.dump(canonical, f, indent=2)


# ---------------------------------------------------------------------------
# Pair generation and chunking
# ---------------------------------------------------------------------------

def compute_pairs(n: int) -> list:
    """
    Returns all unordered pairs (i, j) where i < j, for indices 0..n-1.
    Total: n*(n-1)/2 pairs. The reverse (j, i) is inferred from symmetry.
    """
    return list(itertools.combinations(range(n), 2))


def chunk_pairs(pairs: list, K: int) -> list:
    """Split pairs into chunks of at most K."""
    return [pairs[start:start + K] for start in range(0, len(pairs), K)]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _format_traces_block(traces: dict, enricher=None) -> str:
    lines = ["TRACES:"]
    for idx, t in traces.items():
        if enricher is not None:
            summary = enricher.summarize_trace(t)
            lines.append(f"  T{idx}: {summary}")
        else:
            lines.append(f"  T{idx}: {t}")
    return "\n".join(lines)


def _format_pairs_block(chunk: list) -> str:
    lines = ["PAIRS TO EVALUATE (assign a preference for each):"]
    for i, j in chunk:
        lines.append(f"  (T{i}, T{j})")
    return "\n".join(lines)


def _format_prior_block(prior: list) -> str:
    if not prior:
        return ""
    lines = ["PREVIOUSLY ASSIGNED PREFERENCES (for consistency context):"]
    for entry in prior:
        i, j, p = entry["i"], entry["j"], entry["pref"]
        if p == 1:
            label = f"T{i} > T{j}"
        elif p == -1:
            label = f"T{j} > T{i}"
        else:
            label = f"T{i} = T{j}"
        lines.append(f"  {label}")
    return "\n".join(lines)


def _format_output_spec() -> str:
    return (
        "INSTRUCTIONS:\n"
        "For each pair (Ti, Tj) listed above, assign a preference:\n"
        "  1   if Ti is preferred over Tj\n"
        " -1   if Tj is preferred over Ti\n"
        "  0   if they are equal / indifferent\n\n"
        "Reply with ONLY a JSON array with exactly one entry per pair listed above.\n"
        "Do NOT include the reverse pair. Each element must have keys "
        '"i", "j", "pref". Example for pairs (T0,T1) and (T0,T2):\n'
        '[{"i": 0, "j": 1, "pref": 1}, {"i": 0, "j": 2, "pref": -1}]\n\n'
        "Do not include any explanation, only the JSON array."
    )


def build_prompt(system_prompt: str, traces: list, chunk: list, prior: list, enricher=None) -> str:
    # Include traces referenced in this chunk AND in prior preferences (for context)
    referenced = {idx for pair in chunk for idx in pair}
    referenced |= {p["i"] for p in prior} | {p["j"] for p in prior}
    relevant_traces = {idx: traces[idx] for idx in sorted(referenced)}
    parts = [
        system_prompt.strip(),
        "",
        _format_traces_block(relevant_traces, enricher=enricher),
        "",
    ]
    prior_block = _format_prior_block(prior)
    if prior_block:
        parts += [prior_block, ""]
    parts += [
        _format_pairs_block(chunk),
        "",
        _format_output_spec(),
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_llm(client: anthropic.Anthropic, model: str, prompt: str, retries: int = 4) -> str:
    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=4096,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"[elicitor] LLM error ({e}), retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)


def _parse_response(raw: str, chunk: list) -> list:
    """
    Parse LLM JSON response into a list of preference dicts.
    Falls back to pref=0 for any pair the LLM failed to return.
    """
    # Strip markdown code fences if present
    text = raw
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    try:
        entries = json.loads(text.strip())
    except json.JSONDecodeError as e:
        print(f"[elicitor] JSON parse error: {e}\nRaw response:\n{raw}", file=sys.stderr)
        entries = []

    # Index by (i, j) for lookup
    returned = {(e["i"], e["j"]): e["pref"] for e in entries if "i" in e and "j" in e and "pref" in e}

    results = []
    for i, j in chunk:
        pref = returned.get((i, j), 0)
        if pref not in (1, -1, 0):
            pref = 0
        results.append({"i": i, "j": j, "pref": pref})
        results.append({"i": j, "j": i, "pref": -pref})  # symmetric reverse
    return results


# ---------------------------------------------------------------------------
# Main elicitation loop
# ---------------------------------------------------------------------------

def elicit_preferences(
    system_prompt: str,
    traces: list,
    model: str,
    K: int,
    verbose: bool = False,
    enricher=None,
    chunk_delay: float = 10.0,
) -> list:
    """
    Query the LLM for preferences over all N^2-N pairs of traces.

    Args:
        system_prompt: Task context / role description for the LLM.
        traces:        List of trace strings.
        model:         Anthropic model ID.
        K:             Chunk size (number of pairs per LLM call).
        verbose:       Print progress to stderr.

    Returns:
        List of {"i": int, "j": int, "pref": 1|-1|0} dicts covering all pairs.
    """
    client = anthropic.Anthropic(api_key=load_api_key(model))
    n = len(traces)
    all_pairs = compute_pairs(n)
    chunks = chunk_pairs(all_pairs, K)

    total_pairs = len(all_pairs)
    total_chunks = len(chunks)

    if verbose:
        print(
            f"[elicitor] {n} traces → {total_pairs} unordered pairs ({2*total_pairs} total with symmetry) → {total_chunks} chunks of ≤{K}",
            file=sys.stderr,
        )

    all_prefs = []

    for chunk_idx, chunk in enumerate(chunks):
        if verbose:
            print(
                f"[elicitor] chunk {chunk_idx + 1}/{total_chunks} ({len(chunk)} pairs)...",
                file=sys.stderr,
            )

        if chunk_idx > 0 and chunk_delay > 0:
            time.sleep(chunk_delay)

        prompt = build_prompt(system_prompt, traces, chunk, all_prefs, enricher=enricher)
        raw = _call_llm(client, model, prompt)
        chunk_prefs = _parse_response(raw, chunk)
        all_prefs.extend(chunk_prefs)

        if verbose:
            assigned = sum(1 for p in chunk_prefs if p["pref"] != 0)
            print(
                f"[elicitor]   → {assigned}/{len(chunk)} non-equal preferences",
                file=sys.stderr,
            )

    return all_prefs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Elicit LLM pairwise preferences over a set of traces."
    )
    parser.add_argument("traces", help="Path to traces .txt file (one trace per line).")
    parser.add_argument("--prompt", required=True, help="System prompt string or path to a .txt file containing it.")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001", help="Anthropic model ID.")
    parser.add_argument("--K", type=int, default=50, help="Pairs per LLM chunk (default: 50).")
    parser.add_argument("--output", default="preferences.json", help="Output JSON file path.")
    parser.add_argument(
        "--enrich",
        choices=["kuhn_poker"],
        default=None,
        help="Enrich traces with human-readable AP descriptions before sending to the LLM.",
    )
    parser.add_argument("--chunk-delay", type=float, default=10.0, help="Seconds to wait between chunks to avoid rate limits (default: 10).")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Load prompt from file if it looks like a path
    if os.path.isfile(args.prompt):
        with open(args.prompt) as f:
            system_prompt = f.read()
    else:
        system_prompt = args.prompt

    # Load enricher if requested
    enricher = None
    if args.enrich == "kuhn_poker":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from semantics.kuhn_poker import KuhnPokerEnricher
        enricher = KuhnPokerEnricher()
        if args.verbose:
            print("[elicitor] Using KuhnPokerEnricher for trace descriptions", file=sys.stderr)

    traces = load_traces(args.traces)
    if len(traces) < 2:
        print("Error: need at least 2 traces.", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"[elicitor] Loaded {len(traces)} traces from {args.traces}", file=sys.stderr)

    prefs = elicit_preferences(
        system_prompt=system_prompt,
        traces=traces,
        model=args.model,
        K=args.K,
        verbose=args.verbose,
        enricher=enricher,
        chunk_delay=args.chunk_delay,
    )

    save_preferences(prefs, args.output, traces=traces)
    print(f"[elicitor] Saved {len(prefs)} preferences to {args.output}")


if __name__ == "__main__":
    main()
