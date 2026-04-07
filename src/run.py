#!/usr/bin/env python3
"""
run.py — Full preference-elicitation pipeline (Steps 1–4).

Step 1  Generate traces from an automaton (dot_trace_generator)
Step 2  Elicit pairwise preferences from an LLM (preference_elicitor)
Step 3  Check consistency and resolve cycles (consistency_checker)
Step 4  Build preference power set over all trace prefixes (preference_power_set)

Each step can be skipped if its output file already exists by passing
--skip-traces / --skip-prefs respectively.

Example (mirrors the README full-pipeline block):

  python src/run.py \\
    src/data/Kuhn_Poker/kuhn_poker.dot \\
    --fmt dot \\
    --aps a0,a1,a2,bs,c1hi,c1lo,c2hi,c2lo,cur_bet,deal,m1b0,m1b1,m1b2,m2b0,m2b1,m2b2,p1,p1b,p2b,p2c,p2r \\
    --num 10 --length 8 \\
    --traces-out src/data/Kuhn_Poker/output/kuhn_traces.txt \\
    --prompt "You are evaluating a Kuhn Poker player. Prefer traces where the player bluffs with low cards and value-bets with high cards. Penalize passive play or over-bluffing." \\
    --model claude-haiku-4-5-20251001 \\
    --K 50 --enrich kuhn_poker --chunk-delay 10 \\
    --prefs-out src/data/Kuhn_Poker/output/kuhn_prefs.json \\
    --max-rounds 3 \\
    --clean-out src/data/Kuhn_Poker/output/kuhn_prefs_clean.json \\
    --plot src/data/Kuhn_Poker/output/pref_graph.png \\
    --prefix-prefs-out src/data/Kuhn_Poker/output/prefix_preferences.json \\
    --verbose
"""


import argparse
import json
import os
import sys

# Ensure src/ is on the path so pipeline sub-modules can find api_keys / semantics
_SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC)

from pipeline import dot_trace_generator as _dtg
from pipeline import preference_elicitor as _pe
from pipeline import consistency_checker as _cc
from pipeline import preference_power_set as _pps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_prompt(prompt_arg: str) -> str:
    if os.path.isfile(prompt_arg):
        with open(prompt_arg) as f:
            return f.read()
    return prompt_arg


def _load_enricher(enrich: str):
    if enrich == "kuhn_poker":
        from semantics.kuhn_poker import KuhnPokerEnricher
        return KuhnPokerEnricher()
    return None


# ---------------------------------------------------------------------------
# Step 1 — Generate traces
# ---------------------------------------------------------------------------

def step1_generate_traces(args) -> list:
    """Returns list of trace strings."""
    if args.skip_traces and os.path.isfile(args.traces_out):
        print(f"[run] Step 1 skipped — loading existing traces from {args.traces_out}")
        return _pe.load_traces(args.traces_out)

    print(f"[run] Step 1 — generating {args.num} traces from {args.dot} ...")
    if args.fmt == "json":
        machine = _dtg.load_json(args.dot)
    else:
        machine = _dtg.load_dot(args.dot)

    ap_order = [ap.strip() for ap in args.aps.split(",") if ap.strip()]
    traces = []
    for _ in range(args.num):
        raw = _dtg.generate_trace(machine, length=args.length, cycle=args.cycle)
        traces.append(_dtg.trace_to_spot(raw, ap_order))

    if args.traces_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.traces_out)), exist_ok=True)
        with open(args.traces_out, "w") as f:
            for t in traces:
                f.write(t + "\n")
        print(f"[run] Step 1 done — wrote {len(traces)} traces to {args.traces_out}")
    else:
        print(f"[run] Step 1 done — {len(traces)} traces (no output file specified)")

    return traces


# ---------------------------------------------------------------------------
# Step 2 — Elicit preferences
# ---------------------------------------------------------------------------

def step2_elicit_preferences(args, traces: list) -> list:
    """Returns list of preference dicts."""
    if args.skip_prefs and os.path.isfile(args.prefs_out):
        print(f"[run] Step 2 skipped — loading existing preferences from {args.prefs_out}")
        with open(args.prefs_out) as f:
            return json.load(f)

    print(f"[run] Step 2 — eliciting preferences with {args.model} ...")
    system_prompt = _load_prompt(args.prompt)
    enricher = _load_enricher(args.enrich)

    prefs = _pe.elicit_preferences(
        system_prompt=system_prompt,
        traces=traces,
        model=args.model,
        K=args.K,
        verbose=args.verbose,
        enricher=enricher,
        chunk_delay=args.chunk_delay,
    )

    if args.prefs_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.prefs_out)), exist_ok=True)
        _pe.save_preferences(prefs, args.prefs_out, traces=traces)
        print(f"[run] Step 2 done — saved preferences to {args.prefs_out}")
    else:
        print(f"[run] Step 2 done — {len(prefs)} preferences (no output file specified)")

    return prefs


# ---------------------------------------------------------------------------
# Step 3 — Consistency check
# ---------------------------------------------------------------------------

def step3_check_consistency(args, prefs: list, traces: list) -> list:
    """Returns cleaned preference list."""
    print(f"[run] Step 3 — checking consistency (max {args.max_rounds} rounds) ...")
    system_prompt = _load_prompt(args.prompt)
    enricher = _load_enricher(args.enrich)

    clean = _cc.run(
        prefs=prefs,
        traces=traces,
        system_prompt=system_prompt,
        model=args.model,
        max_rounds=args.max_rounds,
        enricher=enricher,
        verbose=args.verbose,
    )

    if args.plot:
        os.makedirs(os.path.dirname(os.path.abspath(args.plot)), exist_ok=True)
        _cc.plot_graph(prefs, args.plot, enricher=enricher, traces=traces)

    if args.clean_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.clean_out)), exist_ok=True)
        with open(args.clean_out, "w") as f:
            json.dump(clean, f, indent=2)
        print(f"[run] Step 3 done — saved {len(clean)} cleaned preferences to {args.clean_out}")
    else:
        print(f"[run] Step 3 done — {len(clean)} cleaned preferences (no output file specified)")

    return clean


# ---------------------------------------------------------------------------
# Step 4 — Preference power set over prefixes
# ---------------------------------------------------------------------------

def step4_prefix_preferences(args, traces: list, clean_prefs: list) -> list:
    """Returns pairwise prefix preference list."""
    import time
    print(f"[run] Step 4 — building preference power set over trace prefixes ...")

    # Traces from step 1 are semicolon-delimited strings; split into step lists
    parsed_traces = [t.strip().split(";") for t in traces]

    t0 = time.time()
    G = _pps.build_graph(parsed_traces)
    print(f"[run] Step 4   graph built          ({time.time()-t0:.2f}s)")

    t1 = time.time()
    trace_ranks = _pps.build_trace_ranks(clean_prefs, len(parsed_traces))
    print(f"[run] Step 4   trace ranks computed ({time.time()-t1:.2f}s)")

    t2 = time.time()
    prefix_prefs = _pps.compute_prefix_preferences(G, parsed_traces, trace_ranks)
    print(f"[run] Step 4   pairs computed       ({time.time()-t2:.2f}s)  "
          f"{len(prefix_prefs['prefixes'])} prefixes, {len(prefix_prefs['pairs'])} pairs")

    if args.prefix_prefs_out:
        t3 = time.time()
        os.makedirs(os.path.dirname(os.path.abspath(args.prefix_prefs_out)), exist_ok=True)
        with open(args.prefix_prefs_out, "w") as f:
            json.dump(prefix_prefs, f)
        print(f"[run] Step 4   written to disk      ({time.time()-t3:.2f}s)  -> {args.prefix_prefs_out}")
        print(f"[run] Step 4 done (total {time.time()-t0:.2f}s)")
    else:
        print(f"[run] Step 4 done — {len(prefix_prefs['pairs'])} pairs (no output file specified)")

    return prefix_prefs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the full preference-elicitation pipeline (Steps 1–3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Step 1: trace generation ────────────────────────────────────────────
    g1 = p.add_argument_group("Step 1 — Trace generation")
    g1.add_argument("dot", help="Path to automaton file (JSON or DOT).")
    g1.add_argument("--fmt", choices=["json", "dot"], default="dot",
                    help="Automaton file format.")
    g1.add_argument("--aps", required=True,
                    help="Comma-separated APs in fixed order, e.g. a0,a1,a2,...")
    g1.add_argument("--num", "-n", type=int, default=10,
                    help="Number of traces to generate.")
    g1.add_argument("--length", "-l", type=int, default=8,
                    help="Steps per trace.")
    g1.add_argument("--cycle", action="store_true",
                    help="Append cycle{1} to each trace.")
    g1.add_argument("--traces-out", default="data/Kuhn_Poker/output/kuhn_traces.txt",
                    help="Output path for generated traces (.txt).")
    g1.add_argument("--skip-traces", action="store_true",
                    help="Skip Step 1 if --traces-out already exists.")

    # ── Step 2: preference elicitation ──────────────────────────────────────
    g2 = p.add_argument_group("Step 2 — Preference elicitation")
    g2.add_argument("--prompt", required=True,
                    help="System prompt string or path to a .txt file containing it.")
    g2.add_argument("--model", default="claude-haiku-4-5-20251001",
                    help="Anthropic model ID.")
    g2.add_argument("--K", type=int, default=50,
                    help="Pairs per LLM chunk.")
    g2.add_argument("--enrich", choices=["kuhn_poker"], default=None,
                    help="Enrich traces with human-readable descriptions before sending to LLM.")
    g2.add_argument("--chunk-delay", type=float, default=10.0,
                    help="Seconds between LLM chunks to avoid rate limits.")
    g2.add_argument("--prefs-out", default="data/Kuhn_Poker/output/kuhn_prefs.json",
                    help="Output path for raw preferences (.json).")
    g2.add_argument("--skip-prefs", action="store_true",
                    help="Skip Step 2 if --prefs-out already exists.")

    # ── Step 3: consistency checking ────────────────────────────────────────
    g3 = p.add_argument_group("Step 3 — Consistency checking")
    g3.add_argument("--max-rounds", type=int, default=3,
                    help="Max LLM re-query rounds before dropping cycle edges.")
    g3.add_argument("--clean-out", default="data/Kuhn_Poker/output/kuhn_prefs_clean.json",
                    help="Output path for cleaned preferences (.json).")
    g3.add_argument("--plot", default=None,
                    help="Save preference graph as PNG to this path.")

    # ── Step 4: preference power set ────────────────────────────────────────
    g4 = p.add_argument_group("Step 4 — Preference power set")
    g4.add_argument("--prefix-prefs-out", default="data/Kuhn_Poker/output/prefix_preferences.json",
                    help="Output path for prefix pairwise preferences (.json).")
    g4.add_argument("--skip-prefix-prefs", action="store_true",
                    help="Skip Step 4 entirely.")

    # ── General ─────────────────────────────────────────────────────────────
    p.add_argument("--verbose", action="store_true", help="Print progress to stderr.")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    traces      = step1_generate_traces(args)
    prefs       = step2_elicit_preferences(args, traces)
    clean_prefs = step3_check_consistency(args, prefs, traces)
    if not args.skip_prefix_prefs:
        step4_prefix_preferences(args, traces, clean_prefs)


if __name__ == "__main__":
    main()
