"""
Cache Builder  (Pipeline Step 2 — orchestrates 2a + 2b + 2c)

Runs classification, preference pre-computation, and prefix-closed
extension over the corpus, then writes everything to cache.json.

Usage:
    python pipeline/precompute/build_cache.py [--backend deterministic|llm]

Output schema (cache.json):
  {
    "backend":         "deterministic" | "llm",
    "classifications": {
        "<trace_key>": -1 | 0 | 1,     ← complete traces (leaf labels)
        "<prefix_key>": float,          ← internal nodes (minimax-derived)
        ...
    },
    "preferences":     { "<pair_key>": -1 | 0 | 1, ... }
  }

Step 2c extends the classifications dict to be prefix-closed using minimax
aggregation so that L* can look up any prefix it constructs during learning.
Leaf labels (from 2a) take priority over minimax-derived values where both
exist, preserving the LLM's direct judgment on complete traces.
"""

import argparse
import json
import os
import sys

_HERE     = os.path.dirname(os.path.abspath(__file__))  # src/pipeline/precompute/
_PIPELINE = os.path.dirname(_HERE)                       # src/pipeline/
_SRC      = os.path.dirname(_PIPELINE)                   # src/
sys.path.insert(0, _SRC)
sys.path.insert(0, _PIPELINE)

from config import CACHE_PATH
from corpus.generate import load as load_corpus
from precompute.classify import classify_corpus
from precompute.preferences import compute_preferences
from precompute.prefix_tree import build_prefix_classifications


def build(backend: str = "deterministic") -> dict:
    """
    Run the full pre-computation pipeline.

    Args:
        backend: "deterministic" (no LLM) or "llm" (requires API key).

    Returns:
        The cache dict that was written to CACHE_PATH.
    """
    print(f"[cache] Loading corpus...")
    corpus = load_corpus()
    print(f"[cache] {len(corpus)} traces loaded.")

    print(f"\n[cache] Step 2a: classifying traces (backend={backend})...")
    classifications = classify_corpus(corpus, backend=backend)
    print(f"[cache] {len(classifications)} traces classified.")

    print(f"\n[cache] Step 2b: computing pairwise preferences (backend={backend})...")
    preferences = compute_preferences(corpus, classifications, backend=backend)
    print(f"[cache] {len(preferences)} preference pairs computed.")

    print(f"\n[cache] Step 2c: building prefix-closed classifications (minimax)...")
    prefix_classifications = build_prefix_classifications(corpus, classifications)
    # Merge: prefix-derived labels fill in internal nodes.
    # Leaf labels from Step 2a take priority (overwrite) to preserve LLM judgment.
    all_classifications = {**prefix_classifications, **classifications}
    print(f"[cache] {len(all_classifications)} total classification entries "
          f"({len(classifications)} leaves + "
          f"{len(all_classifications) - len(classifications)} prefix nodes).")

    cache = {
        "backend":         backend,
        "classifications": all_classifications,
        "preferences":     preferences,
    }

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"\n[cache] Saved → {CACHE_PATH}")

    return cache


def load() -> dict:
    """Load cache from disk."""
    with open(CACHE_PATH) as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute LLM classifications and preferences.")
    parser.add_argument("--backend", choices=["deterministic", "llm"],
                        default="deterministic",
                        help="'deterministic' needs no API key; 'llm' calls Claude.")
    args = parser.parse_args()
    build(backend=args.backend)
