"""
Pipeline Entry Point

Orchestrates all four steps:
  1. Generate corpus T          (skipped if corpus.json exists)
  2. Pre-compute cache          (skipped if cache.json exists)
  3. Run symbolic L*            (always)
  4. Evaluate learned machine   (always)

Usage:
    # Deterministic run (no API key needed):
    python pipeline/run.py

    # LLM-backed run (requires .env/api_key):
    python pipeline/run.py --backend llm

    # Force regeneration of corpus and/or cache:
    python pipeline/run.py --regen-corpus --regen-cache
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))  # src/pipeline/
_SRC  = os.path.dirname(_HERE)                       # src/
sys.path.insert(0, _SRC)
sys.path.insert(0, _HERE)

from config import CORPUS_PATH, CACHE_PATH, REMAP_PATH
sys.path.insert(0, REMAP_PATH)

from corpus.generate import generate as gen_corpus, save as save_corpus, load as load_corpus
from precompute.build_cache import build as build_cache, load as load_cache
from teacher.cached_teacher import CachedTeacher
from evaluate import evaluate

from lstar import symbolic_lstar


def main():
    parser = argparse.ArgumentParser(description="Kuhn Poker reward machine pipeline.")
    parser.add_argument(
        "--backend", choices=["deterministic", "llm"], default="deterministic",
        help="'deterministic' uses outcome labels (no API key). "
             "'llm' calls Claude for classifications and preferences."
    )
    parser.add_argument("--regen-corpus", action="store_true",
                        help="Regenerate corpus even if corpus.json exists.")
    parser.add_argument("--regen-cache",  action="store_true",
                        help="Rerun pre-computation even if cache.json exists.")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Step 1: Corpus
    # -----------------------------------------------------------------------
    if args.regen_corpus or not os.path.exists(CORPUS_PATH):
        print("=" * 60)
        print("STEP 1: Generating corpus")
        print("=" * 60)
        hands = gen_corpus()
        save_corpus(hands)
    else:
        print(f"[run] Corpus found at {CORPUS_PATH} — skipping generation.")

    corpus = load_corpus()
    print(f"[run] Corpus: {len(corpus)} traces.\n")

    # -----------------------------------------------------------------------
    # Step 2: Pre-compute cache
    # -----------------------------------------------------------------------
    if args.regen_cache or not os.path.exists(CACHE_PATH):
        print("=" * 60)
        print(f"STEP 2: Pre-computing cache (backend={args.backend})")
        print("=" * 60)
        cache = build_cache(backend=args.backend)
    else:
        print(f"[run] Cache found at {CACHE_PATH} — skipping pre-computation.")
        cache = load_cache()

    print(f"[run] Cache: {len(cache['classifications'])} classifications, "
          f"{len(cache['preferences'])} preference pairs.\n")

    # -----------------------------------------------------------------------
    # Step 3: Build teacher and run L*
    # -----------------------------------------------------------------------
    teacher = CachedTeacher(
        corpus          = corpus,
        classifications = cache["classifications"],
        preferences     = cache["preferences"],
    )

    print("\n" + "=" * 60)
    print("STEP 3: Running symbolic L*")
    print("=" * 60)
    hypothesis, data = symbolic_lstar(teacher.sigma_I, teacher.sigma_O, teacher)

    (num_pref, num_ineq, num_seq, num_ecs,
     num_vars, up_shape, lo_shape, num_eq, cex_lens, events) = data

    print(f"\n[run] L* complete.")
    print(f"  Preference queries : {num_pref}")
    print(f"  Equivalence queries: {num_eq}")
    print(f"  States learned     : {len(hypothesis[0])}")
    print(f"  Events             : {events}")
    teacher.print_stats()

    # -----------------------------------------------------------------------
    # Step 4: Evaluate
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Evaluation")
    print("=" * 60)
    evaluate(hypothesis, teacher)


if __name__ == "__main__":
    main()
