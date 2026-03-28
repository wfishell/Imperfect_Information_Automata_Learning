"""
Preference Pre-computation  (Pipeline Step 2b)

Computes pairwise preferences over the corpus T before learning begins,
so that symbolic_lstar never needs to make a live LLM call.

Two strategies (set in config.py):
  "all_pairs"  — compare every (i, j) pair; O(N^2/2) calls.
                 Feasible for N=62 (1891 pairs).
  "quicksort"  — randomised quicksort order; O(N log N) calls.
                 Scales to larger corpora.

Both strategies write their results to cache.json under "preferences":
  { "<key_i>|<key_j>": -1 | 0 | 1, ... }

where the value is:
   1  trace i is preferred over trace j
  -1  trace j is preferred over trace i
   0  equal

Two backends:
  "deterministic" — derives preferences from pre-computed classifications
                    (no LLM calls).
  "llm"           — prompts the LLM to compare each pair.
"""

import os
import random
import sys

_HERE     = os.path.dirname(os.path.abspath(__file__))  # src/pipeline/precompute/
_PIPELINE = os.path.dirname(_HERE)                       # src/pipeline/
_SRC      = os.path.dirname(_PIPELINE)                   # src/
sys.path.insert(0, _SRC)
sys.path.insert(0, _PIPELINE)

import anthropic
from config import MODEL, API_KEY_PATH, PREF_STRATEGY
from precompute.classify import trace_key, _trace_to_text


# -----------------------------------------------------------------------
# Pair key helper
# -----------------------------------------------------------------------

def pair_key(t1: tuple, t2: tuple) -> str:
    """Canonical key for a (t1, t2) pair."""
    return f"{trace_key(t1)}|||{trace_key(t2)}"


# -----------------------------------------------------------------------
# Deterministic preference (from classifications)
# -----------------------------------------------------------------------

def prefer_deterministic(t1: tuple, t2: tuple, classifications: dict) -> int:
    """
    Derive preference from pre-computed classification labels.

    Returns 1 if t1 preferred, -1 if t2 preferred, 0 if equal.
    """
    v1 = classifications.get(trace_key(t1), 0)
    v2 = classifications.get(trace_key(t2), 0)
    if v1 > v2:
        return 1
    if v1 < v2:
        return -1
    return 0


# -----------------------------------------------------------------------
# LLM pairwise preference
# -----------------------------------------------------------------------

_LLM_PREFER_PROMPT = """\
You are evaluating Kuhn Poker hands. Deck: Jack < Queen < King.
Each player antes 1 chip. P1 acts first; P2 acts with knowledge of
their own card and P1's visible action only.

Compare P2's strategic play quality in these two hands:

Hand A:
{text_a}

Hand B:
{text_b}

Which hand shows BETTER strategic play by P2, considering only whether
P2's decisions were correct given their information — not the final outcome.

Reply with ONLY one of: A, B, or EQUAL"""


def prefer_llm(t1: tuple, t2: tuple, client: anthropic.Anthropic) -> int:
    """
    Ask the LLM to compare two traces.

    Returns 1 if t1 preferred, -1 if t2 preferred, 0 if equal.
    """
    prompt = _LLM_PREFER_PROMPT.format(
        text_a=_trace_to_text(t1),
        text_b=_trace_to_text(t2),
    )
    for attempt in range(4):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=10,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            break
        except Exception as e:
            import time
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)

    raw = resp.content[0].text.strip().upper()
    if raw.startswith("A") and "B" not in raw:
        return 1
    if raw.startswith("B") and "A" not in raw:
        return -1
    return 0


# -----------------------------------------------------------------------
# All-pairs strategy
# -----------------------------------------------------------------------

def _all_pairs(corpus: list, prefer_fn) -> dict:
    """Compare every (i, j) pair with i < j."""
    prefs = {}
    n = len(corpus)
    total = n * (n - 1) // 2
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            t1, t2 = corpus[i], corpus[j]
            result = prefer_fn(t1, t2)
            prefs[pair_key(t1, t2)] = result
            prefs[pair_key(t2, t1)] = -result
            count += 1
            if count % 50 == 0:
                print(f"  [prefs] {count}/{total} pairs done")
    return prefs


# -----------------------------------------------------------------------
# Quicksort strategy
# -----------------------------------------------------------------------

def _quicksort_pairs(corpus: list, prefer_fn) -> dict:
    """
    Collect comparisons made during randomised quicksort.
    O(N log N) expected comparisons; covers pairs needed for a total order.
    """
    prefs = {}

    def _compare(t1, t2):
        result = prefer_fn(t1, t2)
        prefs[pair_key(t1, t2)] = result
        prefs[pair_key(t2, t1)] = -result
        return result

    def _qsort(items):
        if len(items) <= 1:
            return items
        pivot = random.choice(items)
        less, equal, greater = [], [pivot], []
        for item in items:
            if item is pivot:
                continue
            cmp = _compare(item, pivot)
            if cmp > 0:
                less.append(item)
            elif cmp < 0:
                greater.append(item)
            else:
                equal.append(item)
        return _qsort(less) + equal + _qsort(greater)

    _qsort(list(corpus))
    return prefs


# -----------------------------------------------------------------------
# Batch compute preferences over entire corpus
# -----------------------------------------------------------------------

def compute_preferences(corpus: list, classifications: dict,
                        backend: str = "deterministic",
                        strategy: str = PREF_STRATEGY) -> dict:
    """
    Pre-compute pairwise preferences for all traces in the corpus.

    Args:
        corpus:          List of trace tuples.
        classifications: Output of classify_corpus() — used by deterministic backend.
        backend:         "deterministic" or "llm".
        strategy:        "all_pairs" or "quicksort".

    Returns:
        dict mapping pair_key(t1, t2) -> int preference.
    """
    if backend == "llm":
        with open(API_KEY_PATH) as f:
            api_key = f.read().strip()
        client = anthropic.Anthropic(api_key=api_key)
        prefer_fn = lambda t1, t2: prefer_llm(t1, t2, client)
    else:
        prefer_fn = lambda t1, t2: prefer_deterministic(t1, t2, classifications)

    print(f"[prefs] Strategy={strategy}, backend={backend}, N={len(corpus)}")

    if strategy == "quicksort":
        return _quicksort_pairs(corpus, prefer_fn)
    else:
        return _all_pairs(corpus, prefer_fn)
