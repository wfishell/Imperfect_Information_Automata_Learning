"""
Cached Teacher  (Pipeline Step 3 dependency)

A drop-in teacher for symbolic_lstar that answers all queries from the
pre-computed cache — no live LLM calls during learning.

preference_query:   looks up (t1, t2) in the preferences cache.
equivalence_query:  draws from the corpus and looks up classifications.
sample_sequences:   returns random draws from the corpus.
set_student:        no-op (cache is already complete; no student needed).

If a preference pair is missing from the cache, the teacher falls back to
comparing the pre-computed classification labels (always available).
"""

import json
import os
import random
import sys

_HERE     = os.path.dirname(os.path.abspath(__file__))  # src/pipeline/teacher/
_PIPELINE = os.path.dirname(_HERE)                       # src/pipeline/
_SRC      = os.path.dirname(_PIPELINE)                   # src/
sys.path.insert(0, _SRC)
sys.path.insert(0, _PIPELINE)

from precompute.classify import trace_key, classify_deterministic
from precompute.preferences import pair_key


class CachedTeacher:
    """
    Teacher for symbolic_lstar backed entirely by pre-computed cache.

    Args:
        corpus:          List of trace tuples (from corpus/generate.py).
        classifications: dict mapping trace_key -> int label {-1, 0, 1}.
        preferences:     dict mapping pair_key  -> int preference {-1, 0, 1}.
    """

    P2_WINS  =  1
    IN_PROG  =  0
    P2_LOSES = -1

    def __init__(self, corpus: list, classifications: dict, preferences: dict):
        self.corpus          = corpus
        self.classifications = classifications
        self.preferences     = preferences

        self.sigma_I = tuple(sorted({sym for trace in corpus for sym in trace}))
        self.sigma_O = (self.P2_LOSES, self.IN_PROG, self.P2_WINS)

        self._cache_hits    = 0
        self._prefix_hits   = 0
        self._fallback_hits = 0

        # Count how many entries are prefix nodes vs leaves
        n_leaves  = len(corpus)
        n_total   = len(classifications)
        n_prefixes = n_total - n_leaves

        print(f"[teacher] CachedTeacher ready.")
        print(f"  corpus:         {len(corpus)} traces")
        print(f"  sigma_I:        {len(self.sigma_I)} symbols")
        print(f"  classifications:{n_total} entries "
              f"({n_leaves} leaves + {n_prefixes} prefix nodes)")
        print(f"  preferences:    {len(preferences)} pairs pre-computed")

    # -----------------------------------------------------------------------
    # preference_query
    # -----------------------------------------------------------------------

    def preference_query(self, s1: tuple, s2: tuple) -> int:
        """
        Return preference for (s1, s2).  Three tiers:

        Tier 1 — preferences cache hit:
            Pre-computed pair (complete hand vs complete hand). Return directly.

        Tier 2 — prefix-closed classifications hit:
            Both sequences found in the extended classifications dict, which
            covers all prefixes of all corpus traces via minimax aggregation.
            Derive preference from their minimax values.

        Tier 3 — last resort fallback:
            Sequence is outside the prefix tree entirely (an impossible game
            sequence L* constructed). Use classify_deterministic which scans
            the symbol names for _W1/_W2 — always safe and consistent.

        Returns 1 if s1 preferred, -1 if s2 preferred, 0 if equal.
        """
        # Tier 1: pre-computed pair
        key = pair_key(s1, s2)
        if key in self.preferences:
            self._cache_hits += 1
            return self.preferences[key]

        # Tier 2: prefix-closed classification lookup
        # Threshold minimax floats to {-1, 0, 1} via sign before comparing.
        # Raw float comparison (e.g., 0.3 vs 0.5) would add strict-ordering
        # constraints that exceed the 3-level sigma_O and cause Z3 UNSAT.
        k1, k2 = trace_key(s1), trace_key(s2)
        if k1 in self.classifications and k2 in self.classifications:
            self._prefix_hits += 1
            raw1 = self.classifications[k1]
            raw2 = self.classifications[k2]
            v1 = 1 if raw1 > 0 else (-1 if raw1 < 0 else 0)
            v2 = 1 if raw2 > 0 else (-1 if raw2 < 0 else 0)
            if v1 > v2: return 1
            if v1 < v2: return -1
            return 0

        # Tier 3: outside prefix tree — use deterministic symbol scan
        self._fallback_hits += 1
        v1 = classify_deterministic(s1)
        v2 = classify_deterministic(s2)
        if v1 > v2: return 1
        if v1 < v2: return -1
        return 0

    ###### Notes From Call w/ Will ######
    # Prefix Closed - Look into that.
    # cache prefixes and compute preferences 
    # Turn into a prefix closed data set -> compute on subtraces and then do preferences there.
    # Enumerating all pairs of traces and compute LLM preferences on that. 
    # Cap traces
    # OR compute LLM on full traces and then back into subtraces.
    # Example: opening: A, look at all the things that start with A, and then compute reward. Assign a preference and the LLM only computes on full traces.
    # See what they share to identify manually derive preferences on subtraces. See what traces have same prefixes and follow the tree. Possible paths to explore.  

    # Compute a bunch of traces for complete games
    # Compute n^2 queries for preferences
    # O(n) queries
    # We need to figure out how to do show that prefixes represent average behavior of traces. 
    # All comparisons should be to similar size traces. For prefixes. 

    # -----------------------------------------------------------------------
    # equivalence_query
    # -----------------------------------------------------------------------

    def equivalence_query(self, states, sigma_I, sigma_O, init_state, delta, output_fnc):
        """
        Exhaustive equivalence check against all corpus traces.

        Compares the teacher's pre-computed classification against the
        hypothesis output for every trace in the corpus.

        Returns (True, None) if all match, else (False, (counterexample, correct_value)).
        """
        for trace in self.corpus:
            teacher_val = self.classifications.get(trace_key(trace), self.IN_PROG)
            hyp_val     = self._run_hypothesis(trace, init_state, delta, output_fnc)
            if teacher_val != hyp_val:
                return False, (trace, teacher_val)
        return True, None

    def _run_hypothesis(self, seq, init_state, delta, output_fnc) -> int:
        """Simulate the current hypothesis on seq and return its output."""
        q = init_state
        for sym in seq:
            q = delta.get(q, {}).get(sym, q)
        return output_fnc.get(q, self.IN_PROG)

    # -----------------------------------------------------------------------
    # sample_sequences
    # -----------------------------------------------------------------------

    def sample_sequences(self, quantity: int) -> list:
        """Return random draws from the corpus."""
        return [random.choice(self.corpus) for _ in range(quantity)]

    # -----------------------------------------------------------------------
    # set_student  (no-op — cache is complete)
    # -----------------------------------------------------------------------

    def set_student(self, init_state, delta, output_fnc):
        """API compatibility only. Cache is pre-computed; student not needed."""
        pass

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def print_stats(self):
        total = self._cache_hits + self._prefix_hits + self._fallback_hits
        print(f"[teacher] preference queries: {total}")
        print(f"[teacher]   tier 1 (cache):   {self._cache_hits}")
        print(f"[teacher]   tier 2 (prefix):  {self._prefix_hits}")
        print(f"[teacher]   tier 3 (fallback):{self._fallback_hits}")


# -----------------------------------------------------------------------
# Factory: build from cache.json + corpus.json
# -----------------------------------------------------------------------

def from_cache_files(corpus_path: str, cache_path: str) -> "CachedTeacher":
    """
    Convenience constructor: load corpus and cache from disk.

    Args:
        corpus_path: Path to corpus/corpus.json.
        cache_path:  Path to precompute/cache.json.

    Returns:
        A ready CachedTeacher instance.
    """
    from corpus.generate import load as load_corpus

    corpus = load_corpus(corpus_path)

    with open(cache_path) as f:
        cache = json.load(f)

    return CachedTeacher(
        corpus          = corpus,
        classifications = cache["classifications"],
        preferences     = cache["preferences"],
    )
