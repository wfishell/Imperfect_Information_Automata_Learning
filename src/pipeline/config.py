"""
Pipeline configuration.

All paths and tuneable parameters live here. Import this module
everywhere instead of hardcoding paths or constants.
"""

import os

# -----------------------------------------------------------------------
# Directory anchors
# -----------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))       # src/pipeline/
_SRC  = os.path.dirname(_HERE)                            # src/
_ROOT = os.path.dirname(_SRC)                             # Imperfect_Information_Automata_Learning/

# -----------------------------------------------------------------------
# Input files
# -----------------------------------------------------------------------

DOT_FILE = os.path.join(_SRC, "data", "Kuhn_Poker", "kuhn_poker.dot")
HOA_FILE = os.path.join(_SRC, "data", "Kuhn_Poker", "Kuhn_Poker.hoa")

# -----------------------------------------------------------------------
# REMAP import path
# -----------------------------------------------------------------------

REMAP_PATH = os.path.join(_SRC, "REMAP", "remap")

# -----------------------------------------------------------------------
# Pipeline artifact paths
# -----------------------------------------------------------------------

CORPUS_PATH    = os.path.join(_HERE, "corpus",     "corpus.json")
CACHE_PATH     = os.path.join(_HERE, "precompute", "cache.json")

# -----------------------------------------------------------------------
# LLM settings
# -----------------------------------------------------------------------

MODEL        = "claude-opus-4-6"
API_KEY_PATH = os.path.join(_ROOT, ".env", "api_key")

# -----------------------------------------------------------------------
# Corpus settings
#
# CORPUS_MODE:
#   "exhaustive" — enumerate all legal hands from the DOT machine (62 for Kuhn Poker).
#   "sampled"    — random walks; set CORPUS_SAMPLE_SIZE to desired N.
# -----------------------------------------------------------------------

CORPUS_MODE        = "exhaustive"
CORPUS_SAMPLE_SIZE = 200          # only used when CORPUS_MODE == "sampled"

# -----------------------------------------------------------------------
# Preference pre-computation strategy
#
# PREF_STRATEGY:
#   "all_pairs"  — compute every (i, j) pair; O(N^2/2) LLM calls.
#                  Fine for N=62, may be expensive for larger corpora.
#   "quicksort"  — randomised quicksort order; O(N log N) LLM calls.
# -----------------------------------------------------------------------

PREF_STRATEGY = "all_pairs"
