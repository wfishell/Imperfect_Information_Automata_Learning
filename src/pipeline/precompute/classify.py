"""
Trace Classification  (Pipeline Step 2a)

Assigns every trace t_i in the corpus a label in sigma_O = {-1, 0, 1}:
  -1  P2 loses  (hand ends with _W1)
   0  in progress  (no terminal outcome symbol)
   1  P2 wins   (hand ends with _W2)

Two backends:
  "deterministic" — reads the outcome directly from the terminal symbol.
                    No LLM calls. Used for ground-truth / debugging.
  "llm"           — asks the LLM to classify each trace based on P2's
                    strategic play quality. Requires API key in config.

The result is stored in cache.json under the key "classifications":
  { "<trace_as_pipe_separated_string>": -1 | 0 | 1, ... }
"""

import os
import sys

_HERE     = os.path.dirname(os.path.abspath(__file__))  # src/pipeline/precompute/
_PIPELINE = os.path.dirname(_HERE)                       # src/pipeline/
_SRC      = os.path.dirname(_PIPELINE)                   # src/
sys.path.insert(0, _SRC)
sys.path.insert(0, _PIPELINE)

import anthropic
from config import MODEL, API_KEY_PATH


# -----------------------------------------------------------------------
# Trace key helper
# -----------------------------------------------------------------------

def trace_key(trace: tuple) -> str:
    """Stable string key for a trace tuple."""
    return "|".join(trace)


# -----------------------------------------------------------------------
# Deterministic classification (outcome from symbol)
# -----------------------------------------------------------------------

def classify_deterministic(trace: tuple) -> int:
    """
    Classify a trace by its terminal outcome symbol.

    Scans the trace in reverse for the first symbol ending in _W1 or _W2.
    Returns 1 (P2 wins), -1 (P2 loses), or 0 (no terminal outcome found).
    """
    for sym in reversed(trace):
        if sym.endswith("_W2"):
            return 1
        if sym.endswith("_W1"):
            return -1
    return 0


# -----------------------------------------------------------------------
# LLM classification
# -----------------------------------------------------------------------

_LLM_CLASSIFY_PROMPT = """\
You are evaluating a single Kuhn Poker hand from the perspective of Player 2 (P2).
Deck: Jack < Queen < King. Each player antes 1 chip.

Hand transcript:
{trace_text}

Based on P2's strategic decisions (considering only what P2 could know at decision time),
classify this hand as one of:
  WIN  — P2 played well and/or won
  LOSE — P2 played poorly and/or lost
  DRAW — P2's result was neutral or ambiguous

Reply with ONLY one word: WIN, LOSE, or DRAW."""

_LABEL_MAP = {"WIN": 1, "LOSE": -1, "DRAW": 0}


def _trace_to_text(trace: tuple) -> str:
    """Convert a trace tuple to a human-readable string for the LLM prompt."""
    lines = []
    for sym in trace:
        p = sym.split("_")
        phase = p[0]
        if phase == "DEAL" and len(p) >= 3:
            lines.append(f"  Deal    : P1 gets {p[1]}, P2 gets {p[2]}")
        elif phase == "P1" and len(p) >= 4:
            lines.append(f"  P1      : {p[3]}  (P1={p[1]}, P2={p[2]})")
        elif phase == "P2" and len(p) >= 5:
            outcome = {"W1": "-> P1 wins", "W2": "-> P2 wins"}.get(p[4], "-> continues")
            lines.append(f"  P2      : {p[3]} {outcome}  (P2 holds {p[2]})")
        elif phase in ("P1B", "P2B") and len(p) >= 5:
            outcome = {"W1": "-> P1 wins", "W2": "-> P2 wins"}.get(p[4], "-> continues")
            lines.append(f"  {phase:4s}    : {p[3]} {outcome}")
    return "\n".join(lines) if lines else "  (empty trace)"


def classify_llm(trace: tuple, client: anthropic.Anthropic) -> int:
    """
    Ask the LLM to classify a single trace.

    Args:
        trace:  The game transcript as a tuple of symbols.
        client: An initialised Anthropic client.

    Returns:
        1, -1, or 0.
    """
    prompt = _LLM_CLASSIFY_PROMPT.format(trace_text=_trace_to_text(trace))
    for attempt in range(4):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=5,
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
    for key, val in _LABEL_MAP.items():
        if raw.startswith(key):
            return val
    return 0  # fallback


# -----------------------------------------------------------------------
# Batch classify entire corpus
# -----------------------------------------------------------------------

def classify_corpus(corpus: list, backend: str = "deterministic") -> dict:
    """
    Classify every trace in the corpus.

    Args:
        corpus:  List of trace tuples from corpus/generate.py.
        backend: "deterministic" or "llm".

    Returns:
        dict mapping trace_key(trace) -> int label.
    """
    client = None
    if backend == "llm":
        with open(API_KEY_PATH) as f:
            api_key = f.read().strip()
        client = anthropic.Anthropic(api_key=api_key)

    classifications = {}
    for i, trace in enumerate(corpus):
        if backend == "llm":
            label = classify_llm(trace, client)
            print(f"  [{i+1}/{len(corpus)}] {trace_key(trace)[:60]}... → {label}")
        else:
            label = classify_deterministic(trace)

        classifications[trace_key(trace)] = label

    return classifications
