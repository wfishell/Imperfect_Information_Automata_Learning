"""
Corpus Generation  (Pipeline Step 1)

Builds the finite trace corpus T = {t_1, ..., t_N} from the Kuhn Poker
DOT machine and writes it to corpus.json.

Two modes (set in config.py):
  "exhaustive" — DFS over the automaton; yields all 62 legal Kuhn Poker hands.
  "sampled"    — random walks from the initial state; yields CORPUS_SAMPLE_SIZE hands.

Output schema (corpus.json):
  {
    "mode":   "exhaustive" | "sampled",
    "count":  N,
    "traces": [
      ["DEAL_King_Queen", "P1_King_Queen_bet", "P2_King_Queen_fold_W1"],
      ...
    ]
  }
"""

import json
import os
import random
import sys

# -- path bootstrap --
_HERE     = os.path.dirname(os.path.abspath(__file__))  # src/pipeline/corpus/
_PIPELINE = os.path.dirname(_HERE)                       # src/pipeline/
_SRC      = os.path.dirname(_PIPELINE)                   # src/
sys.path.insert(0, _SRC)
sys.path.insert(0, _PIPELINE)

from config import CORPUS_MODE, CORPUS_SAMPLE_SIZE, CORPUS_PATH, DOT_FILE
from teacher.deterministic_teacher import (
    _prepare_machine,
    _enumerate_all_hands,
    _to_symbol,
)
from dot_trace_generator import load_dot


def _sampled_hands(machine: dict, n: int) -> list:
    """
    Generate n complete hands via random walks on the cleaned automaton.
    A hand is complete when a transition returns to the initial state.
    """
    initial = machine["initial"]
    trans   = machine["transitions"]
    results = []
    attempts = 0

    while len(results) < n and attempts < n * 20:
        attempts += 1
        state = initial
        path  = []
        for _ in range(15):
            choices = list(trans.get(state, {}).items())
            if not choices:
                break
            inp, (nxt, out) = random.choice(choices)
            path.append(_to_symbol(inp, out))
            if nxt == initial:
                results.append(tuple(path))
                break
            state = nxt

    # pad with repeats if random walks were unlucky
    while len(results) < n and results:
        results.append(random.choice(results))

    return results


def generate(dot_file: str = DOT_FILE, mode: str = CORPUS_MODE,
             sample_size: int = CORPUS_SAMPLE_SIZE) -> list:
    """
    Build and return the corpus as a list of tuples.

    Args:
        dot_file:    Path to the DOT automaton.
        mode:        "exhaustive" or "sampled".
        sample_size: Number of traces when mode == "sampled".

    Returns:
        list[tuple[str, ...]]: Each tuple is one complete game transcript.
    """
    machine = _prepare_machine(load_dot(dot_file))

    if mode == "exhaustive":
        hands = _enumerate_all_hands(machine)
    else:
        hands = _sampled_hands(machine, sample_size)

    return hands


def save(hands: list, path: str = CORPUS_PATH, mode: str = CORPUS_MODE) -> None:
    """Serialise corpus to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "mode":   mode,
        "count":  len(hands),
        "traces": [list(h) for h in hands],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[corpus] Saved {len(hands)} traces → {path}")


def load(path: str = CORPUS_PATH) -> list:
    """Load corpus from JSON. Returns list of tuples."""
    with open(path) as f:
        data = json.load(f)
    return [tuple(t) for t in data["traces"]]


if __name__ == "__main__":
    hands = generate()
    save(hands)
    print(f"[corpus] {len(hands)} traces generated (mode={CORPUS_MODE})")
