"""
Unit test: Corpus generation from kuhn_poker.dot (sampled mode)

Usage:
    python test/pipeline/corpus/generate.py
"""

import os
import sys

# -- path bootstrap --
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
_SRC  = os.path.join(_ROOT, "src")
sys.path.insert(0, _SRC)
sys.path.insert(0, os.path.join(_SRC, "pipeline"))

from pipeline.corpus.generate import generate
from dot_trace_generator import load_dot, generate_trace
from pipeline.config import DOT_FILE


def test_dot_trace_generation():
    machine = load_dot(DOT_FILE)

    n = 10
    traces = [generate_trace(machine) for _ in range(n)]

    print(f"\n{'='*60}")
    print(f"  DOT TRACES  ({n} random traces from kuhn_poker.dot)")
    print(f"{'='*60}")
    for i, trace in enumerate(traces, 1):
        steps = trace.split(";")
        print(f"\n  [{i:>2}]")
        for step in steps:
            print(f"        {step.strip()}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    # Tests Trace Generation of kuhn_poker.dot and writes to corpus.json
    test_dot_trace_generation()
