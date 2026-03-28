"""
Prefix Tree Builder  (Pipeline Step 2c)

Extends the classification dict to be prefix-closed using minimax aggregation.

A prefix-closed set means: if complete trace t is in T, then all prefixes
of t are also in T with derived labels. This is required for L* to work
correctly — it queries sequences at every depth, not just complete traces.

Aggregation uses minimax:
  P2 decision nodes  (children start with P2_ or P2B_) → max over children
  P1 decision nodes  (children start with P1_ or P1B_) → min over children
  Chance nodes       (children start with DEAL_)        → average over children

This reflects rational strategic reasoning:
  P2 is the agent whose quality we evaluate → P2 plays optimally (max)
  P1 is the opponent                        → P1 plays adversarially (min)
  Card dealing is random                    → average

The actor at each node is inferred from the symbols of its children,
which are encoded in the symbol names (P1_, P2_, DEAL_, P1B_, P2B_).
"""

import os
import sys

_HERE     = os.path.dirname(os.path.abspath(__file__))  # src/pipeline/precompute/
_PIPELINE = os.path.dirname(_HERE)                       # src/pipeline/
_SRC      = os.path.dirname(_PIPELINE)                   # src/
sys.path.insert(0, _SRC)
sys.path.insert(0, _PIPELINE)

from precompute.classify import trace_key


# -----------------------------------------------------------------------
# Prefix tree construction
# -----------------------------------------------------------------------

def build_prefix_tree(corpus: list) -> dict:
    """
    Build a mapping from every prefix of every corpus trace to the list
    of complete descendant traces that pass through it.

    Args:
        corpus: List of complete trace tuples.

    Returns:
        dict mapping prefix tuple → list of complete trace tuples.
    """
    tree = {}
    for trace in corpus:
        for i in range(len(trace) + 1):
            prefix = trace[:i]
            tree.setdefault(prefix, []).append(trace)
    return tree


# -----------------------------------------------------------------------
# Actor detection
# -----------------------------------------------------------------------

def _actor_from_children(child_symbols: list) -> str:
    """
    Determine whose turn it is at a node by inspecting the symbols of
    its direct children (the next step after this prefix).

    P1_ or P1B_ prefix → P1 decides → min
    P2_ or P2B_ prefix → P2 decides → max
    DEAL_ prefix       → chance     → average
    """
    for sym in child_symbols:
        if sym.startswith(("P1_", "P1B_")):
            return "P1"
        if sym.startswith(("P2_", "P2B_")):
            return "P2"
        if sym.startswith("DEAL_"):
            return "chance"
    return "chance"


# -----------------------------------------------------------------------
# Minimax aggregation
# -----------------------------------------------------------------------

def _minimax(prefix: tuple, tree: dict, leaf_labels: dict, memo: dict) -> float:
    """
    Recursively compute the minimax value of a prefix node.

    Base case: prefix is a complete trace → return its leaf label directly.
    Recursive case: aggregate children values according to actor type.

    Args:
        prefix:      The current prefix tuple being evaluated.
        tree:        Full prefix tree from build_prefix_tree().
        leaf_labels: Classifications for complete traces (from classify step).
        memo:        Memoization dict to avoid recomputation.

    Returns:
        float: The minimax value for this prefix in [-1, 1].
    """
    if prefix in memo:
        return memo[prefix]

    tk = trace_key(prefix)

    # Base case: this prefix IS a complete trace — use its label directly
    if tk in leaf_labels:
        val = float(leaf_labels[tk])
        memo[prefix] = val
        return val

    # Get all complete traces descending from this prefix
    descendants = tree.get(prefix, [])
    if not descendants:
        memo[prefix] = 0.0
        return 0.0

    # Find direct children by grouping descendants by their next symbol
    depth = len(prefix)
    children = {}  # next_symbol → child_prefix tuple
    for trace in descendants:
        if len(trace) > depth:
            next_sym = trace[depth]
            if next_sym not in children:
                children[next_sym] = prefix + (next_sym,)

    if not children:
        memo[prefix] = 0.0
        return 0.0

    # Determine who acts at this node
    actor = _actor_from_children(list(children.keys()))

    # Recurse into each child
    child_values = [
        _minimax(child_prefix, tree, leaf_labels, memo)
        for child_prefix in children.values()
    ]

    # Aggregate
    if actor == "P2":
        value = max(child_values)
    elif actor == "P1":
        value = min(child_values)
    else:  # chance
        value = sum(child_values) / len(child_values)

    memo[prefix] = value
    return value


# -----------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------

def build_prefix_classifications(corpus: list, leaf_labels: dict) -> dict:
    """
    Build a prefix-closed classification dict using minimax aggregation.

    For every prefix of every corpus trace (including the empty prefix and
    all complete traces), computes a value in [-1, 1] and stores it under
    the same trace_key encoding used by classify.py.

    Leaf labels (LLM or deterministic) take priority over minimax-derived
    values when both exist — the merge in build_cache.py handles this by
    letting the original classifications overwrite the prefix dict.

    Args:
        corpus:      List of complete trace tuples from corpus/generate.py.
        leaf_labels: dict mapping trace_key(trace) → int label from classify step.

    Returns:
        dict mapping trace_key(prefix) → float for ALL prefixes in the tree.
    """
    tree   = build_prefix_tree(corpus)
    memo   = {}
    result = {}

    for prefix in tree:
        value = _minimax(prefix, tree, leaf_labels, memo)
        result[trace_key(prefix)] = value

    print(f"[prefix_tree] {len(result)} prefix nodes computed "
          f"({len(corpus)} leaves + {len(result) - len(corpus)} internal nodes)")

    return result
