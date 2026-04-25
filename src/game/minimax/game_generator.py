#!/usr/bin/env python3
"""
Random minimax game tree generator.

P1 is the environment player (provides inputs).
P2 is the system player (provides responses — what we learn).
Players alternate turns starting with P1.
Each node has a random integer value; trace score = cumulative sum of node values.

Usage:
    python -m src.game.game_generator 4
    python -m src.game.game_generator 6 --branching 3 --seed 42 --scores
    python -m src.game.game_generator 4 --output game.json
"""

import argparse
import random
import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GameNode:
    value: int
    player: str          # 'P1' (environment) or 'P2' (system)
    depth: int
    children: dict = field(default_factory=dict)   # action -> GameNode

    def is_terminal(self) -> bool:
        return len(self.children) == 0


def generate_tree(depth: int, branching: int = 2, seed: Optional[int] = None) -> GameNode:
    """
    Generate a random game tree of given depth.
    P1 and P2 alternate turns starting with P1.
    Node values are random integers in [0, 10].
    """
    if seed is not None:
        random.seed(seed)

    p1_actions = [chr(65 + i) for i in range(branching)]    # A, B, C, ...
    p2_actions = [chr(88 + i) for i in range(branching)]    # X, Y, Z, ...

    def build(d: int, player: str) -> GameNode:
        value = random.randint(0, 10)
        node = GameNode(value=value, player=player, depth=d)

        if d < depth:
            next_player = 'P2' if player == 'P1' else 'P1'
            actions = p1_actions if player == 'P1' else p2_actions
            for action in actions:
                node.children[action] = build(d + 1, next_player)

        return node

    return build(0, 'P1')


def compute_trace_scores(root: GameNode) -> list[tuple[list[str], int]]:
    """Return all root-to-leaf traces paired with their cumulative scores."""
    def _collect(node, path, score):
        score += node.value
        if node.is_terminal():
            return [(path, score)]
        results = []
        for action, child in node.children.items():
            results.extend(_collect(child, path + [action], score))
        return results
            
    return _collect(root, [], 0)


# Untested Function Because It's Just Pretty Printing.
def print_tree(node: GameNode, prefix: str = '', is_last: bool = True,
               action: str = None) -> None:
    """Pretty-print the game tree."""
    terminal_marker = '  (terminal)' if node.is_terminal() else ''
    if action is None:
        # Root node
        print(f'[{node.player}] val={node.value}')
    else:
        connector = '└── ' if is_last else '├── '
        print(f'{prefix}{connector}{action}  →  [{node.player}] val={node.value}{terminal_marker}')

    child_items = list(node.children.items())
    for i, (act, child) in enumerate(child_items):
        child_is_last = i == len(child_items) - 1
        if action is None:
            new_prefix = ''
        else:
            extension = '    ' if is_last else '│   '
            new_prefix = prefix + extension
        print_tree(child, new_prefix, child_is_last, act)


def tree_to_dict(node: GameNode) -> dict:
    """Serialise tree to a dict for JSON output."""
    return {
        'value': node.value,
        'player': node.player,
        'depth': node.depth,
        'children': {
            action: tree_to_dict(child)
            for action, child in node.children.items()
        }
    }


def tree_from_dict(d: dict) -> GameNode:
    """Deserialise tree from a dict (e.g. loaded from JSON)."""
    node = GameNode(value=d['value'], player=d['player'], depth=d['depth'])
    for action, child_dict in d.get('children', {}).items():
        node.children[action] = tree_from_dict(child_dict)
    return node


def main():
    parser = argparse.ArgumentParser(
        description='Generate a random minimax game tree.'
    )
    parser.add_argument('depth', type=int,
                        help='Depth of the game tree')
    parser.add_argument('--branching', '-b', type=int, default=2,
                        help='Branching factor for both players (default: 2)')
    parser.add_argument('--seed', '-s', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Save tree to JSON file')
    parser.add_argument('--scores', action='store_true',
                        help='Print all trace scores sorted descending')
    args = parser.parse_args()

    print(f'Generating game tree  depth={args.depth}  '
          f'branching={args.branching}  seed={args.seed}')
    print()

    root = generate_tree(args.depth, args.branching, args.seed)

    print_tree(root)
    print()

    traces = compute_trace_scores(root)
    scores = [s for _, s in traces]

    if args.scores:
        print('Trace scores (cumulative sum along path):')
        for path, score in sorted(traces, key=lambda x: -x[1]):
            print(f'  {" → ".join(path):30s}  score={score}')
        print()

    print(f'Traces : {len(traces)}')
    print(f'Scores : min={min(scores)}  max={max(scores)}  '
          f'mean={sum(scores)/len(scores):.1f}')

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(tree_to_dict(root), f, indent=2)
        print(f'Saved  : {args.output}')


if __name__ == '__main__':
    main()
