"""
Two-phase MCTS Nim player (apples-to-apples with L*).

Phase 1 — TRAINING. T UCB1 rollouts from the initial state build a
search tree; pairwise oracle.compare() against a random prior rollout
provides binary reward backed up along the selection path.

Phase 2 — EVALUATION. Frozen tree, argmax_a N(s,a) at each P2 turn,
random fallback on out-of-tree states. ZERO oracle calls during eval.

Usage:
    python -m src.scripts.mcts_trained_player_nim --piles 1 2 3 --train-K 1000
"""

import argparse
import math
import random

from src.game.nim.game_nfa          import NimNFA
from src.game.nim.preference_oracle import NimOracle
from src.lstar_mcts.counting_oracle import CountingOracle


def train_uct(nfa, oracle, T: int, c: float, rng: random.Random) -> dict:
    tree: dict   = {}
    buffer: list = []
    for _ in range(T):
        path: list  = []
        trace: list = []
        node = nfa.root
        while True:
            if node is None or node.is_terminal():
                break
            key = tuple(trace)
            if key not in tree:
                tree[key] = {'visits': 0,
                             'children': {a: {'N': 0, 'W': 0} for a in node.children}}
                break
            children = tree[key]['children']
            untried  = [a for a, s in children.items() if s['N'] == 0]
            if untried:
                action = untried[0]
                path.append((key, action)); trace.append(action)
                node = node.children[action]
                break
            total_visits = tree[key]['visits']
            def ucb(a):
                s = children[a]
                return s['W'] / s['N'] + c * math.sqrt(math.log(total_visits) / s['N'])
            action = max(children.keys(), key=ucb)
            path.append((key, action)); trace.append(action)
            node = node.children[action]

        rollout_trace = list(trace); roll_node = node
        while roll_node is not None and not roll_node.is_terminal():
            a = rng.choice(list(roll_node.children.keys()))
            rollout_trace.append(a)
            roll_node = roll_node.children[a]

        if buffer:
            prior  = rng.choice(buffer)
            pref   = oracle.compare(rollout_trace, prior)
            reward = 1 if pref == 't1' else 0
        else:
            reward = 1
        buffer.append(rollout_trace)

        for (key, action) in path:
            tree[key]['visits'] += 1
            tree[key]['children'][action]['N'] += 1
            tree[key]['children'][action]['W'] += reward
    return tree


def policy_action(tree: dict, trace: list, legal: list,
                  rng: random.Random, stats: dict):
    stats['lookups'] += 1
    key = tuple(trace)
    node = tree.get(key)
    if node is None:
        stats['miss'] += 1
        return rng.choice(legal)
    candidates = [(a, node['children'][a]['N']) for a in legal
                  if a in node['children']]
    if not candidates or all(n == 0 for _, n in candidates):
        stats['miss'] += 1
        return rng.choice(legal)
    return max(candidates, key=lambda p: p[1])[0]


def _play_one_game(nfa, tree: dict, rng: random.Random, stats: dict):
    state = nfa.root
    trace: list = []
    while not state.is_terminal():
        if state.player == 'P1':
            action = rng.choice(list(state.children.keys()))
        else:
            action = policy_action(tree, trace, list(state.children.keys()),
                                   rng, stats)
        trace.append(action)
        state = state.children[action]
    return state.winner()


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Two-phase trained MCTS Nim player.'
    )
    parser.add_argument('--piles',        type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--train-K',      dest='train_K', type=int, default=1000)
    parser.add_argument('--c',            type=float, default=1.4)
    parser.add_argument('--games',        type=int, default=200)
    parser.add_argument('--seed',         type=int, default=0)
    parser.add_argument('--oracle-depth', dest='oracle_depth', type=int, default=None)
    args = parser.parse_args()

    nfa    = NimNFA(piles=tuple(args.piles))
    inner  = NimOracle(nfa, depth=args.oracle_depth)
    oracle = CountingOracle(inner)
    rng    = random.Random(args.seed)

    print(f'Trained-MCTS Nim  piles={tuple(args.piles)}  train_K={args.train_K}  '
          f'c={args.c}  games={args.games}  oracle_depth={args.oracle_depth}')
    print()

    t0 = oracle.total_queries
    tree = train_uct(nfa, oracle, args.train_K, args.c, rng)
    train_calls = oracle.total_queries - t0
    print(f'Training:\n  rollouts     : {args.train_K}\n  '
          f'oracle calls : {train_calls}\n  tree size    : {len(tree)} states\n')

    e0 = oracle.total_queries
    fallback = {'lookups': 0, 'miss': 0}
    wins = draws = losses = 0
    for _ in range(args.games):
        w = _play_one_game(nfa, tree, rng, fallback)
        if   w == 'P2': wins   += 1
        elif w == 'P1': losses += 1
        else:           draws  += 1
    eval_calls = oracle.total_queries - e0

    n = wins + draws + losses
    miss_rate = (fallback['miss'] / fallback['lookups']
                 if fallback['lookups'] else 0.0)
    print('Evaluation vs random P1:')
    print(f'  wins={wins}  draws={draws}  losses={losses}')
    print(f'  win rate={wins/n:.1%}  loss rate={losses/n:.1%}')
    print(f'  P2 lookups   : {fallback["lookups"]}')
    print(f'  out-of-tree  : {miss_rate:.1%}')
    print(f'  oracle calls : {eval_calls}  (should be 0)')
    print()
    print(f'Total preference-oracle calls: {oracle.total_queries}')


if __name__ == '__main__':
    main()
