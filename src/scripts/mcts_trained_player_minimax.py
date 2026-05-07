"""
Two-phase MCTS Minimax player (apples-to-apples with L*).

PHASE 1 — TRAINING.
  From the initial state, run T UCB1-driven rollouts that build a search
  tree. Each rollout:
      SELECT  : descend the tree by UCB1 = W/N + c·√(ln N_parent / N_child)
      EXPAND  : add the first untried child as a new leaf
      ROLLOUT : random play from that leaf to terminal
      REWARD  : pairwise oracle.compare(rollout, prior) — 1 iff this
                terminal trace is preferred to a randomly-sampled prior
                rollout
      BACKUP  : walk the path, incrementing N(s,a) and W(s,a) by reward
  Pays oracle calls. Output: a populated tree (the artifact).

PHASE 2 — EVALUATION.
  Tree is frozen. At each P2 turn during a game, look up the current
  trace in the tree and play argmax_a N(s,a). Out-of-tree states fall
  back to uniform random and are counted. Pays ZERO oracle calls.

This is the apples-to-apples analogue of L*+MCTS: both pay oracle
calls upfront to produce a portable artifact, then play games for free.
The artifacts differ — L* yields a total Mealy machine, this script
yields a partial search tree — and that gap is the comparison.

Usage:
    python -m src.scripts.mcts_trained_player_minimax --depth 4 --train-K 1000
    python -m src.scripts.mcts_trained_player_minimax --game-file deep_game.json --train-K 5000
"""

import argparse
import json
import math
import random

from src.game.minimax.game_generator   import generate_tree, tree_from_dict, GameNode
from src.game.minimax.game_nfa         import GameNFA
from src.game.minimax.preference_oracle import PreferenceOracle
from src.lstar_mcts.counting_oracle    import CountingOracle


# ----------------------------------------------------------------------
# Phase 1: train a UCB tree from the initial state
# ----------------------------------------------------------------------

def train_uct(nfa: GameNFA, oracle, T: int, c: float,
              rng: random.Random) -> dict:
    """
    Run T UCB1-rollouts from nfa.root, returning a tree:
        tree[trace_tuple] = {
            'visits':   total visits at this node,
            'children': {action: {'N': int, 'W': int}}
        }
    """
    tree: dict       = {}
    buffer: list     = []   # all terminal rollouts, used as pairwise reward refs

    for _ in range(T):
        path: list   = []          # list of (trace_tuple, action) along selection
        trace: list  = []
        node         = nfa.root

        while True:
            if node is None or node.is_terminal():
                break
            key = tuple(trace)

            if key not in tree:
                tree[key] = {
                    'visits':   0,
                    'children': {a: {'N': 0, 'W': 0} for a in node.children},
                }
                # Newly-added node — break to enter rollout from here
                break

            children = tree[key]['children']
            untried  = [a for a, s in children.items() if s['N'] == 0]
            if untried:
                action = untried[0]
                path.append((key, action))
                trace.append(action)
                node = node.children[action]
                break  # expansion: rollout starts from this new leaf

            total_visits = tree[key]['visits']
            def ucb(a):
                s = children[a]
                return s['W'] / s['N'] + c * math.sqrt(math.log(total_visits) / s['N'])
            action = max(children.keys(), key=ucb)
            path.append((key, action))
            trace.append(action)
            node = node.children[action]

        # ROLLOUT
        rollout_trace = list(trace)
        roll_node = node
        while roll_node is not None and not roll_node.is_terminal():
            a = rng.choice(list(roll_node.children.keys()))
            rollout_trace.append(a)
            roll_node = roll_node.children[a]

        # REWARD via pairwise compare to a random prior rollout
        if buffer:
            prior  = rng.choice(buffer)
            pref   = oracle.compare(rollout_trace, prior)
            reward = 1 if pref == 't1' else 0
        else:
            reward = 1   # first rollout is its own baseline; doesn't affect ranking
        buffer.append(rollout_trace)

        # BACKUP
        for (key, action) in path:
            tree[key]['visits'] += 1
            tree[key]['children'][action]['N'] += 1
            tree[key]['children'][action]['W'] += reward

    return tree


# ----------------------------------------------------------------------
# Phase 2: frozen-policy P2 action lookup
# ----------------------------------------------------------------------

def policy_action(tree: dict, trace: list, legal: list,
                  rng: random.Random, stats: dict) -> str:
    """
    Look up trace in trained tree; pick argmax_a N(s,a) over legal moves.
    If state not in tree, fall back to random and increment stats['miss'].
    """
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


# ----------------------------------------------------------------------
# Score helpers (mirror learner_minimax.evaluate)
# ----------------------------------------------------------------------

def _all_p1_sequences(node: GameNode, seq: list) -> list:
    if node.is_terminal():
        return [list(seq)]
    if node.player == 'P1':
        return [s for a, c in node.children.items()
                for s in _all_p1_sequences(c, seq + [a])]
    return [s for c in node.children.values()
            for s in _all_p1_sequences(c, seq)]


def _optimal_score(node: GameNode) -> float:
    if node.is_terminal():
        return node.value
    cs = {a: _optimal_score(c) for a, c in node.children.items()}
    if node.player == 'P2':
        return node.value + max(cs.values())
    return node.value + sum(cs.values()) / len(cs)


def _play(root: GameNode, p1_seq: list, p2_choice_fn) -> float:
    node, trace, total, idx = root, [], root.value, 0
    while not node.is_terminal():
        if node.player == 'P1':
            if idx >= len(p1_seq): break
            action = p1_seq[idx]; idx += 1
        else:
            action = p2_choice_fn(trace)
        if action is None or action not in node.children:
            break
        trace.append(action)
        node   = node.children[action]
        total += node.value
    return total


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Two-phase trained MCTS Minimax player.'
    )
    parser.add_argument('--game-file', dest='game_file', type=str, default=None,
                        help='Path to JSON-serialized game tree (overrides --depth/--seed)')
    parser.add_argument('--depth',    type=int,   default=4,
                        help='Game tree depth (ignored if --game-file is set)')
    parser.add_argument('--seed',     type=int,   default=0,
                        help='RNG seed for tree generation AND training rollouts')
    parser.add_argument('--train-K',  dest='train_K', type=int, default=1000,
                        help='Total UCB rollouts during training (default: 1000)')
    parser.add_argument('--c',        type=float, default=1.4,
                        help='UCB1 exploration constant (default: 1.4 — matches table_b)')
    args = parser.parse_args()

    if args.game_file:
        with open(args.game_file) as f:
            root = tree_from_dict(json.load(f))
        source = f'file={args.game_file}'
    else:
        root   = generate_tree(args.depth, seed=args.seed)
        source = f'depth={args.depth}  seed={args.seed}'

    nfa    = GameNFA(root)
    inner  = PreferenceOracle(nfa)
    oracle = CountingOracle(inner)
    rng    = random.Random(args.seed)

    print(f'Trained-MCTS Minimax  {source}  train_K={args.train_K}  c={args.c}')
    print()

    # ----- PHASE 1: TRAIN -----
    train_calls_before = oracle.total_queries
    tree = train_uct(nfa, oracle, args.train_K, args.c, rng)
    train_calls = oracle.total_queries - train_calls_before

    print('Training:')
    print(f'  rollouts        : {args.train_K}')
    print(f'  oracle calls    : {train_calls}')
    print(f'  tree size       : {len(tree)} states')
    print()

    # ----- PHASE 2: EVAL (frozen policy, zero oracle calls) -----
    eval_calls_before = oracle.total_queries
    p1_seqs = _all_p1_sequences(root, [])
    fallback_stats = {'lookups': 0, 'miss': 0}

    learned_scores: list = []
    optimal_scores: list = []
    random_scores:  list = []

    for p1_seq in p1_seqs:
        def policy_p2(trace, _legal_fn=lambda t: list(nfa.get_node(t).children.keys())):
            return policy_action(tree, trace, _legal_fn(trace), rng, fallback_stats)

        def opt_p2(trace):
            n = nfa.get_node(trace)
            return None if n is None else max(
                n.children, key=lambda a: _optimal_score(n.children[a]))

        def rand_p2(trace):
            n = nfa.get_node(trace)
            return None if n is None else rng.choice(list(n.children.keys()))

        learned_scores.append(_play(root, p1_seq, policy_p2))
        optimal_scores.append(_play(root, p1_seq, opt_p2))
        random_scores.append(_play(root, p1_seq, rand_p2))

    eval_calls = oracle.total_queries - eval_calls_before
    n     = len(p1_seqs)
    opt   = sum(optimal_scores) / n
    lrn   = sum(learned_scores) / n
    rnd   = sum(random_scores)  / n
    norm  = 1.0 if opt == rnd else (lrn - rnd) / (opt - rnd)
    miss_rate = (fallback_stats['miss'] / fallback_stats['lookups']
                 if fallback_stats['lookups'] else 0.0)

    print('Evaluation:')
    print(f'  P1 sequences    : {n}')
    print(f'  P2 lookups      : {fallback_stats["lookups"]}')
    print(f'  out-of-tree rate: {miss_rate:.1%}  (fell back to random)')
    print(f'  oracle calls    : {eval_calls}  (should be 0)')
    print()
    print('Score (mean over all P1 sequences):')
    print(f'  Optimal   : {opt:.3f}')
    print(f'  MCTS P2   : {lrn:.3f}')
    print(f'  Random    : {rnd:.3f}')
    print(f'  Normalised: {norm:.3f}  (0=random  1=optimal)')
    print()
    print(f'Total preference-oracle calls: {oracle.total_queries}')


if __name__ == '__main__':
    main()
