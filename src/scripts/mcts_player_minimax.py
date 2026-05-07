"""
Pure-MCTS Minimax baseline (no automaton learning).

At each P2 turn during play, the agent runs K UCB1-allocated rollouts
to terminal, scores each via pairwise oracle.compare() against the
running champion trace, and picks the most-visited action. There is no
L*, no automaton, no learning — every decision pays its own oracle cost.

UCB matches the exploration scheme used inside the L* equivalence
oracle (src/lstar_mcts/table_b.py), making per-move oracle cost
apples-to-apples with L*'s training-time cost.

Score is enumerated over every P1 sequence (matching learner_minimax's
evaluator) and normalised: 0 = random P2 baseline, 1 = optimal P2.

Usage:
    python -m src.scripts.mcts_player_minimax --depth 4 --seed 0 --K 20
    python -m src.scripts.mcts_player_minimax --game-file my_game.json --K 50
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
# UCB1 root-level P2 action selection
# ----------------------------------------------------------------------

def _random_rollout(nfa: GameNFA, trace: list, rng: random.Random) -> list:
    """Random play from `trace` until terminal; return the full terminal trace."""
    state = nfa.get_node(trace)
    while state is not None and not state.is_terminal():
        action = rng.choice(list(state.children.keys()))
        trace  = trace + [action]
        state  = state.children[action]
    return trace


def _mcts_p2_action(nfa: GameNFA, oracle, trace: list, K: int,
                    c: float, rng: random.Random) -> str:
    """
    Allocate K rollouts across legal P2 actions via UCB1, with a
    pairwise-tournament reward.

    Seeding: one rollout per arm, all pairs compared once via
    oracle.compare(). UCB phase: each subsequent pull picks the
    highest-ucb1 arm, does one rollout, compares it to a random stored
    rollout from another arm; visits/wins are updated symmetrically.
    Choice = arm with the most visits at the end.
    """
    state = nfa.get_node(trace)
    legal = list(state.children.keys())

    if len(legal) == 1:
        return legal[0]

    visits  = {a: 0 for a in legal}
    wins    = {a: 0 for a in legal}
    storage = {a: [] for a in legal}

    for a in legal:
        storage[a].append(_random_rollout(nfa, trace + [a], rng))
    for i, a1 in enumerate(legal):
        for a2 in legal[i + 1:]:
            pref = oracle.compare(storage[a1][0], storage[a2][0])
            visits[a1] += 1
            visits[a2] += 1
            if   pref == 't1': wins[a1] += 1
            elif pref == 't2': wins[a2] += 1

    for _ in range(max(0, K - len(legal))):
        total = sum(visits.values())
        def ucb(a):
            return wins[a] / visits[a] + c * math.sqrt(math.log(total) / visits[a])
        chosen = max(legal, key=ucb)

        t_new   = _random_rollout(nfa, trace + [chosen], rng)
        storage[chosen].append(t_new)
        other   = rng.choice([a for a in legal if a != chosen])
        t_other = rng.choice(storage[other])

        pref = oracle.compare(t_new, t_other)
        visits[chosen] += 1
        visits[other]  += 1
        if   pref == 't1': wins[chosen] += 1
        elif pref == 't2': wins[other]  += 1

    return max(legal, key=lambda a: visits[a])


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
        description='Pure-MCTS Minimax baseline (no automaton learning).'
    )
    parser.add_argument('--game-file', dest='game_file', type=str, default=None,
                        help='Path to JSON-serialized game tree (overrides --depth/--seed)')
    parser.add_argument('--depth',        type=int, default=4,
                        help='Game tree depth (ignored if --game-file is set)')
    parser.add_argument('--seed',         type=int, default=0,
                        help='RNG seed for tree generation AND rollout RNG')
    parser.add_argument('--K',            type=int,   default=20,
                        help='UCB1 rollout budget per P2 decision')
    parser.add_argument('--c',            type=float, default=1.4,
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

    p1_seqs = _all_p1_sequences(root, [])

    learned_scores: list = []
    optimal_scores: list = []
    random_scores:  list = []

    for p1_seq in p1_seqs:
        def mcts_p2(trace):
            return _mcts_p2_action(nfa, oracle, trace, args.K, args.c, rng)

        def opt_p2(trace):
            n = nfa.get_node(trace)
            return None if n is None else max(
                n.children, key=lambda a: _optimal_score(n.children[a]))

        def rand_p2(trace):
            n = nfa.get_node(trace)
            return None if n is None else rng.choice(list(n.children.keys()))

        learned_scores.append(_play(root, p1_seq, mcts_p2))
        optimal_scores.append(_play(root, p1_seq, opt_p2))
        random_scores.append(_play(root, p1_seq, rand_p2))

    n     = len(p1_seqs)
    opt   = sum(optimal_scores) / n
    lrn   = sum(learned_scores) / n
    rnd   = sum(random_scores)  / n
    norm  = 1.0 if opt == rnd else (lrn - rnd) / (opt - rnd)

    print(f'UCB1-MCTS Minimax  {source}  K={args.K}  c={args.c}')
    print()
    print('Score (mean over all P1 sequences):')
    print(f'  Optimal   : {opt:.3f}')
    print(f'  MCTS P2   : {lrn:.3f}')
    print(f'  Random    : {rnd:.3f}')
    print(f'  Normalised: {norm:.3f}  (0=random  1=optimal)')
    print()
    print('Preference-oracle calls:')
    print(f'  compare()        : {oracle.compare_calls}')
    print(f'  preferred_move() : {oracle.preferred_move_calls}')
    print(f'  total            : {oracle.total_queries}')
    print(f'  per P1 sequence  : {oracle.total_queries / n:.1f}')


if __name__ == '__main__':
    main()
