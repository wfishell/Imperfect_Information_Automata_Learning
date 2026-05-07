"""
Pure-MCTS Hex baseline (no automaton learning).

At each P2 turn during play, the agent runs K UCB1-allocated rollouts
to terminal, scores each via pairwise oracle.compare() against the
running champion trace, and picks the most-visited action. There is no
L*, no automaton, no learning — every decision pays its own oracle cost.

UCB matches the exploration scheme used inside the L* equivalence
oracle (src/lstar_mcts/table_b.py), making per-move oracle cost
apples-to-apples with L*'s training-time cost.

Usage:
    python -m src.scripts.mcts_player_hex --size 3 --K 20 --games 200
"""

import argparse
import math
import random

from src.game.hex.game_nfa          import HexNFA
from src.game.hex.preference_oracle import HexOracle
from src.lstar_mcts.counting_oracle import CountingOracle


# ----------------------------------------------------------------------
# UCB1 root-level P2 action selection
# ----------------------------------------------------------------------

def _random_rollout(nfa: HexNFA, trace: list, rng: random.Random) -> list:
    """Random play from `trace` until terminal; return the full terminal trace."""
    state = nfa.get_node(trace)
    while state is not None and not state.is_terminal():
        action = rng.choice(list(state.children.keys()))
        trace  = trace + [action]
        state  = state.children[action]
    return trace


def _mcts_p2_action(nfa: HexNFA, oracle, trace: list, K: int,
                    c: float, rng: random.Random) -> tuple:
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
# Game loop: MCTS P2 vs random P1
# ----------------------------------------------------------------------

def _play_one_game(nfa: HexNFA, oracle, K: int, c: float,
                   rng: random.Random) -> str | None:
    state = nfa.root
    trace: list = []
    while not state.is_terminal():
        if state.player == 'P1':
            action = rng.choice(list(state.children.keys()))
        else:
            action = _mcts_p2_action(nfa, oracle, trace, K, c, rng)
        trace.append(action)
        state = state.children[action]
    return state.winner()


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Pure-MCTS Hex baseline (no automaton learning).'
    )
    parser.add_argument('--size',         type=int,   default=3,
                        help='Hex board size (default: 3)')
    parser.add_argument('--K',            type=int,   default=20,
                        help='UCB1 rollout budget per P2 decision')
    parser.add_argument('--c',            type=float, default=1.4,
                        help='UCB1 exploration constant (default: 1.4 — matches table_b)')
    parser.add_argument('--games',        type=int,   default=200,
                        help='number of evaluation games (vs random P1)')
    parser.add_argument('--seed',         type=int,   default=0)
    parser.add_argument('--oracle-depth', dest='oracle_depth', type=int, default=None,
                        help='Minimax lookahead for the oracle (default: full search)')
    args = parser.parse_args()

    nfa    = HexNFA(size=args.size)
    inner  = HexOracle(nfa, depth=args.oracle_depth)
    oracle = CountingOracle(inner)
    rng    = random.Random(args.seed)

    wins = draws = losses = 0
    for _ in range(args.games):
        w = _play_one_game(nfa, oracle, args.K, args.c, rng)
        if   w == 'P2': wins   += 1
        elif w == 'P1': losses += 1
        else:           draws  += 1

    n = wins + draws + losses

    print(f'UCB1-MCTS Hex  size={args.size}  K={args.K}  c={args.c}  games={args.games}  '
          f'oracle_depth={args.oracle_depth}')
    print()
    print('Results vs random P1:')
    print(f'  wins={wins}  draws={draws}  losses={losses}')
    print(f'  win rate={wins/n:.1%}  loss rate={losses/n:.1%}')
    print()
    print('Preference-oracle calls:')
    print(f'  compare()        : {oracle.compare_calls}')
    print(f'  preferred_move() : {oracle.preferred_move_calls}')
    print(f'  total            : {oracle.total_queries}')
    print(f'  per game         : {oracle.total_queries / args.games:.1f}')


if __name__ == '__main__':
    main()
