"""
Main entry point: learn P2's strategy automaton for a random minimax game.

Usage:
    python -m src.scripts.random_game.learner 4
    python -m src.scripts.random_game.learner 4 --seed 42 --depth-n 2 --K 100 --verbose
"""

import argparse
from pathlib import Path
from aalpy.learning_algs import run_Lstar
from aalpy.utils import visualize_automaton

from src.game.minimax.game_generator import generate_tree, print_tree, compute_trace_scores
from src.game.minimax.game_nfa import GameNFA
from src.lstar_mcts.preference_oracle import PreferenceOracle
from src.lstar_mcts.game_sul import GameSUL
from src.lstar_mcts.table_b import TableB
from src.lstar_mcts.mcts_oracle import MCTSEquivalenceOracle


def main():
    parser = argparse.ArgumentParser(
        description='Learn P2 strategy automaton via L* + MCTS equivalence oracle.'
    )
    parser.add_argument('depth',    type=int,            help='Game tree depth')
    parser.add_argument('--seed',   type=int, default=0, help='Random seed')
    parser.add_argument('--depth-n', dest='depth_n', type=int, default=None,
                        help='MCTS search depth N (default: game depth)')
    parser.add_argument('--K',      type=int, default=200,
                        help='MCTS rollout budget per equivalence query')
    parser.add_argument('--epsilon', type=float, default=0.05,
                        help='Min advantage to accept a counterexample')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    depth_n = args.depth_n if args.depth_n is not None else args.depth

    # ---------------------------------------------------------------
    # Build game
    # ---------------------------------------------------------------
    print(f'Game tree  depth={args.depth}  seed={args.seed}')
    root = generate_tree(args.depth, seed=args.seed)
    print_tree(root)
    print()

    # ---------------------------------------------------------------
    # Component setup
    # ---------------------------------------------------------------
    nfa    = GameNFA(root)
    oracle = PreferenceOracle(nfa)
    sul    = GameSUL(nfa, oracle)
    table_b = TableB()

    eq_oracle = MCTSEquivalenceOracle(
        sul       = sul,
        nfa       = nfa,
        oracle    = oracle,
        table_b   = table_b,
        depth_N   = depth_n,
        K         = args.K,
        epsilon   = args.epsilon,
        verbose   = args.verbose,
    )

    # Input alphabet for L*: P1's moves (A, B at root)
    p1_inputs = list(root.children.keys())

    # ---------------------------------------------------------------
    # Run L* in a loop: each iteration learns the current oracle,
    # then MCTS checks for improvements.  On improvement, the oracle
    # is updated and L* restarts from scratch so its table is always
    # consistent with the (now-updated) SUL.
    # ---------------------------------------------------------------
    print(f'Running L*  MCTS depth={depth_n}  K={args.K}  epsilon={args.epsilon}')
    print()

    mcts_rounds = 0
    model = None
    while True:
        mcts_rounds += 1
        # L* converges: hypothesis fully matches current oracle strategy
        model = run_Lstar(
            alphabet          = p1_inputs,
            sul               = sul,
            eq_oracle         = eq_oracle,
            automaton_type    = 'mealy',
            print_level       = 2 if args.verbose else 0,
            cache_and_non_det_check = False,
        )
        # MCTS: run K rollouts on the converged hypothesis
        # Note: _deviation_leaves persists across rounds so evidence accumulates
        remaining = {None: args.K}
        for _ in range(args.K):
            eq_oracle._rollout(model, remaining)

        # Check if any deviation beats its shadow — if so, update oracle and restart
        improvement = eq_oracle._check_for_improvement(model)
        if args.verbose:
            print(f'  MCTS round {mcts_rounds}: states={len(model.states)}, '
                  f'improvement={improvement is not None}')
        if improvement is None:
            break   # converged — no better strategy found

    # ---------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------
    print()
    print('Learned automaton:')
    print(f'  States : {len(model.states)}')
    print(f'  Queries: membership={sul.num_queries}  '
          f'equivalence={eq_oracle.num_queries}')
    print()

    scores = evaluate(model, root)
    print(f'Score (mean over all P1 sequences):')
    print(f'  Optimal  : {scores["optimal_mean"]:.2f}')
    print(f'  Learned  : {scores["learned_mean"]:.2f}')
    print(f'  Random   : {scores["random_mean"]:.2f}')
    print(f'  Normalised: {scores["normalised"]:.3f}  '
          f'(0=random, 1=optimal)')
    print()
    print(table_b.summary())

    out_dir = Path(__file__).parents[3] / 'outputs'
    out_dir.mkdir(exist_ok=True)
    model.save(str(out_dir / 'learned_strategy'))
    print(f'Saved: {out_dir / "learned_strategy.dot"}')


def evaluate(model, root) -> dict:
    """
    Score the learned strategy against optimal and random baselines.

    For every possible P1 input sequence we compute the cumulative score
    achieved by:
      - optimal  P2  (backward induction — best P2 response at every node)
      - learned  P2  (the model's output)
      - random   P2  (uniform random choice at every P2 node)

    Returns mean scores and a normalised quality in [0, 1].
    """
    from src.game.minimax.game_generator import GameNode
    import random as rng

    nfa = GameNFA(root)

    # --- Optimal P2 score via backward induction ---
    def optimal_score(node: GameNode) -> float:
        if node.is_terminal():
            return node.value
        child_scores = {a: optimal_score(c) for a, c in node.children.items()}
        if node.player == 'P2':
            best = max(child_scores.values())
        else:
            # P1 is environment — we average over all P1 inputs
            best = sum(child_scores.values()) / len(child_scores)
        return node.value + best

    # --- Enumerate all P1 input sequences and score each strategy ---
    def all_p1_sequences(node: GameNode, seq: list) -> list:
        """Recursively collect all P1 input sequences to terminal states."""
        if node.is_terminal():
            return [list(seq)]
        results = []
        if node.player == 'P1':
            for action, child in node.children.items():
                results.extend(all_p1_sequences(child, seq + [action]))
        else:
            # P2 node: doesn't branch P1's sequence, just pass through
            for child in node.children.values():
                results.extend(all_p1_sequences(child, seq))
        return results

    def score_for_strategy(p1_sequence: list, p2_choice_fn) -> float:
        """
        Follow a P1 input sequence and use p2_choice_fn to pick P2 moves.
        p2_choice_fn(trace) -> P2 action
        Returns cumulative score.
        """
        node  = root
        trace = []
        total = node.value
        p1_idx = 0

        while not node.is_terminal():
            if node.player == 'P1':
                if p1_idx >= len(p1_sequence):
                    break
                action = p1_sequence[p1_idx]
                p1_idx += 1
            else:
                action = p2_choice_fn(trace)

            if action not in node.children:
                break
            trace.append(action)
            node   = node.children[action]
            total += node.value

        return total

    p1_seqs = all_p1_sequences(root, [])

    optimal_scores = []
    learned_scores = []
    random_scores  = []

    for p1_seq in p1_seqs:
        # Optimal: always pick the child with highest subtree value
        def opt_p2(trace, _root=root):
            node = nfa.get_node(trace)
            if node is None:
                return None
            return max(node.children,
                       key=lambda a: optimal_score(node.children[a]))

        # Learned: use the trained model
        model.reset_to_initial()
        _model_outputs = iter(model.step(p) for p in p1_seq)

        def learned_p2(trace, _it=_model_outputs):
            try:
                return next(_it)
            except StopIteration:
                return None

        # Random: uniform choice
        def rand_p2(trace):
            node = nfa.get_node(trace)
            if node is None:
                return None
            return rng.choice(list(node.children.keys()))

        # Re-init learned iterator properly per sequence
        model.reset_to_initial()
        learned_iter = [model.step(p) for p in p1_seq]
        li = iter(learned_iter)

        optimal_scores.append(score_for_strategy(p1_seq, opt_p2))
        learned_scores.append(score_for_strategy(p1_seq,
                                                  lambda t, it=iter(learned_iter): next(it, None)))
        random_scores.append(score_for_strategy(p1_seq, rand_p2))

    opt_mean  = sum(optimal_scores) / len(optimal_scores)
    lrn_mean  = sum(learned_scores) / len(learned_scores)
    rnd_mean  = sum(random_scores)  / len(random_scores)

    if opt_mean == rnd_mean:
        normalised = 1.0
    else:
        normalised = (lrn_mean - rnd_mean) / (opt_mean - rnd_mean)

    return {
        'optimal_mean':  opt_mean,
        'learned_mean':  lrn_mean,
        'random_mean':   rnd_mean,
        'normalised':    normalised,
        'n_sequences':   len(p1_seqs),
    }


if __name__ == '__main__':
    main()
