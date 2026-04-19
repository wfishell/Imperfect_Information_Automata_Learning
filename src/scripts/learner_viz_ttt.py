"""
Visualized L* strategy learner for Tic-Tac-Toe.

Usage:
    python -m src.scripts.learner_viz_ttt
    python -m src.scripts.learner_viz_ttt --depth-n 5 --K 200
"""

import argparse
import random
from pathlib import Path
from aalpy.learning_algs import run_Lstar

from src.game.tic_tac_toe.game_nfa import TicTacToeNFA
from src.game.tic_tac_toe.preference_oracle import TicTacToeOracle
from src.lstar_mcts.game_sul import GameSUL
from src.lstar_mcts.table_b import TableB
from src.lstar_mcts.mcts_oracle import MCTSEquivalenceOracle
from src.viz.visualizer import Visualizer


def main():
    parser = argparse.ArgumentParser(
        description='L* + MCTS strategy learner for Tic-Tac-Toe with Rich visualisation.'
    )
    parser.add_argument('--depth-n', dest='depth_n', type=int, default=5,
                        help='MCTS search depth N (default: 5)')
    parser.add_argument('--K',       type=int,   default=200,
                        help='MCTS rollout budget per equivalence query')
    parser.add_argument('--epsilon', type=float, default=0.05,
                        help='Min advantage to accept a counterexample')
    parser.add_argument('--n-eval',  dest='n_eval', type=int, default=500,
                        help='Games to play when evaluating learned strategy')
    args = parser.parse_args()

    viz = Visualizer()

    nfa     = TicTacToeNFA()
    oracle  = TicTacToeOracle(nfa)
    sul     = GameSUL(nfa, oracle)
    table_b = TableB()

    eq_oracle = MCTSEquivalenceOracle(
        sul=sul, nfa=nfa, oracle=oracle, table_b=table_b,
        depth_N=args.depth_n, K=args.K, epsilon=args.epsilon,
        verbose=False,
    )

    p1_inputs = list(nfa.root.children.keys())  # [0..8]

    # Show the initial empty board
    viz.show_ttt_board(nfa.root, title="Tic-Tac-Toe — initial board")

    # ------------------------------------------------------------------
    # Run L*  (single pass — oracle is fixed minimax, no improvement loop)
    # ------------------------------------------------------------------
    viz.show_round_header(1)

    model = run_Lstar(
        alphabet                = p1_inputs,
        sul                     = sul,
        eq_oracle               = eq_oracle,
        automaton_type          = 'mealy',
        print_level             = 0,
        cache_and_non_det_check = False,
    )

    viz.show_hypothesis(model, p1_inputs)

    # Run MCTS rollouts so Table B is populated
    remaining = {None: args.K}
    for _ in range(args.K):
        eq_oracle._rollout(model, remaining)

    viz.show_table_b(eq_oracle.table_b)
    viz.show_deviations(eq_oracle._deviation_leaves)
    viz.show_improvement(None)   # oracle is minimax — always converged in one pass

    # ------------------------------------------------------------------
    # Evaluate vs random X
    # ------------------------------------------------------------------
    losses, draws, wins = _eval_vs_random(model, nfa, n_games=args.n_eval, seed=0)
    viz.show_ttt_eval(losses, draws, wins, n_games=args.n_eval)

    # Show a sample end-state from O winning
    sample_state = _sample_o_win(nfa, oracle)
    if sample_state:
        viz.show_ttt_board(sample_state, title="Sample O win (minimax optimal)")

    # ------------------------------------------------------------------
    # Final summary and save
    # ------------------------------------------------------------------
    viz.show_final_summary(model, sul, eq_oracle, table_b)

    diagrams_dir = Path(__file__).parents[1] / 'viz' / 'diagrams'
    diagrams_dir.mkdir(parents=True, exist_ok=True)
    save_path = diagrams_dir / 'learned_strategy_ttt'
    model.save(str(save_path))
    viz.console.print(f"[dim]Saved: {save_path}.dot[/dim]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eval_vs_random(model, nfa: TicTacToeNFA, n_games: int,
                    seed: int) -> tuple[int, int, int]:
    """Return (losses, draws, wins) for learned O vs random X."""
    rng = random.Random(seed)
    losses = draws = wins = 0

    for _ in range(n_games):
        state = nfa.root
        model.reset_to_initial()

        while not state.is_terminal():
            p1_move = rng.choice(list(state.children.keys()))
            o_move  = model.step(p1_move)
            state   = state.children[p1_move]
            if state.is_terminal():
                break
            if o_move not in state.children:
                o_move = rng.choice(list(state.children.keys()))
            state = state.children[o_move]

        w = state.winner()
        if w == 'P1':   losses += 1
        elif w == 'P2': wins   += 1
        else:           draws  += 1

    return losses, draws, wins


def _sample_o_win(nfa: TicTacToeNFA, oracle: TicTacToeOracle):
    """Play oracle vs first-legal X and return terminal state if O wins."""
    state = nfa.root
    trace = []
    while not state.is_terminal():
        if state.player == 'P1':
            move = list(state.children.keys())[0]
        else:
            move = oracle.preferred_move(trace)
            if move is None:
                break
        trace.append(move)
        state = state.children[move]
    return state if state.winner() == 'P2' else None


if __name__ == '__main__':
    main()
