"""
Core L* + MCTS learning algorithm — game-agnostic.

Call run_lstar_mcts() from a game-specific script, passing in the game's
NFA, preference oracle, and P1 input alphabet.  All game construction,
CLI parsing, and evaluation logic live in the caller.

Example
-------
    from src.lstar_mcts.learner import run_lstar_mcts

    model, sul, mcts, table_b = run_lstar_mcts(
        nfa       = nfa,
        oracle    = oracle,
        p1_inputs = list(nfa.root.children.keys()),
        depth_n   = 4,
        K         = 200,
        verbose   = True,
    )
    print(f'States: {len(model.states)}')
    print(f'Membership queries: {sul.num_queries}')
    print(f'Equivalence queries: {mcts.num_queries}')
"""

from src.lstar_mcts.game_sul          import GameSUL
from src.lstar_mcts.table_b           import TableB
from src.lstar_mcts.mcts_oracle       import MCTSEquivalenceOracle
from src.lstar_mcts.pac_oracle        import PACEqOracle
from src.lstar_mcts.composite_oracle  import CompositeEqOracle
from src.lstar_mcts.custom_lstar      import MealyLStar


def run_lstar_mcts(
    nfa,
    oracle,
    p1_inputs: list,
    depth_n: int,
    K: int             = 200,
    epsilon: float     = 0.05,
    temperature: float = 1.0,
    verbose: bool      = False,
    use_pac:        bool  = True,
    pac_eps:        float = 0.05,
    pac_delta:      float = 0.05,
    pac_max_walk:   int   = 20,
):
    """
    Run the L* + MCTS learning loop and return the learned Mealy machine.

    Parameters
    ----------
    nfa          : game NFA with .root, .get_node(), .p2_legal_moves()
    oracle       : preference oracle with .preferred_move() and .compare()
    p1_inputs    : list of P1's alphabet symbols
    depth_n      : MCTS rollout depth
    K            : MCTS rollout budget per equivalence query
    epsilon      : exploration parameter for Table B UCB
    temperature  : softmax temperature for Table B sampling
    verbose      : whether to print L* hypothesis sizes
    use_pac      : if True, run PAC validation after every MCTS phase
                   (CompositeEqOracle); if False, use bare MCTSEquivalenceOracle
    pac_eps      : PAC error tolerance (fraction of D)
    pac_delta    : PAC failure probability per round
    pac_max_walk : max P1 inputs per sampled NFA walk

    Returns
    -------
    (model, sul, eq_oracle, table_b)
        eq_oracle is the CompositeEqOracle when use_pac=True, otherwise
        the bare MCTSEquivalenceOracle. The MCTS and PAC sub-oracles can
        be reached as eq_oracle.mcts and eq_oracle.pac respectively.
    """
    table_b = TableB()
    sul     = GameSUL(nfa, oracle, table_b)

    mcts = MCTSEquivalenceOracle(
        sul         = sul,
        nfa         = nfa,
        oracle      = oracle,
        table_b     = table_b,
        depth_N     = depth_n,
        K           = K,
        epsilon     = epsilon,
        temperature = temperature,
        verbose     = verbose,
    )

    if use_pac:
        pac = PACEqOracle(
            alphabet       = p1_inputs,
            sul            = sul,
            nfa            = nfa,
            eps            = pac_eps,
            delta          = pac_delta,
            max_walk_depth = pac_max_walk,
        )
        eq_oracle = CompositeEqOracle(mcts, pac, verbose=verbose)
    else:
        eq_oracle = mcts

    lstar = MealyLStar(
        alphabet  = p1_inputs,
        sul       = sul,
        eq_oracle = eq_oracle,
        verbose   = verbose,
    )

    model = lstar.run()
    return model, sul, eq_oracle, table_b
