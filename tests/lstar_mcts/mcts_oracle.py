"""
Unit tests for MCTSEquivalenceOracle (src/lstar_mcts/mcts_oracle.py).

Run Instructions:

'pytest tests/lstar_mcts/mcts_oracle.py -v'

"""

import pytest
from unittest.mock import MagicMock
from src.lstar_mcts.mcts_oracle import MCTSEquivalenceOracle
from src.game.minimax.game_generator import generate_tree
from src.game.minimax.game_nfa import GameNFA
from src.game.minimax.preference_oracle import PreferenceOracle
from src.lstar_mcts.game_sul import GameSUL
from src.lstar_mcts.table_b import TableB
from aalpy.base import Oracle
from aalpy.learning_algs import run_Lstar
class _NoCexOracle(Oracle):                                                                                                                                                                                               
    def find_cex(self, hypothesis):
        return None  

@pytest.fixture
def oracle():   
    root   = generate_tree(depth=10, seed=41)
    nfa    = GameNFA(root)
    pref   = PreferenceOracle(nfa)
    tableb = TableB()                                                                                                                                                                                                     
    sul    = GameSUL(nfa, pref, tableb)
                                                                                                                                                                                                                        
    p1_alphabet = list(getattr(nfa, 'p1_alphabet', nfa.root.children.keys()))                                                                                                                                             
    hypothesis  = run_Lstar(p1_alphabet, sul, _NoCexOracle(p1_alphabet, sul),
                            automaton_type='mealy', print_level=0)                                                                                                                                                        
                
    o = MCTSEquivalenceOracle(sul, nfa, pref, tableb, depth_N=2)                                                                                                                                                          
    o.hypothesis = hypothesis
    return o 
class TestCheckRollout:
    #Generating sub traces from hypothesis language
    def test_hypothesis_trace_generator(self, oracle):
      for i in range(10):                                                                                                                                                                      
        sub_trace = oracle.GenerateSubTrace()
        print(sub_trace)                                                                                                                                                                                 
        p1_inputs = tuple(sub_trace[0::2])             
        print(p1_inputs)
        assert oracle.sul._cache[p1_inputs] == sub_trace[-1]   

    def test_deviation_points_reflect_ucb_scores(self):
        #TODO Check that the deviation points return
        #Reflected inversly to the UCB Scores
        return None

    def test_collect_deviation_traces_over_P1_actions(self):
        #TODO Check that the player one moves are over all input space on sub tree
        return None

    ### NEED TO BUILD TEST FUNCTIONS FOR THESE THINGS ###
    # TODO: test that CollectTraces returns None when only one action is available at deviation point
    # TODO: test that sampled deviation action is never the same as SubTrace[-1]
    # TODO: test that all returned traces end on a P2 action (never P1)
    # TODO: test that 'Terminal' is appended and zero_prob set when game ends after P1 action
    # TODO: test that P2 visits are recorded for all actions at each P2 node visited
    # TODO: test that returned traces have length <= depth_N * 2 (from deviation point)
    # TODO: test that P1 branching produces one trace per legal P1 action at each frontier step
    # TODO: test that completed traces are included in the return alongside frontier traces
    # TODO: assert that NFA_Node returned by get_current_state matches expected node for known trace
    # TODO: assert that current_state returned by get_current_state matches expected hypothesis state for known trace
    # TODO: assert that all returned traces in Generate_Hypothesis_Language start with SubTrace
    # TODO: assert that all returned traces end on a P2 action (never P1)
    # TODO: assert that returned trace count equals branching_factor^depth_N for a complete tree
    # TODO: assert that terminal traces are collected and not expanded further
    # TODO: assert that hypothesis outputs in returned traces match direct hypothesis.step() calls
    # TODO: assert that AssertionError is raised when a P2 node is encountered mid-BFS

    # TODO: test GenerateCounterExample returns (SubTrace, CE_Traces, majority) tuple
    # TODO: test GenerateCounterExample majority is True when CE traces are consistently preferred
    # TODO: test GenerateCounterExample majority is False when hypothesis traces are consistently preferred

    # TODO: test AssignPreferencesAndPreferenceValues returns majority=False when total_count=0
    # TODO: test AssignPreferencesAndPreferenceValues values dict contains both CE and HE trace keys
    # TODO: test AssignPreferencesAndPreferenceValues returns None values when constraints are contradictory

    # TODO: test PropagateValuesThroughTableB leaf values are written to Table B correctly
    # TODO: test PropagateValuesThroughTableB parent values equal average of children after propagation
    # TODO: test PropagateValuesThroughTableB stops propagating at SubTrace and does not go above it
    # TODO: test PropagateValuesThroughTableB handles single leaf trace correctly

    # TODO: test GenerateSubTrace returns trace ending on P2 action
    # TODO: test GenerateSubTrace returns full trace when no deviation candidates found (empty Table B)
    # TODO: test GenerateSubTrace chosen deviation index is biased toward low UCB weight actions

    # TODO: test get_current_state on empty trace returns root node and initial hypothesis state
    # TODO: test get_current_state stops correctly at terminal node mid-trace


class TestFindCex:
    """Tests for MCTSEquivalenceOracle.find_cex"""
    pass

# Will
class TestRollout:
    """Tests for MCTSEquivalenceOracle._rollout"""
    pass


# Removed in Last Commit
# class TestBudgetCheck:
#     """Tests for MCTSEquivalenceOracle._budget_check"""
#     pass

# Will
class TestShadowTrace:
    """Tests for MCTSEquivalenceOracle._shadow_trace"""
    pass

# Will
class TestCollectTableALeaves:
    """Tests for MCTSEquivalenceOracle._collect_table_a_leaves"""
    pass


class TestHypothesisOutput:
    """Tests for MCTSEquivalenceOracle._hypothesis_output

    Specifically looks at minimax as test:
    ```
        Hypothesis: digraph learnedModel {
            s0 [label="s0"];
            s1 [label="s1"];
            s2 [label="s2"];
            s0 -> s1 [label="A/Y"];
            s0 -> s1 [label="B/X"];
            s1 -> s2 [label="A/Y"];
            s1 -> s2 [label="B/X"];
            s2 -> s2 [label="A/None"];
            s2 -> s2 [label="B/None"];
            __start0 [shape=none, label=""];
            __start0 -> s0 [label=""];
        }
    ```
    """

    def test_returns_last_step_output(self, oracle):
        return None


    def test_returns_none_on_exception(self, oracle):
        return None 
