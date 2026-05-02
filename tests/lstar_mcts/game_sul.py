"""
Unit tests for GameSUL (src/lstar_mcts/game_sul.py).

Run Instructions:
    pytest tests/lstar_mcts/game_sul.py -v
"""
#TODO ADD THE TABLE B to this 
import pytest
from src.lstar_mcts.game_sul import GameSUL
from src.lstar_mcts.preference_oracle import PreferenceOracle
from src.game.minimax.game_generator import generate_tree
from src.game.minimax.game_nfa import GameNFA


@pytest.fixture
def setup():
    root   = generate_tree(depth=2, seed=42)
    nfa    = GameNFA(root)
    oracle = PreferenceOracle(nfa)
    sul    = GameSUL(nfa, oracle)
    return sul, nfa, oracle


# ----------------------------------------------------------------------
# pre / post
# ----------------------------------------------------------------------
class TestPrePost:
    def test_pre_post(self):
        #test that pre() resets _trace to empty
        root   = generate_tree(depth=10, seed=42)
        nfa    = GameNFA(root)
        oracle = PreferenceOracle(nfa)
        sul    = GameSUL(nfa, oracle)
        sul.step('A')
        sul.step('B')
        assert sul._trace !=[]
        sul.pre()
        assert sul._trace == []
    def test_pre_reset_current_p1(self):
        # test that pre() resets _current_p1 to empty
        root   = generate_tree(depth=10, seed=42)
        nfa    = GameNFA(root)
        oracle = PreferenceOracle(nfa)
        sul    = GameSUL(nfa, oracle)
        sul.step('A')
        sul.step('B') 
        sul.step('A')       
        sul.pre()
        assert sul._current_p1 == []
    def test_post_nothing_no_side_effects(self):
        # test that post() does nothing (no side effects)
        root   = generate_tree(depth=10, seed=42)
        nfa    = GameNFA(root)
        oracle = PreferenceOracle(nfa)
        sul    = GameSUL(nfa, oracle)
        sul.step('A')
        sul.step('B') 
        sul.step('A')
        pre_artifact = sul._current_p1
        sul.post()
        post_artifact = sul._current_p1
        assert pre_artifact==post_artifact    
    


# ----------------------------------------------------------------------
# step
# ----------------------------------------------------------------------
class TestStep:
    def test_step_returns_valid_p2_for_known_p1(self):
        # test that step() returns a valid P2 action for a known P1 input
        root   = generate_tree(depth=10, seed=42)
        nfa    = GameNFA(root)
        oracle = PreferenceOracle(nfa)
        sul    = GameSUL(nfa, oracle)
        inputs=['A','B','B','A']
        for input in inputs:
            assert sul.step(input) in ['X', 'Y']

    def test_step_builds_interleaved_trace(self):
        # test that step() builds the interleaved trace correctly after multiple steps
        root   = generate_tree(depth=5, seed=42)
        nfa    = GameNFA(root)
        oracle = PreferenceOracle(nfa)
        sul    = GameSUL(nfa, oracle)

        p1_inputs = ['A', 'B', 'A', 'B']
        for inp in p1_inputs:
            sul.step(inp)

        # trace alternates P1, P2, P1, P2...
        # even indices are P1 actions, odd indices are P2 responses
        # stop at terminal nodes (step returns None there, trace stops alternating)
        for i, action in enumerate(sul._trace):
            node = nfa.get_node(sul._trace[:i])
            if node is None or node.is_terminal():
                break
            expected_player = 'P1' if i % 2 == 0 else 'P2'
            assert node.player == expected_player

    def test_step_Returns_None_When_No_legal_moves(self):
        #test that step() returns None when there are no legal P2 moves
        root   = generate_tree(depth=3, seed=42)
        nfa    = GameNFA(root)
        oracle = PreferenceOracle(nfa)
        sul    = GameSUL(nfa, oracle)
        sul.step('A')
        assert sul.step('B') is None

    def test_step_uses_cache_on_second_call_with_same_P1_input_seq(self):
        #test that step() uses cache on second call with same P1 input sequence
        root   = generate_tree(depth=10, seed=42)
        nfa    = GameNFA(root)
        oracle = PreferenceOracle(nfa)
        sul    = GameSUL(nfa, oracle)
        sul.step('A')
        sul.step('B')
        sul.step('B')
        cache=sul._cache
        sul.pre()
        sul.step('A')
        sul.step('B')
        sul.step('B')
        cache_2=sul._cache
        assert cache == cache_2

    def test_step_uses_cache_on_second_call_with_same_P1_input_seq(self):                                                                                                                                                                                                   
      root   = generate_tree(depth=10, seed=42)                                                                                                                                                                                                                           
      nfa    = GameNFA(root)                                                                                                                                                                                                                                              
      oracle = PreferenceOracle(nfa)                                                                                                                                                                                                                                      
      sul    = GameSUL(nfa, oracle)                                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                          
      # first query — populates cache                                                                                                                                                                                                                                     
      sul.pre()                                                                                                                                                                                                                                                           
      sul.step('A')                                         
      sul.step('B')

      # replace preferred_move with a counting wrapper                                                                                                                                                                                                                    
      call_count = [0]
      original = oracle.preferred_move                                                                                                                                                                                                                                    
      def counting_preferred_move(trace):                                                                                                                                                                                                                                 
          call_count[0] += 1
          return original(trace)                                                                                                                                                                                                                                          
      oracle.preferred_move = counting_preferred_move       
                                                                                                                                                                                                                                                                          
      # second identical query
      sul.pre()                                                                                                                                                                                                                                                           
      sul.step('A')                                         
      sul.step('B')

      assert call_count[0] == 0  # oracle never called — cache was used  

    def test_step_falls_through_to_oracle_when_no_override(self):
        # test that step() falls through to oracle when no override exists
                                                                                                                                                                                                        
        root   = generate_tree(depth=10, seed=42)             
        nfa    = GameNFA(root)                                                                                                                                                                                                                                              
        oracle = PreferenceOracle(nfa)
        sul    = GameSUL(nfa, oracle)                                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                            
        call_count = [0]
        original = oracle.preferred_move                                                                                                                                                                                                                                    
        def counting_preferred_move(trace):                   
            call_count[0] += 1
            return original(trace)
        oracle.preferred_move = counting_preferred_move
                                                                                                                                                                                                                                                                            
        sul.step('A')
                                                                                                                                                                                                                                                                            
        assert call_count[0] == 1  # oracle was called exactly once


# ----------------------------------------------------------------------
# caching
# ----------------------------------------------------------------------
    def test_repeated_step_with_same_inputs(self):
        # test that repeated step() calls with same inputs return same P2 response
        root   = generate_tree(depth=10, seed=42)             
        nfa    = GameNFA(root)                                                                                                                                                                                                                                              
        oracle = PreferenceOracle(nfa)
        sul    = GameSUL(nfa, oracle)
        sul.step('A')
        sul.step('B')
        sul.step('A')
        trace=sul._trace
        sul.pre()
        sul.step('A')
        sul.step('B')
        sul.step('A')
        trace_2=sul._trace
        assert trace == trace_2
    def test_cache_cleared_after_update_strat(self):                                                                                                                                                                                                                        
      root   = generate_tree(depth=10, seed=42)                                                                                                                                                                                                                           
      nfa    = GameNFA(root)                                                                                                                                                                                                                                              
      oracle = PreferenceOracle(nfa)
      sul    = GameSUL(nfa, oracle)                                                                                                                                                                                                                                       
  
      sul.step('A')  # _trace becomes ['A', 'Y'] — 'A' is the P2 decision point                                                                                                                                                                                           
      assert sul._cache != {}                                                                                                                                                                                                                                             
      sul.pre()                                              
      # override at ['A'] with whatever the *other* P2 move is                                                                                                                                                                                                            
      current = sul.current_strategy(['A'])   # what P2 does now
      p2_moves = nfa.p2_legal_moves(['A'])                                                                                                                                                                                                                                
      different = [m for m in p2_moves if m != current][0]                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                         
      sul.update_strategy(['A'], different)   # changes the answer → cache must clear                                                                                                                                                                                     
                                                            
      assert sul._cache == {}
      sul.step('A')
      print(sul._cache)
    def test_cache_not_cleared_same_strat(self):
    # test that cache is NOT cleared when update_strategy() sets the same value
      root   = generate_tree(depth=10, seed=42)                                                                                                                                                                                                                           
      nfa    = GameNFA(root)                                                                                                                                                                                                                                              
      oracle = PreferenceOracle(nfa)
      sul    = GameSUL(nfa, oracle)                                                                                                                                                                                                                                       
  
      sul.step('A')  # _trace becomes ['A', 'Y'] — 'A' is the P2 decision point                                                                                                                                                                                           
      assert sul._cache != {}
      capture=sul.current_strategy(['A'])
      sul.update_strategy(['A'],capture)
      sul.step('A')
      sul.update_strategy(['A'],capture)
      assert sul._cache != {}                                                                                                                                                                                                                                      



# ----------------------------------------------------------------------
# update_strategy / overrides
# ----------------------------------------------------------------------
class TestUpdateStrategy:
    # test that update_strategy() overrides oracle response at given trace prefix
    def test_update_strategy_over_rides_oracle_response(self):
        root   = generate_tree(depth=10, seed=42)                                                                                                                                                                                                                           
        nfa    = GameNFA(root)                                                                                                                                                                                                                                              
        oracle = PreferenceOracle(nfa)
        sul    = GameSUL(nfa, oracle)                                                                                                                                                                                                                                       
  
        sul.step('A')                                                                                                                                                                                                                      
        p2_response = sul._trace[1]                                                                                                                                                                                                            
        sul.step('B')                                                                                                                                                                                                                                                           
        current = sul.current_strategy(['A', p2_response, 'B']) 
        sul.pre()
                                                                                                                                  
        p2_moves = nfa.p2_legal_moves(['A', p2_response, 'B'])

        different = [m for m in p2_moves if m != current][0]

        sul.update_strategy(['A', p2_response, 'B'], different)                                                                                                                                                                                                                 
                                    
        Preferred = oracle.preferred_move(['A', p2_response, 'B'])                                                                                                                                                                                                              
        current_strat = sul.current_strategy(['A', p2_response, 'B'])
        assert Preferred != current_strat        

    def test_update_strategy_with_same_value(self):       
        #test that update_strategy() with same value does not clear cache
                                                                                                                                                                                                                  
        root   = generate_tree(depth=10, seed=42)                                                                                                                                                                                                                           
        nfa    = GameNFA(root)                                                                                                                                                                                                                                              
        oracle = PreferenceOracle(nfa)
        sul    = GameSUL(nfa, oracle)                                                                                                                                                                                                                                       
    
        sul.step('A')                                                                                                                                                                                                                                                       
        p2_response = sul._trace[1]                           
        sul.step('B')
        current = sul.current_strategy(['A', p2_response, 'B'])
        sul.update_strategy(['A', p2_response, 'B'], current)                                                                                                                                                                               
        sul.pre()
        sul.step('A')                                                                                                                                                                                                                                                       
        sul.step('B')                                       
                                                                                                                                                                                                                                                                            
        cache_before = dict(sul._cache)
        assert cache_before != {}                                                                                                                                                                                                                 
        sul.update_strategy(['A', p2_response, 'B'], current)  
        assert sul._cache == cache_before 

    def test_multiple_overrides_at_different_prefixes(self):
        root   = generate_tree(depth=10, seed=42)                                                                                   
        nfa    = GameNFA(root)
        oracle = PreferenceOracle(nfa)                                                                                                                                                                                                                                      
        sul    = GameSUL(nfa, oracle)
                                                                                                                                                                                                                                                                            
        sul.step('A')                                                                                                                                                                                                                                                       
        p2_after_a = sul._trace[1]                                                                                                                                                                                                        
        sul.step('B')                                                                                                                                                                                                                                                       
        p2_after_ab = sul._trace[3]      
                                                                                                                                                                                                                                                                            
        prefix_1 = ['A']                                      
        prefix_2 = ['A', p2_after_a, 'B']                                                                                                                                                                                                                                   
                                                                
        override_1 = [m for m in nfa.p2_legal_moves(prefix_1) if m != sul.current_strategy(prefix_1)][0]
        override_2 = [m for m in nfa.p2_legal_moves(prefix_2) if m != sul.current_strategy(prefix_2)][0]                                                                                                                                                                    
                                                                                                                                                                                                                                                                            
        sul.update_strategy(prefix_1, override_1)                                                                                                                                                                                                                           
        sul.update_strategy(prefix_2, override_2)                                                                                                                                                                                                                           
                                                                
        assert sul.current_strategy(prefix_1) == override_1
        assert sul.current_strategy(prefix_2) == override_2

# ----------------------------------------------------------------------
# current_strategy
# ----------------------------------------------------------------------
class TestCurrentStrategy:
    def test_current_strategy_returns_override_when_one_exists(self):
        # test that current_strategy() returns override when one exists

        root   = generate_tree(depth=10, seed=42)                                                                                                                                                                                                                           
        nfa    = GameNFA(root)                                                                                                                                                                                                                                              
        oracle = PreferenceOracle(nfa)
        sul    = GameSUL(nfa, oracle)

        sul.step('B')
        p2_move=sul._trace[1]
        sul.step('A')

        current=sul.current_strategy(['B',p2_move,'A'])
        p2_moves = nfa.p2_legal_moves(['B', p2_move, 'A'])

        different = [m for m in p2_moves if m != current][0]
        
        sul.update_strategy(['B',p2_move,'A'], different)
        assert different == sul.current_strategy(['B',p2_move,'A'])

    def test_current_strategy_and_oracle_alignement(self):
        # test that current_strategy() falls back to oracle when no override exists
        root   = generate_tree(depth=10, seed=42)                                                                                                                                                                                                                           
        nfa    = GameNFA(root)                                                                                                                                                                                                                                              
        oracle = PreferenceOracle(nfa)
        sul    = GameSUL(nfa, oracle)

        sul.step('A')
        p2_move=sul._trace[1]
        sul.step('A')
        current=sul.current_strategy(['A',p2_move,'A'])
        pref=oracle.preferred_move(['A',p2_move,'A'])
        assert current==pref


# ----------------------------------------------------------------------
# p1_inputs_from_trace
# ----------------------------------------------------------------------
class TestP1Inputs:
    def test_p1_inputs_trace_extracts_every_other_ele_starting_at_0(self):
        #Not extracting P2 actions
        root   = generate_tree(depth=10, seed=42)                                                                                                                                                                                                                           
        nfa    = GameNFA(root)                                                                                                                                                                                                                                              
        oracle = PreferenceOracle(nfa)
        sul    = GameSUL(nfa, oracle)

        sul.step('A')
        sul.step('A')
        sul.step('B')
        sul.step('A')
        assert ['X','Y'] not in sul.p1_inputs_from_trace(sul._trace)
    def test_p1_inputs_on_empty_trace_returns_empty(self):
        #test that p1_inputs_from_trace on empty trace returns empty list
        root   = generate_tree(depth=10, seed=42)                                                                                                                                                                                                                           
        nfa    = GameNFA(root)                                                                                                                                                                                                                                              
        oracle = PreferenceOracle(nfa)
        sul    = GameSUL(nfa, oracle)
        sul.step('A')
        sul.pre()
        assert []==sul.p1_inputs_from_trace(sul._trace)

    def test_p1_inputs_from_trace_on_single_element_returns_single_element(self):
        root   = generate_tree(depth=10, seed=42)                                                                                                                                                                                                                           
        nfa    = GameNFA(root)                                                                                                                                                                                                                                              
        oracle = PreferenceOracle(nfa)
        sul    = GameSUL(nfa, oracle)
        sul.step('A')
        assert len(sul.p1_inputs_from_trace(sul._trace))==1
# TODO: test that p1_inputs_from_trace on single element returns that element


# ----------------------------------------------------------------------
# Table B integration
# ----------------------------------------------------------------------
class TestTableB:
    # TODO: test that after step('A'), the preferred P2 action has visits=1 in Table B
    #       at the trace prefix ['A']

    # TODO: test that after step('A'), all alternative P2 actions at ['A'] have visits=0
    #       in Table B (they exist as entries but were not visited)

    # TODO: test that after step('A'), step('B'), Table B has entries at both
    #       ['A'] and ['A', p2_after_a, 'B'] — one for each P2 decision point

    # TODO: test that a cache hit does NOT add a new visit to Table B —
    #       run the same P1 sequence twice, confirm visit count stays at 1

    # TODO: test that after update_strategy() changes the preferred action at some
    #       prefix, the next step() records a visit on the NEW preferred action,
    #       not the old one

    # TODO: test that the old preferred action (before override) keeps its old visit
    #       count in Table B after update_strategy() — it is not reset to 0

    # TODO: test that alternative actions (visits=0) return HIGH_PRIOR from ucb_score,
    #       confirming they will be prioritised for MCTS exploration
    pass