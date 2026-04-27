"""
Unit tests for PreferenceOracle (src/lstar_mcts/preference_oracle.py).

Run Instructions:
    pytest tests/lstar_mcts/preference_oracle.py -v
"""

import pytest
from src.game.minimax.preference_oracle import PreferenceOracle
from src.game.minimax.game_generator import generate_tree
from src.game.minimax.game_nfa import GameNFA
from src.game.minimax.game_generator import GameNode

@pytest.fixture
def setup():
    root   = generate_tree(depth=2, seed=42)
    nfa    = GameNFA(root)
    oracle = PreferenceOracle(nfa)
    return oracle, nfa

class TestScore:
    def test_score_returns_cumalitive_node(self):
        # test _score returns correct cumulative node value sum for a known trace
        root   = generate_tree(depth=4, seed=42)
        nfa    = GameNFA(root)
        node=nfa.root
        oracle = PreferenceOracle(nfa)
        test_traces=[['A'],
                     ['A','X'],
                     ['B','X','A'],
                     ['A','Y','B','X'],
                     []]
        PreferenceOracle_Scores=[]
        for trace in test_traces:
            PreferenceOracle_Scores.append(oracle._score(trace))
        Manually_Calcualted_Score=[]
        for trace in test_traces:
            score=0
            temp_node=nfa.root
            for action in trace:
                score+=temp_node.value
                temp_node=temp_node.children[action]
            score+=temp_node.value
            Manually_Calcualted_Score.append(score)
        for i in range(len(Manually_Calcualted_Score)):
            assert Manually_Calcualted_Score[i]==PreferenceOracle_Scores[i]

    def test_score_empty_trace(self):
        # test _score on empty trace returns root node value only

        root   = generate_tree(depth=10, seed=2)
        nfa    = GameNFA(root)
        node=nfa.root
        oracle = PreferenceOracle(nfa)
        trace=[]
        assert node.value == oracle._score(trace)   


    def test_raise_key_error_invalid_traces(self):
        # test _score raises KeyError on invalid action in trace
        root   = generate_tree(depth=10, seed=2)
        nfa    = GameNFA(root)
        node=nfa.root
        oracle = PreferenceOracle(nfa)
        test_traces=[['X'],
                     ['A',''],
                     ['B','B','A'],
                     ['X','Y','B','X'],
                     ['','X']]
        for trace in test_traces:
            with pytest.raises(KeyError):
                oracle._score(trace=trace)

class TestPreferenceQueiries:

    def test_preffered_move_when_prefix_is_not_a_p2_node(self):
        #test preferred_move returns None when prefix is not a P2 node
        test_traces=[
                     ['A','X'],
                     ['A','Y','B','X'],
                     []]
        root   = generate_tree(depth=10, seed=2)
        nfa    = GameNFA(root)
        node=nfa.root
        oracle = PreferenceOracle(nfa)
        for trace in test_traces:
            assert oracle.preferred_move(trace) is None
    def test_preffered_move_none_in_terminal(self):
        # test preferred_move returns None when prefix is terminal
        root   = generate_tree(depth=4, seed=2)
        nfa    = GameNFA(root)
        node=nfa.root
        oracle = PreferenceOracle(nfa)  
        trace=['A','Y','B','X']
        assert oracle.preferred_move(trace) is None
    def test_preffered_move_none_in_terminal_2(self):
        # test preferred_move returns None when prefix is terminal
        root   = generate_tree(depth=3, seed=2)
        nfa    = GameNFA(root)
        node=nfa.root
        oracle = PreferenceOracle(nfa)  
        trace=['A','Y','B']
        assert oracle.preferred_move(trace) is None 
    def test_preffered_move_returns_max_move(self):
        # test preferred_move returns the action leading to highest cumulative score
        root   = generate_tree(depth=4, seed=2)
        nfa    = GameNFA(root)
        node=nfa.root
        oracle = PreferenceOracle(nfa)  
        trace=['A','Y','B']
        preference_move = oracle.preferred_move(trace)
        Optimal_Trace=trace.append(preference_move)
        non_optimal_move_traces=[]
        moves=nfa.p2_legal_moves(trace)
        for move in moves:
            if move!=preference_move:
                temp_trace=trace.append(move)
                non_optimal_move_traces.append(temp_trace)
        for trace in non_optimal_move_traces:
            assert oracle._score(Optimal_Trace)>=oracle._score(trace=trace)


    def test_prefferred_move_returns_consistent_tie_breaker(self):
        # test preferred_move breaks ties consistently

        root = GameNode(value=5, player='P1', depth=0)
        p2_node_a = GameNode(value=5, player='P2', depth=1)                                                                             
        p2_node_b = GameNode(value=5, player='P2', depth=1)
        x_node_a = GameNode(value=5, player='P1', depth=2)                                                                                                                                                                                                                      
        y_node_a = GameNode(value=5, player='P1', depth=2)
        x_node_b = GameNode(value=5, player='P1', depth=2)                                                                                                                                                                                                                      
        y_node_b = GameNode(value=5, player='P1', depth=2)
        p2_node_a.children['X'] = x_node_a                                                                                                                                                                                                                                      
        p2_node_a.children['Y'] = y_node_a                                                                                                                                                                                                                                      
        p2_node_b.children['X'] = x_node_b
        p2_node_b.children['Y'] = y_node_b                                                                                                                                                                                                                                      
        root.children['A'] = p2_node_a
        root.children['B'] = p2_node_b                                                                                                                                                                                                                                         
   
        nfa = GameNFA(root)                                                                                                                                                                                                                                                 
        oracle = PreferenceOracle(nfa)

        results = [oracle.preferred_move(['A']) for _ in range(10)]                                                                                                                                                                                                         
        assert len(set(results)) == 1
                                                 
class TestCompareTraces:
    def test_compare_returns_t1_when_greater(self):
        #test compare returns 't1' when trace1 has higher cumulative score
        
        root = GameNode(value=5, player='P1', depth=0)
        p2_node_a = GameNode(value=5, player='P2', depth=1)                                                                             
        p2_node_b = GameNode(value=5, player='P2', depth=1)
        x_node_a = GameNode(value=10, player='P1', depth=2)                                                                                                                                                                                                                      
        y_node_a = GameNode(value=5, player='P1', depth=2)
        x_node_b = GameNode(value=10, player='P1', depth=2)                                                                                                                                                                                                                      
        y_node_b = GameNode(value=5, player='P1', depth=2)
        p2_node_a.children['X'] = x_node_a                                                                                                                                                                                                                                      
        p2_node_a.children['Y'] = y_node_a                                                                                                                                                                                                                                      
        p2_node_b.children['X'] = x_node_b
        p2_node_b.children['Y'] = y_node_b                                                                                                                                                                                                                                      
        root.children['A'] = p2_node_a
        root.children['B'] = p2_node_b                                                                                                                                                                                                                                       
        
        nfa = GameNFA(root)                                                                                                                                                                                                                                                 
        oracle = PreferenceOracle(nfa)
        Trace_1=['A','X']
        Trace_2=['A','Y']
        assert oracle.compare(Trace_1,Trace_2) == 't1'
    def test_compare_returns_t2_when_greater(self):
        #test compare returns 't2' when trace1 has higher cumulative score
        
        root = GameNode(value=5, player='P1', depth=0)
        p2_node_a = GameNode(value=5, player='P2', depth=1)                                                                             
        p2_node_b = GameNode(value=5, player='P2', depth=1)
        x_node_a = GameNode(value=10, player='P1', depth=2)                                                                                                                                                                                                                      
        y_node_a = GameNode(value=15, player='P1', depth=2)
        x_node_b = GameNode(value=10, player='P1', depth=2)                                                                                                                                                                                                                      
        y_node_b = GameNode(value=15, player='P1', depth=2)
        p2_node_a.children['X'] = x_node_a                                                                                                                                                                                                                                      
        p2_node_a.children['Y'] = y_node_a                                                                                                                                                                                                                                      
        p2_node_b.children['X'] = x_node_b
        p2_node_b.children['Y'] = y_node_b                                                                                                                                                                                                                                      
        root.children['A'] = p2_node_a
        root.children['B'] = p2_node_b                                                                                                                                                                                                                                       
        
        nfa = GameNFA(root)                                                                                                                                                                                                                                                 
        oracle = PreferenceOracle(nfa)
        Trace_1=['A','X']
        Trace_2=['A','Y']
        assert oracle.compare(Trace_1,Trace_2) == 't2'

    def test_equal_preference(self):
        root = GameNode(value=5, player='P1', depth=0)
        p2_node_a = GameNode(value=5, player='P2', depth=1)                                                                             
        p2_node_b = GameNode(value=5, player='P2', depth=1)
        x_node_a = GameNode(value=5, player='P1', depth=2)                                                                                                                                                                                                                      
        y_node_a = GameNode(value=5, player='P1', depth=2)
        x_node_b = GameNode(value=10, player='P1', depth=2)                                                                                                                                                                                                                      
        y_node_b = GameNode(value=5, player='P1', depth=2)
        p2_node_a.children['X'] = x_node_a                                                                                                                                                                                                                                      
        p2_node_a.children['Y'] = y_node_a                                                                                                                                                                                                                                      
        p2_node_b.children['X'] = x_node_b
        p2_node_b.children['Y'] = y_node_b                                                                                                                                                                                                                                      
        root.children['A'] = p2_node_a
        root.children['B'] = p2_node_b                                                                                                                                                                                                                                       
        
        nfa = GameNFA(root)                                                                                                                                                                                                                                                 
        oracle = PreferenceOracle(nfa)
        Trace_1=['A','X']
        Trace_2=['A','Y']
        assert oracle.compare(Trace_1,Trace_2) == 'equal' 