from src.lstar_mcts.table_b import TableB, ActionStats
import pytest 
import math
class TestActionStats:

    def test_default_values(self):
        stats = ActionStats()
        assert stats.visits == 0
        assert stats.value == 0.5
        assert stats.zero_prob is False


# ----------------------------------------------------------------------
# _get — foundational, every other method depends on it
# ----------------------------------------------------------------------

class TestGet:

    def test_creates_new_entry_with_defaults(self):
        tb = TableB()
        stats = tb._get(['P1_bet'], 'P2_call')
        assert isinstance(stats, ActionStats)
        assert stats.visits == 0

    def test_returns_same_object_on_repeated_calls(self):
        tb = TableB()
        first = tb._get(['P1_bet'], 'P2_call')
        first.visits = 7
        second = tb._get(['P1_bet'], 'P2_call')
        assert second is first

    def test_empty_trace_is_valid(self):
        tb = TableB()
        stats = tb._get([], 'P1_bet')
        assert isinstance(stats, ActionStats)


# ----------------------------------------------------------------------
# actions_at — read-only lookup
# ----------------------------------------------------------------------

class TestActionsAt:

    def test_empty_trace_returns_empty_dict(self):
        tb = TableB()
        assert tb.actions_at(['P1_bet']) == {}

    def test_returns_recorded_actions(self):
        tb = TableB()
        tb.record_visit(['P1_bet'], 'P2_call')
        actions = tb.actions_at(['P1_bet'])
        assert 'P2_call' in actions


# ----------------------------------------------------------------------
# record_visit — increments, with zero_prob guard
# ----------------------------------------------------------------------

class TestRecordVisit:

    def test_increments_visit_count(self):
        tb = TableB()
        tb.record_visit(['P1_bet'], 'P2_call')
        assert tb._get(['P1_bet'], 'P2_call').visits == 1

    def test_multiple_visits_accumulate(self):
        tb = TableB()
        for _ in range(4):
            tb.record_visit(['P1_bet'], 'P2_call')
        assert tb._get(['P1_bet'], 'P2_call').visits == 4

    def test_zero_prob_action_does_not_increment(self):
        tb = TableB()
        tb.set_zero_prob(['P1_bet'], 'P2_call')
        tb.record_visit(['P1_bet'], 'P2_call')
        assert tb._get(['P1_bet'], 'P2_call').visits == 0


# ----------------------------------------------------------------------
# update_value — straightforward setter
# ----------------------------------------------------------------------

class TestUpdateValue:

    def test_sets_value(self):
        tb = TableB()
        tb.update_value(['P1_bet'], 'P2_call', 0.8)
        assert tb._get(['P1_bet'], 'P2_call').value == 0.8

    def test_overwrites_previous_value(self):
        tb = TableB()
        tb.update_value(['P1_bet'], 'P2_call', 0.3)
        tb.update_value(['P1_bet'], 'P2_call', 0.9)
        assert tb._get(['P1_bet'], 'P2_call').value == 0.9


# ----------------------------------------------------------------------
# set_zero_prob — straightforward setter
# ----------------------------------------------------------------------

class TestSetZeroProb:

    def test_marks_action_as_zero_prob(self):
        tb = TableB()
        tb.set_zero_prob(['P1_bet'], 'P2_call')
        assert tb._get(['P1_bet'], 'P2_call').zero_prob is True


# ----------------------------------------------------------------------
# ucb_score — four branches: zero_prob, unexplored, total==0, normal
# ----------------------------------------------------------------------

class TestUCBScore:

    def test_zero_prob_returns_neg_infinity(self):
        tb = TableB()
        tb.set_zero_prob(['P1_bet'], 'P2_call')
        score = tb.ucb_score(['P1_bet'], 'P2_call', ['P2_call', 'P2_fold'])
        assert score == float('-inf')

    def test_unexplored_returns_high_prior(self):
        tb = TableB()
        score = tb.ucb_score(['P1_bet'], 'P2_call', ['P2_call', 'P2_fold'])
        assert score == float('inf')
    
    def test_record_vist_and_update_value(self):
    # test normal case — record some visits and update values,
    # then check the returned score matches the UCB formula
        tb = TableB()
        tb.record_visit(['P1_bet'], 'P2_call')
        tb.update_value(['P1_bet'], 'P2_call', 0.6)
        tb.record_visit(['P1_bet'], 'P2_fold')
        tb.update_value(['P1_bet'], 'P2_fold', 0.4)
        expected_UCB=.6 +1.4 * math.sqrt(math.log(2)/1)*(1.2**-1)

        assert expected_UCB == tb.ucb_score(['P1_bet'], 'P2_call',  ['P2_call', 'P2_fold'])

    def test_record_visit_and_update_value(self):
        tb = TableB()                                                                                                                                                                                     
        # shallow node at depth 0
        tb.record_visit([], 'P2_call')                                                                                                                                                                    
        tb.record_visit([], 'P2_fold')                                                                                                                                                                    
        tb.update_value([], 'P2_call', 0.6)
                                                                                                                                                                                                            
        # deep node at depth 3
        tb.record_visit(['P1_bet', 'P2_call', 'P1_bet'], 'P2_call')                                                                                                                                       
        tb.record_visit(['P1_bet', 'P2_call', 'P1_bet'], 'P2_fold')                                                                                                                                       
        tb.update_value(['P1_bet', 'P2_call', 'P1_bet'], 'P2_call', 0.6)
                                                                                                                                                                                                            
        shallow_ucb = tb.ucb_score([], 'P2_call', ['P2_call', 'P2_fold'])                                                                                                                                 
        deep_ucb    = tb.ucb_score(['P1_bet', 'P2_call', 'P1_bet'], 'P2_call', ['P2_call', 'P2_fold'])                                                                                                    
                                                                                                                                                                                                            
        assert shallow_ucb > deep_ucb  


# ----------------------------------------------------------------------
# best_action — wraps ucb_score
# ----------------------------------------------------------------------

class TestBestAction:

    def test_returns_none_when_all_pruned(self):
        tb = TableB()
        tb.set_zero_prob(['P1_bet'], 'P2_call')
        tb.set_zero_prob(['P1_bet'], 'P2_fold')
        assert tb.best_action(['P1_bet'], ['P2_call', 'P2_fold']) is None
    def test_unexplored_preference(self):
    # test that unexplored action is preferred over explored low-value one
        tb = TableB()
        val=0.5
        for i in range(3):
            val-=0.1
            tb.record_visit(['P1_call'],'P2_call')
            tb.update_value(['P1_call'],'P2_call',val)
        tb.record_visit(['P1_call'],'P2_fold')
        tb.update_value(['P2_call'],'P2_fold',0.25)
        assert tb.best_action(['P1_call'],['P2_fold','P2_call','P2_raise']) == 'P2_raise'

    def  test_explored_actions_highest_value(self):
    #test that among explored actions, highest value wins
        tb = TableB()
        val=0.5
        for i in range(3):
            val-=0.1
            tb.record_visit(['P1_call'],'P2_call')
            tb.update_value(['P1_call'],'P2_call',val)
        tb.record_visit(['P1_call'],'P2_fold')
        tb.update_value(['P2_call'],'P2_fold',0.25)
        assert tb.best_action(['P1_call'],['P2_fold','P2_call']) == 'P2_fold'

# ----------------------------------------------------------------------
# sample_p2_action — stochastic, needs seeding
# ----------------------------------------------------------------------

class TestSampleP2Action:

    def test_returns_none_when_all_pruned(self):
        tb = TableB()
        tb.set_zero_prob(['P1_bet'], 'P2_call')
        assert tb.sample_p2_action(['P1_bet'], ['P2_call']) is None

    def seed_random_confirm_distribiution(self):
 #  compute theoretical weights using ucb_score for each visited action,
    #   normalise to probabilities, multiply by 10000 for expected counts,
    #    run scipy.stats.chisquare(observed, f_exp=expected), assert p > 0.05
        tb=TableB()
        tb.record_visit(['P1_bet'], 'P2_call')                                                                                                                                                               
        tb.update_value(['P1_bet'], 'P2_call', 0.8)                                                                                                                                                           
        tb.record_visit(['P1_bet'], 'P2_bet')                                                                                                                                                          
        tb.update_value(['P1_bet'], 'P2_bet', 0.4)                                                                                                                                                            
        tb.record_visit(['P1_bet'], 'P2_raise')                                                                                                                                                               
        tb.update_value(['P1_bet'], 'P2_raise', 0.1)                                                                                                                                                                                            
        action_dictionary={}
        for i in range(10000):
            sampled_p2_action=tb.sample_p2_action(['P1_bet'],['P2_call','P2_raise','P2_bet'])
            if sampled_p2_action not in action_dictionary.keys():
                action_dictionary[sampled_p2_action]=1
            else:
                action_dictionary[sampled_p2_action]+=1
        actions = ['P2_call', 'P2_raise', 'P2_bet']                                                                                                                                                           
                                                                                                                                                                                                        
        observed = [action_dictionary[a] for a in actions]                                                                                                                                                                                                                                                                                        
        # theoretical weights from ucb_score                                                                                                                                                                  
        weights = [math.exp(tb.ucb_score(['P1_bet'], a, actions)) for a in actions]
        total = sum(weights)                                                                                                                                                                                  
        expected = [(w / total) * 10000 for w in weights]   
                                                                                                                                                                                                                
        stat, p = chisquare(observed, f_exp=expected)                                                                                                                                                         
        assert p > 0.05    
    
    import numpy as np                                                                                                                                                                                                  
    @pytest.mark.parametrize('low_value', np.linspace(0.05, 0.2, 1000))                                                                                                                                   
    def test_unexplored_beats_explored_low_value(self, low_value):                                                                                                                                        
        tb = TableB()                                                                                                                                                                                     
        tb.record_visit(['P1_bet'], 'P2_call')
        tb.update_value(['P1_bet'], 'P2_call', low_value)                                                                                                                                                 
        counts = {'P2_call': 0, 'P2_bet': 0}
        for _ in range(1000):                                                                                                                                                                             
            a = tb.sample_p2_action(['P1_bet'], ['P2_call', 'P2_bet'])
            counts[a] += 1                                                                                                                                                                                
        assert counts['P2_bet'] > counts['P2_call']   
    # TODO: confirm unexplored actions get higher weight than explored low-value


# ----------------------------------------------------------------------
# prune_below_median — careful with even-length median and strict 
# ----------------------------------------------------------------------

class TestPruneBelowMedian:

    def test_empty_input_returns_zero(self):
        tb = TableB()
        assert tb.prune_below_median([]) == 0

    def test_with_odd_number_of_values(self):
        tb = TableB()
        tb.record_visit(['P1_bet'], 'P2_call')                                                                                                                                                               
        tb.update_value(['P1_bet'], 'P2_call', 0.9)                                                                                                                                                           
        tb.record_visit(['P1_bet'], 'P2_bet')                                                                                                                                                          
        tb.update_value(['P1_bet'], 'P2_bet', 0.5)                                                                                                                                                            
        tb.record_visit(['P1_bet'], 'P2_raise')                                                                                                                                                               
        tb.update_value(['P1_bet'], 'P2_raise', 0.1)
        assert tb.prune_below_median([(['P1_bet'], 'P2_call'), (['P1_bet'], 'P2_bet'), (['P1_bet'], 'P2_raise')]) == 1    
    # : test with odd number of values, e.g. values [0.1, 0.5, 0.9]
    #       median is 0.5, only 0.1 should be pruned
    def test_with_even_number_off_values(self):
    # test with even number, e.g. [0.1, 0.3, 0.7, 0.9]
    #       sorted[len//2] is 0.7 (upper median), so 0.1 and 0.3 prune
        tb = TableB()
        tb.record_visit(['P1_bet'], 'P2_call')                                                                                                                                                               
        tb.update_value(['P1_bet'], 'P2_call', 0.9)                                                                                                                                                           
        tb.record_visit(['P1_bet'], 'P2_bet')                                                                                                                                                          
        tb.update_value(['P1_bet'], 'P2_bet', 0.7)                                                                                                                                                            
        tb.record_visit(['P1_bet'], 'P2_raise')                                                                                                                                                               
        tb.update_value(['P1_bet'], 'P2_raise', 0.1)
        tb.record_visit(['P1_bet'],'P2_fold')
        tb.update_value(['P1_bet'],'P2_fold',0.3)
        assert tb.prune_below_median([(['P1_bet'], 'P2_call'), (['P1_bet'], 'P2_bet'), (['P1_bet'], 'P2_raise'),(['P1_bet'],'P2_fold')]) == 2  


    # TODO: test that ties with median are NOT pruned (strict < comparison)
    def test_ties(self):
        tb = TableB()
        tb.record_visit(['P1_bet'], 'P2_call')                                                                                                                                                               
        tb.update_value(['P1_bet'], 'P2_call', 0.9)                                                                                                                                                           
        tb.record_visit(['P1_bet'], 'P2_bet')                                                                                                                                                          
        tb.update_value(['P1_bet'], 'P2_bet', 0.5)                                                                                                                                                            
        tb.record_visit(['P1_bet'], 'P2_raise')                                                                                                                                                               
        tb.update_value(['P1_bet'], 'P2_raise', 0.5)
        assert tb.prune_below_median([(['P1_bet'], 'P2_call'), (['P1_bet'], 'P2_bet'), (['P1_bet'], 'P2_raise')]) == 0 

# ----------------------------------------------------------------------
# summary — formatting
# ----------------------------------------------------------------------

class TestSummary:

    def test_empty_table_summary(self):
        tb = TableB()
        s = tb.summary()
        assert '0 states' in s
        assert '0 edges' in s

    #: add some entries, confirm counts in summary string match
    def test_summary_with_entries(self):
        tb = TableB()                                                                                                               
        tb.record_visit(['P1_bet'], 'P2_call')
        tb.record_visit(['P1_bet'], 'P2_fold')                                                                                                                                                            
        tb.record_visit(['P1_bet', 'P2_call'], 'P1_raise')
        tb.set_zero_prob(['P1_bet'], 'P2_fold')                                                                                                                                                           
        s = tb.summary()                                                                                                                                                                                  
        assert '2 states' in s
        assert '3 edges' in s                                                                                                                                                                             
        assert '1 pruned' in s         