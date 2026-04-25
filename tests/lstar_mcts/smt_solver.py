"""
Unit tests for SMTValueAssigner (src/lstar_mcts/smt_solver.py).

Run Instructions:

'pytest tests/lstar_mcts/smt_solver.py -v'

"""

import pytest
from src.lstar_mcts.smt_solver import SMTValueAssigner


@pytest.fixture
def smt():
    return SMTValueAssigner()


class TestVar:
    """Tests for SMTValueAssigner._var
    
    Inputs: list[str] | tuple representing a trace of actions:
        - Example: "['B', 'X', 'B', 'Y']" where 'B' is a P1 move, 'X' is a P2 move, etc.

    Outputs: A Z3 variable corresponding to the trace:
        - Example: "v1" for the first unique trace,
                   "v2" for the second unique trace, etc.
    """

    def test_var_creation(self, smt):
            trace1 = ['B', 'X', 'B', 'Y']
            trace2 = ['B', 'X', 'B', 'Y']  # Same as trace1, should return same var
            trace3 = ['B', 'X', 'B']       # Different trace, should return new var
    
            v1 = smt._var(trace1)
            v2 = smt._var(trace2)
            v3 = smt._var(trace3)
    
            assert v1.eq(v2), "Same trace should return the same variable"
            assert v1.eq(v3) == False, "Different traces should return different variables"

    def test_var_normalization(self, smt):
        trace = ['B', 'X', 'B', 'Y']
        v = smt._var(trace)

        # Check that the variable has constraints to be between 0 and 100
        assert any(str(c) == f'{v} >= 0' for c in smt._solver.assertions())                      
        assert any(str(c) == f'{v} <= 100' for c in smt._solver.assertions())

class TestAdd:
    """Tests for SMTValueAssigner.add
    
    Inputs: Two traces and a preference indicating which trace is considered more optimal:
        - t1: ['A', 'Y', 'B', 'Y'] 
        - t2: ['A', 'X', 'B', 'X']
        - Preference: 't2'
    """

    def tets_add_preference_t1(self, smt):
        trace1 = ['A', 'Y', 'B', 'Y']
        trace2 = ['A', 'X', 'B', 'X']
        preference = 't1'

        smt.add(trace1, trace2, preference)

        v1 = smt._var(trace1)
        v2 = smt._var(trace2)

        # Check that the correct constraint was added based on the preference
        assert any(str(c) == f'{v1} > {v2}' for c in smt._solver.assertions()), "Preference 't1' should add constraint v1 > v2"

    def test_add_preference_t2(self, smt):
        trace1 = ['B', 'X', 'B', 'Y']
        trace2 = ['A', 'Y', 'B', 'Y']
        preference = 't2'

        smt.add(trace1, trace2, preference)

        v1 = smt._var(trace1)
        v2 = smt._var(trace2)

        # Check that the correct constraint was added based on the preference
        assert any(str(c) == f'{v2} > {v1}' for c in smt._solver.assertions()), "Preference 't2' should add constraint v1 < v2"

    def test_add_preference_t1(self, smt):
        trace1 = ['B', 'X', 'B', 'X']
        trace2 = ['B', 'Y', 'B', 'X']
        preference = 't1'

        smt.add(trace1, trace2, preference)

        v1 = smt._var(trace1)
        v2 = smt._var(trace2)

        # Check that the correct constraint was added based on the preference
        assert any(str(c) == f'{v1} > {v2}' for c in smt._solver.assertions()), "Preference 't1' should add constraint v1 > v2"

    def test_add_preference_equal(self, smt):
        trace1 = ['A', 'Y', 'A', 'X']
        trace2 = ['B', 'X', 'A', 'Y']
        preference = 'equal'

        smt.add(trace1, trace2, preference)

        v1 = smt._var(trace1)
        v2 = smt._var(trace2)

        # Check that the correct constraint was added based on the preference
        assert any(str(c) == f'{v1} == {v2}' for c in smt._solver.assertions()), "Preference 'equal' should add constraint v1 == v2"

    def test_fall_through(self, smt):
        trace1 = ['A', 'Y', 'A', 'X']
        trace2 = ['B', 'X', 'A', 'Y']
        preference = 'invalid'

        with pytest.raises(ValueError, match="Unknown preference"):
            smt.add(trace1, trace2, preference)

class TestSolve:
    """Tests for SMTValueAssigner.solve"""

    def test_solve_consistent_preferences(self, smt):
        """
        Tests Valid Preferences:
        - ['A', 'Y', 'B', 'Y'] preferred over ['A', 'X', 'B', 'X'] (t1 > t2)
        - ['A', 'X', 'B', 'X'] preferred over ['B', 'X', 'B', 'Y'] (t2 > t3)
        - ['A', 'Y', 'B', 'Y'] preferred over ['B', 'X', 'B', 'Y'] (t1 > t3)

        Expected Ordering: t1 > t2 > t3
        
        """

        # Add consistent preferences
        smt.add(['A', 'Y', 'B', 'Y'], ['A', 'X', 'B', 'X'], 't1')
        smt.add(['A', 'X', 'B', 'X'], ['B', 'X', 'B', 'Y'], 't1')
        smt.add(['A', 'Y', 'B', 'Y'], ['B', 'X', 'B', 'Y'], 't1')

        values = smt.solve()
        assert values is not None, "Expected a solution for consistent preferences"
        assert all(0.0 <= v <= 1.0 for v in values.values()), "All values should be in the range [0, 1]"

    def test_solve_inconsistent_preferences(self, smt):
        """
        Tests Inconsistent Preferences:
        - ['A', 'Y', 'B', 'Y'] preferred over ['A', 'X', 'B', 'X'] (t1 > t2)
        - ['A', 'X', 'B', 'X'] preferred over ['B', 'X', 'B', 'Y'] (t2 > t3)
        - ['B', 'X', 'B', 'Y'] preferred over ['A', 'Y', 'B', 'Y'] (t3 > t1)

        Expected Result: UNSATISFIABLE due to circular preferences
        """

        # Add inconsistent preferences
        smt.add(['A', 'Y', 'B', 'Y'], ['A', 'X', 'B', 'X'], 't1')
        smt.add(['A', 'X', 'B', 'X'], ['B', 'X', 'B', 'Y'], 't1')
        smt.add(['B', 'X', 'B', 'Y'], ['A', 'Y', 'B', 'Y'], 't1')

        values = smt.solve()
        assert values is None, "Expected no solution for inconsistent preferences"
    pass


class TestValue:
    """Tests for SMTValueAssigner.value"""
    pass


class TestIsSatisfiable:
    """Tests for SMTValueAssigner.is_satisfiable"""
    pass


class TestNConstraints:
    """Tests for SMTValueAssigner.n_constraints"""
    pass


class TestKnownTraces:
    """Tests for SMTValueAssigner.known_traces"""
    pass
