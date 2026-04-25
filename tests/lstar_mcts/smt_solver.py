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
    """Tests for SMTValueAssigner.add"""
    pass


class TestSolve:
    """Tests for SMTValueAssigner.solve"""
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
