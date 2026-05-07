"""
RobotGridNFA — wrapper around RobotGridState with the same interface as
the existing game NFAs (DotsAndBoxesNFA, NimNFA, etc.) so the L*+MCTS
pipeline can plug in unchanged.

Key methods:
    nfa.root                    — initial state (player='P1', at home)
    nfa.get_node(prefix)        — replay a P1/P2 trace from root
    nfa.p1_alphabet             — set of observation-digest tuples
    nfa.p2_alphabet             — list of action symbols
    nfa.p1_legal_inputs(prefix) — singleton list (deterministic obs)
    nfa.p2_legal_moves(prefix)  — list of legal actions
    nfa.is_terminal(prefix)
    nfa.current_player(prefix)
"""

from __future__ import annotations
from itertools import product

from src.control_systems.RobotGrid.board import (
    RobotGridState,
    GAS_BANDS,
    N, S, E, W, PICKUP, DROP, REFUEL, GO_TO_REFUEL,
)


class RobotGridNFA:

    def __init__(self,
                 rows: int = 4,
                 cols: int = 4,
                 refuel:  tuple[int, int] | None = None,
                 dropoff: tuple[int, int] | None = None,
                 home:    tuple[int, int] = (0, 0),
                 gas_max:   int = 10,
                 max_steps: int = 50,
                 eligible_cells: tuple | None = None) -> None:
        self.rows      = rows
        self.cols      = cols
        # Default refuel = top-right corner, dropoff = bottom-right corner.
        if refuel is None:
            refuel = (0, cols - 1)
        if dropoff is None:
            dropoff = (rows - 1, cols - 1)
        self.refuel    = refuel
        self.dropoff   = dropoff
        self.home      = home
        self.gas_max   = gas_max
        self.max_steps = max_steps
        # Cells where tasks may spawn (default: all non-special cells).
        if eligible_cells is None:
            eligible_cells = tuple(
                (r, c) for r in range(rows) for c in range(cols)
                if (r, c) != refuel and (r, c) != dropoff
            )
        self.eligible_cells = eligible_cells

        # Root: idle (task_loc=None), at home, full gas. The first P1
        # input is a task arrival event chosen from eligible_cells.
        self.root = RobotGridState(
            rows=rows, cols=cols,
            refuel=refuel, dropoff=dropoff,
            gas_max=gas_max, max_steps=max_steps,
            eligible_cells=eligible_cells,
            pos=home, gas=gas_max, task_loc=None,
            carrying=False, delivered_count=0,
            player='P1', step_count=0,
        )

    # ------------------------------------------------------------------
    # Trace navigation
    # ------------------------------------------------------------------

    def get_node(self, trace: list) -> RobotGridState | None:
        """Replay a P1/P2 trace from root. Returns None on illegal action."""
        state = self.root
        for action in trace:
            if state is None or state.is_terminal():
                return None
            if action not in state.children:
                return None
            state = state.children[action]
        return state

    # ------------------------------------------------------------------
    # Legal-move queries
    # ------------------------------------------------------------------

    def p1_legal_inputs(self, trace: list) -> list:
        state = self.get_node(trace)
        if state is None or state.is_terminal() or state.player != 'P1':
            return []
        return list(state.children.keys())

    def p2_legal_moves(self, trace: list) -> list:
        state = self.get_node(trace)
        if state is None or state.is_terminal() or state.player != 'P2':
            return []
        return list(state.children.keys())

    def is_terminal(self, trace: list) -> bool:
        state = self.get_node(trace)
        return state is not None and state.is_terminal()

    def current_player(self, trace: list) -> str | None:
        state = self.get_node(trace)
        if state is None or state.is_terminal():
            return None
        return state.player

    # ------------------------------------------------------------------
    # Alphabets
    # ------------------------------------------------------------------

    @property
    def p1_alphabet(self) -> list:
        """All possible P1-input symbols.

        Two distinct symbol classes — the Mealy must handle both:
          (1) Observation digests — during an active task or refuel detour
                pos × gas_band × task_loc × carrying
                = (rows × cols) × 4 × (1 + |eligible_cells|) × 2
          (2) Task-arrival events — at idle states
                ('TASK', (r, c)) for each cell in eligible_cells

        Mode is intentionally NOT in the alphabet — see board.py's
        `observation` docstring. State-merging in L* splits normal-mode
        and refuel-mode prefixes via their differing MQ-response
        signatures, so encoding mode here would just double the alphabet
        without gaining expressivity.
        """
        all_cells = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        task_locs = [None] + list(self.eligible_cells)
        observations = [
            (pos, gb, tl, c)
            for pos in all_cells
            for gb  in GAS_BANDS
            for tl  in task_locs
            for c   in (False, True)
        ]
        task_arrivals = [('TASK', loc) for loc in self.eligible_cells]
        return observations + task_arrivals

    @property
    def p2_alphabet(self) -> list:
        return [N, S, E, W, PICKUP, DROP, REFUEL, GO_TO_REFUEL]


# ----------------------------------------------------------------------
# Quick demo
# ----------------------------------------------------------------------

if __name__ == '__main__':
    nfa = RobotGridNFA()
    print(f'NFA: {nfa.rows}x{nfa.cols} grid')
    print(f'  home={nfa.home}  refuel={nfa.refuel}  dropoff={nfa.dropoff}')
    print(f'  eligible task cells (n={len(nfa.eligible_cells)}): {list(nfa.eligible_cells)[:5]} ...')
    print(f'  gas_max={nfa.gas_max}  max_steps={nfa.max_steps}')
    print(f'  |p1_alphabet|={len(nfa.p1_alphabet)}  |p2_alphabet|={len(nfa.p2_alphabet)}')
    print()
    print(f'root (idle): {nfa.root}')
    print(f'  legal P1 inputs (task arrivals): n={len(nfa.p1_legal_inputs([]))}')

    # Walk: arrive task at (2,2) → see observation → first action
    s1 = nfa.root.children[('TASK', (2, 2))]
    print(f'after task arrives at (2,2): {s1}')
    print(f'  legal P2 actions: {list(s1.children.keys())}')
