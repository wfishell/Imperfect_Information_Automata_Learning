"""
Dots and Boxes 2x2 grid — game state and NFA over valid game traces.

Grid layout (3x3 dots, 12 edges, 4 boxes):

  (0,0) -H00- (0,1) -H01- (0,2)
    |            |            |
   V00          V01          V02
    |            |            |
  (1,0) -H10- (1,1) -H11- (1,2)
    |            |            |
   V10          V11          V12
    |            |            |
  (2,0) -H20- (2,1) -H21- (2,2)

Edge indices:
  H00=0  H01=1
  H10=2  H11=3
  H20=4  H21=5
  V00=6  V01=7  V02=8
  V10=9  V11=10 V12=11

Boxes (unit squares, each needs all 4 bounding edges):
  A (top-left):     H00, H10, V00, V01  -> {0, 2, 6, 7}
  B (top-right):    H01, H11, V01, V02  -> {1, 3, 7, 8}
  C (bottom-left):  H10, H20, V10, V11  -> {2, 4, 9, 10}
  D (bottom-right): H11, H21, V11, V12  -> {3, 5, 10, 11}

A word over {0..11} is in the language iff it is a valid complete game:
  - each symbol is an unclaimed edge at that point
  - all 12 edges are claimed by the end
  - player turn alternates unless a box is completed (same player goes again)
"""

NUM_EDGES = 12
NUM_BOXES = 4

EDGE_NAMES = [
    'H00', 'H01',        # horizontal top
    'H10', 'H11',        # horizontal middle
    'H20', 'H21',        # horizontal bottom
    'V00', 'V01', 'V02', # vertical left gap
    'V10', 'V11', 'V12', # vertical right gap
]

EDGE_INDEX = {name: idx for idx, name in enumerate(EDGE_NAMES)}

# Each box is defined by the frozenset of its 4 bounding edge indices
BOXES = [
    frozenset({0, 2, 6, 7}),   # A: top-left
    frozenset({1, 3, 7, 8}),   # B: top-right
    frozenset({2, 4, 9, 10}),  # C: bottom-left
    frozenset({3, 5, 10, 11}), # D: bottom-right
]

# Precompute: for each edge, which boxes does it border?
EDGE_TO_BOXES = [[] for _ in range(NUM_EDGES)]
for box_id, box_edges in enumerate(BOXES):
    for e in box_edges:
        EDGE_TO_BOXES[e].append(box_id)


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

class GameState:
    """
    Immutable game state.

    claimed  : frozenset of edge indices claimed so far
    player   : 0 (P1) or 1 (P2) — whose turn it is
    score    : (p1_boxes, p2_boxes)
    """
    __slots__ = ('claimed', 'player', 'score')

    def __init__(self, claimed=frozenset(), player=0, score=(0, 0)):
        self.claimed = claimed
        self.player  = player
        self.score   = score

    def __eq__(self, other):
        return (self.claimed == other.claimed and
                self.player  == other.player  and
                self.score   == other.score)

    def __hash__(self):
        return hash((self.claimed, self.player, self.score))

    def __repr__(self):
        claimed_names = sorted(EDGE_NAMES[e] for e in self.claimed)
        return (f"GameState(player=P{self.player+1}, score={self.score}, "
                f"claimed={claimed_names})")


INITIAL_STATE = GameState()


def legal_moves(state):
    """Return list of edge indices that are legal to claim from this state."""
    return [e for e in range(NUM_EDGES) if e not in state.claimed]


def transition(state, edge):
    """
    Apply edge claim to state. Returns the next GameState, or None if illegal.
    """
    if edge in state.claimed:
        return None  # illegal move

    new_claimed = state.claimed | {edge}

    # Count how many boxes this edge completes
    newly_completed = sum(
        1 for box_id in EDGE_TO_BOXES[edge]
        if BOXES[box_id] <= new_claimed
    )

    s0, s1 = state.score
    if state.player == 0:
        new_score = (s0 + newly_completed, s1)
    else:
        new_score = (s0, s1 + newly_completed)

    # Same player goes again if a box was completed, otherwise switch
    new_player = state.player if newly_completed > 0 else 1 - state.player

    return GameState(new_claimed, new_player, new_score)


def is_terminal(state):
    """True when all 12 edges have been claimed."""
    return len(state.claimed) == NUM_EDGES


def winner(state):
    """
    Returns 0 (P1 wins), 1 (P2 wins), or None (draw).
    Only meaningful at terminal states.
    """
    s0, s1 = state.score
    if s0 > s1:
        return 0
    elif s1 > s0:
        return 1
    return None


# ---------------------------------------------------------------------------
# NFA over valid game traces
# ---------------------------------------------------------------------------

class DotsAndBoxesNFA:
    """
    NFA (deterministic transitions) whose language is the set of all valid
    complete Dots and Boxes 2x2 games.

    Alphabet : integers 0..11  (edge indices)
    States   : GameState objects  (plus a dead/reject state represented as None)
    q0       : INITIAL_STATE
    F        : all states where is_terminal(state) is True

    A word w = (e0, e1, ..., e11) is accepted iff:
      - each eᵢ is a legal move from the state reached after e0..eᵢ₋₁
      - len(w) == 12  (all edges claimed)
    """

    def __init__(self):
        self.alphabet      = list(range(NUM_EDGES))
        self.initial_state = INITIAL_STATE

    # -- Core NFA interface --------------------------------------------------

    def delta(self, state, symbol):
        """
        Transition function. Returns next state or None (dead/reject).
        This NFA is actually deterministic: each (state, symbol) has at
        most one successor.
        """
        if state is None:
            return None
        return transition(state, symbol)

    def is_accepting(self, state):
        return state is not None and is_terminal(state)

    # -- Derived helpers -----------------------------------------------------

    def accepts(self, word):
        """Check whether a sequence of edge indices is a valid complete game."""
        state = self.initial_state
        for symbol in word:
            state = self.delta(state, symbol)
            if state is None:
                return False
        return self.is_accepting(state)

    def run(self, word):
        """
        Run the NFA on a word.
        Returns the list of states visited (length len(word)+1),
        with None indicating rejection.
        """
        states = [self.initial_state]
        state  = self.initial_state
        for symbol in word:
            state = self.delta(state, symbol)
            states.append(state)
        return states

    def legal_moves_at(self, state):
        """All legal next symbols from a given state (the membership oracle)."""
        if state is None or is_terminal(state):
            return []
        return legal_moves(state)

    def state_after(self, word):
        """Return the state reached after reading word from q0, or None."""
        state = self.initial_state
        for symbol in word:
            state = self.delta(state, symbol)
            if state is None:
                return None
        return state

    # -- Visualisation -------------------------------------------------------

    def display_state(self, state):
        """Print a human-readable board for a given state."""
        if state is None:
            print("[dead state]")
            return

        c = state.claimed

        def h(e):   return '---' if e in c else '   '
        def v(e):   return '|'   if e in c else ' '

        # Row of dots + horizontal edges
        row0 = f"*{h(0)}*{h(1)}*"
        row1 = f"*{h(2)}*{h(3)}*"
        row2 = f"*{h(4)}*{h(5)}*"

        # Column of vertical edges + box labels
        def box_label(box_id):
            if BOXES[box_id] <= c:
                owner = None
                # find who owns: re-derive from score isn't straightforward,
                # so just mark as claimed
                return 'X'
            return ' '

        mid0 = f"{v(6)} {box_label(0)} {v(7)} {box_label(1)} {v(8)}"
        mid1 = f"{v(9)} {box_label(2)} {v(10)} {box_label(3)} {v(11)}"

        print(row0)
        print(mid0)
        print(row1)
        print(mid1)
        print(row2)
        print(f"  P{state.player+1} to move | Score: P1={state.score[0]} P2={state.score[1]}")
