# Nim — Implementation TODO

Work through these steps in order.  Each step has a test file to verify before moving on.
Run tests from the project root with: `python -m pytest tests/game/nim/ -v`

Start with **3 piles of sizes [1, 2, 3]** (classic Nim) — small enough for full minimax,
large enough to be non-trivial.  Parameterise `piles` throughout so larger configurations
can be swapped in for benchmark experiments.

---

## Background — Nim rules

Players alternate removing objects from piles.  On each turn a player **must** remove at
least one object from **exactly one** pile (any number up to the full pile).  The player
who takes the **last object wins** (normal-play / last-player-wins convention).

**Key difference from TTT and D&B:** the game tree branches on *(pile_index, count)*
pairs instead of board squares or edges.  The alphabet is all legal *(pile, count)* pairs
reachable from the initial configuration — fixed at construction time.

**Optimal strategy:** the nim-sum (XOR of all pile sizes) is the classical solution.
A position is losing for the player to move iff `xor(piles) == 0`.  The oracle can use
this directly (depth=None) or bounded minimax with a heuristic for the locally-greedy
variant.

**No extra turns:** Nim is strictly alternating — P1 and P2 take turns with no
consecutive-move mechanic.  `GameSUL` can be reused **unchanged**.

---

## Action encoding

Each legal move is a tuple `(pile_index, count)` where
`0 <= pile_index < len(piles)` and `1 <= count <= piles[pile_index]`.

For initial piles `[1, 2, 3]`:
```
(0,1)                          # remove 1 from pile 0
(1,1), (1,2)                   # remove 1 or 2 from pile 1
(2,1), (2,2), (2,3)            # remove 1, 2, or 3 from pile 2
```
Total alphabet size: `sum(piles)` = 6 for [1, 2, 3].

The alphabet is **fixed** as all `(pile, count)` pairs where `count <= initial_piles[pile]`.
Moves that are legal in the initial state but illegal in a later state (pile already
smaller) are no-ops: `children` simply will not contain that key, and `GameSUL.step`
will return PASS / fall back gracefully.

---

## Step 1 — Board state (`src/game/nim/board.py`)

Implement `NimState` — the core game object.

**Must have:**
- `piles: tuple[int, ...]` — pile sizes (immutable)
- `player: str` — `'P1'` or `'P2'`
- `children: dict[tuple, NimState]` — `{(pile_index, count): next_state}`
  - Key is a `(pile_index, count)` tuple (hashable, usable as L* alphabet symbol)
  - `next_state.piles` has `pile_index` reduced by `count`
  - Player alternates: `'P1' ↔ 'P2'` (no extra-turn mechanic)
  - Only legal moves: `0 <= count <= piles[pile_index]`, `count >= 1`
  - Terminal states have no children
- `is_terminal() -> bool` — True when all piles are 0
- `winner() -> str | None` — the player who **just moved** wins when all piles are 0
  - Equivalently: the player currently **to move** loses when all piles are 0
  - `winner()` returns `None` when the game is not yet over
- `value -> int` — from P2's perspective: `+1` P2 win, `-1` P1 win, `0` draw
  - Nim has no draws; `value` is always ±1 at terminal nodes
- Use `@cached_property` for `children` (immutable state, same pattern as TTT/D&B)

**Helpers:**
```python
def _apply_move(piles: tuple, pile_index: int, count: int) -> tuple:
    """Return new piles tuple with count removed from pile_index."""
```

**Test file:** `tests/game/nim/test_board.py`
- [ ] Initial state `[1,2,3]`: not terminal, P1 to move
- [ ] `children` has exactly `sum(piles)` entries for initial state
- [ ] Each child has the correct pile reduced
- [ ] Player alternates after every move
- [ ] `is_terminal` True when all piles are 0
- [ ] `winner` returns `None` on non-terminal state
- [ ] `winner` returns the correct player (whoever just moved — the other player from
      the one `to_move` in that terminal state)
- [ ] `value` is `+1` when P2 wins, `-1` when P1 wins
- [ ] Single-pile `[3]`: only 3 children `(0,1)`, `(0,2)`, `(0,3)`
- [ ] Pile-of-1: taking it reaches terminal immediately
- [ ] Chained take: removing an entire pile leaves remaining piles unchanged

---

## Step 2 — NFA interface (`src/game/nim/game_nfa.py`)

Implement `NimNFA` — mirrors `TicTacToeNFA` exactly.

**Must have:**
- `root: NimState` — initial state, P1 to move
- `alphabet: list[tuple]` — all `(pile, count)` pairs where `count <= initial_piles[pile]`
  - This is `list(root.children.keys())` at construction time
  - Fixed alphabet: used directly as L*'s input alphabet
- `get_node(trace: list) -> NimState | None`
  - Replays moves from root; returns `None` if any move is illegal
- `p1_legal_inputs(trace) -> list`
- `p2_legal_moves(trace) -> list`
- `is_terminal(trace) -> bool`
- `current_player(trace) -> str | None`

**Note:** No PASS mechanic needed.  `GameSUL` can be imported and used directly in
the learner script without modification.

**Test file:** `tests/game/nim/test_game_nfa.py`
- [ ] `root` is initial state, P1 to move
- [ ] `alphabet` equals `list(root.children.keys())`
- [ ] `get_node([])` returns root
- [ ] `get_node([(pile, count)])` returns correct next state
- [ ] `get_node` returns `None` for illegal move (count > current pile size)
- [ ] `get_node` returns `None` for repeated exhausted pile move
- [ ] `p1_legal_inputs` returns legal moves when P1 to move
- [ ] `p1_legal_inputs` returns `[]` when P2 to move
- [ ] `p2_legal_moves` returns legal moves when P2 to move
- [ ] `p2_legal_moves` returns `[]` when P1 to move
- [ ] `is_terminal` True when all piles are 0
- [ ] `current_player` returns `None` on terminal state
- [ ] 3-pile `[1,2,3]`: alphabet has 6 elements

---

## Step 3 — Preference oracle (`src/game/nim/preference_oracle.py`)

Implement `NimOracle` — mirrors `TicTacToeOracle` with bounded minimax and a
nim-sum heuristic at the depth cutoff.

**Must have:**
- `__init__(nfa, depth=None)` — `None` = unbounded (globally optimal via nim-sum minimax)
- `preferred_move(prefix: list) -> tuple | None`
  - Best P2 `(pile, count)` at the current position; `None` if not a P2 decision point
- `compare(trace1, trace2) -> str`
  - `'t1'`, `'t2'`, or `'equal'` — same convention as TTT/D&B oracles

**Bounded minimax:**
```
_minimax(state, depth):
    if terminal        → state.value
    if depth == 0      → _heuristic(state)       ← NOT 0
    if P2's turn       → max over children
    if P1's turn       → min over children
```

Cache key: `(piles, player, depth)` — `piles` is already a hashable tuple.

**Heuristic at depth cutoff — nim-sum advantage:**
```python
@staticmethod
def _heuristic(state: NimState) -> float:
    nim_xor = 0
    for p in state.piles:
        nim_xor ^= p
    # nim_xor == 0 → current player is in a losing position
    # Normalise to (-1, 1): winning → positive (from P2's perspective)
    if state.player == 'P2':
        return 0.5 if nim_xor != 0 else -0.5   # P2 to move: nonzero XOR = P2 wins
    else:
        return -0.5 if nim_xor != 0 else 0.5   # P1 to move: nonzero XOR = P1 wins
```

This directly encodes Nim's mathematical winning condition without full lookahead,
giving bounded minimax real signal rather than a flat 0 cutoff.

**Test file:** `tests/game/nim/test_preference_oracle.py`
- [ ] `preferred_move` takes a winning move when XOR ≠ 0 (P2 in winning position)
- [ ] `preferred_move` returns some legal move even when P2 is in losing position
- [ ] `preferred_move` returns `None` on P1's turn
- [ ] `preferred_move` returns `None` on terminal state
- [ ] `preferred_move` returns `None` on invalid trace
- [ ] `compare` returns `'t2'` when trace2 leads to P2 win, trace1 to P2 loss
- [ ] `compare` returns `'equal'` for two symmetric traces with same value
- [ ] `_heuristic` returns positive when P2 to move and nim_xor ≠ 0
- [ ] `_heuristic` returns negative when P1 to move and nim_xor ≠ 0
- [ ] `_heuristic` returns negative when P2 to move and nim_xor == 0
- [ ] With `depth=None`, `preferred_move` always picks a nim-sum-zeroing move when
      one exists (optimal play)
- [ ] With `depth=1`, `preferred_move` still finds a winning move one step ahead

---

## Step 4 — Learner script (`src/scripts/learner_nim.py`)

Create a script that runs L* + MCTS on Nim end-to-end, reusing `GameSUL`, `TableB`,
and `MCTSEquivalenceOracle` unchanged (no PASS mechanic needed).

**What changes from `learner_ttt.py`:**
- Import `NimNFA` and `NimOracle` instead of TTT equivalents
- `--piles` argument: space-separated pile sizes (default `1 2 3`)
- `--oracle-depth` argument (default `None`)
- `p1_inputs = nfa.alphabet`  (already computed in NFA constructor)
- Evaluation: play learned automaton vs random P1, report win/loss/draw counts

**Structure:**
```python
def main():
    nfa    = NimNFA(piles=tuple(args.piles))
    oracle = NimOracle(nfa, depth=args.oracle_depth)
    sul    = GameSUL(nfa, oracle)         # reused unchanged
    table_b = TableB()
    eq = MCTSEquivalenceOracle(sul, nfa, oracle, table_b, ...)
    p1_inputs = nfa.alphabet

def evaluate_vs_random(model, nfa, n_games, seed) -> tuple[int, int, int]:
    """Play learned P2 vs random P1; return (losses, draws, wins)."""
```

**Test file:** `tests/game/nim/test_learner_nim.py`
- [ ] `run_Lstar` completes without error on [1,2,3] piles
- [ ] Learned automaton has at least 1 state
- [ ] `evaluate_vs_random` returns non-negative integers summing to n_games
- [ ] With `depth=None` oracle, learned P2 wins > 80% of games vs random P1 (100 games)

---

## Step 5 — Benchmark sweep (`src/scripts/benchmark_nim.py`)

Mirror `benchmark_dab.py` for Nim with a pile-configuration sweep axis.

**Additional sweep axis vs TTT benchmark:**
- `pile_configs: [(1,2,3), (2,3,4), (3,4,5)]` — test scaling to larger alphabets and
  deeper game trees

**Figures to produce:**
- `nim_score_K.png`           — normalised score vs round, varying K     (fix depth_n, [1,2,3])
- `nim_states_K.png`          — automaton states  vs round, varying K
- `nim_score_dn.png`          — normalised score  vs round, varying depth_n (fix K)
- `nim_states_dn.png`         — automaton states  vs round, varying depth_n
- `nim_score_piles.png`       — score vs oracle_depth, rows=pile_config (shows scaling)

---

## Implementation notes

**Alphabet is fixed at construction:** `NimNFA.__init__` computes `alphabet =
list(root.children.keys())`.  This is the full L* input alphabet — all `(pile, count)`
pairs valid from the initial configuration.  A move that becomes illegal mid-game (pile
too small) simply won't appear in a later state's `children`; `GameSUL` handles this by
returning PASS for unknown inputs.

**No PASS mechanic:** Unlike D&B, Nim is strictly alternating.  `GameSUL` is reused
without modification.  Do not create a `NimSUL`.

**Nim-sum heuristic precision:** The heuristic returns ±0.5 (not ±1) to stay strictly
inside the terminal value range.  This ensures the minimax tree correctly prefers a
guaranteed win (value 1.0) over a heuristically-good non-terminal (value 0.5).

**Larger piles / more piles:** Alphabet size = `sum(piles)`.  For [3,4,5] that's 12
symbols — same as 2×2 D&B.  State count grows quickly with pile depth, so bounded
minimax (`depth ≤ 4`) is the intended operating mode for anything beyond [1,2,3].

**Cache key:** `(piles, player, depth)` — `piles` is a `tuple[int,...]`, already
hashable.  Include `player` because the heuristic is player-dependent.
