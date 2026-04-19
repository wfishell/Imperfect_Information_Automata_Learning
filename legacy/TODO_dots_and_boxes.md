# Dots and Boxes — Implementation TODO

Work through these steps in order. Each step has a test file to verify before moving on.
Run tests from the project root with: `python -m pytest tests/game/dots_and_boxes/ -v`

Start with a **2×2 box grid** (3×3 dots, 12 edges) — small enough for full minimax,
large enough to be interesting. Parameterise rows/cols throughout so 3×3 boxes can be
swapped in for benchmark experiments.

---

## Background — Dots and Boxes rules

Players alternate drawing one edge between two adjacent dots.  When a player draws the
**fourth side** of a box they **claim it** and **take another turn** (potentially
chaining multiple boxes in one turn).  The player with more boxes when all edges are
drawn wins.

**Key difference from Tic-Tac-Toe:** a player may move multiple times in a row.
The existing `MCTSEquivalenceOracle`, `GameSUL`, and `TableB` handle this transparently
as long as `DotsAndBoxesState.player` always reflects whose turn it actually is —
including after box completions.

---

## Edge indexing convention

For a grid of `R` rows × `C` cols of boxes (`(R+1)×(C+1)` dots):

```
Horizontal edges: (R+1) * C total
  edge index  =  r * C + c          r ∈ [0, R],  c ∈ [0, C-1]

Vertical edges: R * (C+1) total  (offset by (R+1)*C)
  edge index  =  (R+1)*C + r*(C+1) + c    r ∈ [0, R-1], c ∈ [0, C]
```

For 2×2 boxes: 6 horizontal (0–5) + 6 vertical (6–11) = **12 edges total**.
The L* input alphabet is `[0, 1, ..., 11]`.

Box `(r, c)` (0-indexed, row-major) is bounded by:
- top    : horizontal edge `r * C + c`
- bottom : horizontal edge `(r+1) * C + c`
- left   : vertical edge `(R+1)*C + r*(C+1) + c`
- right  : vertical edge `(R+1)*C + r*(C+1) + c + 1`

---

## Step 1 — Board state (`src/game/dots_and_boxes/board.py`)

Implement `DotsAndBoxesState` — the core game object the NFA and oracle will use.

**Must have:**
- `rows: int`, `cols: int` — grid dimensions (default 2×2)
- `edges: tuple[bool, ...]` — length `(R+1)*C + R*(C+1)`, True if edge is drawn
- `player: str` — `'P1'` or `'P2'`
- `p1_boxes: int`, `p2_boxes: int` — boxes claimed so far
- `children: dict[int, DotsAndBoxesState]` — `{edge_index: next_state}` for each
  undrawn edge
  - Drawing an edge that completes ≥1 boxes: `next_state.player` = **same player**
  - Drawing an edge that completes 0 boxes:  `next_state.player` = **other player**
  - Box score updated in `next_state` to reflect newly completed boxes
- `is_terminal() -> bool` — True when all edges are drawn
- `winner() -> str | None` — `'P1'`, `'P2'`, `'draw'`, or `None`
- `value -> int` — from P2's perspective: `+1` win, `0` draw, `-1` loss
  (only meaningful at terminal nodes)

**Helper:** `_completed_boxes(edges, drawn_edge, rows, cols) -> int`
Returns how many new boxes are completed by adding `drawn_edge` to the current
edge set.  Used inside the `children` property to determine turn hand-off.

**Test file:** `tests/game/dots_and_boxes/test_board.py`
- [ ] Initial state: all edges False, P1 to move, p1_boxes=0, p2_boxes=0
- [ ] `children` at root has 12 entries for 2×2 grid
- [ ] Drawing a non-completing edge switches player
- [ ] Drawing the 4th side of a box keeps the same player and increments their score
- [ ] Chained box completion: drawing one edge that completes two boxes simultaneously
      keeps same player and increments score by 2
- [ ] `is_terminal` False on empty board
- [ ] `is_terminal` True when all edges drawn
- [ ] `winner` returns `'P1'` when p1_boxes > p2_boxes at terminal
- [ ] `winner` returns `'P2'` when p2_boxes > p1_boxes at terminal
- [ ] `winner` returns `'draw'` when scores are equal at terminal
- [ ] `value` is `+1` P2 win, `-1` P1 win, `0` draw

---

## Step 2 — NFA interface (`src/game/dots_and_boxes/game_nfa.py`)

Implement `DotsAndBoxesNFA` — mirrors `TicTacToeNFA` exactly.

**Must have:**
- `root: DotsAndBoxesState` — initial empty board, P1 to move
  - `root.children.keys()` = L* input alphabet (edge indices 0–11 for 2×2)
- `get_node(trace: list) -> DotsAndBoxesState | None`
  - Replays moves from root; returns `None` if any move is illegal (edge already drawn)
- `p1_legal_inputs(trace) -> list`
- `p2_legal_moves(trace) -> list`
- `is_terminal(trace) -> bool`
- `current_player(trace) -> str | None`

**Why the extra-turn mechanic is transparent here:**
`MCTSEquivalenceOracle._rollout` checks `node.player` at each step and branches on
`'P1'` vs `'P2'` — it does not assume strict alternation.  As long as
`DotsAndBoxesState.player` is correct after each move, no changes are needed upstream.

**Test file:** `tests/game/dots_and_boxes/test_game_nfa.py`
- [ ] `root` is initial state, P1 to move, 12 children (2×2 grid)
- [ ] `get_node([])` returns root
- [ ] `get_node([e])` returns correct state for each valid first edge `e`
- [ ] `get_node` returns `None` for illegal move (edge already drawn)
- [ ] After a box-completing move, `current_player` still returns the same player
- [ ] `p1_legal_inputs` returns `[]` when it is P2's turn
- [ ] `p2_legal_moves` returns `[]` when it is P1's turn
- [ ] `is_terminal` True when all 12 edges drawn (2×2 grid)
- [ ] Alphabet (`root.children.keys()`) contains all 12 edge indices

---

## Step 3 — Preference oracle (`src/game/dots_and_boxes/preference_oracle.py`)

Implement `DotsAndBoxesOracle` — mirrors `TicTacToeOracle` with bounded minimax
and a box-score heuristic at the depth cutoff.

**Must have:**
- `__init__(nfa, depth=None)` — `None` = unbounded (globally optimal)
- `preferred_move(prefix: list) -> int | None`
  - Best P2 edge at the current position; `None` if not a P2 decision point
- `compare(trace1, trace2) -> str`
  - `'t1'`, `'t2'`, or `'equal'` based on minimax value from P2's perspective

**Bounded minimax:**
```
_minimax(state, depth):
    if terminal        → state.value
    if depth == 0      → _heuristic(state)       ← NOT 0
    if P2's turn       → max over children
    if P1's turn       → min over children
```

Cache key: `(edges, player, p1_boxes, p2_boxes, depth)` — include scores since
box counts affect the heuristic.

**Heuristic at depth cutoff — box score differential:**
```
_heuristic(state) = (p2_boxes - p1_boxes) / total_boxes
```
Range: `[-1, 1]`, consistent in scale with terminal values `±1`.
Captures the current lead without lookahead — directly reflects who is winning.
Better than returning 0 because two states with different box counts will now
compare differently, giving `compare()` real signal.

**Test file:** `tests/game/dots_and_boxes/test_preference_oracle.py`
- [ ] `preferred_move` takes the edge that wins a box immediately when available
- [ ] `preferred_move` blocks P1 from completing a box on their next move (depth ≥ 2)
- [ ] `preferred_move` returns `None` on P1's turn
- [ ] `preferred_move` returns `None` on terminal state
- [ ] `compare` returns `'t1'` when trace1 ends with P2 winning, trace2 with draw
- [ ] `compare` returns `'t2'` when trace2 ends with P2 winning, trace1 with P1 winning
- [ ] `compare` returns `'equal'` for two terminal draws
- [ ] `_heuristic` returns `0.0` on empty board (equal scores)
- [ ] `_heuristic` returns positive when P2 leads in boxes
- [ ] `_heuristic` returns negative when P1 leads in boxes
- [ ] With `depth=1`, `compare` correctly distinguishes positions by immediate box count
- [ ] With `depth=None`, behaves identically to full minimax

---

## Step 4 — Wire up a learner script (`src/scripts/learner_dab.py`)

Create a script that runs L* + MCTS on Dots and Boxes end-to-end,
reusing `GameSUL`, `TableB`, `MCTSEquivalenceOracle` unchanged.

**What changes from `learner_ttt.py`:**
- Import `DotsAndBoxesNFA` and `DotsAndBoxesOracle` instead of TTT equivalents
- Add `--rows` / `--cols` arguments (default 2×2)
- Evaluation: play learned automaton against random P1, report win/draw/loss counts
  and final box score differential

**Test file:** `tests/game/dots_and_boxes/test_learner_dab.py`
- [ ] End-to-end: `run_Lstar` completes without error on 2×2 grid
- [ ] Learned automaton has at least 1 state
- [ ] With `depth=None` oracle, learned P2 never loses to a random P1 in 100 games
      on 2×2 grid (optimal play should achieve this)

---

## Step 5 — Benchmark sweep (`src/scripts/benchmark_dab.py`)

Mirror `benchmark_ttt.py` for Dots and Boxes.

**Additional sweep axis vs TTT benchmark:**
- `grid_size: [(2,2), (3,3)]` — test that bounded oracle + MCTS scales to larger grids
  where full minimax becomes expensive

**Figures to produce:**
- `dab_score_oracle_depth.png` — normalised score vs oracle lookahead, lines=K
- `dab_states_oracle_depth.png` — automaton states vs oracle lookahead
- `dab_score_K.png` — score vs K, lines=oracle_depth
- `dab_states_K.png` — states vs K
- `dab_score_grid.png` — score vs oracle_depth, rows=grid_size (shows scaling)

---

## Implementation notes

**State immutability:** Use `tuple[bool, ...]` for `edges` (hashable, cacheable).
`children` is a `@cached_property` on the state object, same as `TicTacToeState`.

**Extra-turn chains:** The chain can be arbitrarily long (all remaining boxes in a
near-complete grid).  The `children` property handles this recursively — each child
state correctly sets `player` based on whether a box was completed.  No special-casing
needed in the NFA or oracle.

**3×3 grid tractability:** 3×3 boxes = 20 edges = up to 2^20 ≈ 1M configurations.
Full minimax (`depth=None`) will be slow — bounded minimax (`depth ≤ 4`) is the
intended operating mode at this size.  This is the primary motivation for the
heuristic: it makes bounded depth genuinely useful rather than a fallback.

**Alphabet size:** 12 edges (2×2) vs 20 edges (3×3).  L* complexity scales with
alphabet size, so expect larger automata than TTT.  This is expected and interesting
to report in the benchmark.
