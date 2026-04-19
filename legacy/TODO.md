# Tic-Tac-Toe — Implementation TODO

Work through these steps in order. Each step has a test file to verify before moving on.
Run tests from the project root with: `python -m pytest tests/game/tic_tac_toe/ -v`

---

## Step 1 — Board state (`src/game/tic_tac_toe/board.py`)

Implement `TicTacToeState` — the core game object the NFA and oracle will use.

**Must have:**
- Board represented as a 9-tuple of `0` (empty), `1` (X/P1), `2` (O/P2)
- `player: str` — `'P1'` or `'P2'` (whose turn it is)
- `children: dict` — `{square_index: TicTacToeState}` for each legal move
  - This is what `MCTSEquivalenceOracle` iterates over during rollouts
- `is_terminal() -> bool` — True if someone won or board is full
- `winner() -> str | None` — `'P1'`, `'P2'`, `'draw'`, or `None`
- `value: int` — terminal score from P2's perspective: `+1` win, `0` draw, `-1` loss
  - Non-terminal nodes return `0` (the algorithm only scores terminal traces)

**Test file:** `tests/game/tic_tac_toe/test_board.py`
- [ ] Initial board: 9 empty squares, P1 to move
- [ ] `children` at root has 9 entries (all squares legal)
- [ ] `children` after 1 move has 8 entries
- [ ] `current_player` alternates P1 → P2 → P1 ...
- [ ] `is_terminal` False on empty board
- [ ] `is_terminal` True when X wins (row, col, diagonal — test all 8 win patterns)
- [ ] `is_terminal` True when O wins
- [ ] `is_terminal` True on full board (draw)
- [ ] `winner` returns correct result for each case
- [ ] `value` is `+1` for O win, `-1` for X win, `0` for draw

---

## Step 2 — NFA interface (`src/game/tic_tac_toe/game_nfa.py`)

Implement `TicTacToeNFA` — the object the algorithm navigates during L* and MCTS.

**Must have:**
- `root: TicTacToeState` — the initial empty board state (P1 to move)
  - `root.children.keys()` becomes the L* input alphabet (squares 0–8)
- `get_node(trace: list) -> TicTacToeState | None`
  - Replays the move sequence from root, returns resulting state
  - Returns `None` if any move in the trace is illegal
- `p1_legal_inputs(trace) -> list` — legal moves when it is P1's turn
- `p2_legal_moves(trace) -> list` — legal moves when it is P2's turn
- `is_terminal(trace) -> bool`
- `current_player(trace) -> str | None`

**Why this works with the existing algorithm:**
`MCTSEquivalenceOracle._rollout` calls `self.nfa.get_node(trace)` and then accesses
`node.children`, `node.player`, `node.is_terminal()` — exactly what `TicTacToeState` provides.
No changes needed to `MCTSEquivalenceOracle`, `GameSUL`, or `TableB`.

**Test file:** `tests/game/tic_tac_toe/test_game_nfa.py`
- [ ] `root` is initial state, P1 to move, 9 children
- [ ] `get_node([])` returns root
- [ ] `get_node([4])` returns state after X plays center
- [ ] `get_node([4, 0])` returns state after X center, O top-left
- [ ] `get_node` returns `None` for illegal move (replaying same square)
- [ ] `p1_legal_inputs` returns correct squares on P1's turn
- [ ] `p2_legal_moves` returns correct squares on P2's turn
- [ ] `p1_legal_inputs` returns `[]` on P2's turn (and vice versa)
- [ ] `is_terminal` True on a won/drawn trace
- [ ] Alphabet (`root.children.keys()`) contains all 9 squares

---

## Step 3 — Preference oracle (`src/game/tic_tac_toe/preference_oracle.py`)

Implement `TicTacToeOracle` — tells the algorithm which P2 moves are better.

**Must have:**
- `preferred_move(prefix: list) -> int | None`
  - Returns the best P2 (O) move from current position using minimax
  - Returns `None` if not a P2 decision point
- `compare(trace1: list, trace2: list) -> str`
  - Returns `'t1'`, `'t2'`, or `'equal'` based on terminal scores
  - Used by `SMTValueAssigner` to build preference constraints

**Implementation note:** use a simple minimax with alpha-beta over `TicTacToeState.children`.
The game tree is tiny (≤9! = 362,880 nodes), so no caching needed, but you can add it.

**How this plugs in:** `GameSUL.step()` calls `oracle.preferred_move(self._trace)` —
same call signature as `PreferenceOracle` in minimax. No SUL changes needed.

**Test file:** `tests/game/tic_tac_toe/test_preference_oracle.py`
- [ ] `preferred_move` takes winning move when O can win immediately
- [ ] `preferred_move` blocks X from winning on next move
- [ ] `preferred_move` returns `None` on P1's turn
- [ ] `preferred_move` returns `None` on terminal state
- [ ] `compare` returns `'t1'` when trace1 ends in O win, trace2 ends in draw
- [ ] `compare` returns `'t2'` when trace2 ends in O win, trace1 ends in X win
- [ ] `compare` returns `'equal'` for two draws
- [ ] `compare` returns `'t2'` when trace1 is X win, trace2 is O win

---

## Step 4 — Wire up a learner script (`src/scripts/learner_ttt.py`)

Create a script that runs L* + MCTS on tic-tac-toe end-to-end,
reusing `GameSUL`, `TableB`, `MCTSEquivalenceOracle` unchanged.

**What changes from `learner.py`:**
- Import `TicTacToeNFA` and `TicTacToeOracle` instead of minimax equivalents
- No `generate_tree` call — `TicTacToeNFA()` constructs its own root
- Evaluation: play the learned automaton against random X and count wins/draws/losses
  instead of the minimax normalised score

**Test file:** `tests/game/tic_tac_toe/test_learner_ttt.py`
- [ ] End-to-end: `run_Lstar` completes without error on tic-tac-toe
- [ ] Learned automaton has at least 1 state
- [ ] Learned automaton never loses to a random opponent in 100 games
  (a correct minimax oracle means O should never lose)

---

## Step 5 — (Optional) Visualized learner (`src/scripts/learner_viz_ttt.py`)

Copy `learner_viz.py`, swap in tic-tac-toe components, add a board printer
to `src/viz/visualizer.py` so the game tree shows as a grid instead of a tree.

No new tests needed — covered by Step 4.
