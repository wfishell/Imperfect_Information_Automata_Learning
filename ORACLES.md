# Preference Oracle Descriptions

Each game in this project has a preference oracle that acts as the "teacher" in the L\* learning framework. The oracle answers two kinds of questions: **membership queries** ("what move should P2 make here?") and **comparison queries** ("which of these two game traces is better for P2?"). This document describes, in natural language, how each oracle works.

---

## Shared Interface

All oracles expose the same two public methods:

- **`preferred_move(prefix)`** — Given a sequence of moves (a trace prefix) that leads to a P2 decision point, return the single move P2 should make. This is used by the SUL to answer L\* membership queries: when L\* asks "what does the system output on input sequence X?", the SUL replays the game using each player's moves and calls `preferred_move` at every P2 turn.

- **`compare(trace1, trace2)`** — Given two complete (or partial) game traces, return which one P2 prefers: `'t1'`, `'t2'`, or `'equal'`. This is used by the MCTS equivalence oracle when deciding whether a newly discovered game trajectory represents a genuine improvement over what the current hypothesis automaton would have played.

---

## Random Minimax Game (`PreferenceOracle`)

**File:** `src/lstar_mcts/preference_oracle.py`

### What the game looks like

The random minimax game generates a tree of fixed depth where P1 and P2 alternate turns. Every node in the tree carries a random integer value drawn uniformly from [0, 10]. There are no win/loss conditions — the quality of a game trace is measured by the sum of all node values visited along that path from root to leaf.

### How `preferred_move` works

The oracle navigates to the state reached by the prefix, then considers each move P2 could legally make. For each candidate move, it computes the **cumulative score** of extending the prefix by that move: it sums the node values along the resulting path from the root down to the new state. P2 picks the move that leads to the highest cumulative sum. This is not a full lookahead — it scores only the immediate next state's contribution, but because every node's value is already fixed in the tree, the cumulative sum captures everything on the path so far.

### How `compare` works

To compare two traces, the oracle computes the cumulative node-value sum for each trace independently and returns whichever is larger. Ties are `'equal'`. No game-theoretic reasoning is involved — the preference is purely based on which path accumulated more value.

### Key characteristics

- **No depth limit.** The entire game tree is precomputed and stored in memory; scoring is O(depth) per query.
- **No terminal values or win conditions.** Quality is entirely determined by the sum of random node values along the path.
- **Deterministic.** Given the same seed, the tree and therefore all oracle answers are fully determined.

---

## Tic-Tac-Toe (`TicTacToeOracle`)

**File:** `src/game/tic_tac_toe/preference_oracle.py`

### What the game looks like

Tic-Tac-Toe is played on a 3×3 grid. P1 plays X and P2 plays O, alternating turns. P2 (O) wins if it places three O's in a row, column, or diagonal. The terminal value for a state is +1 if O wins, −1 if X wins, and 0 for a draw. The board is represented as a 9-element tuple of cell values (empty, X, or O).

### How `preferred_move` works

From the state reached by the prefix, the oracle scores each legal P2 move by running **minimax** on the resulting child state. The minimax evaluates the full game tree recursively: at P2's turns it maximises the value, and at P1's turns it minimises. The oracle returns whichever move leads to the child with the highest minimax value. An optional `depth` parameter caps the lookahead. When `depth=None` (the default), the search is unbounded and the oracle plays optimally. When a depth limit is hit before reaching a terminal state, the oracle falls back to a **heuristic**.

### The heuristic (used when `depth` > 0 and the limit is reached)

At the depth cutoff, each of the eight lines (three rows, three columns, two diagonals) is evaluated independently. A line that contains only O pieces and empty cells contributes positively, proportional to how many O's are already in it (specifically, `count_of_O / 3`). A line containing only X pieces and empty cells subtracts the same amount. Lines that contain both X and O are dead and contribute nothing. The total raw score is normalised by dividing by 8 (the maximum possible contribution), keeping the result in the range (−1, 1) — consistent in scale with the terminal values ±1.

### How `compare` works

Each trace is evaluated by calling minimax on the state it leads to. Whichever trace ends in the higher-valued state is preferred. Illegal traces (those that reach no valid state) are assigned value −1, treating them as worst-case for P2.

### Key characteristics

- **Fully optimal by default.** With `depth=None`, the oracle is a perfect Tic-Tac-Toe player.
- **Memoised.** Results are cached by `(board, player, depth)` so repeated queries on the same state are O(1).
- **Tunable lookahead.** Shallow depths make the oracle weaker and faster, useful for benchmarking how learning quality degrades with oracle quality.

---

## Nim (`NimOracle`)

**File:** `src/game/nim/preference_oracle.py`

### What the game looks like

Nim is played with several piles of objects. On each turn, the active player removes any positive number of objects from exactly one pile. The player who takes the last object wins (normal play convention). P2 is the player we are learning a strategy for. The game state is a tuple of pile sizes. Terminal value is +1 if P2 wins and −1 if P1 wins.

### How `preferred_move` works

From the state reached by the prefix, the oracle scores each legal P2 move using **minimax**: it recursively evaluates child states, maximising at P2's turns and minimising at P1's turns. The move leading to the highest minimax value is returned. With `depth=None` (the default), the oracle plays the provably optimal Nim strategy via full lookahead. When a depth limit is set and reached, it falls back to a **heuristic**.

### The heuristic (used when `depth` > 0 and the limit is reached)

At the depth cutoff, the oracle uses the **Nim-sum** (the bitwise XOR of all pile sizes) to classify the current position. A non-zero Nim-sum means the player to move is in a winning position under optimal play; a zero Nim-sum means they are in a losing position. The heuristic returns ±0.5 (strictly inside the terminal ±1) based on whose turn it is and the Nim-sum:

- P2 to move, Nim-sum ≠ 0 → P2 is winning → +0.5
- P2 to move, Nim-sum = 0 → P2 is losing → −0.5
- P1 to move, Nim-sum ≠ 0 → P1 is winning (bad for P2) → −0.5
- P1 to move, Nim-sum = 0 → P1 is losing (good for P2) → +0.5

### How `compare` works

Each trace is evaluated by calling minimax on the state it leads to. Whichever trace ends in the higher-valued state is preferred from P2's perspective. Illegal traces are assigned value −1.

### Key characteristics

- **Theoretically solved.** With `depth=None`, the oracle implements the provably optimal Nim strategy and will never lose from a winning position.
- **Heuristic grounded in combinatorial game theory.** The Nim-sum gives the exact winning/losing classification for any position, making the heuristic precise even without full lookahead.
- **Memoised by `(piles, player, depth)`.** Repeated queries on equivalent pile configurations are O(1).
- **Moves are `(pile_index, amount)` tuples** identifying which pile to reduce and by how much.

---

## Dots and Boxes (`DotsAndBoxesOracle`)

**File:** `src/game/dots_and_boxes/preference_oracle.py`

### What the game looks like

Dots and Boxes is played on a rectangular grid of dots. Players take turns drawing a single edge between two adjacent dots. When a player completes the fourth side of a unit box, they score that box and **immediately take another turn** (the extra-turn rule). The player with more boxes when all edges are drawn wins. P2 is the player we learn a strategy for. Terminal value is +1 if P2 has more boxes, −1 if P1 has more, and 0 for a tie.

The game state is a flat boolean tuple of all edge slots (drawn or undrawn), together with the current player and each player's box count. Horizontal edges are indexed row-by-row first; vertical edges follow. Because completing a box grants an extra turn, the turn structure is non-trivial: when P2 completes a box, the state stays on P2's turn. The SUL handles this with a special `PASS` input — when it is still P2's turn after P2 scored, P1's only legal input is `PASS`, signalling to the learned Mealy machine that P2 should act again without P1 having moved.

### How `preferred_move` works

From the state reached by the prefix, the oracle scores each legal P2 edge using **minimax**: it recursively evaluates all child states, maximising at P2's turns (including P2's bonus turns after scoring) and minimising at P1's turns. The edge with the highest minimax value is returned. With `depth=None`, the search is unbounded and globally optimal. When a depth limit is set and reached, it falls back to a **heuristic**.

### The heuristic (used when `depth` > 0 and the limit is reached)

At the depth cutoff, the oracle estimates position quality as the **box score differential** from P2's perspective, normalised to (−1, 1):

```
(p2_boxes − p1_boxes) / total_boxes
```

A positive value means P2 currently holds more boxes; a negative value means P1 leads. Normalising by `total_boxes` keeps the result on the same scale as terminal values (±1), so `compare` gives meaningful signal even at the cutoff depth.

### How `compare` works

Each trace is evaluated by calling minimax on the state it leads to. Whichever trace ends in the higher-valued state is preferred. Illegal traces are assigned value −1.0.

### Key characteristics

- **Extra-turn rule is fully handled.** The minimax stays on P2's branch when P2 completes a box, so multi-box chains and captures are evaluated correctly.
- **Memoised by `(edges, player, p1_boxes, p2_boxes, depth)`.** The cache key includes box counts because two states with the same drawn edges but acquired under different turn orders can differ in value.
- **Moves are integer edge indices** identifying which edge P2 should draw.
- **Tunable lookahead.** Shallow oracle depths make benchmarking feasible on larger grids (e.g., 3×3) where unbounded minimax would be prohibitively slow.
