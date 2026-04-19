# Preference-Guided Automata Learning with MCTS Equivalence Querying

## Project Description

This project develops an automata learning algorithm for strategy synthesis in imperfect-information games. The goal is to learn a finite automaton encoding the system's (P2's) optimal response strategy across all possible environment (P1) inputs.

**P1 is the environment**: provides inputs to the system. We make no assumptions about P1 and treat all possible P1 inputs as equally valid. P1 is not optimised — it is enumerated.

**P2 is the system**: produces outputs in response to P1's inputs. This is what we are learning. The automaton maps P1's input history to P2's best response at each step.

The algorithm extends L* with two key ideas:

1. **A history-conditioned preference oracle** that evaluates candidate P2 responses by looking back across the full trace seen so far, assigning local preferences that are globally consistent over all explored traces.

2. **An MCTS-based equivalence oracle** that searches for P1 input sequences where P2's current strategy is suboptimal — cases where an alternative P2 response leads to better aggregate outcomes across all P1 continuations.

### The Two Observation Tables

**Table A** — the current strategy hypothesis. Membership queries always return the max-preference P2 response at each prefix (greedy with respect to the oracle). Maps P1 input histories to P2's recommended response.

**Table B** — the exploration record. Stores all visited branches with their visit counts and SMT-derived preference values. Represents the partial game tree explored so far, including alternative P2 responses to what Table A currently prescribes.

### Preference Representation

Preferences are generated locally at each step as pairwise orderings over (input, output) pairs: `t1 > t2`. The oracle conditions its preference on the full trace history to that point. An SMT solver derives consistent numeric values from the collected pairwise orderings at each depth level, enabling comparison and aggregation. Values are normalized within each depth level — scores at depth k are comparable to each other but not to depth k+1.

### Variable-Length Games

Games may end at different depths depending on how they are played. Terminal states are modeled as absorbing sink states in the automaton — any extension of a terminal trace maps to the same sink, satisfying the L* closedness condition automatically for those rows.

Terminal payoffs are treated as **depth-independent**: a payoff of 80 at depth 4 and a payoff of 80 at depth 8 are directly comparable. This is the standard minimax assumption — outcomes have inherent value regardless of how many moves it took to reach them. Terminal payoffs are recorded raw and skip the within-depth normalization applied to intermediate state preferences.

MCTS rollouts terminate early on reaching a terminal state rather than forcing to depth N. Discounting terminal payoffs by depth (preferring faster wins) is a future extension.

### MCTS Tree Search as Contextual Bandits with Regret

The MCTS search is formalized as a contextual bandit problem. The context at each node is the trace prefix σ leading to that node. The reward is the SMT-derived preference value of the resulting trace at depth N.

The algorithm minimizes cumulative regret over P2's choices. Known preferences over already-explored P2 responses directly inform which unexplored responses are worth visiting next — an unexplored P2 response adjacent to high-preference explored ones gets a higher prior and is visited sooner.

#### Asymmetric Traversal: P1 vs P2 Nodes

The traversal is fundamentally different at P1 and P2 nodes:

**At P1 nodes (environment inputs):** P1 is not optimised. UCB is used purely for *coverage* — to ensure that across K rollouts, all P1 inputs get visited. This is not strategic selection; it is exploration to ensure no P1 input sequence is neglected.

```
UCB_coverage(σ, a) = c * sqrt(log(total_visits(σ)) / visits(σ, a)) * α^(-depth(σ))
```

**At P2 nodes (system responses):** P2 choices are explored probabilistically, not greedily. A softmax over SMT values is used, with unexplored actions assigned a high prior (they have unknown value and deserve exploration). As more P2 responses are explored and SMT values are assigned, the probability mass concentrates toward higher-value responses while maintaining exploration of uncertain regions.

```
P(P2 chooses a at σ) ∝ exp(SMT_value(σ, a) / τ)   for explored actions
P(P2 chooses a at σ) = high_prior                    for unexplored actions
```

Where τ is a temperature parameter controlling exploitation vs exploration.

This asymmetry is intentional: we want coverage of all P1 inputs (because any of them could be played) but we want intelligent, value-guided exploration of P2 responses (because we are learning the best one).

#### Zero-Probability Pruning at Depth N

After collecting SMT values for all explored leaves at depth N, any leaf whose value falls below the median of all explored leaves at depth N is assigned probability 0 in Table B. This node is never selected again. The assumption is that if a P2 response leads to below-median outcomes across the explored P1 continuations, it is not worth further exploration.

#### Mid-Exploration Pruning

When following a deviation from Table A (a P2 response different from what Table A prescribes), check at each intermediate depth d whether the exploration should continue:

- Collect all traces explored so far within this deviation subtree at depth d
- Compute the fraction that are less preferred than Table A's traces at depth d (using current SMT values)
- If this fraction exceeds 0.5 (majority below Table A), scale the remaining budget for this subtree by `(1 - fraction_below)`
- If the remaining budget falls below a minimum threshold, abandon this subtree entirely and redirect budget elsewhere

This is pessimistic pruning: if the evidence already points against a subtree being a counterexample at an intermediate depth, do not wait until depth N to find out. The freed budget is redistributed to other unexplored deviation subtrees.

Together, mid-exploration pruning and depth-N zero-probability pruning form a layered system: gentle proportional budget reduction at intermediate depths, hard elimination at depth N.

**Fixed search budget**: every equivalence query receives exactly K MCTS rollouts total. K is a configurable parameter that directly controls quality vs cost.

### Counterexample Detection and Acceptance

Within the K-rollout budget, MCTS explores alternative subtrees — branches where P1 deviates from Table A's current choice at some depth k — down to depth N. After the budget is exhausted:

- Collect all leaves at depth N for the alternative subtree and Table A's subtree
- Assign SMT values to all leaves from the pairwise preference ordering at depth N
- Compute mean value uniformly over all explored P2 responses in each subtree
- Accept counterexample if: `mean(alternative) > mean(table_A) + ε`

No assumption is made about how P2 plays — all P2 responses are weighted equally. The ε threshold guards against committing to an update when the preference ordering at depth N is still too sparse to be stable.

### Table B Update — Regardless of Counterexample

**Table B is updated on every equivalence query**, whether or not a counterexample is found. Each rollout adds:

- Incremented visit counts on all nodes along the rollout path
- Updated value estimates via backpropagation
- New pairwise preferences at depth N, fed back to the SMT solver
- Updated SMT values across the depth N leaves (re-solved incrementally)
- New zero-probability assignments for leaves falling below the median

This means the MCTS tree accumulates information continuously. Later equivalence queries benefit from all prior exploration, focusing budget on regions not yet pruned and not yet well-understood.

### Language Closure After Update

When Table A is updated at branching point k, the algorithm explores all P2 responses from k+1 onward on the new branch. This closes the observation table under the full language — the learned automaton must encode P1's strategy for every possible P2 continuation, not just the one MCTS happened to find.

---

## Implementation Steps

### Step 1: Random Minimax Game Generator
- Generate random two-player zero-sum game trees with configurable branching factor and depth
- Each node stores available actions for the current player
- Leaf nodes have numeric payoff values
- Export game trees in a format compatible with the NFA and oracle interfaces

### Step 2: Preference Oracle
- Implement a history-conditioned preference oracle
- Input: full trace to current node + two candidate next actions
- Output: pairwise preference ordering `a > b`
- Oracle evaluates by simulating forward from each candidate and comparing resulting subtree values
- Must be consistent: if `a > b` and `b > c` then `a > c`

### Step 3: SMT Value Assignment
- Collect pairwise preferences at each depth level across all explored traces
- Use an SMT solver (Z3) to find a consistent numeric assignment satisfying all ordering constraints
- Normalize values within each depth level
- Re-solve incrementally as new preferences are added

### Step 4: Table B — Exploration Tree
- Implement as a trie indexed by trace prefix
- Each node stores: visit count, current SMT value, set of explored actions, set of unexplored actions, zero-probability flag
- Support UCB score computation per node with depth discounting:
  ```
  UCB(σ, a) = value(σ, a) + c * sqrt(log(total_visits(σ)) / visits(σ, a)) * α^(-depth(σ))
  ```
- Unexplored actions always score higher than explored ones (forced first visit)
- Zero-probability nodes are excluded from UCB selection permanently

### Step 5: MCTS Counterexample Search — Explicit Algorithm

Each equivalence query runs exactly K rollouts. The algorithm returns the best counterexample found, or None.

```
MCTS_EQUIVALENCE_QUERY(Table_A, Table_B, K, N, ε, τ, budget_threshold):

  remaining_budget ← dict()   // per deviation subtree

  for rollout = 1 to K:

    // --- SELECTION ---
    node ← root
    path ← []
    active_deviation ← None

    while depth(node) < N and not terminal(node):

      if is_P1_node(node):
        // Environment: UCB for coverage, skip zero-probability nodes
        a ← argmax_{a: not zero_prob(node,a)} [
              c * sqrt(log(visits(node)) / visits(node, a)) * α^(-depth(node))
            ]

      else:  // P2 node
        table_a_action ← Table_A.lookup(path)

        // Probabilistic P2 selection
        // Unexplored actions get high prior; explored get softmax over SMT values
        probs ← {}
        for a in actions(node):
          if visits(node, a) == 0:
            probs[a] ← HIGH_PRIOR
          else:
            probs[a] ← exp(SMT.value(node, a) / τ)
        normalise(probs)
        a ← sample(probs)

        if a ≠ table_a_action and active_deviation is None:
          active_deviation ← (node, a)
          remaining_budget[active_deviation] ← remaining_budget.get(
                                                 active_deviation, K)

      path.append((node, a))
      node ← transition(node, a)

      // --- MID-EXPLORATION PRUNING ---
      if active_deviation is not None:
        d ← depth(node)
        subtree_traces ← Table_B.traces_in_subtree(active_deviation, depth=d)

        if len(subtree_traces) > 0:
          table_a_mean ← mean(SMT.value(t)
                              for t in Table_A.traces_at_depth(d))
          fraction_below ← count(SMT.value(t) < table_a_mean
                                  for t in subtree_traces) / len(subtree_traces)

          if fraction_below > 0.5:
            remaining_budget[active_deviation] *= (1 - fraction_below)

            if remaining_budget[active_deviation] < budget_threshold:
              break  // abandon rollout, redirect budget

    // --- EVALUATION ---
    trace ← extract_trace(path)
    reference ← Table_A.sample_trace_at_depth(depth(node))
    ordering ← oracle.compare(trace, reference)
    SMT.add_ordering(ordering)

    // --- BACKPROPAGATION ---
    smt_values ← SMT.solve()
    for (node, a) in path:
      Table_B.increment_visits(node, a)
      Table_B.update_value(node, a, smt_values)

  // --- DEPTH-N ZERO-PROBABILITY PRUNING ---
  leaves ← Table_B.leaves_at_depth(N)
  median_val ← median(SMT.value(l) for l in leaves)
  for leaf in leaves:
    if SMT.value(leaf) < median_val:
      Table_B.set_zero_probability(leaf)

  // --- COUNTEREXAMPLE DETECTION ---
  best_ce ← None
  best_advantage ← ε

  for deviation in Table_B.all_deviation_points():
    alt_leaves ← Table_B.leaves_in_subtree(deviation, depth=N)
    alt_leaves ← [l for l in alt_leaves if not zero_prob(l)]
    if len(alt_leaves) == 0: continue

    alt_mean ← mean(SMT.value(l) for l in alt_leaves)
    ta_mean  ← mean(SMT.value(l)
                    for l in Table_A.corresponding_leaves(deviation, depth=N))

    if alt_mean - ta_mean > best_advantage:
      best_advantage ← alt_mean - ta_mean
      best_ce ← deviation

  return best_ce   // None if no deviation clears the ε threshold
```

### Step 6: Subtree Aggregate Comparison

The counterexample detection (final block of the algorithm above) compares:

- **Alternative subtree**: all non-zero-probability leaves at depth N reachable via the deviation P2 action, averaged uniformly over P1 continuations
- **Table A subtree**: leaves at depth N from Table A's current P2 choice at the same branching point, averaged uniformly over P1 continuations

All P1 inputs are weighted equally — no assumption is made about how P1 behaves. The deviation with the largest mean advantage above ε is returned as the counterexample. If none clears ε, the query returns None.

### Step 7: Table A Update and Closure
- On counterexample acceptance: update Table A at the branching point — P2 now takes the alternative action there
- From the branching point onward, enumerate all P1 inputs on the new branch
- For each P1 input continuation, run the greedy membership query to determine P2's best response
- Add all resulting traces to the observation table
- Re-run L* consistency and closedness checks

### Step 8: Integration with L*
- Plug the MCTS equivalence oracle into the L* loop as a drop-in replacement for random walk
- Membership queries always run to a terminal game state (leaf node) — they are never truncated at depth N
- Equivalence queries run exactly K MCTS rollouts to the current search depth N, returning a counterexample or None
- Table B persists and accumulates across all equivalence queries and all depth levels — it is never reset
- Termination: when Table A has not been updated for M consecutive equivalence queries **at the full game depth D**

### Iterative Deepening

The MCTS search depth N starts shallow and increases as the algorithm converges at each level. This prevents premature termination — the algorithm cannot conclude it has learned the game until it has reached the leaf nodes where terminal payoffs live.

**The deepening schedule:**
1. Run equivalence queries at depth N until M consecutive queries produce no Table A update
2. Increase N (e.g. N ← N + 1 or N ← 2N)
3. Resume equivalence queries — new counterexamples may appear at the deeper depth that were invisible before
4. Repeat until N = D (full game depth) and M stable queries are achieved there

**Why this is necessary:** if N is fixed below D, the preference oracle evaluates intermediate states rather than terminal outcomes. The algorithm can stabilize at M consecutive stable queries having never seen the end of any game — it has learned a partial strategy, not the full game. Iterative deepening ensures convergence is always grounded in complete traces.

**Efficiency:** convergence at shallow depths is fast — the tree is small and K rollouts cover a large fraction of it. The earlier depth levels pre-populate Table B with useful structure before the expensive deep search begins. Each deepening step starts with a warm Table B rather than from scratch.

**Membership queries are decoupled from N.** The search depth N is the MCTS horizon for equivalence queries only. Membership queries in L* always follow a trace to its terminal state regardless of N.

### Convergence and Termination

The termination signal — M consecutive equivalence queries at depth N=D without a Table A update — becomes strictly more meaningful as the algorithm progresses. This is not an arbitrary stopping rule: it reflects two compounding effects that emerge naturally from the search.

**Effect 1: The live search space shrinks.**
Zero-probability pruning permanently closes off depth-D leaves that fall below the median after each query. Over time, a growing fraction of the tree is marked dead. K rollouts now cover a higher fraction of the *remaining* live nodes, so the probability that a counterexample exists in the live space but was missed by K rollouts decreases monotonically.

**Effect 2: Fewer counterexamples exist.**
As Table A approaches the optimal strategy, the advantage gap `mean(alternative) - mean(table_A)` shrinks. Most alternative branches no longer clear the ε threshold. There are genuinely fewer counterexamples to find.

These two effects compound: the search space is smaller *and* the target is harder to hit. M stable queries early in the run is weak evidence — the MCTS simply may not have reached the relevant parts of the tree yet. M stable queries late in the run, when most of the tree has been pruned and Table A has not changed in many iterations, is strong evidence that no ε-significant counterexample remains in the live search space.

This gives the algorithm a PAC character: the effective probability δ of missing a valid counterexample decreases throughout the run as a function of pruning coverage and Table A stability. The user-facing parameters K, M, and ε together determine the confidence of the termination guarantee — larger K and M, and smaller ε, yield a tighter bound on the probability of stopping prematurely.

---

## Experiment Design

### Games

Generate random two-player zero-sum minimax game trees with:
- Branching factor b ∈ {2, 3, 4}
- Depth d ∈ {4, 6, 8, 10}
- Leaf payoffs sampled uniformly from [0, 100]
- Players alternate turns (P1 maximizes, P2 is unconstrained)

Generate N=50 random games per (b, d) configuration.

### Experiment 1: MCTS Aggressiveness vs Strategy Quality

**Question**: does more aggressive MCTS exploration yield a learned automaton whose strategy approaches optimal?

**Vary**: MCTS rollout budget per equivalence query — {100, 500, 1000, 5000, 10000}

**Measure** for each budget:
- Average P1 payoff of the learned strategy across all 50 games, over 1000 plays against random P2
- Average P1 payoff against minimax-optimal P2
- Number of states in the learned automaton
- Number of equivalence queries before convergence

**Expected result**: higher rollout budget → strategy closer to optimal, more states in the automaton reflecting finer strategic distinctions.

### Experiment 2: Game Complexity vs Automaton Complexity

**Question**: how does the size of the search space affect the complexity of the learned automaton and the difficulty of learning it?

**Vary**: game tree size via (b, d) configurations

**Measure** for each configuration:
- Number of states in the learned automaton
- Total oracle queries (membership + equivalence) to convergence
- Gap between learned strategy payoff and minimax-optimal payoff at fixed MCTS budget

**Expected result**: larger games → more automaton states, more queries, larger optimality gap at fixed budget.

### Evaluation Metric

For both experiments, strategy quality is measured as:

```
normalized_payoff = (learned_payoff - random_payoff) / (optimal_payoff - random_payoff)
```

Where:
- `random_payoff` = average payoff of a uniformly random P1 strategy
- `optimal_payoff` = minimax-optimal payoff (computed by backward induction on the known tree)
- `learned_payoff` = average payoff of the learned automaton strategy

A score of 0 is random play, 1 is optimal. This normalizes across games with different payoff scales.
