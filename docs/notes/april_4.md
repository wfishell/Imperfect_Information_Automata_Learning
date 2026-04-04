# Notes: April 4th

I want to use this document as a chance to formalize the current problem we are tackling:

**The Problem:**

We are provided a game trace. We can define a game trace as a series of player inputs and game outputs.

Let us take an example from Kuhn Poker where players can bet:

```
"!a0 & !a1 & !a2 & !bs & !c1hi & c1lo & c2hi & !c2lo & !cur_bet & deal & m1b0 & m1b1 & !m1b2 & m2b0 & m2b1 & !m2b2 & !p1 & !p1b & !p2b & !p2c & !p2r/!game_over & !win1 & !win2"
```

Where the format is inputs/outputs. The trace here consists of atomic propositions. This is one hand. 

We might play another hand and yield:

```
"!a0 & a1 & !a2 & !bs & !c1hi & c1lo & c2hi & !c2lo & !deal & !m1b0 & !m1b1 & !m1b2 & p1 & !p1b & !p2b & !p2c & !p2r/!win1 & !win2",
```

A series of these input/output pairs make up a singular trace.

We can imagine a series of traces as a path graph.

inp/output step * - - - - - * inp/output step - - - - * inp/output step

                ^^^^^^^ Full Game Trace ^^^^^^^'

If two traces share the same inp/output step, that is a unique vertex shared by both paths. 

In the end, we get a graph structure, representing the different generated traces.

**Prompt**

I want you to build a file in test/pipeline/corpus that converts the corpus.json schema from data/Kuhn_Poker:

```
 "_schema": {
    "description": "Sampled trace corpus generated from kuhn_poker.dot via dot_trace_generator.py",
    "count": "number of traces",
    "traces": "list of traces; each trace is a list of step strings in 'input_formula/output_formula' format"
  },
```

into a graphical structure in the manner I showed using NetworkX. 

That way, I can begin to perform graphical analysis.

## Further Notes

This is really cool. We now have a graphical representation of our traces.

Our next problem is the following: 

Given that we have a set of traces, represented graphically. We want the closure over all the traces. That is to say, I want to look at every possible subtrace in our set of traces. These make up our entire set. We can assume that we have values assigned to each trace such that we have a total order over all the traces.

This indicates how we prefer traces over other traces.

For instance, let us assume we have the following traces: T_1, T_2, and T_3. 

For the sake of example, we assume this ordering: T_1 > T_2 > T_3. This indicates how we prefer traces over other traces. Now, here is the big catch:

How do we go about determining the preference values of sub traces? 

We may prefer T_1 over T_2 despite a specific node in T_1 being actually less optimal than a player move in T_2. 

What graphical algorithms can I use to tackle this? Give me some ideas.

**LLM Response**

⏺ This is a classic credit assignment problem on a DAG. Here are the most relevant algorithmic families, roughly ordered from simple to sophisticated:

  ---
  1. Weighted Flow / Score Propagation

  Model trace preference as flow injected at START. Each trace injects rank(T_i) units of flow; flow is split proportionally at shared nodes.

  - Forward pass: each node accumulates a weighted average of the preference scores of all traces passing through it.
  - Result: a node inherits the "average quality" of the traces it belongs to.
  - Limitation: uniform — doesn't distinguish where in a trace a node sits.

  This is the baseline. It's fast and interpretable but ignores positional context.

  ---
  2. Shapley Value Attribution

  Treat each step/node as a "player" and each trace as a "coalition" with value = rank(T_i). The Shapley value of a node is its average marginal contribution across all traces
  containing it.

  φ(v) = Σ over traces T containing v: [rank(T) / |T|]

  This is the fairness-optimal solution — it distributes credit across nodes such that no node is systematically over- or under-credited. The "locally suboptimal node in a globally
   preferred trace" problem is handled by averaging across the full coalition.

  The subtrace version: Shapley value of a subtrace = sum of Shapley values of its nodes.

  ---
  3. Linear Programming (Preference Constraints)

  The most principled direct approach. Define a scalar value x_v ≥ 0 for each node v. Define the value of a trace as:

  val(T_i) = Σ x_v  for v ∈ T_i

  Then encode your total order as linear constraints:

  val(T_1) > val(T_2) > val(T_3) > ...

  i.e., val(T_i) - val(T_{i+1}) ≥ ε for some margin ε > 0. Solve as an LP or QP.

  - Feasible region = set of node-value assignments consistent with your ordering.
  - If infeasible, it tells you the ordering is contradictory given the graph structure (two traces share nodes in a way that makes it impossible to assign consistent local
  values).
  - You can add a max-margin objective (SVM-style) to find the most "confident" consistent assignment.

  This directly encodes the tension you identified — the LP will either find consistent local values or declare the contradiction irresolvable.

  ---
  4. Topological Sort + Dynamic Programming (Value Iteration)

  Since the graph is a DAG (acyclic by construction — traces don't loop), you can do backward induction:

  - Assign terminal nodes a value derived from which traces they end.
  - Propagate backwards: V(v) = weighted average of V(successors of v), weighted by edge flow.

  This is essentially the Bellman equation for a deterministic MDP, where preference rank plays the role of return. It gives each node a "future value" — how good are the outcomes
  reachable from here, in expectation over the trace distribution.

  This naturally handles your tension: a locally suboptimal node early in T_1 still gets a high value if T_1's terminal outcome is preferred.

  ---
  5. Personalized / Preference-Weighted PageRank

  Run PageRank where the teleportation distribution is biased toward nodes that appear in high-preference traces. Concretely:

  - Personalization vector p_v ∝ Σ rank(T_i) for traces T_i containing v.
  - Steady-state distribution = importance score per node.

  This is especially natural because your graph already has the right structure (directed, weighted edges). It's robust to cycles if your corpus ever has them.

  ---
  Recommendation for Your Setting

  Given that you already have a total order and a DAG, I'd start with:

  1. LP approach — because it directly answers "is there a consistent local assignment?" and if not, tells you where the contradiction lies. This feeds directly into automata
  learning (inconsistent preferences = counterexample for the learner).
  2. Shapley values — as a fallback when LP is infeasible or underdetermined; gives you a principled "best guess" at node quality even under contradictory global preferences.

  The LP approach is particularly interesting for your research context because infeasibility corresponds to imperfect information creating irreducible ambiguity — which is exactly
   the phenomenon you're trying to model with the automata learner.