# Thoughts: Looking For New Ideas for Testing Games

We currently have game theory-based games where we aim to learn a system player (controller) that can play optimally using automata learning with preference queries that represent suboptimal greedy play. For instance, in tic tac toe, we use a prefernce oracle to determine what traces are better given a greedy heuristic but still recover an optimal strategy through MCTS.

Abstract:

Reinforcement learning underpins modern AI, from robotic control to game-playing agents to LLM post-training, but suffers from poor verifiability, sample inefficiency, and lack of interpretability. Symbolic RL mitigates these issues by representing policies as discrete, structured objects, but existing approaches either fail to scale or fall short of delivering full interpretability and sample efficiency. We present a modified active automata learning algorithm that extracts symbolic policies for a system player, where the system under learning is not static but evolves through the composition of an NFA over the action space and a preference oracle. A globally optimal preference oracle would reduce learning to standard L*, but realistic oracles are only locally consistent. To recover deeper policies missed by the oracle, we interleave L* with MCTS rollouts seeded from a secondary observation table of UCB-scored alternatives. When a rollout reveals that a deviation leads to more preferential downstream strategies, that deviation is returned as a counterexample that refines both the oracle and the hypothesis, yielding a Mealy-style controller amenable to model checking. We prove that under structural assumptions on the policy space, this procedure converges to the optimal policy without exhaustive search. On classic game-theory benchmarks, our preference-guided active learning extracts controllers competitive with MCTS and Q-learning, demonstrating efficient extraction of high-quality policies. We further extract safe, verifiable controllers with optimal behavior on opaque OpenAI Gymnasium environments, showing the method generalizes beyond fully specified games. We pair this algorithm with a model checking layer to learn adaptive, optimal policies that satisfy a given set of safety specifications, establishing automata learning as a viable path toward symbolic RL.


We want to expand this to games where we can have a suboptimal preference oracle that with MCTS will learn a globally optimal solution in a controller that we can model check and inject safety conditions to show that our learned controller can be checked and injected a saftey condition to prevent reward hacking. 

I want to think of a few games that are more intersting. The big issue is how we encode the input space for the enviornment. If the input space is large, the states explode and MCTS explores too much and we never converge. We see this in dots and boxes on a 4 by 4 grid. The question then is, are there more intersting RL games like Atari Games that we can figure out a clever encoding for to prevent state explosion with safety conditions that show our automata learning can learn a controller for? We really just need a input space that will not blow up. See dots and boxes for example. But I think less grid based games could work with a clever encoding. 


# Idea: Frogger




