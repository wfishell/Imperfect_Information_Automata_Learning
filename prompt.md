# Benchmarking Suite for L* MCTS 

We will be implementing a benchmarking suite for the L* MCTS (Monte Carlo Tree Search) to demonstrate how our L* MCTS performs against a traditional MCTS.

## High Level Overview

We currently have 5 games: 

1. Dots and Boxes
2. Hex
3. Minimax
4. Nim
5. Tic Tac Toe

The goal is to identify how the automata learned from L* MCTS performs against traditional MCTS. We want to compare how they perform in the five games given above. We will do this by learning an automata from L* MCTS under different parametrizations. We will then evaluate each of these variations of automata against different variations of environment players. We will then evaluate how a traditional MCTS performs against the same variations of environment player. The goal is to evaluate how our L* MCTS's game performance compares to an MCTS approach.

## Game Variations

### Dots and Boxes

**Parameters**:

1. Rows, Cols: 
    - (2, 1), (2, 2), (3, 2)
  
2. Depth_n
   -  1, 2, 4, 6
  
3. MCTS Roll Out 'k'
   -  10, 50, 100, 200

4. Oracle Depth
   - None (Globally Optimal), 0, 1, 2

### Hex

**Parameters**:

1. Size:
   - (3,3), (4,4) 

2. Depth_n
    - 1, 2, 3, 4, 6

3. MCTS Roll Out 'k'
   - 10, 50, 100, 200 

4. Oracle Depth
    - None (Globally Optimal), 0, 1, 2

### Minimax

**Parameters**:

1. Depth
   - 4, 6, 8, 10, 12

2. Seed
   - 42

3. Depth_n
    - 1, 2, 4, 6

4. MCTS Roll Out 'k'
   - 10, 50, 100, 200

### Nim

**Parameters**:

1. Piles
   - (1,2,3), (1,3,5), (2,3,4), (1,2,3,4)

2. Depth_n
    - 1, 2, 4, 6

3. MCTS Roll Out 'k'
   - 10, 50, 100, 200 

4. Oracle Depth
    - None (Globally Optimal), 0, 1, 2

### Tic Tac Toe

**Parameters**:

1. Depth_n
  - 1, 2, 4, 6

2. MCTS Roll Out 'k'
   - 10, 50, 100, 200 
       
3. Oracle Depth
    - None (Globally Optimal), 0, 1, 2


## Benchmarking Game Variations

The goal again is to compare how our L* MCTS derived automata performs on the above games based on all combinations of the above parameterizations against a traditional MCTS against the following:

**Core Comparisons**:

1. Query Cost: 
 - Number of 'k' roll outs * depth of roll out. Number of MCTS roll outs and depth versus number of L* MCTS roll outs and their respective depths of each roll out.

2. Robustness against opponent:   
   - MCTS and L* MCTS against random environment player.
   - MCTS and L* MCTS against greedy environment player.
   - MCTS and L* MCTS against optimal environment player.
    **Core Metric**: Game Win Rate
    **Policy**: Both MCTS and L* MCTS use the same MCTS policies to keep comparisons fair.
    **Game Count**: N = 500 on fixed grid.

3. Interpretability of Artifact:
   - We want to show how the variations alter the interpretability of the mealy machine output. How does the states and transitions change?

4. Sensitivity to Oracle Quality:
   - How does the win-lose rate alter under quality of the preference oracle? Does a preference oracle with further look-ahead alter win-lose rate?

## Environment Player Information

Greedy environment player is based on the existing heuristics used in the greedy/sub-optimal preference oracles.

## Benchmarking Specific Information

- We should test every combination for each variation.
- There should be a 10 minute time out for each combination. We should utilize a smart timeout. That is, if a combination times out, we should skip combinations that only increase parameters as they are likely to time out too. Ex. If we find a depth_n of 4 times out for a given combination, we likely do not want to waste time testing a depth_n of 6 for that already timing out combination.
- For every combination round, output a csv and graphs for desired comparisons. That is, we should not have to wait for the entire benchmark to finish to start verifying outputs. I want to be able to view csv's and graphs after each variation run.
- Ensure combination sweep is ordered in a manner that is monotonically increasing to avoid running into many time outs.