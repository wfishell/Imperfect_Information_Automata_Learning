"""
MCTS-based equivalence oracle for AALpy.

At each equivalence query the oracle runs K MCTS rollouts to depth N.
Each rollout:
  - At P1 nodes  : selects via UCB (coverage across all P1 inputs)
  - At P2 nodes  : samples probabilistically (high prior for unexplored,
                   softmax over SMT values for explored)
  - Deviations from the current hypothesis are tracked as candidate
    counterexamples

After K rollouts the oracle:
  1. Collects all deviation leaves (depth N) and their Table A shadows
  2. Feeds all pairwise preferences into the SMT solver
  3. Updates Table B softmax probabilities from the SMT solution
  4. Prunes depth-N Table B leaves below the median SMT value
  5. For each deviation point, counts oracle.compare(dev_leaf, shadow) == 't1'
  6. If majority (>50%) of pairs prefer the deviation:
       - updates the SUL strategy override at the best deviation point
       - returns the P1 input sequence as a counterexample to AALpy

Table B persists across all equivalence queries.
"""

from __future__ import annotations
import random
import time
from aalpy.base import Oracle

from src.lstar_mcts.smt_solver import SMTValueAssigner
from src.game.minimax.game_nfa import GameNFA
from src.game.minimax.preference_oracle import PreferenceOracle
from src.lstar_mcts.game_sul import GameSUL
from src.lstar_mcts.table_b import TableB


class MCTSEquivalenceOracle(Oracle):

    def __init__(
        self,
        sul: GameSUL,
        nfa: GameNFA,
        oracle: PreferenceOracle,
        table_b: TableB,
        depth_N: int,
        max_trace_length: int = 20,
        K: int             = 200,
        epsilon: float     = 0.05,
        temperature: float = 1.0,
        verbose: bool      = False,
    ) -> None:
        alphabet = list(nfa.root.children.keys())   # P1's top-level inputs
        super().__init__(alphabet, sul)
        self.sul              = sul
        self.nfa              = nfa
        self.oracle           = oracle
        self.table_b          = table_b
        self.depth_N          = depth_N
        self.max_trace_length = max_trace_length
        self.K                = K
        self.epsilon          = epsilon
        self.temperature      = temperature
        self.verbose          = verbose
        self.hypothesis       = None  # set at the start of each find_cex call
        self.last_cex_p1: list[str] = []  # P1 inputs of last returned CEX

        # {deviation_point (tuple): [trace_at_depth_N (list), ...]}
        self._deviation_leaves: dict[tuple, list[list[str]]] = {}
    
    def find_cex(self, hypothesis):
        self.hypothesis = hypothesis
        self.num_queries += 1

        for i in range(self.K):
            subtrace, ce_traces, majority, deviation_action = self.GenerateCounterExample()
            print(f'[rollout {i}] subtrace_len={len(subtrace)}  ce_traces={"None" if ce_traces is None else len(ce_traces)}  majority={majority if ce_traces is not None else "-"}')

            if ce_traces is None:
                continue

            if majority:
                print('updating strategy...')
                deviation_start = len(subtrace) - 1
                for trace in ce_traces:
                    for j in range(deviation_start, len(trace), 2):
                        self.sul.update_strategy(trace[:j], trace[j])
                cex = self.sul.p1_inputs_from_trace(subtrace)
                self.last_cex_p1 = cex
                print('returning cex...')
                return cex

        return None

    def GenerateCounterExample(self):                                                                                                                                                                                                                                       
      SubTrace = self.GenerateSubTrace()                                                                                          

      t0 = time.perf_counter()
      CE_Traces, deviation_action = self.CollectTraces(SubTrace)
      print(f'  CollectTraces:          {time.perf_counter()-t0:.3f}s  traces={len(CE_Traces) if CE_Traces else None}')
      if CE_Traces is None:
          return SubTrace, None, False, None

      t0 = time.perf_counter()
      Hypothesis_Traces = self.Generate_Hypothesis_Language(SubTrace)
      print(f'  GenerateHypothesis:     {time.perf_counter()-t0:.3f}s  traces={len(Hypothesis_Traces)}')

      t0 = time.perf_counter()
      values, majority = self.AssignPreferencesAndPreferenceValues(Hypothesis_Traces, CE_Traces)
      print(f'  AssignPreferences:      {time.perf_counter()-t0:.3f}s')

      t0 = time.perf_counter()
      self.PropagateValuesThroughTableB(SubTrace, values)
      print(f'  PropagateValues:        {time.perf_counter()-t0:.3f}s')

      return SubTrace, CE_Traces, majority, deviation_action


    def get_current_state(self, trace: list[str]) -> tuple:                                                                                                                                                                                                                         
      self.hypothesis.reset_to_initial()                                                                                          
      node = self.nfa.root                                                                                                                                                                                                                                                        
      for i, action in enumerate(trace):                                                                                                                                                                                                                                          
          if i % 2 == 0:  # P1 action                                                                                                                                                                                                                                             
              self.hypothesis.step(action)                                                                                                                                                                                                                                        
          node = node.children.get(action)                                                                                                                                                                                                                                        
          if node is None or node.is_terminal():                                                                                                                                                                                                                                  
              break                                                                                                                                                                                                                                                               
      return node, self.hypothesis.current_state

    def GenerateSubTrace(self) -> list[str]:
        self.hypothesis.reset_to_initial()
        node = self.nfa.root
        trace = []
        deviation_candidates = []  # (index_into_trace, weight)

        while len(trace) < self.max_trace_length and node is not None and not node.is_terminal():
            if node.player == 'P1':
                action = random.choice(list(node.children.keys()))
                p2_response = self.hypothesis.step(action)

                trace.append(action)
                node = node.children[action]

                action_space = list(self.table_b.actions_at(trace).keys())
                if action_space:
                    total_score = sum(self.table_b.ucb_score(trace, a, action_space) for a in action_space)
                    weight = self.table_b.ucb_score(trace, p2_response, action_space) / total_score if total_score > 0 else 0.5
                else:
                    weight = 0.5

                deviation_candidates.append((len(trace), weight))

                if node is None or node.is_terminal():
                    break
                trace.append(p2_response)
                node = node.children.get(p2_response)
            else:
                break

        indices, weights = zip(*deviation_candidates)
        inverse_weights = [1.0 / (w + 1e-9) for w in weights]
        chosen_index = random.choices(indices, weights=inverse_weights, k=1)[0]

        return trace[:chosen_index + 1]
                                          
    def CollectTraces(self, SubTrace: list[str]) -> list[list[str]]:  
        '''
        Given a deviation we create a sub tree that updates Table B and searches vals for P2 moves
        '''

        SubTrace_Dev = SubTrace[:-1]                                                                                                                                                                                         
        available = list(self.nfa.get_node(SubTrace_Dev).children.keys())                                                                                                                                                    
                                                                                                                                                                                                                            
        sampled_action = self.table_b.sample_p2_action(SubTrace_Dev, available)
        Sampler_Break=0                                                                                                                            
        while sampled_action == SubTrace[-1]:                                                                                                                                                                                
            sampled_action = self.table_b.sample_p2_action(SubTrace_Dev, available)
            Sampler_Break+=1
            if Sampler_Break==15:
                #assert here 
                print("Something is going wrong there are no other actions currently")
                return None, None
                                                                                                                                                                                                                            
        frontier = [SubTrace_Dev + [sampled_action]]
        completed = []                                                                                                                                                                                                       
                    
        for _ in range(self.depth_N - 1):
            new_frontier = []
            for trace in frontier:
                node = self.nfa.get_node(trace)                                                                                                                                                                              
                if node is None or node.is_terminal():
                    completed.append(trace)                                                                                                                                                                                  
                    continue

                for p1_action, p2_node in node.children.items():
                    p1_trace = trace + [p1_action]
                                                                                                                                                                                                                            
                    if p2_node is None or p2_node.is_terminal():                                                                                                                                                                             
                        self.table_b.set_zero_prob(p1_trace, 'Terminal')                                                                            
                        completed.append(p1_trace + ['Terminal'])                                                                                                                                                                            
                        continue                                                                                                                                                                                                             
                    

                    p2_actions = list(p2_node.children.keys())                                                                                                                                                               
                    for p2_action in p2_actions:
                        self.table_b.record_visit(p1_trace, p2_action)                                                                                                                                                       
                    
                    sampled_p2 = self.table_b.sample_p2_action(p1_trace, p2_actions)                                                                                                                       
                    if sampled_p2 is not None:
                        new_frontier.append(p1_trace + [sampled_p2])                                                                                                                                                         
                    
            frontier = new_frontier
            if not frontier:
                break

        return completed + frontier, sampled_action

    

    def Generate_Hypothesis_Language(self, SubTrace: list[str]) -> list[list[str]]:                                                                                                                                                                                         
      NFA_Node, current_state = self.get_current_state(SubTrace)                                                                                                                                                                                                          
      Collect_Traces = []
      Queue = [[NFA_Node, SubTrace, 0]]                                                                                                                                                                                                                                   
                                                                                                                                                                                                                                                                          
      while Queue:                                                                                                                                                                                                                                                        
          current_node, current_trace, depth = Queue.pop(0)

          if current_node is None or current_node.is_terminal():
              Collect_Traces.append(current_trace)
              continue                                                                                                                                                                                                                                                    
  
          if depth == self.depth_N:                                                                                                                                                                                                                                       
              Collect_Traces.append(current_trace)
              continue

          if current_node.player == 'P2':
              raise AssertionError(f"Expected P1 node but got P2 at trace: {current_trace}")
                                                                                                                                                                                                                                                                          
          for p1_input in current_node.children.keys():
              self.hypothesis.reset_to_initial()
              p1_inputs = (current_trace + [p1_input])[::2]
              output_action = None
              for inp in p1_inputs:
                  output_action = self.hypothesis.step(inp)

              p2_node = current_node.children.get(p1_input)
              if output_action is None or p2_node is None or output_action not in p2_node.children:
                  Collect_Traces.append(current_trace + [p1_input])
                  continue

              new_subtrace = current_trace + [p1_input, output_action]
              New_NFA_Node, new_current_state = self.get_current_state(new_subtrace)
              Queue.append([New_NFA_Node, new_subtrace, depth + 1])                                                                                                                                                                                                       
  
      return Collect_Traces      
            
    def AssignPreferencesAndPreferenceValues(self, Hypothesis_Traces, Counter_Example_Traces):
        smt = SMTValueAssigner()
        preferred_count = 0
        total_count = 0

        for CE in Counter_Example_Traces:
            ce_clean = [a for a in CE if a != 'Terminal']
            for HE in Hypothesis_Traces:
                he_clean = [a for a in HE if a != 'Terminal']
                Preference = self.oracle.compare(ce_clean, he_clean)
                smt.add(ce_clean, he_clean, Preference)
                if Preference == 't1':
                    preferred_count += 1
                total_count += 1
        print('begin solve')
        values = smt.solve()
        print('end solve')
        majority = (preferred_count / total_count) > 0.5 if total_count > 0 else False
        return values, majority

    def PropagateValuesThroughTableB(self, SubTrace, Values):                                                                                                                                                                                                               
        # Step 1: update leaf values                                                                                                                                                                                                                                        
        Traces_Prop = []                                                                                                                                                                                                                                                    
        for Trace in Values.keys():                                                                                                                                                                                                                                         
            trace_list = list(Trace)                                                                                                                                                                                                                                        
            self.table_b.update_value(trace_list[:-1], trace_list[-1], Values[Trace])                                                                                                                                                                                       
            Traces_Prop.append(trace_list)                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                            
        # Step 2: propagate backwards level by level up to SubTrace
        current_level = Traces_Prop                                                                                                                                                                                                                                         
        while True: 
            # Step up one level (drop last P1+P2 pair)                                                                                                                                                                                                                      
            parent_level = [t[:-2] for t in current_level if len(t) - 2 >= len(SubTrace)]
            parent_level = list({tuple(p): p for p in parent_level}.values())  # deduplicate                                                                                                                                                                                
                                                                                                                                                                                                                                                                            
            if not parent_level:                                                                                                                                                                                                                                            
                break                                                                                                                                                                                                                                                       
                    
            for parent in parent_level:                                                                                                                                                                                                                                     
                children = self.table_b.actions_at(parent)
                if not children:                                                                                                                                                                                                                                            
                    continue
                avg = sum(s.value for s in children.values()) / len(children)
                self.table_b.update_value(parent[:-1], parent[-1], avg)                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                            
            current_level = parent_level                                                                                                                                                                                                                                    
                                                                                                                                                                                                                                                                            
            if all(len(t) <= len(SubTrace) for t in current_level):
                break