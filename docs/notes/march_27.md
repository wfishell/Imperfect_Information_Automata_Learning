# Notes: March 27th 2026

[REMAP CODE HERE](https://github.com/eric-hsiung/REMAP)

First step:
- Use REMAP where LLM is the oracle.
- What are the inputs into the REMAP system?

## Repository Review

### Dot Trace Generator

[dot_trace_generator.py](https://github.com/wfishell/Imperfect_Information_Automata_Learning/blob/master/dot_trace_generator.py) takes a DOT file or JSON and outputs SPOT traces of random walks across the inputted automata.

I wanna say these are the traces I am expecting to get out:
 
$$t_1 = \sigma_1, \lambda_1, \sigma_2, \lambda_2, \sigma_3, \lambda_3
\qquad \text{and} \qquad
t_2 = \sigma_1, \lambda_1', \sigma_2', \lambda_2', \sigma_3', \lambda_3'$$

It should be noted, the trace output is in the SPOT format.

I want to do a test run of `dot_trace_generator.py` w/ the Kuhn Poker data to learn more about how this works.

Current Pipeline:

TSLF -> DOT Automaton / HOA File -> test_kuhn.py