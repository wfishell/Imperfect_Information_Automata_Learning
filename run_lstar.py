#!/usr/bin/env python3
import sys

sys.path.insert(0, "REMAP/remap")

from lstar import symbolic_lstar
from deterministic_teacher import DeterministicKuhnPokerTeacher

teacher = DeterministicKuhnPokerTeacher(
    dot_file="Kuhn_Poker/kuhn_poker.dot",
    seq_sample_size=200,
)

print(f"sigma_I: {len(teacher.sigma_I)} symbols")
print(f"sigma_O: {teacher.sigma_O}")
print("\nStarting L*...\n")

hypothesis, data = symbolic_lstar(teacher.sigma_I, teacher.sigma_O, teacher)

# Unpack
states, sigma_I, sigma_O, init_state, delta, output_fnc = hypothesis
(
    num_pref,
    num_ineq,
    num_seq,
    num_ecs,
    num_vars,
    up_shape,
    lo_shape,
    num_eq,
    cex_lens,
    events,
) = data

print("\n=== Results ===")
print(f"States         : {len(states)}")
print(f"Preference Q   : {num_pref}")
print(f"Equivalence Q  : {num_eq}")
print(f"CEX lengths    : {cex_lens}")
print(f"Upper table    : {up_shape}")
print(f"Lower table    : {lo_shape}")
print(f"Events         : {events}")

print("\n=== Learned output function ===")
for state, out in output_fnc.items():
    label = {1: "P2 wins", -1: "P1 wins", 0: "in progress"}.get(out, str(out))
    print(f"  state {state} -> {out} ({label})")

print("\n=== Ground-truth reward machine ===")
teacher.print_reward_machine()
