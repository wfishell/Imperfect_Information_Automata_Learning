"""
Evaluation  (Pipeline Step 4)

Compares the learned hypothesis against the ground-truth reward machine
produced by build_kuhn_reward_machine() and prints a summary.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))  # src/pipeline/
_SRC  = os.path.dirname(_HERE)                       # src/
sys.path.insert(0, _SRC)

from teacher.deterministic_teacher import build_kuhn_reward_machine


def evaluate(hypothesis: tuple, teacher) -> bool:
    """
    Print a comparison between the learned hypothesis and the ground truth.

    Args:
        hypothesis: The (states, sigma_I, sigma_O, init_state, delta, output_fnc)
                    tuple returned by symbolic_lstar.
        teacher:    The CachedTeacher instance (provides sigma_I and corpus).

    Returns:
        True if the learned output function matches the ground truth on all
        corpus traces, False otherwise.
    """
    states, sigma_I, sigma_O, init_state, delta, output_fnc = hypothesis

    print("\n" + "=" * 60)
    print("LEARNED REWARD MACHINE")
    print("=" * 60)
    print(f"  States : {len(states)}")
    label = {1: "P2 wins", -1: "P1 wins", 0: "in progress"}
    for state, out in output_fnc.items():
        print(f"  {state} -> {out}  ({label.get(out, str(out))})")

    print("\n" + "=" * 60)
    print("GROUND-TRUTH REWARD MACHINE")
    print("=" * 60)
    gt = build_kuhn_reward_machine(teacher.sigma_I)
    print(f"  States : {gt['state_names']}")
    print(f"  Output : {gt['output']}")

    # -----------------------------------------------------------------------
    # Trace-level agreement check
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRACE-LEVEL AGREEMENT (learned vs. ground truth)")
    print("=" * 60)

    mismatches = []
    for trace in teacher.corpus:
        # Run learned hypothesis
        q = init_state
        for sym in trace:
            q = delta.get(q, {}).get(sym, q)
        learned_out = output_fnc.get(q, 0)

        # Run ground-truth reward machine
        gt_q = gt["initial"]
        for sym in trace:
            gt_q = gt["delta"].get(gt_q, {}).get(sym, gt_q)
        gt_out = gt["output"].get(gt_q, 0)

        if learned_out != gt_out:
            mismatches.append((trace, learned_out, gt_out))

    total = len(teacher.corpus)
    passed = total - len(mismatches)
    print(f"  {passed}/{total} traces agree")

    if mismatches:
        print(f"\n  First {min(5, len(mismatches))} mismatches:")
        for trace, lo, go in mismatches[:5]:
            print(f"    {' | '.join(trace)}")
            print(f"      learned={lo}  ground_truth={go}")
        return False

    print("\n  ✓ Learned machine agrees with ground truth on all corpus traces.")
    return True
