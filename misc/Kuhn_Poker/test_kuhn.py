#!/usr/bin/env python3
import argparse
import json
import re
import sys
from itertools import permutations, product

# ============================================================
# Constants
# ============================================================
J, Q, K = 0, 1, 2
CARD_NAMES = {0: "J", 1: "Q", 2: "K"}

NOOP = (0, 0, 0)
CHECK = (1, 0, 0)
BET1 = (0, 1, 0)
BET2 = (1, 1, 0)
CALL = (0, 0, 1)
FOLD = (1, 0, 1)
ACTION_NAMES = {
    NOOP: "noop",
    CHECK: "check",
    BET1: "bet1",
    BET2: "bet2",
    CALL: "call",
    FOLD: "fold",
}

PH_DEAL = "deal"
PH_P1 = "p1"
PH_P2 = "p2"
PH_P1B = "p1b"
PH_P2B = "p2b"
PH_END = "end"

ALL_INPUT_SIGNALS = [
    "c1lo",
    "c1hi",
    "c2lo",
    "c2hi",
    "a0",
    "a1",
    "a2",
    "deal",
    "p1",
    "p2",
    "p1b",
    "p2b",
    "end",
]
OUTPUT_SIGNAL_NAMES = {"win1", "win2"}

PHASE_BITS_NO_END = {
    PH_DEAL: {"deal": 1, "p1": 0, "p2": 0, "p1b": 0, "p2b": 0},
    PH_P1: {"deal": 0, "p1": 1, "p2": 0, "p1b": 0, "p2b": 0},
    PH_P2: {"deal": 0, "p1": 0, "p2": 1, "p1b": 0, "p2b": 0},
    PH_P1B: {"deal": 0, "p1": 0, "p2": 0, "p1b": 1, "p2b": 0},
    PH_P2B: {"deal": 0, "p1": 0, "p2": 0, "p1b": 0, "p2b": 1},
}

PHASE_BITS_WITH_END = {
    PH_DEAL: {"deal": 1, "p1": 0, "p2": 0, "p1b": 0, "p2b": 0, "end": 0},
    PH_P1: {"deal": 0, "p1": 1, "p2": 0, "p1b": 0, "p2b": 0, "end": 0},
    PH_P2: {"deal": 0, "p1": 0, "p2": 1, "p1b": 0, "p2b": 0, "end": 0},
    PH_P1B: {"deal": 0, "p1": 0, "p2": 0, "p1b": 1, "p2b": 0, "end": 0},
    PH_P2B: {"deal": 0, "p1": 0, "p2": 0, "p1b": 0, "p2b": 1, "end": 0},
    PH_END: {"deal": 0, "p1": 0, "p2": 0, "p1b": 0, "p2b": 0, "end": 1},
}


def expected_winner(c1, c2, phase, action):
    """
    Determine the expected winner of a Kuhn Poker hand.

    On a FOLD action, the player who did NOT fold wins:
    - If P2 or P2B folds, P1 wins (1, 0).
    - Otherwise (P1 or P1B folded), P2 wins (0, 1).
    On any showdown action, the higher card wins.

    Args:
        c1 (int): P1's card (J=0, Q=1, K=2).
        c2 (int): P2's card.
        phase (str): The game phase when the terminal action occurred.
        action (tuple): The 3-bit action tuple (a0, a1, a2).

    Returns:
        tuple[int, int]: (win1, win2) where exactly one is 1.
    """
    if action == FOLD:
        return (1, 0) if phase in (PH_P2, PH_P2B) else (0, 1)
    return (1, 0) if c1 > c2 else (0, 1)


def is_terminal_no_end(phase, action):
    """
    Check whether a (phase, action) pair ends the hand in the no-end-phase spec.

    In the Mealy (no 'end' phase) variant, the winner output is emitted on the
    same step as the terminal action, so we need to know which actions are terminal:
    - PH_P2 + CHECK/CALL/FOLD: P2's response always ends the hand unless P2 bets.
    - PH_P1B + CALL/FOLD: P1's response to P2's bet ends the hand (no raise path).
    - PH_P2B: any action at this phase is terminal (final re-raise resolution).

    Args:
        phase (str): Current game phase.
        action (tuple): The 3-bit action tuple.

    Returns:
        bool: True if this is a terminal step.
    """
    if phase == PH_P2 and action in (CHECK, CALL, FOLD):
        return True
    if phase == PH_P1B and action in (CALL, FOLD):
        return True
    if phase == PH_P2B:
        return True
    return False


def make_step(c1, c2, action, phase, has_end):
    """
    Build the input signal dictionary for one automaton step.

    Cards are 2-bit encoded: lo = bit 0, hi = bit 1.
      J=0b00, Q=0b01, K=0b10
    Action bits (a0, a1, a2) come directly from the action tuple.
    Phase bits are looked up from PHASE_BITS_WITH_END or PHASE_BITS_NO_END.

    Args:
        c1 (int): P1's card value (0–2).
        c2 (int): P2's card value (0–2).
        action (tuple): 3-bit action (a0, a1, a2).
        phase (str): Current game phase string (e.g. PH_DEAL, PH_P1).
        has_end (bool): Whether to use the spec variant that includes an 'end' phase.

    Returns:
        dict[str, int]: Mapping of signal name -> 0 or 1 for this step.
    """
    signals = {}
    signals["c1lo"] = c1 & 1
    signals["c1hi"] = (c1 >> 1) & 1
    signals["c2lo"] = c2 & 1
    signals["c2hi"] = (c2 >> 1) & 1
    signals["a0"], signals["a1"], signals["a2"] = action
    pb = PHASE_BITS_WITH_END if has_end else PHASE_BITS_NO_END
    signals.update(pb[phase])
    return signals


# ============================================================
# Trace Generation
# ============================================================


def generate_traces(has_end_phase=False):
    """
    Generate all valid Kuhn Poker game traces as sequences of automaton steps.

    Iterates over all 6 card deal permutations (P1/P2 each get one of J/Q/K)
    and all legal action sequences from each game state, building a complete
    test suite covering every reachable play path.

    Each trace is a list of (signals, expected_win1, expected_win2) tuples.
    In the no-end-phase (Mealy) variant, win signals are emitted on the terminal
    step itself. In the with-end-phase variant, a final PH_END step carries the
    win signals.

    Args:
        has_end_phase (bool): If True, generate traces for the spec variant that
            uses a dedicated 'end' phase to emit the winner. Default is False
            (standard Mealy semantics).

    Returns:
        list[tuple[str, list]]: Each entry is (description, steps) where
            description is a human-readable string of the play sequence and
            steps is a list of (signal_dict, win1, win2) tuples.
    """
    traces = []
    for c1, c2 in permutations([J, Q, K], 2):
        for p1a in [CHECK, BET1, BET2]:

            def build(after_p1):
                steps = []
                steps.append((make_step(c1, c2, NOOP, PH_DEAL, has_end_phase), 0, 0))
                steps.append((make_step(c1, c2, p1a, PH_P1, has_end_phase), 0, 0))
                for phase, action in after_p1:
                    if has_end_phase:
                        steps.append((make_step(c1, c2, action, phase, True), 0, 0))
                    else:
                        if is_terminal_no_end(phase, action):
                            w1, w2 = expected_winner(c1, c2, phase, action)
                            steps.append(
                                (make_step(c1, c2, action, phase, False), w1, w2)
                            )
                        else:
                            steps.append(
                                (make_step(c1, c2, action, phase, False), 0, 0)
                            )
                if has_end_phase:
                    tp, ta = after_p1[-1]
                    w1, w2 = expected_winner(c1, c2, tp, ta)
                    steps.append((make_step(c1, c2, NOOP, PH_END, True), w1, w2))
                desc = f"P1={CARD_NAMES[c1]} P2={CARD_NAMES[c2]}"
                desc += f" -> p1:{ACTION_NAMES[p1a]}"
                for ph, ac in after_p1:
                    desc += f" -> {ph}:{ACTION_NAMES[ac]}"
                return (desc, steps)

            if p1a == CHECK:
                for p2a in [CHECK, BET1, BET2]:
                    if p2a == CHECK:
                        traces.append(build([(PH_P2, CHECK)]))
                    elif p2a == BET1:
                        for p1ba in [CALL, FOLD, BET2]:
                            if p1ba == BET2:
                                for p2ba in [CALL, FOLD]:
                                    traces.append(
                                        build(
                                            [
                                                (PH_P2, BET1),
                                                (PH_P1B, BET2),
                                                (PH_P2B, p2ba),
                                            ]
                                        )
                                    )
                            else:
                                traces.append(build([(PH_P2, BET1), (PH_P1B, p1ba)]))
                    elif p2a == BET2:
                        for p1ba in [CALL, FOLD]:
                            traces.append(build([(PH_P2, BET2), (PH_P1B, p1ba)]))
            elif p1a == BET1:
                for p2a in [CALL, FOLD, BET2]:
                    if p2a == BET2:
                        for p1ba in [CALL, FOLD]:
                            traces.append(build([(PH_P2, BET2), (PH_P1B, p1ba)]))
                    else:
                        traces.append(build([(PH_P2, p2a)]))
            elif p1a == BET2:
                for p2a in [CALL, FOLD]:
                    traces.append(build([(PH_P2, p2a)]))
    return traces


# ============================================================
# HOA Parser
# ============================================================


def parse_hoa(filename):
    """
    Parse a Hanoi Omega Automata (HOA) file into an automaton structure.

    Reads the HOA header for metadata and the --BODY-- section for transitions.
    Transition guards are stored as raw strings for lazy evaluation by eval_guard.

    Args:
        filename (str): Path to the .hoa file.

    Returns:
        tuple:
            num_states (int): Total number of states declared.
            start_state (int): Index of the initial state.
            ap_list (list[str]): Ordered list of atomic proposition names.
            controllable_aps (list[int]): AP indices marked as outputs
                (from the 'controllable-ap' header field).
            states (dict[int, list[dict]]): Maps state index to a list of
                transition dicts, each with keys 'guard' (str) and 'dest' (int).
    """
    with open(filename, "r") as f:
        content = f.read()
    lines = content.strip().split("\n")
    num_states = None
    start_state = None
    ap_list = []
    controllable_aps = []
    states = {}
    current_state = None
    in_body = False

    for line in lines:
        line = line.strip()
        if line == "--BODY--":
            in_body = True
            continue
        if line == "--END--":
            break
        if not in_body:
            if line.startswith("States:"):
                num_states = int(line.split(":")[1].strip())
            elif line.startswith("Start:"):
                start_state = int(line.split(":")[1].strip())
            elif line.startswith("AP:"):
                parts = line.split('"')
                ap_list = [parts[i] for i in range(1, len(parts), 2)]
            elif line.startswith("controllable-ap:"):
                ctrl = line.split(":")[1].strip()
                if ctrl:
                    controllable_aps = [int(x) for x in ctrl.split()]
        else:
            if line.startswith("State:"):
                parts = line.split()
                current_state = int(parts[1])
                states[current_state] = []
            elif line.startswith("[") and current_state is not None:
                bracket_end = line.index("]")
                guard_str = line[1:bracket_end]
                rest = line[bracket_end + 1 :].strip()
                dest_match = re.match(r"(\d+)", rest)
                dest = int(dest_match.group(1)) if dest_match else 0
                states[current_state].append({"guard": guard_str, "dest": dest})

    return num_states, start_state, ap_list, controllable_aps, states


# ============================================================
# Guard Evaluator
# ============================================================


def tokenize(s):
    """
    Tokenize a HOA guard string into a list of (type, value) pairs.

    Token types:
        '('  / ')'  : parentheses
        '&'  / '|'  : logical AND / OR
        '!'         : logical NOT
        'V'         : boolean literal (True for 't', False for 'f')
        'N'         : integer AP index

    Args:
        s (str): A HOA guard expression, e.g. "0&!1&2" or "t".

    Returns:
        list[tuple[str, Any]]: Ordered token list.
    """
    tokens = []
    i = 0
    while i < len(s):
        c = s[i]
        if c in " \t":
            i += 1
        elif c == "(":
            tokens.append(("(", None))
            i += 1
        elif c == ")":
            tokens.append((")", None))
            i += 1
        elif c == "&":
            tokens.append(("&", None))
            i += 1
        elif c == "|":
            tokens.append(("|", None))
            i += 1
        elif c == "!":
            tokens.append(("!", None))
            i += 1
        elif c == "t" and (i + 1 >= len(s) or not s[i + 1].isalnum()):
            tokens.append(("V", True))
            i += 1
        elif c == "f" and (i + 1 >= len(s) or not s[i + 1].isalnum()):
            tokens.append(("V", False))
            i += 1
        elif c.isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            tokens.append(("N", int(s[i:j])))
            i = j
        else:
            i += 1
    return tokens


def eval_guard(guard_str, vals):
    """
    Evaluate a HOA guard expression against a valuation vector.

    Parses and evaluates a Boolean formula over AP indices. The formula uses
    integer AP indices (not names), so 'vals' must be indexed by AP position.

    Args:
        guard_str (str): A HOA guard string, e.g. "0&!1&2" or "t".
        vals (list[bool]): Truth values indexed by AP index.

    Returns:
        bool: True if the guard is satisfied by the given valuation.
    """
    s = guard_str.strip()
    if s == "t":
        return True
    if s == "f":
        return False
    tokens = tokenize(s)
    r, _ = _p_or(tokens, 0, vals)
    return r


def _p_or(t, p, v):
    """Parse and evaluate a disjunction (lowest precedence)."""
    l, p = _p_and(t, p, v)
    while p < len(t) and t[p][0] == "|":
        p += 1
        r, p = _p_and(t, p, v)
        l = l or r
    return l, p


def _p_and(t, p, v):
    """Parse and evaluate a conjunction."""
    l, p = _p_not(t, p, v)
    while p < len(t) and t[p][0] == "&":
        p += 1
        r, p = _p_not(t, p, v)
        l = l and r
    return l, p


def _p_not(t, p, v):
    """Parse and evaluate a negation or delegate to atom."""
    if p < len(t) and t[p][0] == "!":
        p += 1
        r, p = _p_atom(t, p, v)
        return not r, p
    return _p_atom(t, p, v)


def _p_atom(t, p, v):
    """Parse and evaluate a leaf: parenthesized subexpr, boolean literal, or AP index."""
    if p >= len(t):
        return False, p
    if t[p][0] == "(":
        p += 1
        r, p = _p_or(t, p, v)
        if p < len(t) and t[p][0] == ")":
            p += 1
        return r, p
    if t[p][0] == "V":
        return t[p][1], p + 1
    if t[p][0] == "N":
        return v[t[p][1]], p + 1
    return False, p + 1


# ============================================================
# HOA Simulator
# ============================================================


def simulate_hoa(hoa_file, traces):
    """
    Simulate a HOA automaton against a set of test traces and report results.

    For each step in each trace, the simulator:
    1. Sets input AP values from the step's signal dict.
    2. Tries all 2^(num_outputs) output combinations.
    3. For each combination, checks all transitions from the current state.
    4. Accepts the first transition whose guard is satisfied AND whose output
       matches the expected (win1, win2) values.

    A trace PASSES if every step finds such a matching transition.
    A trace FAILS on the first step where no matching transition exists.

    Args:
        hoa_file (str): Path to the .hoa file to test.
        traces (list[tuple[str, list]]): Test traces from generate_traces().

    Returns:
        bool: True if all traces passed, False if any failed.
    """
    num_states, start_state, ap_list, controllable_aps, states = parse_hoa(hoa_file)

    print(f"\nParsed HOA: {hoa_file}")
    print(f"  States: {num_states}")
    print(f"  Start:  {start_state}")
    print(f"  APs ({len(ap_list)}): {ap_list}")

    # Identify output APs
    output_ap_set = set(controllable_aps)
    if not controllable_aps:
        print(f"  No controllable-ap declared; identifying outputs by name...")
        for i, name in enumerate(ap_list):
            if name in OUTPUT_SIGNAL_NAMES:
                output_ap_set.add(i)

    output_ap_list = sorted(output_ap_set)
    input_ap_list = [i for i in range(len(ap_list)) if i not in output_ap_set]

    print(f"  Inputs:  {[(i, ap_list[i]) for i in input_ap_list]}")
    print(f"  Outputs: {[(i, ap_list[i]) for i in output_ap_list]}")

    has_end = "end" in ap_list
    print(f"  Has 'end' phase: {has_end}")
    print()

    name_to_ap = {name: idx for idx, name in enumerate(ap_list)}

    # Check that we can map all needed signals
    needed_input_sigs = set()
    for _, steps in traces:
        for signals, _, _ in steps:
            needed_input_sigs.update(signals.keys())
    missing = needed_input_sigs - set(ap_list) - OUTPUT_SIGNAL_NAMES
    if missing:
        print(
            f"  WARNING: These input signals from traces are NOT in HOA APs: {missing}"
        )
        print(f"  (They will default to False/0)")
    extra_aps = set(ap_list) - needed_input_sigs - OUTPUT_SIGNAL_NAMES
    if extra_aps:
        print(f"  NOTE: Extra APs in HOA not in traces: {extra_aps}")
        print(f"  (They will default to False/0)")
    print()

    passed = 0
    failed = 0
    fail_details = []

    for trace_id, (desc, steps) in enumerate(traces):
        state = start_state
        trace_ok = True

        for step_idx, (signals, exp_w1, exp_w2) in enumerate(steps):
            # Set input AP values from signal dict
            base_vals = [False] * len(ap_list)
            for sig_name, sig_val in signals.items():
                if sig_name in name_to_ap and name_to_ap[sig_name] not in output_ap_set:
                    base_vals[name_to_ap[sig_name]] = bool(sig_val)

            # Search over output combinations
            found = False
            actual_outputs = None

            n_out = len(output_ap_list)
            for out_combo in range(1 << n_out):
                test_vals = list(base_vals)
                for bit_idx, ap_idx in enumerate(output_ap_list):
                    test_vals[ap_idx] = bool((out_combo >> bit_idx) & 1)

                for trans in states.get(state, []):
                    if eval_guard(trans["guard"], test_vals):
                        w1_idx = name_to_ap.get("win1")
                        w2_idx = name_to_ap.get("win2")
                        got_w1 = int(test_vals[w1_idx]) if w1_idx is not None else 0
                        got_w2 = int(test_vals[w2_idx]) if w2_idx is not None else 0

                        if got_w1 == exp_w1 and got_w2 == exp_w2:
                            state = trans["dest"]
                            found = True
                            break
                        else:
                            if actual_outputs is None:
                                actual_outputs = (got_w1, got_w2, trans["dest"])

                if found:
                    break

            if not found:
                if actual_outputs:
                    aw1, aw2, nxt = actual_outputs
                    detail = (
                        f"  FAIL trace {trace_id} step {step_idx}: {desc}\n"
                        f"    State={state} -> {nxt} | "
                        f"expected w1={exp_w1} w2={exp_w2} | "
                        f"actual w1={aw1} w2={aw2}"
                    )
                    state = nxt  # advance to continue checking
                else:
                    detail = (
                        f"  FAIL trace {trace_id} step {step_idx}: {desc}\n"
                        f"    State={state} | "
                        f"expected w1={exp_w1} w2={exp_w2} | "
                        f"NO MATCHING TRANSITION FOUND"
                    )
                fail_details.append(detail)
                trace_ok = False
                break

        if trace_ok:
            passed += 1
        else:
            failed += 1

    print(f"{'='*70}")
    print(f"RESULTS: {passed}/{passed+failed} passed, {failed} failed")
    print(f"{'='*70}")

    if fail_details:
        print(f"\nFailures (showing first {min(20, len(fail_details))}):")
        for d in fail_details[:20]:
            print(d)
        if len(fail_details) > 20:
            print(f"  ... and {len(fail_details) - 20} more")

    if failed == 0:
        print("\n✓ ALL TRACES PASSED")
    else:
        print(f"\n✗ {failed} TRACES FAILED")

    return failed == 0


# ============================================================
# CSV / JSON / Traces
# ============================================================


def output_csv(traces, has_end):
    """
    Print all trace steps as CSV rows to stdout.

    Columns are input signal names (excluding 'end' if not used),
    followed by expected_win1, expected_win2, trace_id, and step index.

    Args:
        traces (list[tuple[str, list]]): Test traces from generate_traces().
        has_end (bool): Whether the 'end' phase signal should be included.
    """
    sigs = [s for s in ALL_INPUT_SIGNALS if has_end or s != "end"]
    print(",".join(sigs + ["expected_win1", "expected_win2", "trace_id", "step"]))
    for tid, (_, steps) in enumerate(traces):
        for sid, (signals, w1, w2) in enumerate(steps):
            print(
                ",".join(str(signals.get(s, 0)) for s in sigs)
                + f",{w1},{w2},{tid},{sid}"
            )


def output_json(traces):
    """
    Print all traces as a JSON document to stdout.

    Structure: {"traces": [{"id": int, "description": str, "steps": [
        {"inputs": {signal: val}, "expected": {"win1": int, "win2": int}}
    ]}]}

    Args:
        traces (list[tuple[str, list]]): Test traces from generate_traces().
    """
    out = {"traces": []}
    for tid, (desc, steps) in enumerate(traces):
        t = {"id": tid, "description": desc, "steps": []}
        for signals, w1, w2 in steps:
            t["steps"].append({"inputs": signals, "expected": {"win1": w1, "win2": w2}})
        out["traces"].append(t)
    print(json.dumps(out, indent=2))


def output_traces(traces):
    """
    Print a human-readable summary of all traces to stdout.

    For each step, shows the phase, action name, and win outcome.

    Args:
        traces (list[tuple[str, list]]): Test traces from generate_traces().
    """
    print(f"{'='*80}\nKUHN POKER — {len(traces)} traces\n{'='*80}\n")
    for tid, (desc, steps) in enumerate(traces):
        print(f"--- Trace {tid}: {desc} ---")
        for sid, (signals, w1, w2) in enumerate(steps):
            phase = next(
                (
                    p
                    for p in [PH_DEAL, PH_P1, PH_P2, PH_P1B, PH_P2B, PH_END]
                    if signals.get(p, 0) == 1
                ),
                "?",
            )
            act = ACTION_NAMES.get(
                (signals.get("a0", 0), signals.get("a1", 0), signals.get("a2", 0)),
                "???",
            )
            win = " -> P1 WINS" if w1 else (" -> P2 WINS" if w2 else "")
            print(f"  {sid}: {phase:5s} {act:6s} | w1={w1} w2={w2}{win}")
        print()


# ============================================================
# Main
# ============================================================


def main():
    """
    CLI entry point for the Kuhn Poker test harness.

    Modes (select one flag):
        --hoa FILE   Parse the HOA automaton and run all generated traces against it,
                     printing pass/fail results. The 'end' phase is auto-detected.
        --csv        Print all trace steps as CSV to stdout.
        --json       Print all traces as JSON to stdout.
        --traces     Print a human-readable trace summary to stdout.

    Optional:
        --with-end   Use the spec variant that includes an explicit 'end' phase
                     (only applies to --csv, --json, --traces).
    """
    parser = argparse.ArgumentParser(description="Kuhn Poker test harness v2")
    parser.add_argument("--hoa", type=str, help="Test a HOA automaton file")
    parser.add_argument("--csv", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--traces", action="store_true")
    parser.add_argument(
        "--with-end", action="store_true", help="Use old spec with 'end' phase"
    )
    args = parser.parse_args()

    if args.hoa:
        _, _, ap_list, _, _ = parse_hoa(args.hoa)
        has_end = "end" in ap_list
        print(
            f"Auto-detected spec: {'with end phase' if has_end else 'no end phase (Mealy)'}"
        )
        traces = generate_traces(has_end_phase=has_end)
        print(f"Generated {len(traces)} test traces")
        simulate_hoa(args.hoa, traces)
    elif args.csv:
        output_csv(generate_traces(args.with_end), args.with_end)
    elif args.json:
        output_json(generate_traces(args.with_end))
    elif args.traces:
        output_traces(generate_traces(args.with_end))
    else:
        print("Usage:")
        print("  python3 test_kuhn.py --hoa FILE.hoa")
        print("  python3 test_kuhn.py --csv [--with-end]")
        print("  python3 test_kuhn.py --json [--with-end]")
        print("  python3 test_kuhn.py --traces [--with-end]")


if __name__ == "__main__":
    main()
