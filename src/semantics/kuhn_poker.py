"""
Kuhn Poker Trace Enricher

Converts raw Spot-format AP traces into human-readable text by assigning
a semantic label to each AP independently based on its boolean value.

Each AP is described on its own — no APs are grouped or jointly interpreted.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from semantics import TraceEnricher


# ---------------------------------------------------------------------------
# Per-AP semantic labels
# Each AP maps to (label_when_true, label_when_false)
# ---------------------------------------------------------------------------

AP_SEMANTICS = {
    # --- Phase flags ---
    "deal":    ("Phase: cards are being dealt and antes taken",
                "Phase: not the deal step"),
    "p1":      ("Phase: Player 1 is making their opening action (check or bet)",
                "Phase: not Player 1's opening action"),
    "p2c":     ("Phase: Player 2 acts after Player 1 CHECKED (may check or bet)",
                "Phase: not Player 2's post-check action"),
    "p2r":     ("Phase: Player 2 acts after Player 1 BET (may call or fold only)",
                "Phase: not Player 2's post-bet response"),
    "p1b":     ("Phase: Player 1 responds to Player 2's bet (call, fold, or raise)",
                "Phase: not Player 1's response to a bet"),
    "p2b":     ("Phase: Player 2 responds to Player 1's raise (call or fold)",
                "Phase: not Player 2's response to a raise"),

    # --- Cards — P1 ---
    "c1hi":    ("Player 1 holds the King (high bit set)",
                "Player 1 does not hold the King (holds Queen or Jack)"),
    "c1lo":    ("Player 1 holds the Queen (low bit set)",
                "Player 1 does not hold the Queen (holds King or Jack)"),

    # --- Cards — P2 ---
    "c2hi":    ("Player 2 holds the King (high bit set)",
                "Player 2 does not hold the King (holds Queen or Jack)"),
    "c2lo":    ("Player 2 holds the Queen (low bit set)",
                "Player 2 does not hold the Queen (holds King or Jack)"),

    # --- Action bits ---
    "a0":      ("Action bit 0 is set",  "Action bit 0 is not set"),
    "a1":      ("Action bit 1 is set",  "Action bit 1 is not set"),
    "a2":      ("Action bit 2 is set",  "Action bit 2 is not set"),

    # --- Bet size ---
    "bs":      ("Bet/raise size: 2 chips",  "Bet/raise size: 1 chip"),

    # --- Outstanding bet ---
    "cur_bet": ("There is a 2-chip outstanding bet to match",
                "The outstanding bet is 1 chip (or no bet outstanding)"),

    # --- P1 chip count (3-bit binary: m1b2=MSB, m1b0=LSB) ---
    "m1b2":    ("Player 1 chip count bit 2 (MSB) is set   [contributes +4]",
                "Player 1 chip count bit 2 (MSB) is not set"),
    "m1b1":    ("Player 1 chip count bit 1 is set          [contributes +2]",
                "Player 1 chip count bit 1 is not set"),
    "m1b0":    ("Player 1 chip count bit 0 (LSB) is set   [contributes +1]",
                "Player 1 chip count bit 0 (LSB) is not set"),

    # --- P2 chip count (3-bit binary: m2b2=MSB, m2b0=LSB) ---
    "m2b2":    ("Player 2 chip count bit 2 (MSB) is set   [contributes +4]",
                "Player 2 chip count bit 2 (MSB) is not set"),
    "m2b1":    ("Player 2 chip count bit 1 is set          [contributes +2]",
                "Player 2 chip count bit 1 is not set"),
    "m2b0":    ("Player 2 chip count bit 0 (LSB) is set   [contributes +1]",
                "Player 2 chip count bit 0 (LSB) is not set"),

    # --- Outcomes ---
    "win1":      ("Player 1 wins this hand",      "Player 1 does not win this hand"),
    "win2":      ("Player 2 wins this hand",      "Player 2 does not win this hand"),
    "game_over": ("Game over: a player has reached 0 chips",
                  "Game is still active"),
}

# Canonical AP order for consistent display
AP_ORDER = [
    "deal", "p1", "p2c", "p2r", "p1b", "p2b",
    "c1hi", "c1lo", "c2hi", "c2lo",
    "a0", "a1", "a2", "bs", "cur_bet",
    "m1b2", "m1b1", "m1b0",
    "m2b2", "m2b1", "m2b0",
    "win1", "win2", "game_over",
]

# Action decoding table (a2, a1, a0) -> action name
_ACTION_DECODE = {
    (False, False, True):  "CHECK",
    (False, True,  False): "BET",
    (False, True,  True):  "RAISE",
    (True,  False, False): "CALL",
    (True,  False, True):  "FOLD",
}


# ---------------------------------------------------------------------------
# Enricher
# ---------------------------------------------------------------------------

class KuhnPokerEnricher(TraceEnricher):
    """
    Enriches each AP in a Kuhn Poker trace step independently.

    Each AP's boolean value is described using its own semantic label.
    A decoded action summary and chip counts are appended as convenience lines.
    """

    def enrich_step(self, aps: dict) -> str:
        lines = []
        for ap in AP_ORDER:
            if ap not in aps:
                continue
            value = aps[ap]
            true_label, false_label = AP_SEMANTICS[ap]
            label = true_label if value else false_label
            lines.append(f"  {label}  [{ap}={'True' if value else 'False'}]")

        # Include any APs present in the trace but not in AP_ORDER
        for ap, value in aps.items():
            if ap not in AP_ORDER:
                lines.append(f"  {ap}={'True' if value else 'False'}")

        # Convenience summaries
        a2 = aps.get("a2", False)
        a1 = aps.get("a1", False)
        a0 = aps.get("a0", False)
        action = _ACTION_DECODE.get((a2, a1, a0))
        if action:
            bs = aps.get("bs", False)
            size_note = f" ({2 if bs else 1} chip{'s' if (2 if bs else 1) > 1 else ''})" if action in ("BET", "RAISE") else ""
            lines.append(f"  >> Decoded action: {action}{size_note}")

        # Chip counts
        m1 = 4 * aps.get("m1b2", False) + 2 * aps.get("m1b1", False) + aps.get("m1b0", False)
        m2 = 4 * aps.get("m2b2", False) + 2 * aps.get("m2b1", False) + aps.get("m2b0", False)
        if any(k in aps for k in ("m1b2", "m1b1", "m1b0", "m2b2", "m2b1", "m2b0")):
            lines.append(f"  >> Chip counts: Player 1 = {m1}, Player 2 = {m2}")

        return "\n".join(lines)

    def enrich_trace(self, trace: str) -> str:
        steps = [s.strip() for s in trace.split(";") if s.strip() and "cycle{" not in s]
        blocks = []
        for idx, step in enumerate(steps, start=1):
            aps = self.parse_step(step)
            blocks.append(f"Step {idx}:\n{self.enrich_step(aps)}")
        return "\n\n".join(blocks) if blocks else "(empty trace)"

    def summarize_trace(self, trace: str) -> str:
        """
        Produce a compact one-line summary of a trace, e.g.:
          P1=King P2=Queen | DEAL -> P1:BET(1) -> P2:FOLD | P1 wins (P1=2 P2=4)
        """
        steps = [s.strip() for s in trace.split(";") if s.strip() and "cycle{" not in s]

        cards = ("?", "?")
        actions = []
        outcome = ""
        final_chips = ""

        _CARD = {(True, False): "King", (False, True): "Queen", (False, False): "Jack"}
        _PHASE_ACTOR = {"p1": "P1", "p2c": "P2", "p2r": "P2", "p1b": "P1", "p2b": "P2"}

        for step in steps:
            aps = self.parse_step(step)

            # Cards (stable after deal)
            c1 = _CARD.get((aps.get("c1hi", False), aps.get("c1lo", False)), "?")
            c2 = _CARD.get((aps.get("c2hi", False), aps.get("c2lo", False)), "?")
            if c1 != "?" and c2 != "?":
                cards = (c1, c2)

            # Active phase
            phase = next((p for p in _PHASE_ACTOR if aps.get(p, False)), None)

            # Action
            a2, a1, a0 = aps.get("a2", False), aps.get("a1", False), aps.get("a0", False)
            action = _ACTION_DECODE.get((a2, a1, a0))
            if phase and action:
                actor = _PHASE_ACTOR[phase]
                bs = aps.get("bs", False)
                size = f"({2 if bs else 1})" if action in ("BET", "RAISE") else ""
                actions.append(f"{actor}:{action}{size}")

            # Outcome
            if aps.get("win1", False):
                outcome = "P1 wins"
            elif aps.get("win2", False):
                outcome = "P2 wins"
            if aps.get("game_over", False):
                outcome = "GAME OVER"

            # Final chip counts
            m1 = 4*aps.get("m1b2",False) + 2*aps.get("m1b1",False) + aps.get("m1b0",False)
            m2 = 4*aps.get("m2b2",False) + 2*aps.get("m2b1",False) + aps.get("m2b0",False)
            if any(k in aps for k in ("m1b2","m1b1","m1b0")):
                final_chips = f"P1={m1} P2={m2} chips"

        card_str = f"P1={cards[0]} P2={cards[1]}"
        action_str = " -> ".join(actions) if actions else "no actions"
        parts = [card_str, action_str]
        if outcome:
            parts.append(outcome)
        if final_chips:
            parts.append(final_chips)
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Enrich Kuhn Poker traces -- one semantic label per AP."
    )
    parser.add_argument("traces", help="Path to traces .txt file (one Spot trace per line).")
    parser.add_argument("--numbered", action="store_true", help="Print trace index before each trace.")
    args = parser.parse_args()

    enricher = KuhnPokerEnricher()
    enriched = enricher.enrich_file(args.traces)

    for idx, text in enumerate(enriched):
        if args.numbered:
            print(f"=== Trace {idx} ===")
        print(text)
        print()
