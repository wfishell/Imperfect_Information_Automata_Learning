"""
Semantics package

Each game has its own enrichment module that converts raw AP-valuation traces
(Spot format) into human-readable text suitable for LLM preference queries.

To add a new game:
  1. Create src/semantics/<game_name>.py
  2. Subclass TraceEnricher and implement enrich_step() and enrich_trace()
"""

from abc import ABC, abstractmethod


class TraceEnricher(ABC):
    """
    Base class for game-specific trace enrichment.

    Input traces are in Spot semicolon-separated format:
        "ap1&!ap2&ap3; !ap1&ap2&!ap3; ..."

    Each step is a conjunction of signed AP literals representing
    the full boolean valuation of all APs at that time step.
    """

    @abstractmethod
    def enrich_step(self, aps: dict) -> str:
        """
        Convert a single step's AP valuation dict to a human-readable string.

        Args:
            aps: dict mapping AP name (str) → bool value

        Returns:
            Human-readable description of this time step.
        """

    @abstractmethod
    def enrich_trace(self, trace: str) -> str:
        """
        Convert a full Spot-format trace string to a human-readable string.

        Args:
            trace: semicolon-separated Spot trace, e.g.
                   "a0&!a1&c1hi / !win1&!win2; ..."

        Returns:
            Multi-line human-readable description of the trace.
        """

    # ------------------------------------------------------------------
    # Shared parsing utility
    # ------------------------------------------------------------------

    @staticmethod
    def parse_step(step: str) -> dict:
        """
        Parse a single Spot step into an AP → bool dict.

        Handles both input-only steps ("a0&!a1&...") and
        input/output steps ("a0&!a1 / !win1&win2").
        Strips cycle{} annotations if present.
        """
        # Drop cycle annotations
        if "cycle{" in step:
            step = step.split("cycle{")[0]

        # Merge input and output sides
        if "/" in step:
            inp, out = step.split("/", 1)
        else:
            inp, out = step, ""

        aps = {}
        for side in (inp, out):
            for token in side.split("&"):
                token = token.strip(" ()")
                if not token:
                    continue
                if token.startswith("!"):
                    aps[token[1:].strip()] = False
                else:
                    aps[token] = True
        return aps

    def enrich_file(self, path: str) -> list:
        """
        Enrich all traces in a text file (one trace per line).

        Returns a list of enriched trace strings.
        """
        with open(path) as f:
            lines = [l.rstrip("\n") for l in f if l.strip()]
        return [self.enrich_trace(line) for line in lines]
