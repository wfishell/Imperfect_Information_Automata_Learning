from __future__ import annotations
from functools import cached_property


def _apply_move(piles: tuple, pile_index: int, count: int) -> tuple:
    return piles[:pile_index] + (piles[pile_index] - count,) + piles[pile_index + 1:]


class NimState:

    def __init__(self, piles: tuple = (1, 2, 3), player: str = 'P1') -> None:
        self.piles  = piles
        self.player = player

    # ------------------------------------------------------------------
    # Children
    # ------------------------------------------------------------------

    @cached_property
    def children(self) -> dict[tuple, NimState]:
        if self.is_terminal():
            return {}
        next_player = 'P2' if self.player == 'P1' else 'P1'
        result = {}
        for pile_index, size in enumerate(self.piles):
            for count in range(1, size + 1):
                new_piles = _apply_move(self.piles, pile_index, count)
                result[(pile_index, count)] = NimState(
                    piles=new_piles, player=next_player
                )
        return result

    # ------------------------------------------------------------------
    # Terminal detection
    # ------------------------------------------------------------------

    def is_terminal(self) -> bool:
        return all(p == 0 for p in self.piles)

    def winner(self) -> str | None:
        if not self.is_terminal():
            return None
        # The player to move has no moves — the other player made the last move and wins
        return 'P2' if self.player == 'P1' else 'P1'

    # ------------------------------------------------------------------
    # Value — from P2's perspective
    # ------------------------------------------------------------------

    @property
    def value(self) -> int:
        w = self.winner()
        if w == 'P2': return  1
        if w == 'P1': return -1
        return 0

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f'NimState(piles={self.piles}, player={self.player})'
