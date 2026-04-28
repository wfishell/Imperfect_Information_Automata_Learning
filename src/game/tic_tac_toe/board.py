
from __future__ import annotations
from functools import cached_property

EMPTY, X, O = 0, 1, 2

LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6),             # diagonals
]


class TicTacToeState:

    def __init__(self, board: tuple = None, player: str = 'P1') -> None:
        self.board  = board if board is not None else (EMPTY,) * 9
        self.player = player

    # ------------------------------------------------------------------
    # Children — the key interface the algorithm uses
    # ------------------------------------------------------------------

    @cached_property
    def children(self) -> dict[int, TicTacToeState]:
        if self.is_terminal():
            return {}
        token      = X if self.player == 'P1' else O

        next_player = 'P2' if self.player == 'P1' else 'P1'
        return {
            sq: TicTacToeState(
                board  = self.board[:sq] + (token,) + self.board[sq + 1:],
                player = next_player,
            )
            for sq in range(9)
            if self.board[sq] == EMPTY
        }

    # ------------------------------------------------------------------
    # Terminal detection
    # ------------------------------------------------------------------

    def is_terminal(self) -> bool:
        return self.winner() is not None

    def winner(self) -> str | None:
        for a, b, c in LINES:
            if self.board[a] == self.board[b] == self.board[c] != EMPTY:
                return 'P1' if self.board[a] == X else 'P2'
        if EMPTY not in self.board:
            return 'draw'
        return None

    # ------------------------------------------------------------------
    # Score — from P2 (O) perspective
    # ------------------------------------------------------------------

    @property
    def value(self) -> int:
        w = self.winner()
        if w == 'P2':   return  1
        if w == 'P1':   return -1
        return 0

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        symbols = {EMPTY: '.', X: 'X', O: 'O'}
        rows = []
        for r in range(3):
            rows.append(' '.join(symbols[self.board[r * 3 + c]] for c in range(3)))
        return '\n'.join(rows) + f'\n({self.player} to move)'
