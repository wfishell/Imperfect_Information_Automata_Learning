from __future__ import annotations
from collections import deque
from functools import cached_property

EMPTY, X, O = 0, 1, 2


def _neighbors(cell: int, size: int) -> list[int]:
    """Return the (up to 6) hex-grid neighbors of cell on a size×size board.

    Adjacency for cell (r, c):
        (r-1, c)  (r-1, c+1)
        (r,   c-1)            (r,   c+1)
        (r+1, c-1)  (r+1, c)
    """
    r, c = divmod(cell, size)
    candidates = [
        (r - 1, c), (r - 1, c + 1),
        (r,     c - 1), (r,     c + 1),
        (r + 1, c - 1), (r + 1, c),
    ]
    return [nr * size + nc for nr, nc in candidates
            if 0 <= nr < size and 0 <= nc < size]


def _connected(board: tuple, token: int, size: int) -> bool:
    """Return True if token has a winning connection across the board.

    X (P1) wins by connecting top row (r=0) to bottom row (r=size-1).
    O (P2) wins by connecting left col (c=0) to right col (c=size-1).
    """
    if token == X:
        starts = [c for c in range(size) if board[c] == X]

        def is_goal(cell: int) -> bool:
            return cell >= size * (size - 1)
    else:
        starts = [r * size for r in range(size) if board[r * size] == O]

        def is_goal(cell: int) -> bool:
            return cell % size == size - 1

    visited: set = set(starts)
    queue: deque = deque(starts)

    while queue:
        cell = queue.popleft()
        if is_goal(cell):
            return True
        for nb in _neighbors(cell, size):
            if nb not in visited and board[nb] == token:
                visited.add(nb)
                queue.append(nb)
    return False


class HexState:
    """
    A single game state in Hex on a size×size board.

    board  : flat tuple of length size*size, values EMPTY/X/O.
    player : 'P1' (X, connects top row → bottom row) or
             'P2' (O, connects left col → right col).

    Hex has no draws: one player always wins.
    """

    def __init__(
        self,
        size:   int          = 3,
        board:  tuple | None = None,
        player: str          = 'P1',
    ) -> None:
        self.size   = size
        self.board  = board if board is not None else (EMPTY,) * (size * size)
        self.player = player

    @cached_property
    def children(self) -> dict[int, HexState]:
        if self.is_terminal():
            return {}
        token       = X if self.player == 'P1' else O
        next_player = 'P2' if self.player == 'P1' else 'P1'
        return {
            cell: HexState(
                size   = self.size,
                board  = self.board[:cell] + (token,) + self.board[cell + 1:],
                player = next_player,
            )
            for cell in range(self.size * self.size)
            if self.board[cell] == EMPTY
        }

    def is_terminal(self) -> bool:
        return self.winner() is not None

    def winner(self) -> str | None:
        if _connected(self.board, X, self.size):
            return 'P1'
        if _connected(self.board, O, self.size):
            return 'P2'
        return None

    @property
    def value(self) -> int:
        w = self.winner()
        if w == 'P2':
            return  1
        if w == 'P1':
            return -1
        return 0

    def __repr__(self) -> str:
        symbols = {EMPTY: '.', X: 'X', O: 'O'}
        rows = []
        for r in range(self.size):
            indent = ' ' * r
            row = ' '.join(symbols[self.board[r * self.size + c]]
                           for c in range(self.size))
            rows.append(indent + row)
        return '\n'.join(rows) + f'\n({self.player} to move)'
