from __future__ import annotations
from functools import cached_property


def _h_edge(r: int, c: int, cols: int) -> int:
    """Index of the horizontal edge at grid row r, column c."""
    return r * cols + c


def _v_edge(r: int, c: int, rows: int, cols: int) -> int:
    """Index of the vertical edge at grid row r, column c."""
    return (rows + 1) * cols + r * (cols + 1) + c


def _box_borders(r: int, c: int, rows: int, cols: int) -> tuple[int, int, int, int]:
    """Return the four edge indices that border box (r, c)."""
    return (
        _h_edge(r,     c, cols),          # top
        _h_edge(r + 1, c, cols),          # bottom
        _v_edge(r, c,     rows, cols),    # left
        _v_edge(r, c + 1, rows, cols),    # right
    )


def _adjacent_boxes(edge: int, rows: int, cols: int) -> list[tuple[int, int]]:
    """Return (row, col) indices of all boxes that share the given edge."""
    h_count = (rows + 1) * cols
    boxes = []
    if edge < h_count:
        r, c = divmod(edge, cols)
        if r > 0:       boxes.append((r - 1, c))
        if r < rows:    boxes.append((r,     c))
    else:
        idx = edge - h_count
        r, c = divmod(idx, cols + 1)
        if c > 0:       boxes.append((r, c - 1))
        if c < cols:    boxes.append((r, c))
    return boxes


def _boxes_completed_by(edges: tuple, new_edge: int,
                        rows: int, cols: int) -> int:
    """
    Count how many boxes are completed by drawing new_edge into the current
    edge set.  A box is newly complete if new_edge is one of its borders and
    all other three borders are already drawn.
    """
    count = 0
    for r, c in _adjacent_boxes(new_edge, rows, cols):
        borders = _box_borders(r, c, rows, cols)
        if all(edges[e] or e == new_edge for e in borders):
            count += 1
    return count


class DotsAndBoxesState:
    """
    A single game state in Dots and Boxes.

    Edges are stored as a flat boolean tuple.  Indexing convention:

      Horizontal edges  (rows+1 rows × cols cols):
        h_edge(r, c) = r * cols + c

      Vertical edges  (rows rows × cols+1 cols), offset by (rows+1)*cols:
        v_edge(r, c) = (rows+1)*cols + r*(cols+1) + c

    For the default 2×2 box grid this gives 12 edges (indices 0–11).
    """

    def __init__(
        self,
        rows: int = 2,
        cols: int = 2,
        edges: tuple | None = None,
        player: str = 'P1',
        p1_boxes: int = 0,
        p2_boxes: int = 0,
    ) -> None:
        self.rows     = rows
        self.cols     = cols
        self.edges    = edges if edges is not None else (False,) * self._n_edges(rows, cols)
        self.player   = player
        self.p1_boxes = p1_boxes
        self.p2_boxes = p2_boxes

    @staticmethod
    def _n_edges(rows: int, cols: int) -> int:
        return (rows + 1) * cols + rows * (cols + 1)

    @property
    def total_boxes(self) -> int:
        return self.rows * self.cols

    # ------------------------------------------------------------------
    # Children — one entry per undrawn edge
    # ------------------------------------------------------------------

    @cached_property
    def children(self) -> dict[int, DotsAndBoxesState]:
        if self.is_terminal():
            return {}
        result = {}
        for i, drawn in enumerate(self.edges):
            if drawn:
                continue
            completed  = _boxes_completed_by(self.edges, i, self.rows, self.cols)
            new_edges  = self.edges[:i] + (True,) + self.edges[i + 1:]
            new_p1     = self.p1_boxes + (completed if self.player == 'P1' else 0)
            new_p2     = self.p2_boxes + (completed if self.player == 'P2' else 0)
            # same player keeps turn if they completed a box, otherwise swap
            if completed > 0:
                next_player = self.player
            else:
                next_player = 'P2' if self.player == 'P1' else 'P1'
            result[i] = DotsAndBoxesState(
                rows=self.rows, cols=self.cols,
                edges=new_edges, player=next_player,
                p1_boxes=new_p1, p2_boxes=new_p2,
            )
        return result

    # ------------------------------------------------------------------
    # Terminal detection
    # ------------------------------------------------------------------

    def is_terminal(self) -> bool:
        return all(self.edges)

    def winner(self) -> str | None:
        if not self.is_terminal():
            return None
        if self.p1_boxes > self.p2_boxes:
            return 'P1'
        if self.p2_boxes > self.p1_boxes:
            return 'P2'
        return 'draw'

    # ------------------------------------------------------------------
    # Score — from P2's perspective
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
        lines = []
        for r in range(self.rows + 1):
            # horizontal edges
            row = ''
            for c in range(self.cols):
                top = '.' + ('---' if self.edges[_h_edge(r, c, self.cols)] else '   ')
            row = '.'.join(
                '---' if self.edges[_h_edge(r, c, self.cols)] else '   '
                for c in range(self.cols)
            )
            lines.append('.' + row + '.')
            if r < self.rows:
                # vertical edges + box markers
                vert = ''
                for c in range(self.cols + 1):
                    vert += '|' if self.edges[_v_edge(r, c, self.rows, self.cols)] else ' '
                    if c < self.cols:
                        vert += '   '
                lines.append(vert)
        lines.append(f'({self.player} to move  P1:{self.p1_boxes}  P2:{self.p2_boxes})')
        return '\n'.join(lines)
