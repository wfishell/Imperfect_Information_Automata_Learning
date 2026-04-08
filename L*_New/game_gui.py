"""
Dots and Boxes 2x2 — interactive GUI.

You play as P1 (blue). The learned L* strategy plays as P2 (red).
Click on any edge between two dots to claim it.
"""

import tkinter as tk
import pickle, os, time

from dots_and_boxes import (
    DotsAndBoxesNFA, INITIAL_STATE, EDGE_NAMES, EDGE_INDEX,
    BOXES, legal_moves, transition, is_terminal, winner, NUM_EDGES,
)
from strategy_learner import learn_strategy

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
BG       = '#F0F0F0'
DOT_COL  = '#2C3E50'
FREE_COL = '#C8C8C8'
HOVER_COL= '#27AE60'

P1_EDGE  = '#2980B9'
P2_EDGE  = '#C0392B'
P1_FILL  = '#AED6F1'
P2_FILL  = '#FADBD8'

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
CELL   = 160   # pixels between dots
MARGIN = 90
DOT_R  = 9
EDGE_W = 10
HOVER_W= 6
THRESH = 18    # click detection distance (px)

CANVAS_W = MARGIN * 2 + CELL * 2
CANVAS_H = MARGIN * 2 + CELL * 2


class DotsAndBoxesApp:
    def __init__(self, root, learned):
        self.root    = root
        self.learned = learned
        root.title("Dots & Boxes 2×2  —  You (Blue) vs AI (Red)")
        root.resizable(False, False)
        root.configure(bg=BG)

        self.score_var = tk.StringVar()
        tk.Label(root, textvariable=self.score_var,
                 font=('Helvetica', 18, 'bold'), bg=BG).pack(pady=(12, 0))

        self.canvas = tk.Canvas(root, width=CANVAS_W, height=CANVAS_H,
                                bg=BG, highlightthickness=0)
        self.canvas.pack(padx=20, pady=8)

        self.status_var = tk.StringVar()
        self.status_label = tk.Label(root, textvariable=self.status_var,
                                     font=('Helvetica', 13), bg=BG)
        self.status_label.pack()

        tk.Button(root, text="New Game", font=('Helvetica', 12),
                  command=self.reset).pack(pady=10)

        self.canvas.bind('<Motion>',   self.on_hover)
        self.canvas.bind('<Button-1>', self.on_click)

        self.reset()

    # ------------------------------------------------------------------ reset
    def reset(self):
        self.state      = INITIAL_STATE
        self.edge_owner = {}   # edge_idx -> 0 (P1) or 1 (P2)
        self.box_owner  = {}   # box_id   -> 0 or 1
        self.hover      = None
        self.game_over  = False
        self.learned.reset_to_initial()
        self.draw()
        self.set_status("Your turn — click an edge to claim it", P1_EDGE)

    # -------------------------------------------------------------- geometry
    def dot_pos(self, r, c):
        return MARGIN + c * CELL, MARGIN + r * CELL

    def edge_endpoints(self, idx):
        name = EDGE_NAMES[idx]
        r, c = int(name[1]), int(name[2])
        if name[0] == 'H':
            return self.dot_pos(r, c), self.dot_pos(r, c + 1)
        else:
            return self.dot_pos(r, c), self.dot_pos(r + 1, c)

    def dist_to_edge(self, px, py, idx):
        (x1, y1), (x2, y2) = self.edge_endpoints(idx)
        dx, dy = x2 - x1, y2 - y1
        t = max(0, min(1, ((px-x1)*dx + (py-y1)*dy) / (dx*dx + dy*dy)))
        nx, ny = x1 + t*dx, y1 + t*dy
        return ((px-nx)**2 + (py-ny)**2) ** 0.5

    def nearest_free_edge(self, px, py):
        best, best_d = None, THRESH
        for idx in range(NUM_EDGES):
            if idx in self.edge_owner:
                continue
            d = self.dist_to_edge(px, py, idx)
            if d < best_d:
                best_d, best = d, idx
        return best

    # ------------------------------------------------------------------- draw
    def draw(self):
        self.canvas.delete('all')

        # box fills
        for box_id in range(4):
            if box_id not in self.box_owner:
                continue
            r, ci = box_id // 2, box_id % 2
            x1, y1 = self.dot_pos(r,     ci)
            x2, y2 = self.dot_pos(r + 1, ci + 1)
            fill = P1_FILL if self.box_owner[box_id] == 0 else P2_FILL
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline='')

        # box labels
        for box_id in range(4):
            if box_id not in self.box_owner:
                continue
            r, ci = box_id // 2, box_id % 2
            cx = MARGIN + (ci + 0.5) * CELL
            cy = MARGIN + (r  + 0.5) * CELL
            owner = self.box_owner[box_id]
            self.canvas.create_text(cx, cy,
                                    text='P1' if owner == 0 else 'P2',
                                    font=('Helvetica', 22, 'bold'),
                                    fill=P1_EDGE if owner == 0 else P2_EDGE)

        # edges
        for idx in range(NUM_EDGES):
            (x1, y1), (x2, y2) = self.edge_endpoints(idx)
            if idx in self.edge_owner:
                col   = P1_EDGE if self.edge_owner[idx] == 0 else P2_EDGE
                width = EDGE_W
            elif idx == self.hover and self.state.player == 0 and not self.game_over:
                col, width = HOVER_COL, HOVER_W
            else:
                col, width = FREE_COL, 3
            self.canvas.create_line(x1, y1, x2, y2,
                                    fill=col, width=width, capstyle=tk.ROUND)

        # dots
        for r in range(3):
            for ci in range(3):
                x, y = self.dot_pos(r, ci)
                self.canvas.create_oval(x - DOT_R, y - DOT_R,
                                        x + DOT_R, y + DOT_R,
                                        fill=DOT_COL, outline='')

        s = self.state.score
        self.score_var.set(f"You (P1)  {s[0]} — {s[1]}  AI (P2)")

    # ------------------------------------------------------------------ apply
    def apply_move(self, edge, player):
        """Claim an edge, update ownership, detect newly completed boxes."""
        prev_claimed = self.state.claimed
        self.state = transition(self.state, edge)
        self.edge_owner[edge] = player
        for box_id, box_edges in enumerate(BOXES):
            if box_id not in self.box_owner and box_edges <= self.state.claimed:
                self.box_owner[box_id] = player

    # ------------------------------------------------------------------ P2 AI
    def p2_move(self, p2_out):
        """
        Apply one or more P2 moves driven by the learned machine output.
        p2_out: the output returned by learned.step() after P1's move.
        """
        while True:
            if is_terminal(self.state):
                self.end_game()
                return

            if self.state.player == 0:
                # It's P1's turn again (P1 bonus or P2 finished)
                self.set_status("Your turn — click an edge to claim it", P1_EDGE)
                return

            # P2 should move — decode machine output
            legal = legal_moves(self.state)
            edge = None
            if p2_out in EDGE_INDEX:
                candidate = EDGE_INDEX[p2_out]
                if candidate in legal:
                    edge = candidate

            if edge is None:
                print(f"[AUTOMATON ERROR] machine output '{p2_out}' is not a legal move. "
                      f"claimed={len(self.state.claimed)} edges, "
                      f"legal={[EDGE_NAMES[e] for e in legal]}")
                self.set_status("Automaton error — see terminal", '#E74C3C')
                self.game_over = True
                return

            self.apply_move(edge, player=1)
            self.draw()
            self.root.update()
            time.sleep(0.4)

            if is_terminal(self.state):
                self.end_game()
                return

            if self.state.player == 1:
                # P2 bonus turn — ask the machine (PASS input = no P1 move)
                p2_out = self.learned.step('PASS')
            else:
                break

        self.set_status("Your turn — click an edge to claim it", P1_EDGE)

    # --------------------------------------------------------------- handlers
    def on_hover(self, event):
        if self.state.player != 0 or self.game_over:
            return
        edge = self.nearest_free_edge(event.x, event.y)
        if edge != self.hover:
            self.hover = edge
            self.draw()

    def on_click(self, event):
        if self.state.player != 0 or self.game_over:
            return
        edge = self.nearest_free_edge(event.x, event.y)
        if edge is None:
            return

        self.apply_move(edge, player=0)
        self.hover = None

        # Feed P1's move into the learned machine; get P2's response
        p2_out = self.learned.step(EDGE_NAMES[edge])
        self.draw()

        if is_terminal(self.state):
            self.end_game()
            return

        if self.state.player == 0:
            self.set_status("You completed a box — go again!", HOVER_COL)
            return

        self.set_status("AI is thinking...", P2_EDGE)
        self.root.update()
        time.sleep(0.5)
        self.p2_move(p2_out)

    # ---------------------------------------------------------------- end game
    def end_game(self):
        self.game_over = True
        s = self.state.score
        w = winner(self.state)
        if w == 0:
            msg, col = f"You win!  {s[0]}–{s[1]}", P1_EDGE
        elif w == 1:
            msg, col = f"AI wins!  {s[0]}–{s[1]}", P2_EDGE
        else:
            msg, col = f"Draw!  {s[0]}–{s[1]}", '#7F8C8D'
        self.set_status(f"Game over — {msg}", col)
        self.draw()

    def set_status(self, msg, col='#2C3E50'):
        self.status_var.set(msg)
        self.status_label.configure(fg=col)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
CACHE = os.path.join(os.path.dirname(__file__), 'learned_strategy.pkl')

def load_or_learn():
    if os.path.exists(CACHE):
        print("Loading cached strategy...")
        with open(CACHE, 'rb') as f:
            return pickle.load(f)
    print("No cache found — running L* learner (takes ~10s)...")
    learned, *_ = learn_strategy(verbose=True)
    with open(CACHE, 'wb') as f:
        pickle.dump(learned, f)
    print("Strategy cached.")
    return learned


if __name__ == '__main__':
    learned = load_or_learn()

    dot_path = os.path.join(os.path.dirname(__file__), 'strategy.dot')
    learned.save(file_path=dot_path[:-4])
    print(f"DOT file saved to {dot_path}")
    print(f"Render with: dot -Tpng strategy.dot -o strategy.png")

    root = tk.Tk()
    app  = DotsAndBoxesApp(root, learned)
    root.mainloop()
