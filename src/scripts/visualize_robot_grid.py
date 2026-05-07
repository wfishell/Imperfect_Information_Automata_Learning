"""
Interactive pygame viewer for a learned RobotGrid Mealy.

Loads the .dot Mealy from outputs/ and lets you drive episodes
manually:
  - Click an eligible (light-blue) cell to drop a package there;
    the controller will start moving toward it.
  - Press SPACE to advance one Mealy step at a time.
  - Press A to toggle auto-step (controller moves on its own).
  - Press R to reset the env (also resets the Mealy).
  - Press ESC to quit.

Gas bar, dropoff, refuel, and home positions are fixed — the env
config matches the args you trained with. Pass the same --rows /
--cols / --gas-max / --task-cells as the learner.

Usage (matches your learner CLI):
    python -m src.scripts.visualize_robot_grid \\
        --rows 4 --cols 4 --gas-max 10 --task-cells "(1,1)"

    python -m src.scripts.visualize_robot_grid \\
        --rows 3 --cols 3 --gas-max 5 --task-cells "(1,1)"
"""

from __future__ import annotations
import argparse
import re
import sys
import tempfile
from pathlib import Path

import pygame
from aalpy.utils import load_automaton_from_file

from src.control_systems.RobotGrid import (
    RobotGridNFA, RobotGridState,
    N, S, E, W, PICKUP, DROP, REFUEL,
)


# ----------------------------------------------------------------------
# Visual constants
# ----------------------------------------------------------------------

CELL_SIZE         = 110
GRID_PAD          = 40
INFO_PANEL_HEIGHT = 200

COLORS = {
    'bg':              (28, 28, 36),
    'grid_line':       (90, 90, 110),
    'cell':            (55, 55, 70),
    'eligible':        (70, 95, 130),
    'home':            (60, 170, 100),
    'refuel':          (235, 200, 70),
    'dropoff':         (220, 100, 100),
    'task':            (255, 165, 0),
    'robot':           (110, 180, 255),
    'robot_carrying':  (110, 255, 200),
    'text':            (235, 235, 240),
    'text_dim':        (160, 160, 175),
    'gas_full':        (60, 200, 110),
    'gas_mid':         (200, 220, 80),
    'gas_low':         (240, 160, 60),
    'gas_critical':    (240, 90, 90),
    'event_panel':     (38, 38, 50),
    'success':         (110, 240, 150),
    'failure':         (240, 110, 110),
}


# ----------------------------------------------------------------------
# Robust Mealy loader. aalpy's save() writes:
#     q0 [shape=doublecircle];
#     q1 [shape=circle];
#     ...
# but its load() expects:
#     __start0 -> q0;
#     q0 [label="q0"];
# i.e. state-definition lines need a `label=...` AND there has to be an
# explicit __start0 edge for the initial state. We patch both, write to
# a temp file, then defer to aalpy's loader.
# ----------------------------------------------------------------------

def load_mealy_dot(path: Path):
    content = path.read_text()

    # 1) Find which state is initial (doublecircle marker), record it,
    #    AND rewrite every `qN [shape=...];` line to also include a label
    #    so aalpy's loader registers the state.
    initial_state = None

    def fix_state_def(match):
        nonlocal initial_state
        sid    = match.group(1)
        shape  = match.group(2)
        if shape == 'doublecircle' and initial_state is None:
            initial_state = sid
        return f'{sid} [label="{sid}", shape={shape}];'

    content = re.sub(
        r'(\w+)\s*\[shape=(\w+)\];',
        fix_state_def,
        content,
    )

    if initial_state is None:
        raise RuntimeError(f'No initial state (doublecircle) found in {path}')

    # 2) Add an explicit __start0 -> initial edge if not already there.
    if '__start0' not in content:
        marker = (
            f'  __start0 [label="" shape=none];\n'
            f'  __start0 -> {initial_state};\n'
        )
        content = re.sub(
            r'(digraph\s+\w+\s*\{\n)',
            r'\1' + marker,
            content,
            count=1,
        )

    with tempfile.NamedTemporaryFile(mode='w', suffix='.dot', delete=False) as f:
        f.write(content)
        tmp = f.name
    return load_automaton_from_file(tmp, automaton_type='mealy')


# ----------------------------------------------------------------------
# Step the Mealy once: P1 input → P2 output → env transitions twice.
# ----------------------------------------------------------------------

def step_once(state: RobotGridState, p1_input,
              model, trace_log=None) -> tuple[RobotGridState, str | None, bool]:
    """
    Feed one P1 input to the env and the model. Return:
        (new_state, action_taken, used_fallback)

    `action_taken` is the P2 output the model emitted (or the fallback
    action if the model emitted something illegal at the resulting
    P2 state). Strict P1 → P2 → P1 alternation.

    The ONLY decision source is `model.step(...)`. There is no oracle,
    no manhattan recomputation, no policy fallback to anything other
    than "first legal child" if the Mealy emits an illegal output.
    Pass `trace_log=True` to log each (input, mealy-output, used)
    triple to stdout so you can audit the controller's behaviour.
    """
    if state.player != 'P1' or state.is_terminal():
        return state, None, False
    if p1_input not in state.children:
        return state, None, False

    # Mealy operates on string symbols (post-load); env operates on tuples.
    p2_output_str = model.step(str(p1_input))
    p2_output     = _coerce_action(p2_output_str)
    state_p2      = state.children[p1_input]
    used_fallback = False
    final_action  = p2_output

    if state_p2.is_terminal():
        new_state = state_p2
    elif p2_output in state_p2.children:
        new_state = state_p2.children[p2_output]
    else:
        # Mealy emitted an illegal action at this state — fall back.
        fallback     = next(iter(state_p2.children))
        new_state    = state_p2.children[fallback]
        final_action = fallback
        used_fallback = True

    if trace_log:
        tag = '[FALLBACK]' if used_fallback else ''
        print(f'  mealy.step({p1_input!r:55s})  →  {p2_output!r:10s}  '
              f'{tag}', flush=True)

    return new_state, final_action, used_fallback


# Action symbols saved as strings come back as raw strings — but the
# env's children dict is keyed by string-typed actions (N, S, E, W, ...)
# already, so they already match. This is a no-op for our P2 alphabet
# but makes the conversion explicit if the alphabet ever changes.
def _coerce_action(s):
    return s


# ----------------------------------------------------------------------
# Pygame helpers
# ----------------------------------------------------------------------

def cell_rect(row: int, col: int) -> pygame.Rect:
    return pygame.Rect(
        GRID_PAD + col * CELL_SIZE,
        GRID_PAD + row * CELL_SIZE,
        CELL_SIZE, CELL_SIZE,
    )


def gas_color(gas: int, gas_max: int) -> tuple[int, int, int]:
    pct = gas / max(gas_max, 1)
    if gas <= 0:    return COLORS['gas_critical']
    if pct >= 0.7:  return COLORS['gas_full']
    if pct >= 0.4:  return COLORS['gas_mid']
    if pct >= 0.2:  return COLORS['gas_low']
    return COLORS['gas_critical']


def draw_grid(screen, font, big_font, nfa: RobotGridNFA,
              state: RobotGridState) -> None:
    eligible = set(nfa.eligible_cells)
    for r in range(nfa.rows):
        for c in range(nfa.cols):
            rect = cell_rect(r, c)
            if (r, c) == nfa.home:    color = COLORS['home']
            elif (r, c) == nfa.refuel: color = COLORS['refuel']
            elif (r, c) == nfa.dropoff: color = COLORS['dropoff']
            elif (r, c) in eligible:  color = COLORS['eligible']
            else:                      color = COLORS['cell']
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, COLORS['grid_line'], rect, 2)

            label = None
            if (r, c) == nfa.home:    label = 'H'
            elif (r, c) == nfa.refuel: label = 'R'
            elif (r, c) == nfa.dropoff: label = 'D'
            if label:
                text = big_font.render(label, True, COLORS['text'])
                screen.blit(text, text.get_rect(center=rect.center))

            if state.task_loc == (r, c):
                inner = rect.inflate(-CELL_SIZE // 2, -CELL_SIZE // 2)
                pygame.draw.rect(screen, COLORS['task'], inner)
                pkg = big_font.render('P', True, (40, 40, 40))
                screen.blit(pkg, pkg.get_rect(center=inner.center))


def draw_robot(screen, state: RobotGridState) -> None:
    r, c = state.pos
    x = GRID_PAD + c * CELL_SIZE + CELL_SIZE // 2
    y = GRID_PAD + r * CELL_SIZE + CELL_SIZE // 2
    color = COLORS['robot_carrying'] if state.carrying else COLORS['robot']
    pygame.draw.circle(screen, color, (x, y), CELL_SIZE // 4)
    pygame.draw.circle(screen, (15, 15, 25), (x, y), CELL_SIZE // 4, 2)


def draw_info_panel(screen, font, big_font, nfa: RobotGridNFA,
                    state: RobotGridState, last_action: str | None,
                    auto_advance: bool, used_fallback: bool) -> None:
    panel_y = GRID_PAD + nfa.rows * CELL_SIZE + 10
    panel_rect = pygame.Rect(
        GRID_PAD, panel_y,
        nfa.cols * CELL_SIZE, INFO_PANEL_HEIGHT - 20,
    )
    pygame.draw.rect(screen, COLORS['event_panel'], panel_rect, border_radius=8)

    # Gas bar
    bar_x = panel_rect.x + 12
    bar_y = panel_rect.y + 12
    bar_w = panel_rect.width - 24
    pygame.draw.rect(screen, COLORS['cell'], (bar_x, bar_y, bar_w, 22), border_radius=4)
    pct = state.gas / max(nfa.gas_max, 1)
    fill = max(0, int(bar_w * pct))
    pygame.draw.rect(screen, gas_color(state.gas, nfa.gas_max),
                     (bar_x, bar_y, fill, 22), border_radius=4)
    gas_text = font.render(
        f'Gas: {state.gas}/{nfa.gas_max}', True, COLORS['text'],
    )
    screen.blit(gas_text, (bar_x + 6, bar_y + 3))

    # Status lines
    text_y = bar_y + 32
    lines = [
        ('pos',          str(state.pos)),
        ('carrying',     'YES' if state.carrying else 'no'),
        ('task',         str(state.task_loc) if state.task_loc else '— idle —'),
        ('delivered',    str(state.delivered_count)),
        ('step',         f'{state.step_count}/{nfa.max_steps}'),
        ('last action',  str(last_action) if last_action else '—'),
    ]
    for i, (label, value) in enumerate(lines):
        col = i % 2
        row = i // 2
        x = panel_rect.x + 12 + col * (panel_rect.width // 2)
        y = text_y + row * 22
        screen.blit(font.render(f'{label}:', True, COLORS['text_dim']), (x, y))
        screen.blit(font.render(value, True, COLORS['text']), (x + 100, y))

    # Outcome banner if terminal
    if state.is_terminal():
        outcome = state.winner()
        color = (COLORS['failure'] if outcome == 'failed'
                 else COLORS['success'] if outcome == 'delivered'
                 else COLORS['text_dim'])
        msg = {'failed': 'GAS OUT', 'timeout': 'TIMEOUT'}.get(outcome, str(outcome))
        text = big_font.render(msg, True, color)
        screen.blit(text, (panel_rect.right - text.get_width() - 16,
                           panel_rect.y + 8))

    # Fallback warning
    if used_fallback:
        warn = font.render('!!Mealy emitted illegal output (fallback used)',
                           True, COLORS['failure'])
        screen.blit(warn, (panel_rect.x + 12, panel_rect.bottom - 22))


def draw_help(screen, font, panel_top: int, screen_w: int) -> None:
    lines = [
        'Click eligible cell → drop a package',
        'SPACE   step | A   toggle auto | R   reset | ESC   quit',
    ]
    y = panel_top - 22 * len(lines) - 4
    for line in lines:
        text = font.render(line, True, COLORS['text_dim'])
        screen.blit(text, (GRID_PAD, y))
        y += 22


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _parse_task(s: str) -> tuple[int, int]:
    s = s.strip().lstrip('(').rstrip(')')
    r, c = (int(x.strip()) for x in s.split(','))
    return (r, c)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows',     type=int, default=4)
    parser.add_argument('--cols',     type=int, default=4)
    parser.add_argument('--gas-max',  type=int, default=10)
    parser.add_argument('--max-steps', type=int, default=50)
    parser.add_argument('--home',     type=_parse_task, default=(0, 0))
    parser.add_argument('--refuel',   type=_parse_task, default=None)
    parser.add_argument('--dropoff',  type=_parse_task, default=None)
    parser.add_argument('--task-cells', nargs='+', type=_parse_task, default=None,
                        help='Eligible task cells (must match training). '
                             'Default: every non-special cell.')
    parser.add_argument('--model',    type=str,
                        default='outputs/learned_strategy_robot_grid.dot')
    parser.add_argument('--step-delay', type=int, default=350,
                        help='ms between auto-steps')
    parser.add_argument('--auto', action='store_true',
                        help='Start with auto-step on')
    parser.add_argument('--trace', action='store_true',
                        help='Log every (input, mealy_output) decision to stdout')
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load Mealy + build env (matching the learner config).
    # ------------------------------------------------------------------
    model_path = Path(args.model)
    if not model_path.exists():
        print(f'ERROR: model file not found: {model_path}')
        print('Run learner_robot_grid.py first to produce one.')
        sys.exit(1)
    model = load_mealy_dot(model_path)
    model.reset_to_initial()
    file_size = model_path.stat().st_size
    file_mtime = model_path.stat().st_mtime
    import time as _time, hashlib as _hash
    file_hash  = _hash.md5(model_path.read_bytes()).hexdigest()[:12]
    print(f'================== CONTROLLER VERIFICATION ==================')
    print(f'  Source:    .dot file (NO oracle, NO live computation)')
    print(f'  Path:      {model_path}')
    print(f'  Modified:  {_time.ctime(file_mtime)}')
    print(f'  Size:      {file_size} bytes')
    print(f'  MD5 hash:  {file_hash}')
    print(f'  Mealy:     {len(model.states)} states, '
          f'initial_state={model.initial_state.state_id}')
    n_inputs = len(model.initial_state.transitions)
    print(f'  Initial-state out-edges: {n_inputs} (sample of inputs)')
    sample = list(model.initial_state.transitions.items())[:3]
    for inp, dest in sample:
        out = model.initial_state.output_fun.get(inp, '<no-output>')
        print(f'      on input {inp!r}  →  emit {out!r}, goto {dest.state_id}')
    print(f'=============================================================')

    eligible = tuple(args.task_cells) if args.task_cells else None
    nfa = RobotGridNFA(
        rows=args.rows, cols=args.cols,
        home=args.home, refuel=args.refuel, dropoff=args.dropoff,
        gas_max=args.gas_max, max_steps=args.max_steps,
        eligible_cells=eligible,
    )
    eligible = nfa.eligible_cells   # resolve from NFA's default if not passed

    # ------------------------------------------------------------------
    # Pygame setup.
    # ------------------------------------------------------------------
    pygame.init()
    width  = GRID_PAD * 2 + nfa.cols * CELL_SIZE
    height = GRID_PAD     + nfa.rows * CELL_SIZE + INFO_PANEL_HEIGHT
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('RobotGrid Mealy Viewer')
    font     = pygame.font.SysFont('Arial', 16)
    big_font = pygame.font.SysFont('Arial', 26, bold=True)

    state = nfa.root
    last_action: str | None = None
    pending_p1_input = None
    last_step_ms = 0
    auto_advance = args.auto
    used_fallback = False

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    state = nfa.root
                    model.reset_to_initial()
                    last_action = None
                    used_fallback = False
                    pending_p1_input = None
                elif event.key == pygame.K_a:
                    auto_advance = not auto_advance
                elif event.key == pygame.K_SPACE and not state.is_terminal():
                    if state.player == 'P1':
                        if state.task_loc is None:
                            pass  # need user click first
                        else:
                            pending_p1_input = state.observation
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if state.player == 'P1' and state.task_loc is None and not state.is_terminal():
                    mx, my = event.pos
                    col = (mx - GRID_PAD) // CELL_SIZE
                    row = (my - GRID_PAD) // CELL_SIZE
                    if 0 <= row < nfa.rows and 0 <= col < nfa.cols:
                        if (row, col) in eligible:
                            pending_p1_input = ('TASK', (row, col))

        # Auto-advance: feed the deterministic observation when the env
        # is in an active P1 turn.
        if (auto_advance and pending_p1_input is None
                and not state.is_terminal()
                and state.player == 'P1' and state.task_loc is not None):
            now = pygame.time.get_ticks()
            if now - last_step_ms >= args.step_delay:
                pending_p1_input = state.observation

        # Apply pending input.
        if pending_p1_input is not None and not state.is_terminal():
            new_state, action, used_fallback = step_once(
                state, pending_p1_input, model,
                trace_log=args.trace,
            )
            state = new_state
            last_action = action
            last_step_ms = pygame.time.get_ticks()
            pending_p1_input = None

        # Render.
        screen.fill(COLORS['bg'])
        draw_grid(screen, font, big_font, nfa, state)
        draw_robot(screen, state)
        panel_top = GRID_PAD + nfa.rows * CELL_SIZE + 10
        draw_help(screen, font, panel_top, width)
        draw_info_panel(screen, font, big_font, nfa,
                         state, last_action, auto_advance, used_fallback)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == '__main__':
    main()
