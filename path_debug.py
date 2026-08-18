"""Standalone step-through debugger for the snake pathfinding.

Visualizes, one expansion/decision at a time:
  * A* search to the apple  - explored (cell,time) cells, open frontier, the
    path currently being examined, and the final chosen path.
  * can_reach_tail flood     - the cells reachable from the head accounting for
    the retreating tail, and whether the tail itself is reached.
  * Decision status          - the mode Main would pick (safe apple path /
    tail-follow survival / stuck), computed with the correct path orientation.

The stepped searches mirror the real functions in AlgorithmRework so what you
see matches what the game does.

Controls
  SPACE  step (hold to auto-repeat)  TAB   switch view (A* apple <-> A* tail)
  F      run current view to end     P     cycle preset scenarios
  R      restart current view        C     clear snake
  L      load captured game state (debug_state.json, written by Main.py's C key)
  left-click  extend snake (each click adjacent to the head; last click = head)
  right-click set apple              BACKSPACE undo last snake segment
  ESC / close  quit

Run:  python3 path_debug.py [grid_w grid_h]
      python3 path_debug.py --load [debug_state.json]
"""
import sys
import json
from collections import deque
from heapq import heappush, heappop
from itertools import count

import pygame

STATE_FILE = "debug_state.json"     # written by Main.py (press C in the game)
COMPLETE_MAX_FREE = 25              # matches Main.complete_max_free (gate for the complete search)

from Algorithm import (
    create_adjacent_grid,
    A_star_path,
    A_star_path_complete,
    A_star_path_tail,
    can_reach_tail,
    snake_after_eating,
    get_new_snake,
    manhattan_distance,
    BFS_path,
)

# -----------------------------------------------------------------------------
# Instrumented searches: identical logic to AlgorithmRework, but they `yield`
# their internal state after each step so we can render it.
# -----------------------------------------------------------------------------

def astar_steps(G, snake, apple):
    """Mirror of A_star_path (fast (cell,time) search), yielding state per pop."""
    snake_head = snake[0]
    if snake_head == apple or apple is None:
        yield {"phase": "done", "result": [], "safe": False,
               "path": [], "visited": set(), "frontier": set()}
        return

    max_time = len(G)
    tie = count()
    body0 = set(snake)
    def openness(cell):
        return sum(1 for nb in G[cell] if nb not in body0)
    open_heap = [(manhattan_distance(snake_head, apple), openness(snake_head), next(tie), [snake_head])]
    visited = set()

    while open_heap:
        _, _, _, path = heappop(open_heap)
        v = path[-1]
        t = len(path) - 1
        if (v, t) in visited:
            continue
        visited.add((v, t))

        yield {"phase": "pop", "path": list(path),
               "visited": {c for (c, _t) in visited},
               "frontier": {p[-1] for (_f, _o, _c, p) in open_heap}}

        if v == apple:
            if can_reach_tail(G, snake_after_eating(snake, path)):
                yield {"phase": "done", "result": path[:0:-1], "safe": True,
                       "path": list(path), "visited": {c for (c, _t) in visited},
                       "frontier": {p[-1] for (_f, _o, _c, p) in open_heap}}
                return
            continue

        if t >= max_time:
            continue

        for node in G[v]:
            new_path = path + [node]
            new_snake = get_new_snake(snake, new_path)
            if node in set(new_snake[1:]):
                continue
            g = len(new_path) - 1
            if (node, g) in visited:
                continue
            f = g + manhattan_distance(node, apple)
            heappush(open_heap, (f, openness(node), next(tie), new_path))

    yield {"phase": "done", "result": [], "safe": False,
           "path": [], "visited": {c for (c, _t) in visited}, "frontier": set()}


def astar_complete_steps(G, snake, apple, max_states=3000):
    """Mirror of A_star_path_complete: A* keyed on the FULL body configuration
    (head cell + ordered body) so no body-config is pruned, yielding state per
    pop. Bounded by max_states expansions."""
    snake = list(snake)
    head = snake[0]
    if apple is None or head == apple:
        yield {"phase": "done", "result": [], "safe": False, "path": [],
               "visited": set(), "frontier": set(), "states": 0, "capped": False}
        return

    tie = count()
    body0 = set(snake)
    def openness(cell):
        return sum(1 for nb in G[cell] if nb not in body0)
    open_heap = [(manhattan_distance(head, apple), openness(head), next(tie), [head])]
    visited = set()          # (cell, full ordered body)
    visited_cells = set()    # for display only
    states = 0

    while open_heap and states < max_states:
        _, _, _, path = heappop(open_heap)
        v = path[-1]
        cur = get_new_snake(snake, path)
        key = (v, tuple(cur))
        if key in visited:
            continue
        visited.add(key)
        states += 1
        visited_cells.add(v)

        yield {"phase": "pop", "path": list(path), "states": states,
               "visited": set(visited_cells),
               "frontier": {p[-1] for (_f, _o, _c, p) in open_heap}}

        if v == apple:
            if can_reach_tail(G, snake_after_eating(snake, path)):
                yield {"phase": "done", "result": path[:0:-1], "safe": True, "path": list(path),
                       "visited": set(visited_cells), "frontier": set(), "states": states, "capped": False}
                return
            continue

        if len(path) - 1 >= len(G):
            continue

        for node in G[v]:
            new_snake = get_new_snake(snake, path + [node])
            if node in set(new_snake[1:]):
                continue
            f = len(path) + manhattan_distance(node, apple)
            heappush(open_heap, (f, openness(node), next(tie), path + [node]))

    yield {"phase": "done", "result": [], "safe": False, "path": [],
           "visited": set(visited_cells), "frontier": set(),
           "states": states, "capped": states >= max_states}


def flood_steps(G, snake):
    """Mirror of can_reach_tail, yielding the reachable set as it grows."""
    snake_head = snake[0]
    tail = snake[-1]
    if snake_head == tail:
        yield {"phase": "done", "reachable": set(), "current": None, "tail_reached": True}
        return

    L = len(snake)
    vacate_time = {snake[L - 1 - k]: k + 1 for k in range(L)}
    Q = deque([(snake_head, 0)])
    visited = {(snake_head, 0)}
    reachable = {snake_head}

    while Q:
        cell, t = Q.popleft()
        real_nt = t + 1
        nt = real_nt if real_nt < L else L
        yield {"phase": "pop", "reachable": set(reachable), "current": cell, "tail_reached": False}
        for node in G[cell]:
            if node in vacate_time and vacate_time[node] > real_nt:
                continue
            if node == tail:
                reachable.add(node)
                yield {"phase": "done", "reachable": set(reachable), "current": cell, "tail_reached": True}
                return
            state = (node, nt)
            if state in visited:
                continue
            visited.add(state)
            reachable.add(node)
            Q.append(state)

    yield {"phase": "done", "reachable": set(reachable), "current": None, "tail_reached": False}


def astar_tail_steps(G, snake):
    """Accurate 'can the head reach its tail?' search, yielding state per pop.

    Targets the tail's FIXED original cell (which frees up as the tail retreats)
    and tracks the snake's body per path via get_new_snake, so the head cannot
    overlap its own trail. This is what makes it accurate where the optimistic
    flood was not: it will NOT claim a coil is escapable when the head's own
    body would trap it, yet it still confirms open, safe states.

    (Note: this deliberately does NOT mirror A_star_path_tail, which chases the
    *moving* tail and so can never 'catch' it on an open board - reporting safe
    states as unreachable.)
    """
    head = snake[0]
    tail0 = snake[-1]
    if head == tail0:
        yield {"phase": "done", "result": [], "found": True, "target": tail0,
               "visited": {head}, "frontier": set()}
        return

    max_time = len(G)
    tie = count()
    body0 = set(snake)
    def openness(cell):
        return sum(1 for nb in G[cell] if nb not in body0)
    open_heap = [(manhattan_distance(head, tail0), openness(head), next(tie), [head])]
    visited = set()

    while open_heap:
        _, _, _, path = heappop(open_heap)
        v = path[-1]
        t = len(path) - 1
        if (v, t) in visited:
            continue
        visited.add((v, t))

        yield {"phase": "pop", "path": list(path), "target": tail0,
               "visited": {c for (c, _t) in visited},
               "frontier": {p[-1] for (_f, _o, _c, p) in open_heap}}

        if v == tail0:
            yield {"phase": "done", "result": path[:0:-1], "found": True, "path": list(path),
                   "target": tail0, "visited": {c for (c, _t) in visited}, "frontier": set()}
            return

        if t >= max_time:
            continue

        for node in G[v]:
            new_snake = get_new_snake(snake, path + [node])
            if node in set(new_snake[1:]):        # tail cell frees up via get_new_snake
                continue
            g = len(path)
            if (node, g) in visited:
                continue
            f = g + manhattan_distance(node, tail0)
            heappush(open_heap, (f, openness(node), next(tie), path + [node]))

    yield {"phase": "done", "result": [], "found": False, "target": tail0,
           "visited": {c for (c, _t) in visited}, "frontier": set()}


def reach_tail_cell(G, snake):
    """Bool version of the accurate tail-reachability check (final result only)."""
    last = None
    for last in astar_tail_steps(G, snake):
        pass
    return bool(last and last.get("found"))


def compute_decision(G, snake, apple):
    """Mirror of Main.decide_path's classification: fast A* to the apple, then the
    full-body complete search (only in the tight endgame, free <= COMPLETE_MAX_FREE),
    else survival. Returns (mode, head_first_path)."""
    if apple is None:
        return "no apple", []
    path = A_star_path(G, snake, apple)                 # apple-first, head-excluded
    if not path and (len(G) - len(snake)) <= COMPLETE_MAX_FREE:
        path = A_star_path_complete(G, snake, apple)    # completeness fallback
    if path:
        return "SAFE apple path", [snake[0]] + path[::-1]
    if A_star_path_tail(G, snake):                      # can we still reach our tail?
        return "TAIL survival (reachable)", []
    return "STUCK - no safe path", []


# -----------------------------------------------------------------------------
# Scenario presets
# -----------------------------------------------------------------------------

def boustrophedon(gx, gy):
    cells = []
    for j in range(gy):
        cols = range(gx) if j % 2 == 0 else range(gx - 1, -1, -1)
        for i in cols:
            cells.append((i, j))
    return cells


def presets(gx, gy):
    scenarios = []
    # 0: open board, short snake
    scenarios.append((
        [(gx // 2, gy // 2), (gx // 2 - 1, gy // 2), (gx // 2 - 2, gy // 2)],
        (gx - 1, gy - 1),
    ))
    # 1: medium coil (~half the board), apple ahead
    order = boustrophedon(gx, gy)
    n = (gx * gy) // 2
    snake = list(reversed(order[:n]))                  # head = order[n-1]
    scenarios.append((snake, order[-1]))
    # 2: tight (~70% of board) - forces survival decisions
    n2 = int(gx * gy * 0.7)
    snake2 = list(reversed(order[:n2]))
    free = [c for c in order if c not in set(snake2)]
    scenarios.append((snake2, free[len(free) // 2] if free else None))
    return scenarios


# -----------------------------------------------------------------------------
# Colors
# -----------------------------------------------------------------------------
C_BG        = (24, 24, 28)
C_CELL      = (38, 38, 44)
C_GRID      = (55, 55, 62)
C_VISITED   = (44, 62, 96)
C_FRONTIER  = (0, 170, 175)
C_REACH     = (28, 92, 92)
C_CURRENT   = (235, 225, 110)
C_POPPATH   = (200, 190, 70)
C_RESULT    = (70, 205, 110)
C_SNAKE     = (46, 160, 66)
C_SNAKE_HI  = (120, 225, 140)   # body near the head
C_SNAKE_LO  = (26, 92, 44)      # body near the tail
C_SPINE     = (16, 70, 32)      # line threaded through the body
C_HEAD      = (70, 130, 235)
C_HEAD_RING = (200, 220, 255)
C_TAIL_OK   = (80, 225, 130)
C_TAIL_NO   = (225, 80, 80)
C_APPLE     = (215, 65, 65)
C_HUD_BG    = (16, 16, 20)
C_TEXT      = (232, 232, 236)
C_DIM       = (150, 150, 158)


class Debugger:
    def __init__(self, gx, gy, block=34):
        self.gx, self.gy, self.bs = gx, gy, block
        self.grid = create_adjacent_grid(gx, gy)
        self.hud_h = 150
        self.W = gx * block
        self.H = gy * block + self.hud_h

        self.presets = presets(gx, gy)
        self.preset_i = 0
        self.snake, self.apple = self.presets[0]
        self.snake = list(self.snake)

        self.view = "astar"          # or "flood"
        self.gen = None
        self.state = None
        self.flood_label = ""
        self.flood_snake = None      # the post-eat snake the flood runs on

        # hold-to-step (spacebar auto-repeat)
        self.space_was_down = False
        self.hold_timer = 0.0
        self.repeating = False
        self.repeat_delay = 0.30     # seconds held before auto-repeat starts
        self.repeat_interval = 0.04  # seconds between auto-repeated steps

        pygame.init()
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("Snake pathfinding debugger")
        self.font = pygame.font.SysFont("menlo,consolas,monospace", 16)
        self.font_b = pygame.font.SysFont("menlo,consolas,monospace", 18, bold=True)
        self.rebuild()

    # --- search lifecycle ---------------------------------------------------
    def rebuild(self):
        """(Re)create the generator for the current view + board state."""
        self.state = None
        if not self.snake:
            self.gen = None
            return
        if self.view == "astar":
            self.gen = astar_steps(self.grid, list(self.snake), self.apple)
            self.flood_label = ""
            self.flood_snake = None
        elif self.view == "complete":
            self.gen = astar_complete_steps(self.grid, list(self.snake), self.apple)
            self.flood_label = ""
            self.flood_snake = None
        else:  # accurate A* search to the tail (on the post-eat snake)
            raw = BFS_path(self.grid, list(self.snake), self.apple) if self.apple else []
            if raw:
                head_first = [self.snake[0]] + raw[::-1]
                self.flood_snake = snake_after_eating(list(self.snake), head_first)
                self.flood_label = "post-eat (shortest path to apple)"
            else:
                self.flood_snake = list(self.snake)
                self.flood_label = "current snake"
            self.gen = astar_tail_steps(self.grid, list(self.flood_snake))
        self.step()                  # show the first state immediately

    def step(self):
        if self.gen is None:
            return
        try:
            self.state = next(self.gen)
        except StopIteration:
            pass

    def finish(self):
        if self.gen is None:
            return
        for st in self.gen:
            self.state = st
            if st.get("phase") == "done":
                break

    # --- loading a captured game state --------------------------------------
    def load_state(self, gx, gy, snake, apple):
        """Adopt a board state (resizing the grid/window if needed)."""
        if (gx, gy) != (self.gx, self.gy):
            self.gx, self.gy = gx, gy
            self.grid = create_adjacent_grid(gx, gy)
            self.presets = presets(gx, gy)
            self.W = gx * self.bs
            self.H = gy * self.bs + self.hud_h
            self.screen = pygame.display.set_mode((self.W, self.H))
        self.snake = [tuple(c) for c in snake]
        self.apple = tuple(apple) if apple else None
        self.rebuild()

    def load_state_file(self, path=STATE_FILE):
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            print(f"[load] could not read {path}: {e}")
            return
        self.load_state(data["gx"], data["gy"], data["snake"], data.get("apple"))
        print(f"[load] {path}: {self.gx}x{self.gy}, snake len {len(self.snake)}, apple {self.apple}")

    # --- editing ------------------------------------------------------------
    def cell_at(self, mx, my):
        if my >= self.gy * self.bs:
            return None
        return (mx // self.bs, my // self.bs)

    def add_snake(self, cell):
        if cell is None:
            return
        if not self.snake:
            self.snake = [cell]
        elif cell not in self.snake and cell in self.grid[self.snake[0]]:
            self.snake.insert(0, cell)               # new head, adjacent to old head
        if self.apple == cell:
            self.apple = None
        self.rebuild()

    def set_apple(self, cell):
        if cell is not None and cell not in self.snake:
            self.apple = cell
            self.rebuild()

    # --- drawing ------------------------------------------------------------
    def rect(self, cell, pad=0):
        x, y = cell
        return pygame.Rect(x * self.bs + pad, y * self.bs + pad,
                           self.bs - 2 * pad, self.bs - 2 * pad)

    def center(self, cell):
        x, y = cell
        return (x * self.bs + self.bs // 2, y * self.bs + self.bs // 2)

    def fill_cells(self, cells, color):
        for c in cells:
            pygame.draw.rect(self.screen, color, self.rect(c))

    def draw_path(self, head_first_path, color, width, dots=True):
        if not head_first_path or len(head_first_path) < 2:
            return
        pts = [self.center(c) for c in head_first_path]
        pygame.draw.lines(self.screen, color, False, pts, width)
        if dots:
            for p in pts:
                pygame.draw.circle(self.screen, color, p, max(3, width))

    @staticmethod
    def _lerp(a, b, f):
        return tuple(int(a[i] + (b[i] - a[i]) * f) for i in range(3))

    def draw_snake(self, snake, tail_color):
        """Draw a snake head-first with a head->tail gradient, a connecting
        spine, a ringed head, and a clearly marked tail."""
        if not snake:
            return
        n = len(snake)
        radius = max(4, self.bs // 4)
        # spine threaded through segment centers shows it is one continuous body
        if n >= 2:
            pygame.draw.lines(self.screen, C_SPINE, False,
                              [self.center(c) for c in snake], max(2, self.bs // 5))
        for i, c in enumerate(snake):
            if i == 0:
                col = C_HEAD
            else:
                f = i / max(1, n - 1)                 # 0 at head-end .. 1 at tail
                col = self._lerp(C_SNAKE_HI, C_SNAKE_LO, f)
            pygame.draw.rect(self.screen, col, self.rect(c, 2), border_radius=radius)
        # head ring
        pygame.draw.rect(self.screen, C_HEAD_RING, self.rect(snake[0], 2), 2, border_radius=radius)
        # tail: filled inner square in the given color (green/red in flood view)
        if n >= 2:
            pygame.draw.rect(self.screen, tail_color, self.rect(snake[-1], self.bs // 4),
                             border_radius=max(2, self.bs // 8))

    def draw(self):
        self.screen.fill(C_BG)
        # base grid
        for i in range(self.gx):
            for j in range(self.gy):
                pygame.draw.rect(self.screen, C_CELL, self.rect((i, j), 1))

        st = self.state or {}

        # The tail search runs on the post-eat snake, so draw THAT body there.
        if self.view == "tail" and self.flood_snake:
            shown_snake = self.flood_snake
        else:
            shown_snake = self.snake
        shown_set = set(shown_snake)

        # Both views are A* searches: shade visited cells, outline the frontier,
        # trace the path being examined and the final chosen path.
        self.fill_cells((st.get("visited", set()) - shown_set), C_VISITED)
        for c in (st.get("frontier", set()) - shown_set):
            pygame.draw.rect(self.screen, C_FRONTIER, self.rect(c, 2), 2)
        if st.get("phase") == "pop":
            self.draw_path(st.get("path", []), C_POPPATH, 3, dots=False)
        if st.get("phase") == "done" and st.get("result"):
            res_hf = [shown_snake[0]] + list(st["result"])[::-1]
            self.draw_path(res_hf, C_RESULT, 5)
        # in the tail view, mark the target (the tail cell being sought)
        if self.view == "tail" and st.get("target"):
            pygame.draw.rect(self.screen, C_CURRENT, self.rect(st["target"], 2), 3)

        # grid lines
        for i in range(self.gx + 1):
            pygame.draw.line(self.screen, C_GRID, (i * self.bs, 0), (i * self.bs, self.gy * self.bs))
        for j in range(self.gy + 1):
            pygame.draw.line(self.screen, C_GRID, (0, j * self.bs), (self.gx * self.bs, j * self.bs))

        # apple (skip when it is the shown snake's head, e.g. the post-eat body)
        if self.apple and (not shown_snake or self.apple != shown_snake[0]):
            pygame.draw.rect(self.screen, C_APPLE, self.rect(self.apple, 3))

        # snake body: green/red tail marker in the tail view once the search ends
        tail_col = C_DIM
        if self.view == "tail" and st.get("phase") == "done":
            tail_col = C_TAIL_OK if st.get("found") else C_TAIL_NO
        self.draw_snake(shown_snake, tail_col)

        self.draw_hud(st)
        pygame.display.flip()

    def draw_hud(self, st):
        y0 = self.gy * self.bs
        pygame.draw.rect(self.screen, C_HUD_BG, pygame.Rect(0, y0, self.W, self.hud_h))
        pygame.draw.line(self.screen, C_GRID, (0, y0), (self.W, y0))

        mode, dec_path = compute_decision(self.grid, list(self.snake), self.apple) if self.snake else ("-", [])
        done = st.get("phase") == "done"

        free_cells = len(self.grid) - len(self.snake) if self.snake else 0
        lines = []
        if self.view == "astar":
            lines.append((f"VIEW: A* -> apple (fast, cell,time)   snake len {len(self.snake)}   apple {self.apple}", C_TEXT))
            lines.append((f"visited {len(st.get('visited', ()))}   frontier {len(st.get('frontier', ()))}"
                          + ("   [running]" if not done else ""), C_DIM))
            if done:
                if st.get("safe"):
                    lines.append((f"RESULT: SAFE path, length {len(st.get('result', []))}", C_RESULT))
                else:
                    lines.append(("RESULT: no SAFE path found (fast search) - try TAB to the complete view", C_TAIL_NO))
        elif self.view == "complete":
            gated = free_cells <= COMPLETE_MAX_FREE
            lines.append((f"VIEW: A* complete (full-body config)   free {free_cells}"
                          + (f"  <= {COMPLETE_MAX_FREE} (would run in game)" if gated
                             else f"  > {COMPLETE_MAX_FREE} (game would SKIP this - too costly)"), C_TEXT))
            lines.append((f"states {st.get('states', 0)}   frontier {len(st.get('frontier', ()))}"
                          + ("   [running]" if not done else ""), C_DIM))
            if done:
                if st.get("safe"):
                    lines.append((f"RESULT: SAFE path, length {len(st.get('result', []))} (found what the fast search missed)", C_RESULT))
                elif st.get("capped"):
                    lines.append((f"RESULT: hit state cap ({st.get('states',0)}) - gave up", C_TAIL_NO))
                else:
                    lines.append(("RESULT: no safe path exists (searched every body-config)", C_TAIL_NO))
        else:
            lines.append((f"VIEW: A* -> tail (accurate) on {self.flood_label}", C_TEXT))
            lines.append((f"visited {len(st.get('visited', ()))}   frontier {len(st.get('frontier', ()))}"
                          + ("   [running]" if not done else ""), C_DIM))
            if done:
                if st.get("found"):
                    lines.append((f"RESULT: tail REACHABLE, path length {len(st.get('result', []))} - safe", C_TAIL_OK))
                else:
                    lines.append(("RESULT: tail NOT reachable - unsafe (head would trap itself)", C_TAIL_NO))

        col_mode = C_RESULT if mode.startswith("SAFE") else (C_CURRENT if "TAIL" in mode else C_TAIL_NO)
        lines.append((f"Decision (Main should): {mode}", col_mode))
        lines.append(("SPACE step  F finish  R restart  TAB view  P preset  L load-game  "
                      "click=snake  Rclick=apple  Bksp undo  C clear  ESC quit", C_DIM))

        y = y0 + 8
        for text, col in lines:
            surf = (self.font_b if col != C_DIM else self.font).render(text, True, col)
            self.screen.blit(surf, (10, y))
            y += 24

    # --- main loop ----------------------------------------------------------
    def run(self):
        clock = pygame.time.Clock()
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit(); return
                if e.type == pygame.KEYDOWN:
                    if e.key in (pygame.K_ESCAPE,):
                        pygame.quit(); return
                    elif e.key == pygame.K_f:
                        self.finish()
                    elif e.key == pygame.K_r:
                        self.rebuild()
                    elif e.key == pygame.K_TAB:
                        views = ["astar", "complete", "tail"]
                        self.view = views[(views.index(self.view) + 1) % len(views)]
                        self.rebuild()
                    elif e.key == pygame.K_p:
                        self.preset_i = (self.preset_i + 1) % len(self.presets)
                        s, a = self.presets[self.preset_i]
                        self.snake, self.apple = list(s), a
                        self.rebuild()
                    elif e.key == pygame.K_c:
                        self.snake = []
                        self.rebuild()
                    elif e.key == pygame.K_l:
                        self.load_state_file()
                    elif e.key == pygame.K_BACKSPACE:
                        if self.snake:
                            self.snake.pop(0)
                            self.rebuild()
                if e.type == pygame.MOUSEBUTTONDOWN:
                    cell = self.cell_at(*e.pos)
                    if e.button == 1:
                        self.add_snake(cell)
                    elif e.button == 3:
                        self.set_apple(cell)

            # Spacebar: single step on press, then auto-repeat while held.
            dt = clock.tick(60) / 1000.0
            space_down = pygame.key.get_pressed()[pygame.K_SPACE]
            if space_down and not self.space_was_down:
                self.step()                                  # initial press
                self.hold_timer = 0.0
                self.repeating = False
            elif space_down:
                self.hold_timer += dt
                threshold = self.repeat_interval if self.repeating else self.repeat_delay
                if self.hold_timer >= threshold:
                    self.step()
                    self.hold_timer = 0.0
                    self.repeating = True
            self.space_was_down = space_down

            self.draw()


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--load":                 # python3 path_debug.py --load [file]
        path = args[1] if len(args) > 1 else STATE_FILE
        with open(path) as f:
            data = json.load(f)
        dbg = Debugger(data["gx"], data["gy"])        # size the window to the saved board
        dbg.load_state_file(path)
        dbg.run()
    elif len(args) >= 2:                              # python3 path_debug.py W H
        Debugger(int(args[0]), int(args[1])).run()
    else:
        Debugger(12, 12).run()
