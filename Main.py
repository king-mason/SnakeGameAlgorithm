import pygame
from collections import deque
import random
from Algorithm import create_adjacent_grid, A_star_path, A_star_path_complete, \
    follow_tail, A_star_path_tail, can_reach_tail, snake_after_eating, DFS_long_path, BFS_path, BFS_path_complete
import sys
import json

STATE_FILE = "debug_state.json"

BLACK = (0, 0, 0)
WHITE = (200, 200, 200)
GREEN = "green"
RED = "red"
BLUE = "blue"

class SnakeGame:
    def __init__(self, x, y, block_size, border_size=None, delay=5,
                 head_color=(60, 120, 230), tail_color=(46, 160, 66)):
        self.x = x
        self.y = y
        self.block_size = block_size
        self.border_size = border_size if border_size else block_size / 30
        self.window_height = self.y * self.block_size
        self.window_width = self.x * self.block_size

        # Snake colouring: body is a gradient from head_color (at the head) to
        # tail_color (at the tail), so it eases toward the tail colour as it grows.
        self.head_color = head_color
        self.tail_color = tail_color

        self.snake = deque([(2, 1), (1, 1)])
        self.start_len = len(self.snake)   # for the score (apples eaten)
        self.grid = create_adjacent_grid(self.x, self.y)
        self.all_points = list(self.grid.keys())
        self.apple = self.choose_apple(self.all_points, self.snake)
        self.original_delay = delay
        self.delay = delay
        self.paused = False

        # Survival-mode state (see decide_path): a committed path we follow rather
        # than recomputing the expensive tail/DFS search every tick, plus stall
        # detection so a tail-chasing cycle is broken by a space-filling reorg.
        self.survival_path = []          # committed survival steps (Main pops from it)
        self.survival_ticks = 0          # ticks in the current survival stint
        self.survival_seen = set()       # snake configs seen this stint (cycle detection)
        self.survival_stalled = False
        self.stall_limit = len(self.grid)  # safety-net cap; tunable
        # Only run the expensive full-body completeness search once free space is
        # this small (its cost explodes with free-cell count). Tunable.
        self.complete_max_free = 25

        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        self.clock = pygame.time.Clock()
        self.screen.fill(BLACK)
        self.font = pygame.font.Font(None, self.window_width // 4)
        self.small_font = pygame.font.Font(None, max(20, self.block_size))
        
    @staticmethod
    def choose_apple(all_points, snake_deque):
        """Return a random free point or None if no free cells remain."""
        free = list(set(all_points) - set(snake_deque))
        if not free:
            return None
        return random.choice(free)

    def dump_state(self):
        """Save the current board state to STATE_FILE (and clipboard, if possible)
        so it can be loaded into path_debug.py with `L` or `--load`."""
        state = {
            "gx": self.x,
            "gy": self.y,
            "snake": [list(c) for c in self.snake],   # head-first
            "apple": list(self.apple) if self.apple else None,
        }
        text = json.dumps(state)
        with open(STATE_FILE, "w") as f:
            f.write(text)
        print(f"[state] saved to {STATE_FILE}  (len {len(self.snake)}, apple {self.apple})")
        try:                                          # best-effort clipboard copy (macOS)
            import subprocess
            subprocess.run(["pbcopy"], input=text.encode(), check=True)
            print("[state] copied to clipboard")
        except Exception:
            pass

    def handle_events(self):
        """Process window/key events. SPACE pauses, C copies the state."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    print("[game] paused" if self.paused else "[game] resumed")
                elif event.key == pygame.K_c:
                    self.dump_state()
                if event.key == pygame.K_TAB and self.original_delay > 5:
                    self.delay = 5
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_TAB:
                    self.delay = self.original_delay

    def decide_path(self):
        """Next path to follow.

        Every tick we cheaply re-check for a safe path to the apple. When there
        isn't one we go into survival mode, but instead of recomputing the
        expensive tail/DFS search every tick we COMMIT to a survival path and just
        follow it, only recomputing when it runs out. If survival keeps cycling
        without opening an apple route (a repeated configuration, or too many
        ticks), we switch to a space-filling DFS reorganization to break the
        stall. Every path returned is collision-free, so the snake never dies by
        choice.
        """
        if self.apple is None:
            return []

        # (1) Cheap recheck: has a safe path to the apple opened up?
        path = A_star_path(self.grid, self.snake, self.apple)
        if not path:
            # The fast (cell,time) search can prune a body-config that still
            # reaches a safe apple (e.g. a tail-chase detour). Retry with the
            # complete full-body search -- but ONLY once free space is small
            # (self.complete_max_free). Its cost stays cheap there because the
            # search runs out of room and terminates fast; above ~30 free cells
            # it explodes (a state cap does NOT bound it -- each state is O(len)).
            # When lots of cells are open we skip it and tail-follow instead; a
            # safe path in open space, if one exists, is found by the fast search.
            free_cells = len(self.grid) - len(self.snake)
            if free_cells <= self.complete_max_free:
                print("running BFS complete")
                path = BFS_path_complete(self.grid, self.snake, self.apple)
            else:
                print("running BFS")
                path = BFS_path(self.grid, self.snake, self.apple)
        if path:
            print("path found")
            self._reset_survival()
            return path

        # (2) Survival — no safe apple path.
        self.survival_ticks += 1
        config = tuple(self.snake)
        if config in self.survival_seen or self.survival_ticks > self.stall_limit:
            self.survival_stalled = True     # cycling / stuck: reorganize next recompute
        self.survival_seen.add(config)

        # Follow the committed survival path if it still has steps.
        if self.survival_path:
            return self.survival_path

        # Committed path exhausted: recompute survival navigation.
        if self.survival_stalled:
            print("stalled, running DFS longest path")
            self.survival_path = DFS_long_path(self.grid, self.snake, self.apple)
            self.survival_stalled = False    # re-arm; the reorg gets a chance to open a route
        else:
            print("finding survival path, running A star path to tail")
            self.survival_path = A_star_path_tail(self.grid, self.snake)
        # Last resort: no full survival path, but if ANY legal move exists, take a
        # single safe step rather than giving up. Only [] when truly boxed in.
        if not self.survival_path:
            print("Last resort...")
            self.survival_path = follow_tail(self.grid, self.snake, self.apple)
        return self.survival_path            # [] only when no legal move remains

    def _reset_survival(self):
        self.survival_path = []
        self.survival_ticks = 0
        self.survival_seen.clear()
        self.survival_stalled = False

    def reset_game(self):
        """Reset to a fresh game (used by 'play again')."""
        self.snake = deque([(2, 1), (1, 1)])
        self.start_len = len(self.snake)
        self.apple = self.choose_apple(self.all_points, self.snake)
        self.paused = False
        self._reset_survival()

    def score(self):
        return len(self.snake) - self.start_len

    def draw_score(self):
        text = self.small_font.render(f"Score: {self.score()}", True, WHITE)
        self.screen.blit(text, (6, 4))

    def draw_pause_symbol(self):
        # two vertical bars in the top-right corner
        bw = max(4, self.block_size // 3)
        gap = bw
        h = max(12, self.block_size)
        x = self.window_width - 2 * bw - gap - 8
        y = 6
        for i in range(2):
            pygame.draw.rect(self.screen, WHITE,
                             pygame.Rect(x + i * (bw + gap), y, bw, h))

    def main(self):
        while True:
            won = self.play_one_game()
            if not self.wait_for_restart(won):   # blocks until R (restart) or quit
                break
            self.reset_game()

    def play_one_game(self):
        """Run one game to completion. Returns True if the board was filled (win)."""
        current_path = self.decide_path()

        while True:
            self.drawGrid()
            self.drawSnake(self.snake)
            self.draw_score()
            if not current_path:
                return len(self.snake) == len(self.grid)
            self.handle_events()

            if not self.paused:
                next_space = current_path.pop()

                if next_space == self.apple:
                    self.snake.appendleft(self.apple) # type: ignore
                    self.apple = self.choose_apple(self.all_points, self.snake)
                    # Eating grows the snake, so any committed survival path is now
                    # stale — start the next decision fresh.
                    self._reset_survival()
                    current_path = self.decide_path()
                else:
                    self.snake.appendleft(next_space)
                    self.snake.pop()
                    # Following the apple path? keep going. Otherwise (survival) we
                    # re-decide each tick: cheaply recheck for a safe apple path and
                    # follow the committed survival path.
                    if not current_path or current_path[0] != self.apple:
                        current_path = self.decide_path()
            else:
                self.draw_pause_symbol()

            pygame.display.update()
            pygame.time.wait(self.delay)

    def wait_for_restart(self, won):
        """Show the game-over screen; return True to play again, False to quit."""
        self.drawGrid()
        self.drawSnake(self.snake)
        if won:
            win_text = self.font.render("Game Won!", True, "gold")
            rect = win_text.get_rect(center=(self.window_width // 2, self.window_height // 4))
            self.screen.blit(win_text, rect)
        score_text = self.font.render(str(self.score()), True, "blue")
        rect = score_text.get_rect(center=(self.window_width // 2, self.window_height // 2))
        self.screen.blit(score_text, rect)
        again = self.small_font.render("Press R to play again", True, WHITE)
        rect = again.get_rect(center=(self.window_width // 2, self.window_height * 3 // 4))
        self.screen.blit(again, rect)
        pygame.display.update()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    return True
            pygame.time.wait(20)

    def drawGrid(self):

        for x in range(0, self.window_width, self.block_size):
            for y in range(0, self.window_height, self.block_size):

                rect = pygame.Rect(x, y, self.block_size, self.block_size)
                pygame.draw.rect(self.screen, BLACK, rect) # change to WHITE for grid

        # draw apple
        if self.apple:
            body_rect = pygame.Rect(self.apple[0]*self.block_size, self.apple[1]*self.block_size, self.block_size, self.block_size)
            pygame.draw.rect(self.screen, RED, body_rect)

    @staticmethod
    def lerp_color(a, b, f):
        """Blend colour a -> b by fraction f in [0, 1]."""
        return tuple(round(a[i] + (b[i] - a[i]) * f) for i in range(3))

    def drawSnake(self, snake):
        # draw the snake as a gradient: head_color at the head, easing toward
        # tail_color at the tail. The longer the snake, the further the far end
        # reaches toward the tail colour.
        n = len(snake)
        for i, (body_x, body_y) in enumerate(snake):
            f = i / (n - 1) if n > 1 else 0.0
            color = self.lerp_color(self.head_color, self.tail_color, f)
            body_rect = pygame.Rect(body_x*self.block_size, body_y*self.block_size, self.block_size, self.block_size)
            pygame.draw.rect(self.screen, color, body_rect)

        # draw snake borders
        for i in range(len(snake)):
            body_x, body_y = snake[i]
            prev_body = None
            next_body = None
            if i != 0:
                prev_body = snake[i-1]
            if i != len(snake) - 1:
                next_body = snake[i+1]

            if (body_x, body_y - 1) != prev_body and (body_x, body_y - 1) != next_body:
                self.drawBorder(body_x, body_y, 'top')
            if (body_x - 1, body_y) != prev_body and (body_x - 1, body_y) != next_body:
                self.drawBorder(body_x, body_y, 'left')
            if (body_x, body_y + 1) != prev_body and (body_x, body_y + 1) != next_body:
                self.drawBorder(body_x, body_y, 'bottom')
            if (body_x + 1, body_y) != prev_body and (body_x + 1, body_y) != next_body:
                self.drawBorder(body_x, body_y, 'right')

    def drawBorder(self, x, y, location='top', color='black'):
        a = b = length = height = 0
        if location == 'top':
            length = self.block_size
            height = self.border_size
            a = 0
            b = 0
        elif location == 'bottom':
            length = self.block_size
            height = self.border_size
            a = 0
            b = self.block_size - height
        elif location == 'left':
            length = self.border_size
            height = self.block_size
            a = 0
            b = 0
        elif location == 'right':
            length = self.border_size
            height = self.block_size
            a = self.block_size - length
            b = 0

        border_rect = pygame.Rect(x*self.block_size+a, y*self.block_size+b, length, height)
        pygame.draw.rect(self.screen, color, border_rect)



if __name__ == "__main__":
    random.seed(42)
    game = SnakeGame(10, 10, block_size=30, border_size=3, delay=5,
                     head_color=(90, 180, 255), tail_color=(240, 90, 255))
                    #  head_color=(60, 120, 230), tail_color=(46, 160, 66))
    game.main()
