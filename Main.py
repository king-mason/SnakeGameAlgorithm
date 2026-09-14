import pygame
from collections import deque
import random
from Algorithm import create_adjacent_grid, SearchContext
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
                 head_color=(60, 120, 230), tail_color=(46, 160, 66),
                 headless=False, verbose=None, program_seed=None,
                 first_game_seed=None):
        # headless: skip pygame entirely (no window, no drawing, no frame delay) --
        # used by the benchmarker to run games at full speed. verbose: print the
        # decide_path debug lines; defaults to True for normal play, False when
        # headless (a benchmark run would otherwise flood stdout).
        self.headless = headless
        self.verbose = verbose if verbose is not None else not headless

        if program_seed:
            self.program_seed = random.Random(program_seed)
        else: 
            self.program_seed = random.Random(random.random())
        if first_game_seed:
            self.game_seed = first_game_seed
        else:
            self.game_seed = self.program_seed.randrange(2**32)
        self.game_random = random.Random(self.game_seed)

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
        self.ticks = 0                     # moves taken this game (benchmark timing)
        self.avg_point_ticks = 0            # average moves per point (benchmark timing)
        self.point_tick_history = []       # moves taken to reach each successive point (benchmark timing)
        self.grid = create_adjacent_grid(self.x, self.y)
        self.all_points = list(self.grid.keys())
        self.apple = self.choose_apple(self.all_points, self.snake)
        self.original_delay = delay
        self.delay = delay
        self.paused = False
        self.game_over = False

        self.search = SearchContext(self.apple, self.grid, verbose=self.verbose)

        if not self.headless:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            self.clock = pygame.time.Clock()
            self.screen.fill(BLACK)
            self.font = pygame.font.Font(None, self.window_width // 4)
            self.small_font = pygame.font.Font(None, max(20, self.block_size))
        
    def choose_apple(self, all_points, snake_deque):
        """Return a random free point or None if no free cells remain."""
        free = list(set(all_points) - set(snake_deque))
        if not free:
            return None
        return self.game_random.choice(free)

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
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    print("[game] paused" if self.paused else "[game] resumed")
                elif event.key == pygame.K_c:
                    print("game state saved")
                    self.dump_state()
                elif event.key == pygame.K_s:
                    print("game seed:", self.game_seed)
                elif event.key == pygame.K_r and self.game_over:
                    self.game_over = False
                elif event.key == pygame.K_TAB and self.original_delay > 5:
                    self.delay = 5
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_TAB:
                    self.delay = self.original_delay

    def reset_game(self):
        """Reset to a fresh game (used by 'play again')."""
        self.snake = deque([(2, 1), (1, 1)])
        self.start_len = len(self.snake)
        self.ticks = 0
        self.avg_point_ticks = 0
        self.point_tick_history = []
        self.game_seed = self.program_seed.randrange(2**32)
        self.game_random = random.Random(self.game_seed)
        self.apple = self.choose_apple(self.all_points, self.snake)
        self.paused = False
        self.search.new_apple(self.apple)

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
            self.wait_for_restart(won)   # blocks until R (restart) or quit
            print("Playing new game")
            self.reset_game()

    def play_one_game(self):
        """Run one game to completion. Returns True if the board was filled (win)."""
        current_path = self.search.decide_path(self.snake)
        last_point_ticks = 0

        while True:
            if not self.headless:
                self.drawGrid()
                self.drawSnake(self.snake)
                self.draw_score()
            if not current_path:
                return len(self.snake) == len(self.grid)
            if not self.headless:
                self.handle_events()

            if not self.paused:
                self.ticks += 1
                next_space = current_path.pop()

                if next_space == self.apple:
                    self.snake.appendleft(self.apple) # type: ignore
                    interval = self.ticks - last_point_ticks
                    self.point_tick_history.append(interval)
                    self.avg_point_ticks += (interval - self.avg_point_ticks) / self.score()
                    last_point_ticks = self.ticks
                    self.apple = self.choose_apple(self.all_points, self.snake)
                    self.search.new_apple(self.apple)
                    current_path = self.search.decide_path(self.snake)
                else:
                    self.snake.appendleft(next_space)
                    self.snake.pop()
                    # Following the apple path? keep going. Otherwise (survival) we
                    # re-decide each tick: cheaply recheck for a safe apple path and
                    # follow the committed survival path.
                    if not current_path or current_path[0] != self.apple:
                        current_path = self.search.decide_path(self.snake)
            elif not self.headless:
                self.draw_pause_symbol()

            if not self.headless:
                pygame.display.update()
                pygame.time.wait(self.delay)

    def wait_for_restart(self, won):
        """Show the game-over screen; return True to play again, False to quit."""
        self.drawGrid()
        self.drawSnake(self.snake)
        self.game_over = True
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
        while self.game_over:
            self.handle_events()
        return True

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
    game = SnakeGame(20, 15, block_size=30, border_size=3, delay=10,
                     head_color=(90, 180, 255), tail_color=(60, 255, 100),
                     program_seed=64, first_game_seed=2045084184)
                    #  head_color=(60, 120, 230), tail_color=(46, 160, 66))
    game.main()
