from collections import deque
from math import sqrt
from random import randint, shuffle
import queue
import threading


from heapq import heappush, heappop
from itertools import count, islice


class SearchContext:
    def __init__(self, apple, G, complete_max_free=25, lost_confirmation_count=3, verbose=True):
        self.apple = apple
        self.G = G
        self.verbose = verbose

        # Cache of searched snake configurations
        self.searched = set()

        # Survival-mode state (see decide_path): a committed path we follow rather
        # than recomputing the expensive tail/DFS search every tick, plus stall
        # detection so a tail-chasing cycle is broken by a space-filling reorg.
        self.survival_path = []          # committed survival steps (Main pops from it)
        self.survival_ticks = 0          # ticks in the current survival stint
        self.survival_seen = set()       # snake configs seen this stint (cycle detection)
        self.survival_stalled = False
        self.stall_limit = len(self.G)  # safety-net cap; tunable
        # Only run the expensive full-body completeness search once free space is
        # this small (its cost explodes with free-cell count). Tunable.
        self.complete_max_free = complete_max_free

        # Background complete-search (see _manage_background_search): runs the
        # full, uninterrupted BFS_path_complete search on a worker thread so the
        # snake never has to hold still for it. Rooted at the snake configuration
        # AFTER the currently-committed survival_path finishes playing out (a
        # fixed, known-in-advance target). A finished result is held and applied
        # exactly once the real snake reaches that configuration (survival_path
        # always plays out in full, so this is a certainty, not a one-shot race)
        # -- or discarded only if a new apple makes the target moot.
        self._bg_thread = None
        self._bg_result_queue = queue.Queue(maxsize=1)
        self._bg_root = None          # tuple: exact snake config the thread is rooted at
        self._bg_apple = None         # apple value captured when the thread was spawned
        self._bg_pending_result = None  # (path, exhausted) once finished, held until root matches

        # "Lost" detection: each time the complete search EXHAUSTS (its queue
        # empties out on its own, rather than being cut off by max_states) with
        # no path found, that's a real proof no safe route existed from that
        # exact root -- not a guess. One such proof isn't conclusive on its own
        # (a different root, reached via different survival/reorg choices,
        # might still work -- see the DFS_long_path reorg), but several in a
        # row, from the different roots that reorg naturally produces, is
        # strong evidence the game is genuinely lost. Once that count is
        # reached, decide_path stops requiring safety and just grabs whatever
        # score is still reachable (see A_star_path's require_safe). A single
        # later found path is treated as contradicting evidence and resets it.
        self.game_lost = False
        self._consecutive_exhausted = 0
        self.lost_confirmation_count = lost_confirmation_count

    def new_apple(self, apple):
        self.apple = apple
        self.searched.clear()
        # Can't force-kill a live thread; just detach so a late result is
        # recognized as stale and ignored (see _manage_background_search). The
        # orphaned thread finishes harmlessly on its own -- it only touches
        # local state and the read-only grid.
        self._bg_root = None
        self._bg_apple = None
        self._bg_pending_result = None
        # A new apple is a new situation -- re-earn "lost" status rather than
        # carrying over a conclusion drawn about the old target.
        self.game_lost = False
        self._consecutive_exhausted = 0
        self._reset_survival()

    def _reset_survival(self):
        self.survival_path = []
        self.survival_ticks = 0
        self.survival_seen.clear()
        self.survival_stalled = False

    def decide_path(self, snake):
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

        # (1) Cheap recheck: has a (safe, unless the game is confirmed lost --
        # see (3)/game_lost) path to the apple opened up? The full complete
        # search (the expensive fallback for what this prunes) no longer runs
        # synchronously here -- see (3) below -- so the snake is never blocked
        # waiting on it.
        path = self.A_star_path(snake, require_safe=not self.game_lost)
        if not path:
            free_cells = len(self.G) - len(snake)
            # Above complete_max_free the complete search would explode anyway
            # (a state cap does NOT bound it -- each state is O(len)), so open
            # space just falls back to plain tail-following via BFS_path.
            if free_cells > self.complete_max_free:
                if self.verbose: print("running BFS")
                path = self.BFS_path(snake)
        if path:
            if self.verbose: print("path found")
            self._reset_survival()
            return path

        # (2) Survival — no safe apple path via the cheap checks.
        self.survival_ticks += 1
        config = tuple(snake)
        if config in self.survival_seen or self.survival_ticks > self.stall_limit:
            self.survival_stalled = True     # cycling / stuck: reorganize next recompute
        self.survival_seen.add(config)

        if not self.survival_path:
            if self.survival_stalled:
                if self.verbose: print("stalled, running DFS longest path")
                self.survival_path = self.DFS_long_path(snake)
                self.survival_stalled = False    # re-arm; the reorg gets a chance to open a route
                self.survival_ticks = 0
                # Fresh window for cycle detection: configs seen before this reorg
                # are stale history, not evidence the NEW path is cycling too --
                # without this, a config revisited from a prior stint re-flags
                # survival_stalled before the new path even has a chance, causing
                # DFS_long_path to rerun back-to-back on an almost-identical path
                # forever (the observed infinite stall/DFS/stall loop).
                self.survival_seen.clear()
            else:
                if self.verbose: print("finding survival path, running A star path to tail")
                self.survival_path = self.A_star_path_tail(snake)
                if self.survival_path and self.verbose: print("survival path found")
            # Last resort: no full survival path, but if ANY legal move exists,
            # take a single safe step rather than giving up. Only [] when truly
            # boxed in.
            if not self.survival_path:
                if self.verbose: print("Last resort...")
                self.survival_path = self.DFS_long_path(snake)

        # (3) The fast (cell,time) search can prune a body-config that still
        # reaches a safe apple (e.g. a tail-chase detour); the complete
        # full-body search is what catches that. It runs on a background
        # thread (see _manage_background_search) so it never blocks movement --
        # the snake keeps following survival_path (just settled above) every
        # tick regardless of whether/how long the search takes.
        if self.survival_path:
            found = self._manage_background_search(snake, len(self.G) - len(snake))
            if found:
                if self.verbose: print("path found (background complete search)")
                self._reset_survival()
                return found

        return self.survival_path            # [] only when no legal move remains

    def _manage_background_search(self, snake, free_cells):
        """Check on / (re)spawn the background complete-search thread.

        Returns a found path once the current snake exactly matches the root a
        finished search was rooted at, and it found one -- otherwise None
        (caller falls back to following survival_path). Also updates the
        "lost" evidence counter (see game_lost) as soon as a result comes in,
        regardless of whether its root matches yet.

        A finished result is HELD (self._bg_pending_result), not discarded, if
        the snake hasn't reached its root yet -- the search very often
        finishes before the currently-committed survival_path (which the root
        is the end of) has fully played out. Since that path always plays out
        in full once committed (nothing in decide_path replaces it early), the
        match isn't a one-shot check that must land on the exact tick the
        thread happens to finish -- it's just a matter of waiting for the
        snake to actually get there. Discarding on the first mismatch (the
        previous version of this method) meant an already-solved answer was
        thrown away almost every time, right before it would have applied.
        """
        if self._bg_thread is not None and not self._bg_thread.is_alive():
            try:
                self._bg_pending_result = self._bg_result_queue.get_nowait()
            except queue.Empty:
                self._bg_pending_result = None
            self._bg_thread = None

        if self._bg_pending_result is not None:
            path, exhausted = self._bg_pending_result
            if tuple(snake) == self._bg_root and self.apple == self._bg_apple:
                self._bg_pending_result = None
                self._bg_root = None
                self._bg_apple = None
                if path:
                    # A real solution, and the snake is exactly at the config
                    # it was computed for -- directly contradicts "probably
                    # lost".
                    self._consecutive_exhausted = 0
                    self.game_lost = False
                    return path
                elif exhausted:
                    # A genuine proof: this exact configuration -- something the
                    # snake actually reached -- has no safe route. Not
                    # conclusive alone (a different reorg root might still
                    # work), but several in a row is strong evidence.
                    self._consecutive_exhausted += 1
                    if self._consecutive_exhausted >= self.lost_confirmation_count:
                        self.game_lost = True
                # else: capped by max_states, inconclusive -- no update either way.

        if self._bg_thread is None and self._bg_pending_result is None and free_cells <= self.complete_max_free:
            # BFS_basic is a cheap flood fill that already models tail-retreat
            # timing (a body cell unblocks once enough moves have passed for it
            # to have vacated -- see its depth check), so "apple not in here" is
            # a sound proof no real path can exist yet -- a safe gate to avoid
            # spawning threads that are provably going to fail.
            if self.apple in self.BFS_basic(snake):
                # get_new_snake expects a head-first path (path[0] == current
                # head, growing forward) -- survival_path is the opposite
                # convention (destination-first, head excluded, next move via
                # .pop()), so it must be un-reversed and given the head back
                # before being passed in.
                head_first_path = [snake[0]] + self.survival_path[::-1]
                root = tuple(get_new_snake(snake, head_first_path))
                self._bg_root = root
                self._bg_apple = self.apple
                self._bg_thread = threading.Thread(
                    target=self._bg_search_entry, args=(root, self.apple), daemon=True)
                self._bg_thread.start()

        return None

    def _bg_search_entry(self, root_snake, apple):
        """Entry point run on the background thread. Only ever touches local
        state, the read-only grid, and other apple-parameterized/pure methods
        -- never self.apple, self.searched, or self.survival_path -- so it's
        safe to run concurrently with the main thread."""
        result = self._complete_search_worker(root_snake, apple)
        try:
            self._bg_result_queue.put_nowait(result)
        except queue.Full:
            pass

    def A_star_path(self, snake, require_safe=True):
        """Time-aware A* shortest path from the snake's head to the apple.

        Returns `path` ordered so the next move is path.pop() (apple-first, head
        excluded), or [] if the apple is unreachable. Whether eating is actually
        safe is decided separately by can_reach_tail -- unless `require_safe` is
        False, in which case the first (shortest) arrival is returned regardless
        of what happens after. Used once the game is confirmed lost (see
        decide_path / self.game_lost): safety no longer matters, so grab
        whatever extra score is still reachable instead of continuing to route
        around a fate that's already sealed.

        The search state is (cell, time), not just cell. Two arrivals at the same
        cell at different times are genuinely different states, because which cells
        are blocked depends on how many moves have elapsed (the body has shifted).
        Collisions are tested against the exact body configuration via get_new_snake,
        so the tail retreating out of a cell is modeled correctly instead of being
        treated as a permanent wall.

        Completeness caveat: (cell, time) does NOT capture the full body, so two paths
        reaching the same (cell, time) with different trailing bodies collapse to one.
        In rare cases this can prune the body-configuration that would have led to a
        safe apple arrival, so a safe path that exists may be missed. A full fix needs
        the whole body in the state (exponential); in practice it is rare and
        self-correcting (next tick the snake has moved and the search reruns), and the
        survival loop in Main.decide_path absorbs a transient miss.
        """
        snake_head = snake[0]
        if snake_head == self.apple:
            return []

        max_time = len(self.G)             # a shortest path to the apple never needs more
                                    # moves than there are cells; also bounds the
                                    # search when the apple is unreachable.

        tie = count()                 # unique counter so the heap never has to
                                    # compare paths once (f, ...) already differ.

        # Compactness tie-breaker: among equal-cost routes, prefer cells that hug
        # walls or the body (fewer free neighbours). Hugging keeps the remaining free
        # space in one connected blob instead of carving it into isolated holes.
        body = set(snake)
        def openness(cell):
            return 4 - sum(1 for nb in self.G[cell] if nb not in body)
        def straight(cell):
            x_diff = bool(snake[1][0] - snake_head[0])
            if x_diff:
                return int(cell[0] - snake_head[0])
            else:  # y diff
                return int(cell[1] - snake_head[1])

        # Heap entries: (f = g + h, openness, tie, path). path is head-first.
        open_heap = [(manhattan_distance(snake_head, self.apple), straight(snake_head), 
                      openness(snake_head), next(tie), [snake_head])]
        visited = set()               # (cell, time) states already expanded.

        while open_heap:
            _, _, _, _, path = heappop(open_heap)
            v = path[-1]
            t = len(path) - 1         # moves made so far == g-cost.

            if (v, t) in visited:
                continue
            visited.add((v, t))

            if v == self.apple:
                if not require_safe or self.can_reach_tail(snake_after_eating(snake, path)):
                    return path[:0:-1]              # shortest (safe, unless require_safe=False) path: apple = first move
                # Don't expand past the apple: a valid path eats it (and stops) on
                # arrival. Longer *safe* routes are still found via other branches
                # that reach the apple later without revisiting it.
                continue

            if t >= max_time:
                continue

            if tuple(snake) in self.searched:
                continue

            for node in self.G[v]:
                new_path = path + [node]
                new_snake = get_new_snake(snake, new_path)
                # Overlap check
                if node in set(new_snake[1:]):
                    continue

                g = len(new_path) - 1
                if (node, g) in visited:
                    continue
                f = g + manhattan_distance(node, self.apple)
                heappush(open_heap, (f, straight(node), openness(node), next(tie), new_path))

        return []                     # no survivable path to the apple found

    
    def A_star_path_tail(self, snake):
        snake_head = snake[0]
        snake_tail = snake[-1]

        max_time = len(self.G)             # a shortest path to the apple never needs more
                                    # moves than there are cells; also bounds the
                                    # search when the apple is unreachable.

        tie = count()                 # unique counter so the heap never has to
                                    # compare paths once (f, ...) already differ.

        # Compactness tie-breaker (see A_star_path): while heading for the tail, hug
        # walls/body so survival fills space instead of carving out holes.
        body = set(snake)
        def openness(cell):
            return sum(1 for nb in self.G[cell] if nb not in body)
        def straight(path):
            """0 if `cell` continues in the same direction as the last move, 1 otherwise"""
            cell = path[-1]
            if len(path) < 3:
                path += list(islice(snake, 0, 3))
                if len(path) < 3:
                    return 1
            prev = path[-2]
            dx = prev[0] - path[-3][0]
            dy = prev[1] - path[-3][1]
            return int((cell[0] - prev[0], cell[1] - prev[1]) != (dx, dy))

        # Heap entries: (f = g + h, openness, tie, path). path is head-first.
        open_heap = [(manhattan_distance(snake_head, snake_tail), straight([snake_head]), openness(snake_head), next(tie), [snake_head])]
        visited = set()               # (cell, time) states already expanded.

        while open_heap:
            _, _, _, _, path = heappop(open_heap)
            v = path[-1]
            t = len(path) - 1         # moves made so far == g-cost.

            if (v, t) in visited:
                continue
            visited.add((v, t))

            if v == snake_tail:
                return path[:0:-1]

            if t >= max_time:
                continue

            for node in self.G[v]:
                new_path = path + [node]
                if self.apple in new_path:
                    new_snake = snake_after_eating(snake, new_path)
                else:
                    new_snake = get_new_snake(snake, new_path)
                
                if node in set(new_snake[1:]):
                    continue

                g = len(path)
                if (node, g) in visited:
                    continue

                f = g + manhattan_distance(node, snake_tail)
                heappush(open_heap, (f, straight(new_path), openness(node), next(tie), path + [node]))

        return []                    # no survivable path to the tail found


    def A_star_path_complete(self, snake, max_states=3000):
        """Completeness fallback for A_star_path.

        A_star_path keys its visited set on (cell, time), which collapses two paths
        that reach the same cell at the same move count but with DIFFERENT bodies --
        so it can prune the body-configuration that would have reached a safe apple
        (e.g. a short tail-chase detour), reporting no safe path when one exists.

        This search keys visited on the FULL snake configuration (head cell + ordered
        body), so no body-config is ever pruned -- it is complete. That is
        exponential in the free-cell count (the full-body state space), so its cost
        is only acceptable when free space is SMALL. The caller must gate it on that
        (see Main.decide_path); `max_states` is just a hard safety bound. It is only
        meant as a fallback when A_star_path returns [].
        """
        snake = list(snake)
        snake_head = snake[0]
        if snake_head == self.apple:
            return []

        tie = count()
        body0 = set(snake)
        def openness(cell):
            return sum(1 for nb in self.G[cell] if nb not in body0)

        # Heap entries: (f = g + h, openness, tie, path). path is head-first.
        open_heap = [(manhattan_distance(snake_head, self.apple), openness(snake_head), next(tie), [snake_head])]
        visited = set()               # (cell, full ordered body) states already expanded
        states = 0

        while open_heap and states < max_states:
            _, _, _, path = heappop(open_heap)
            v = path[-1]
            cur_snake = get_new_snake(snake, path)
            key = (v, tuple(cur_snake))
            if key in visited:
                continue
            visited.add(key)
            states += 1

            if v == self.apple:
                if self.can_reach_tail(snake_after_eating(snake, path)):
                    return path[:0:-1]
                continue

            if len(path) - 1 >= len(self.G):
                continue

            for node in self.G[v]:
                new_snake = get_new_snake(snake, path + [node])
                if node in set(new_snake[1:]):
                    continue
                f = len(path) + manhattan_distance(node, self.apple)   # len(path) == child g
                heappush(open_heap, (f, openness(node), next(tie), path + [node]))

        return []                     # no safe path found (or hit the state cap)


    def BFS_basic(self, snake):
        snake_head = snake[0]

        Q = deque([snake_head])
        depth = {snake_head: 0}
        E = set()
        snake_set = set(snake)

        while Q:
            v = Q.popleft()
            if depth[v]+1 <= len(snake):
                if snake[-(depth[v]+1)] in E:
                    E.remove(snake[-(depth[v]+1)])
            for node in self.G[v]:
                if node not in E:
                    E.add(node)
                    depth[node] = depth[v] + 1
                    if node not in snake_set:
                        Q.append(node)
        return E

    def _tail_flood_reachable(self, snake):
        """Optimistic O(cells) time-expanded flood used only as a fast pre-filter.

        It ignores where the head's own trail goes, so it OVER-estimates reachability:
        a False result means the tail is genuinely unreachable (a cheap, sound
        reject), while a True result must still be confirmed by the accurate search.
        """
        head = snake[0]
        tail = snake[-1]
        L = len(snake)
        vacate_time = {snake[L - 1 - k]: k + 1 for k in range(L)}
        Q = deque([(head, 0)])
        visited = {(head, 0)}
        while Q:
            cell, t = Q.popleft()
            real_nt = t + 1
            nt = real_nt if real_nt < L else L
            for node in self.G[cell]:
                if node in vacate_time and vacate_time[node] > real_nt:
                    continue
                if node == tail:
                    return True
                state = (node, nt)
                if state in visited:
                    continue
                visited.add(state)
                Q.append(state)
        return False


    def can_reach_tail(self, snake):
        """Accurate check: can the head still reach its own tail via real moves?

        This is the survival gate. It runs a time-aware A* from the head to the
        tail's FIXED original cell (which frees up as the tail retreats), tracking
        the snake's body per path via get_new_snake so the head can never overlap its
        own future trail. Collisions are tested exactly as in A_star_path: a step is
        blocked only if it lands on a body segment that has not yet vacated.

        This is exact where a plain flood is optimistic: a flood lets the head "wait"
        for a coil to unzip while ignoring that its own trail fills the space, so it
        wrongly reports tightly coiled traps as safe. Tracking the body per path
        fixes that, while still confirming open, survivable states (including ones
        that need a longer-than-shortest route as the tail retreats).

        A cheap optimistic flood runs first as a sound fast-reject: because it can
        only over-estimate reachability, a False from it settles the answer without
        the expensive search.
        """
        snake = list(snake)
        head = snake[0]
        tail = snake[-1]
        if head == tail:                       # length-1 snake is trivially safe.
            return True
        if not self._tail_flood_reachable(snake):  # optimistic superset: False => truly False
            return False

        max_time = len(self.G)                      # a path never needs more moves than cells.
        tie = count()                          # unique tie-breaker so paths aren't compared.
        open_heap = [(manhattan_distance(head, tail), next(tie), [head])]
        visited = set()                        # (cell, time) states already expanded.

        while open_heap:
            _, _, path = heappop(open_heap)
            v = path[-1]
            t = len(path) - 1                  # moves made so far.
            if (v, t) in visited:
                continue
            visited.add((v, t))

            if v == tail:                      # reached the (now-vacated) tail cell.
                return True
            if t >= max_time:
                continue

            for node in self.G[v]:
                new_snake = get_new_snake(snake, path + [node])
                # Blocked if the new head overlaps any body cell that has not vacated;
                # get_new_snake already drops the retreated tail, so the tail cell is
                # reachable once enough moves have passed.
                if node in set(new_snake[1:]):
                    continue
                g = len(path)                  # moves made after taking this step.
                if (node, g) in visited:
                    continue
                f = g + manhattan_distance(node, tail)
                heappush(open_heap, (f, next(tie), path + [node]))

        return False

    def follow_tail(self, snake, apple=None):
        """Pick a single survival step toward the tail, to be re-evaluated next tick.

        Used when no safe path to the apple exists: instead of committing to a long
        buy-time path, take one step that keeps the snake alive and heads toward its
        own tail, then reassess. Among the head's legal neighbours (any cell except a
        body segment that has not yet vacated), prefer moves that keep the tail
        reachable, then the move closest to the tail (chase it). Returns a one-move
        path (next move = path.pop()) or [] if no legal move remains.
        """
        snake = list(snake)
        head = snake[0]
        tail = snake[-1]
        blocked = set(snake[:-1])              # body stays put except the tail, which moves.

        best = None
        best_key = None
        for node in self.G[head]:
            if node in blocked:
                continue
            if node == apple:
                moved = [node] + snake         # eating grows: the tail does not retreat.
            else:
                moved = [node] + snake[:-1]    # normal step: the tail retreats.
            # Prefer moves that keep the tail reachable, then the one closest to it.
            key = (self.can_reach_tail(moved), -manhattan_distance(node, tail))
            if best_key is None or key > best_key:
                best_key = key
                best = node

        return [best] if best is not None else []



    def BFS_path(self, snake, apple=None):
        """Time-aware breadth-first shortest path from the snake's head to the apple.

        `apple` defaults to self.apple; the background complete-search worker
        passes it explicitly instead (see _complete_search_worker) so this stays
        safe to call from a thread even if self.apple changes concurrently.

        Returns `path` apple-first / head-excluded (next move is path.pop()),
        matching A_star_path, or [] if the apple is unreachable. Because BFS expands
        in order of moves made, the first path to reach the apple is a shortest one.

        Collisions are checked per path against get_new_snake, so each branch sees
        its own body (including the retreating tail) rather than a single shared
        blocked set. The visited state is keyed on (cell, time): the same cell is a
        distinct state at a different move count, since the body has shifted.
        """
        if apple is None:
            apple = self.apple
        snake_head = snake[0]

        if snake_head == apple:
            return []

        max_time = len(self.G)                      # bounds the search if apple is unreachable.

        Q = deque([[snake_head]])
        visited = {(snake_head, 0)}            # (cell, time) states already enqueued.

        while Q:
            path = Q.popleft()
            v = path[-1]  # (row, col)
            t = len(path) - 1                  # moves made so far.

            if t >= max_time:
                continue

            # shuffle(self.G[v])

            for node in self.G[v]:
                new_path = path + [node]
                new_snake = get_new_snake(snake, new_path)
                # The new head must not overlap the rest of its own body at that
                # moment. get_new_snake already drops the vacated tail cell.
                if node in set(new_snake[1:]):
                    continue

                if node == apple:
                    if self.can_reach_tail(snake_after_eating(snake, new_path)):
                        return new_path[:0:-1]
                    continue

                state = (node, len(new_path) - 1)
                if state in visited:
                    continue
                visited.add(state)
                Q.append(new_path)

        return []


    def BFS_path_complete(self, snake, max_states=1000):
        """Complete breadth-first fallback for BFS_path / A_star_path, run
        synchronously to completion. See _complete_search_worker for the
        thread-safe version _manage_background_search actually uses.
        """
        path, _exhausted = self._complete_search_worker(snake, self.apple, max_states=max_states)
        return path

    def _complete_search_worker(self, snake, apple, max_states=1000):
        """Thread-safe complete search: same algorithm as BFS_path_complete,
        but takes `apple` explicitly (never reads self.apple) and uses a local
        `searched` set (never touches self.searched), so it's safe to run
        concurrently with the main thread on a background thread -- see
        _manage_background_search / _bg_search_entry.

        Like A_star_path_complete, this keys its visited set on the FULL snake
        configuration (the ordered body, head first) instead of (cell, time), so no
        body-configuration is ever pruned -- it is complete. Because it expands
        breadth-first, the first safe apple arrival it reaches is a SHORTEST one
        (fewest moves).

        The full-body state space is exponential in the free-cell count, so the
        search is bounded by `max_states` expansions. It is only meant as a fallback
        when the (cell, time) searches return [], and only when free space is small
        enough for it to stay cheap (the caller gates it -- see decide_path).

        Returns `(path, exhausted)`: `path` is apple-first / head-excluded (next
        move is path.pop()), or [] if none was found. `exhausted` is True only
        when the search ran out of states to expand on its own (Q emptied) --
        a genuine proof no safe route exists from `snake` -- as opposed to being
        cut off by `max_states`, which is inconclusive. See game_lost.
        """
        snake_head = snake[0]

        if snake_head == apple:
            return [], False

        Q = deque([[snake_head]])
        searched = set()
        states = 0

        while Q and states < max_states:
            path = Q.popleft()
            v = path[-1]  # (row, col)

            states += 1

            # shuffle(self.G[v])

            for node in self.G[v]:
                new_path = path + [node]
                new_snake = get_new_snake(snake, new_path)
                # The new head must not overlap the rest of its own body at that
                # moment. get_new_snake already drops the vacated tail cell.
                if node in set(new_snake[1:]):
                    continue

                if node == apple:
                    if self.can_reach_tail(snake_after_eating(snake, new_path)):
                        return new_path[:0:-1], False
                    continue

                # Dedup on the FULL configuration (this is what makes it complete but
                # bounds the search): two paths reaching the same head+body are
                # equivalent for the future, so keep only the first (shortest).
                key = tuple(new_snake)
                if key in searched:
                    continue

                if final_path := self.BFS_path(new_snake, apple):
                    return final_path + new_path[:0:-1], False


                searched.add(key)
                Q.append(new_path)

        return [], not Q   # Q empty => exhausted (proof); else cut off by max_states (inconclusive)


    # Provides the snake with a longer path until it can find a safe shortest path
    # If there is no safe one, return the safest option
    # If there is no way to survivie, return the longest path
    def DFS_long_path(self, snake):
        snake_head = snake[0]

        Q = [[snake_head]]
        E = set()
        longest_path = [snake_head]

        accessible_nodes = self.BFS_basic(snake)
        target = snake_head
        dist_from_tail = len(snake) - 1
        for i in range(len(snake)):
            body = snake[len(snake) - (i+1)]
            if body in accessible_nodes:
                target = body
                dist_from_tail = i
                break
        
        while Q:
            path = Q.pop()
            v = path[-1]
            E.add(v)

            # Sort a copy, not self.G[v] in place: DFS_long_path runs on the main
            # thread while a background complete-search thread may concurrently
            # be reading self.G[v] (see _complete_search_worker) -- mutating the
            # shared adjacency list here would race with that.
            adj_nodes = sorted(self.G[v], key = lambda x: manhattan_distance(x, target))
            for node in adj_nodes:
                new_path = path + [node]
                new_snake = get_new_snake(snake, new_path)
                # Add snake to explored
                if node not in (E | set(new_snake[1:])):  # head is current node
                    if len(set(new_snake)) != len(new_snake):
                        # i.e. there are duplicate positions in the new snake
                        print("SNAKE:", snake)
                        print("PATH:", new_path)
                        print("NEW SNAKE:", new_snake)
                        raise RuntimeError("Duplicate positions in snake")
                    Q.append(new_path)
                    # Find out if at any point the snake could reach the target point
                    # as the tail leaves that point
                    if manhattan_distance(node, target) >= dist_from_tail - len(new_path):
                        escape_path = self.BFS_path(new_snake)
                        # Accept this detour as soon as, after buying time and eating,
                        # the snake can still reach its own tail.
                        if escape_path and self.can_reach_tail(snake_after_eating(new_snake, escape_path)):
                            return escape_path + new_path[:0:-1]
                    longest_path = max(longest_path, new_path, key=len)

        new_snake = get_new_snake(snake, longest_path)
        return self.BFS_path(new_snake) + longest_path[:0:-1]

def get_new_snake(snake, path):
    # note: snake is head first, path is opposite
    if len(snake) >= len(path):
        # note: snake head is included in path
        return path[::-1] + [snake[i+1] for i in range(len(snake)-len(path))]
    else:
        return path[:-(len(snake)+1):-1]

def absolute_distance(a, b):
    return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def manhattan_distance(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def create_adjacent_grid(x, y):
    grid = {}
    for i in range(x):
        for j in range(y):
            node = (i, j)
            neighbors = []

            # Add the neighboring nodes to the list (excluding diagonals)
            for dx, dy in [(0, -1), (-1, 0), (1, 0), (0, 1)]:
                new_x, new_y = i + dx, j + dy
                if 0 <= new_x < x and 0 <= new_y < y and (new_x, new_y) != node:
                    neighbors.append((new_x, new_y))

            grid[node] = neighbors

    return grid

def snake_after_eating(snake, path):
    """The snake body right after following `path` and eating the apple at its end.

    `path` is head-first (snake head .. apple). Eating grows the snake by one
    (the head advances but the tail does NOT retreat on the eating move, matching
    Main.py's appendleft-without-pop), so the result is len(snake) + 1 cells,
    head-first, with the head on the apple.
    """
    # path[::-1] is apple .. current-head; snake[1:] is the body behind the head.
    # Together they are the full trajectory newest-first; keep the leading L + 1.
    return (path[::-1] + list(snake)[1:])[:len(snake) + 1]