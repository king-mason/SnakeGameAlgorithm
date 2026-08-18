from collections import deque
from math import sqrt
from random import randint, shuffle


from heapq import heappush, heappop
from itertools import count

def A_star_path(G, snake, apple):
    """Time-aware A* shortest path from the snake's head to the apple.

    Returns `path` ordered so the next move is path.pop() (apple-first, head
    excluded), or [] if the apple is unreachable. Whether eating is actually
    safe is decided separately by can_reach_tail.

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
    if snake_head == apple:
        return []

    max_time = len(G)             # a shortest path to the apple never needs more
                                  # moves than there are cells; also bounds the
                                  # search when the apple is unreachable.

    tie = count()                 # unique counter so the heap never has to
                                  # compare paths once (f, ...) already differ.

    # Compactness tie-breaker: among equal-cost routes, prefer cells that hug
    # walls or the body (fewer free neighbours). Hugging keeps the remaining free
    # space in one connected blob instead of carving it into isolated holes.
    body = set(snake)
    def openness(cell):
        return 4 - sum(1 for nb in G[cell] if nb not in body)

    # Heap entries: (f = g + h, openness, tie, path). path is head-first.
    open_heap = [(manhattan_distance(snake_head, apple), openness(snake_head), randint(1, 10000), next(tie), [snake_head])]
    visited = set()               # (cell, time) states already expanded.

    while open_heap:
        _, _, _, _, path = heappop(open_heap)
        v = path[-1]
        t = len(path) - 1         # moves made so far == g-cost.

        if (v, t) in visited:
            continue
        visited.add((v, t))

        if v == apple:
            if can_reach_tail(G, snake_after_eating(snake, path)):
                return path[:0:-1]              # shortest SAFE path: apple = first move (next move = path.pop())
            # Don't expand past the apple: a valid path eats it (and stops) on
            # arrival. Longer *safe* routes are still found via other branches
            # that reach the apple later without revisiting it.
            continue

        if t >= max_time:
            continue

        for node in G[v]:
            new_path = path + [node]
            new_snake = get_new_snake(snake, new_path)
            # The new head must not overlap the rest of the body at that moment.
            # get_new_snake already drops the vacated tail, so stepping into the
            # cell the tail is leaving is correctly allowed.
            if node in set(new_snake[1:]):
                continue

            g = len(new_path) - 1
            if (node, g) in visited:
                continue
            f = g + manhattan_distance(node, apple)
            heappush(open_heap, (f, openness(node), randint(1, 10000), next(tie), new_path))

    return []                     # no survivable path to the apple found


def A_star_path_complete(G, snake, apple, max_states=3000):
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
    if snake_head == apple:
        return []

    tie = count()
    body0 = set(snake)
    def openness(cell):
        return sum(1 for nb in G[cell] if nb not in body0)

    # Heap entries: (f = g + h, openness, tie, path). path is head-first.
    open_heap = [(manhattan_distance(snake_head, apple), openness(snake_head), next(tie), [snake_head])]
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

        if v == apple:
            if can_reach_tail(G, snake_after_eating(snake, path)):
                return path[:0:-1]
            continue

        if len(path) - 1 >= len(G):
            continue

        for node in G[v]:
            new_snake = get_new_snake(snake, path + [node])
            if node in set(new_snake[1:]):
                continue
            f = len(path) + manhattan_distance(node, apple)   # len(path) == child g
            heappush(open_heap, (f, openness(node), next(tie), path + [node]))

    return []                     # no safe path found (or hit the state cap)




def BFS_basic(G, snake):
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
        for node in G[v]:
            if node not in E:
                E.add(node)
                depth[node] = depth[v] + 1
                if node not in snake_set:
                    Q.append(node)
    return E

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

def _tail_flood_reachable(G, snake):
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
        for node in G[cell]:
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


def can_reach_tail(G, snake):
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
    if not _tail_flood_reachable(G, snake):  # optimistic superset: False => truly False
        return False

    max_time = len(G)                      # a path never needs more moves than cells.
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

        for node in G[v]:
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

def follow_tail(G, snake, apple=None):
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
    for node in G[head]:
        if node in blocked:
            continue
        if node == apple:
            moved = [node] + snake         # eating grows: the tail does not retreat.
        else:
            moved = [node] + snake[:-1]    # normal step: the tail retreats.
        # Prefer moves that keep the tail reachable, then the one closest to it.
        key = (can_reach_tail(G, moved), -manhattan_distance(node, tail))
        if best_key is None or key > best_key:
            best_key = key
            best = node

    return [best] if best is not None else []

def A_star_path_tail(G, snake):
    snake_head = snake[0]
    snake_tail = snake[-1]

    max_time = len(G)             # a shortest path to the apple never needs more
                                  # moves than there are cells; also bounds the
                                  # search when the apple is unreachable.

    tie = count()                 # unique counter so the heap never has to
                                  # compare paths once (f, ...) already differ.

    # Compactness tie-breaker (see A_star_path): while heading for the tail, hug
    # walls/body so survival fills space instead of carving out holes.
    body = set(snake)
    def openness(cell):
        return sum(1 for nb in G[cell] if nb not in body)

    # Heap entries: (f = g + h, openness, tie, path). path is head-first.
    open_heap = [(manhattan_distance(snake_head, snake_tail), openness(snake_head), next(tie), [snake_head])]
    visited = set()               # (cell, time) states already expanded.

    while open_heap:
        _, _, _, path = heappop(open_heap)
        v = path[-1]
        t = len(path) - 1         # moves made so far == g-cost.

        if (v, t) in visited:
            continue
        visited.add((v, t))

        if v == snake_tail:
            return path[:0:-1]

        if t >= max_time:
            continue

        for node in G[v]:
            new_snake = get_new_snake(snake, path + [node])
            # Same collision model as A_star_path / can_reach_tail: the new head
            # must not overlap the body at the moment it arrives. get_new_snake
            # drops the vacated tail, so stepping into the retreating tail cell is
            # correctly allowed -- which is exactly how the head reaches its tail.
            if node in set(new_snake[1:]):
                continue

            g = len(path)
            if (node, g) in visited:
                continue

            f = g + manhattan_distance(node, snake_tail)
            heappush(open_heap, (f, openness(node), next(tie), path + [node]))

    return []                    # no survivable path to the tail found



def BFS_path(G, snake, apple):
    """Time-aware breadth-first shortest path from the snake's head to the apple.

    Returns `path` apple-first / head-excluded (next move is path.pop()),
    matching A_star_path, or [] if the apple is unreachable. Because BFS expands
    in order of moves made, the first path to reach the apple is a shortest one.

    Collisions are checked per path against get_new_snake, so each branch sees
    its own body (including the retreating tail) rather than a single shared
    blocked set. The visited state is keyed on (cell, time): the same cell is a
    distinct state at a different move count, since the body has shifted.
    """
    snake_head = snake[0]

    if snake_head == apple:
        return []

    max_time = len(G)                      # bounds the search if apple is unreachable.

    Q = deque([[snake_head]])
    visited = {(snake_head, 0)}            # (cell, time) states already enqueued.

    while Q:
        path = Q.popleft()
        v = path[-1]  # (row, col)
        t = len(path) - 1                  # moves made so far.

        if t >= max_time:
            continue

        shuffle(G[v])

        for node in G[v]:
            new_path = path + [node]
            new_snake = get_new_snake(snake, new_path)
            # The new head must not overlap the rest of its own body at that
            # moment. get_new_snake already drops the vacated tail cell.
            if node in set(new_snake[1:]):
                continue

            if node == apple:
                if can_reach_tail(G, snake_after_eating(snake, new_path)):
                    return new_path[:0:-1]
                continue

            state = (node, len(new_path) - 1)
            if state in visited:
                continue
            visited.add(state)
            Q.append(new_path)

    return []


def BFS_path_complete(G, snake, apple, max_states=500):
    """
    """
    snake_head = snake[0]

    if snake_head == apple:
        return []

    Q = deque([[snake_head]])
    # visited = {tuple([snake_head])}
    states = 0

    while Q and states < max_states:
        path = Q.popleft()
        v = path[-1]  # (row, col)

        states += 1

        shuffle(G[v])

        for node in G[v]:
            new_path = path + [node]
            # if tuple(new_path) in visited:
            #     continue
            new_snake = get_new_snake(snake, new_path)
            # The new head must not overlap the rest of its own body at that
            # moment. get_new_snake already drops the vacated tail cell.
            if node in set(new_snake[1:]):
                continue

            if node == apple:
                if can_reach_tail(G, snake_after_eating(snake, new_path)):
                    return new_path[:0:-1]
                continue

            # visited.add(tuple(new_path))
            Q.append(new_path)

    return []


# Provides the snake with a longer path until it can find a safe shortest path
# If there is no safe one, return the safest option
# If there is no way to survivie, return the longest path
def DFS_long_path(G, snake, apple):
    snake_head = snake[0]

    Q = [[snake_head]]
    E = set()
    longest_path = [snake_head]

    accessible_nodes = BFS_basic(G, snake)
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

        adj_nodes = G[v]
        adj_nodes.sort(key = lambda x: manhattan_distance(x, target))
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
                    escape_path = BFS_path(G, new_snake, apple)
                    # Accept this detour as soon as, after buying time and eating,
                    # the snake can still reach its own tail.
                    if escape_path and can_reach_tail(G, snake_after_eating(new_snake, escape_path)):
                        return escape_path + new_path[:0:-1]
                longest_path = max(longest_path, new_path, key=len)

    new_snake = get_new_snake(snake, longest_path)
    return BFS_path(G, new_snake, apple) + longest_path[:0:-1]

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

