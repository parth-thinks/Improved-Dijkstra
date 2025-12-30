import numpy as np
import heapq
import matplotlib.pyplot as plt
import time
from scipy.ndimage import distance_transform_edt


# Inflate obstacles 

def inflate_obstacles(grid, inflation_radius=2):
    obstacle_mask = grid >= 253
    dist = distance_transform_edt(~obstacle_mask)

    inflated = np.zeros_like(grid, dtype=float)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if obstacle_mask[i, j]:
                inflated[i, j] = 255
            elif dist[i, j] <= inflation_radius:
                inflated[i, j] = (inflation_radius - dist[i, j] + 1) * 8
            else:
                inflated[i, j] = 0
    return inflated


# Dijkstra 

def dijkstra(grid, start, goal):
    rows, cols = grid.shape
    costmap = inflate_obstacles(grid)

    g_cost = {start: 0.0}
    parent = {}
    direction = {}

    pq = [(0.0, start, None)]

    motions = [
        (-1, 0, 1.0), (1, 0, 1.0),
        (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, 1.414), (-1, 1, 1.414),
        (1, -1, 1.414), (1, 1, 1.414)
    ]

    TURN_PENALTY = 0.3

    while pq:
        cost, current, prev_dir = heapq.heappop(pq)

        if current == goal:
            break

        for dx, dy, step_cost in motions:
            nr, nc = current[0] + dx, current[1] + dy

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            if costmap[nr, nc] >= 255:
                continue

            # prevent diagonal corner cutting
            if abs(dx) == 1 and abs(dy) == 1:
                if grid[current[0], nc] >= 253 or grid[nr, current[1]] >= 253:
                    continue

            turn_cost = 0
            if prev_dir is not None and prev_dir != (dx, dy):
                turn_cost = TURN_PENALTY

            new_cost = (
                g_cost[current]
                + step_cost
                + costmap[nr, nc]
                + turn_cost
            )

            neighbor = (nr, nc)

            if neighbor not in g_cost or new_cost < g_cost[neighbor]:
                g_cost[neighbor] = new_cost
                parent[neighbor] = current
                direction[neighbor] = (dx, dy)
                heapq.heappush(pq, (new_cost, neighbor, (dx, dy)))

    # reconstruct

    path = []
    node = goal
    while node != start:
        path.append(node)
        node = parent.get(node)
        if node is None:
            return None
    path.append(start)
    path.reverse()
    return path


# Bézier smoothing

def quadratic_bezier(p0, p1, p2, t):
    return (1 - t)**2 * p0 + 2*(1 - t)*t*p1 + t**2 * p2


def collision(pt, grid):
    r, c = int(round(pt[0])), int(round(pt[1]))
    if r < 0 or c < 0 or r >= grid.shape[0] or c >= grid.shape[1]:
        return True
    return grid[r, c] >= 253


def smooth_path_bezier(path, grid, samples=40):
    if len(path) < 3:
        return np.array(path)

    smooth = [np.array(path[0], float)]

    for i in range(1, len(path) - 1):
        p0 = np.array(path[i - 1], float)
        p1 = np.array(path[i], float)
        p2 = np.array(path[i + 1], float)

        curve = []
        valid = True

        for t in np.linspace(0, 1, samples):
            pt = quadratic_bezier(p0, p1, p2, t)
            if collision(pt, grid):
                valid = False
                break
            curve.append(pt)

        if valid:
            smooth.extend(curve[1:])
        else:
            smooth.append(p1)

    smooth.append(np.array(path[-1], float))
    return np.array(smooth)


# ENVIRONMENT GENERATOR

def generate_random_environment(size=50, obstacle_count=90, seed=3):
    rng = np.random.default_rng(seed)
    grid = np.zeros((size, size), dtype=np.uint8)

    placed = 0
    while placed < obstacle_count:
        r = rng.integers(2, size - 2)
        c = rng.integers(2, size - 2)

        # avoid start & goal zones
        if (r < 4 and c < 4) or (r > size - 5 and c > size - 5):
            continue

        # avoid tight clusters
        if np.sum(grid[max(0, r-2):r+3, max(0, c-2):c+3]) > 0:
            continue

        grid[r, c] = 255
        placed += 1

    return grid


# Build environment

grid = generate_random_environment(
    size=50,
    obstacle_count=100,
    seed=7
)

start = (2, 2)
goal = (47, 47)


# Run planner

start_time = time.time()
path = dijkstra(grid, start, goal)
exec_time = time.time() - start_time

smooth_path = smooth_path_bezier(path, grid)

print(f"Execution time: {exec_time:.6f}s")
print(f"Dijkstra nodes: {len(path)}")
print(f"Bezier points: {len(smooth_path)}")


# Plot

plt.figure(figsize=(10, 10))
plt.imshow(grid, cmap="gray_r")

# Dijkstra path (RED)
plt.plot(
    [p[1] for p in path],
    [p[0] for p in path],
    color="red",
    linestyle="--",
    linewidth=2,
    marker="o",
    markersize=3,
    label="Dijkstra Path"
)

# Bézier path (CYAN)
plt.plot(
    smooth_path[:, 1],
    smooth_path[:, 0],
    color="cyan",
    linewidth=3,
    label="Bezier Smoothed Path"
)

plt.scatter(start[1], start[0], c="green", s=120, label="Start")
plt.scatter(goal[1], goal[0], c="blue", s=120, label="Goal")

plt.title("Dijkstra - Complex Environment")
plt.legend()
plt.grid(True)
plt.show()
