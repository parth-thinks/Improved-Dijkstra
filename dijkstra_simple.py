import numpy as np
import heapq
import matplotlib.pyplot as plt
import time
from scipy.ndimage import distance_transform_edt


# Obstacle inflation

def inflate_obstacles(grid, inflation_radius=2):
    obstacle_mask = grid >= 253
    dist = distance_transform_edt(~obstacle_mask)

    inflated = np.zeros_like(grid, dtype=float)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if obstacle_mask[i, j]:
                inflated[i, j] = 255
            elif dist[i, j] <= inflation_radius:
                inflated[i, j] = (inflation_radius - dist[i, j] + 1) * 10
            else:
                inflated[i, j] = 0
    return inflated


# Dijkstra 

def dijkstra(grid, start, goal):
    rows, cols = grid.shape
    costmap = inflate_obstacles(grid, inflation_radius=2)

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
        current_cost, current, prev_dir = heapq.heappop(pq)

        if current == goal:
            break

        for dx, dy, motion_cost in motions:
            nr, nc = current[0] + dx, current[1] + dy

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            if costmap[nr, nc] >= 255:
                continue

            # prevent diagonal corner cutting
            if abs(dx) == 1 and abs(dy) == 1:
                if grid[current[0], nc] >= 253 or grid[nr, current[1]] >= 253:
                    continue

            turn_cost = 0.0
            if prev_dir is not None and prev_dir != (dx, dy):
                turn_cost = TURN_PENALTY

            new_cost = (
                g_cost[current]
                + motion_cost
                + costmap[nr, nc]
                + turn_cost
            )

            neighbor = (nr, nc)

            if neighbor not in g_cost or new_cost < g_cost[neighbor]:
                g_cost[neighbor] = new_cost
                parent[neighbor] = current
                direction[neighbor] = (dx, dy)
                heapq.heappush(pq, (new_cost, neighbor, (dx, dy)))

    # Reconstruct path
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
    return (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2


def is_collision(pt, grid):
    r, c = int(round(pt[0])), int(round(pt[1]))
    if r < 0 or c < 0 or r >= grid.shape[0] or c >= grid.shape[1]:
        return True
    return grid[r, c] >= 253


def smooth_path_bezier(path, grid, samples=25):
    if len(path) < 3:
        return np.array(path)

    smooth = [np.array(path[0], dtype=float)]

    for i in range(1, len(path) - 1):
        p0 = np.array(path[i - 1], dtype=float)
        p1 = np.array(path[i], dtype=float)
        p2 = np.array(path[i + 1], dtype=float)

        curve_segment = []
        valid = True

        for t in np.linspace(0, 1, samples):
            pt = quadratic_bezier(p0, p1, p2, t)
            if is_collision(pt, grid):
                valid = False
                break
            curve_segment.append(pt)

        if valid:
            smooth.extend(curve_segment[1:])
        else:
            smooth.append(p1)

    smooth.append(np.array(path[-1], dtype=float))
    return np.array(smooth)


# Grid map

grid = np.array([
    [0,   0,   0,   0,   0,   0,   0,   0],
    [0, 255, 255, 255, 20,  20,  20,  0],
    [0,   0,   0,   0,   0, 255,  0,   0],
    [0,  20, 255, 255,  0, 255,  0,   0],
    [0,  20,  20,   0,   0,   0,   0,   0],
    [0, 255,  0, 255, 255, 255, 255,  0],
    [0,   0,   0,   0,   0,   0,   0,   0],
    [0,   0,   0,   0,   0,   0,   0,   0]
], dtype=np.uint8)

start = (0, 0)
goal = (7, 7)


# Run planner

start_time = time.time()
path = dijkstra(grid, start, goal)
exec_time = time.time() - start_time

smooth_path = smooth_path_bezier(path, grid)

print(f"Execution Time: {exec_time:.6f} s")


# Visualization 

plt.figure(figsize=(6, 6))
plt.imshow(grid, cmap="gray_r")

# RAW Dijkstra path 
px = [p[1] for p in path]
py = [p[0] for p in path]
plt.plot(
    px, py,
    linestyle="--",
    color="red",
    linewidth=2,
    marker="o",
    markersize=4,
    label="Dijkstra Path",
    zorder=2
)

# Bézier smooth path
plt.plot(
    smooth_path[:, 1],
    smooth_path[:, 0],
    color="cyan",
    linewidth=3,
    label="Bezier Smoothed Path",
    zorder=3
)

# Start & goal
plt.scatter(start[1], start[0], c="green", s=120, label="Start", zorder=4)
plt.scatter(goal[1], goal[0], c="blue", s=120, label="Goal", zorder=4)

plt.title("Dijkstra - Simple Environment")
plt.legend()
plt.grid(True)
plt.show()
