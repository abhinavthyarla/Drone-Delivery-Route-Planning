"""
DRONE DELIVERY AI PROJECT
CO1 -> State Representation, PEAS concepts
CO2 -> BFS, DFS, UCS, Greedy, A*
CO3 -> CSP Backtracking + MRV + LCV
CO4 -> Minimax + Alpha-Beta Pruning
CO5 -> Bayesian Reasoning + Markov Weather Prediction
CO6 -> Integrated Pipeline + Explainable Logs

Run:
    python drone_delivery_ai_COwise.py
"""

import heapq
from collections import deque
from dataclasses import dataclass
import time

# ==========================================================
# CO1 : STATE REPRESENTATION
# ==========================================================

@dataclass
class DroneState:
    position: tuple
    battery: int
    delivered: bool = False


class Logger:
    """CO6 Explainable Reasoning Trace"""
    def __init__(self):
        self.logs = []

    def add(self, msg):
        self.logs.append(msg)

    def show(self):
        print("\n========== REASONING TRACE ==========")
        for step in self.logs:
            print(step)


logger = Logger()

# ==========================================================
# USER INPUT ENVIRONMENT
# ==========================================================

def create_environment():
    rows = int(input("Enter rows: "))
    cols = int(input("Enter cols: "))

    grid = [[0 for _ in range(cols)] for _ in range(rows)]

    print("\nStart Position")
    sx = int(input("Row: "))
    sy = int(input("Col: "))

    print("\nGoal Position")
    gx = int(input("Row: "))
    gy = int(input("Col: "))

    obs = int(input("\nNumber of obstacles: "))

    for i in range(obs):
        print(f"\nObstacle {i+1}")
        r = int(input("Row: "))
        c = int(input("Col: "))
        if (r, c) != (sx, sy) and (r, c) != (gx, gy):
            grid[r][c] = 1

    battery = int(input("\nBattery %: "))

    return grid, (sx, sy), (gx, gy), battery


# ==========================================================
# COMMON FUNCTIONS
# ==========================================================

def neighbors(grid, node):
    rows = len(grid)
    cols = len(grid[0])

    x, y = node

    result = []

    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        nx, ny = x + dx, y + dy

        if (
            0 <= nx < rows and
            0 <= ny < cols and
            grid[nx][ny] == 0
        ):
            result.append((nx, ny))

    return result


# ==========================================================
# CO2 : BFS
# ==========================================================

def bfs(grid, start, goal):

    queue = deque([[start]])
    visited = set()

    while queue:

        path = queue.popleft()
        node = path[-1]

        if node == goal:
            return path

        if node not in visited:

            visited.add(node)

            for nbr in neighbors(grid, node):
                queue.append(path + [nbr])


# ==========================================================
# CO2 : DFS
# ==========================================================

def dfs(grid, start, goal):

    stack = [[start]]
    visited = set()

    while stack:

        path = stack.pop()
        node = path[-1]

        if node == goal:
            return path

        if node not in visited:

            visited.add(node)

            for nbr in neighbors(grid, node):
                stack.append(path + [nbr])


# ==========================================================
# CO2 : UCS
# ==========================================================

def ucs(grid, start, goal):

    pq = [(0, start, [start])]
    visited = set()

    while pq:

        cost, node, path = heapq.heappop(pq)

        if node == goal:
            return path, cost

        if node not in visited:

            visited.add(node)

            for nbr in neighbors(grid, node):

                heapq.heappush(
                    pq,
                    (cost + 1, nbr, path + [nbr])
                )


# ==========================================================
# CO2 : GREEDY SEARCH
# ==========================================================

def heuristic(node, goal):
    return abs(node[0]-goal[0]) + abs(node[1]-goal[1])

def greedy(grid, start, goal):

    pq = [(heuristic(start, goal), start, [start])]
    visited = set()

    while pq:

        _, node, path = heapq.heappop(pq)

        if node == goal:
            return path

        if node not in visited:

            visited.add(node)

            for nbr in neighbors(grid, node):

                heapq.heappush(
                    pq,
                    (heuristic(nbr, goal), nbr, path+[nbr])
                )


# ==========================================================
# CO2 : A*
# ==========================================================

def astar(grid, start, goal):

    pq = [(0, start, [start], 0)]
    visited = set()

    while pq:

        _, node, path, g = heapq.heappop(pq)

        if node == goal:
            return path, g

        if node in visited:
            continue

        visited.add(node)

        for nbr in neighbors(grid, node):

            new_g = g + 1

            heapq.heappush(
                pq,
                (
                    new_g + heuristic(nbr, goal),
                    nbr,
                    path+[nbr],
                    new_g
                )
            )


# ==========================================================
# CO3 : CSP
# ==========================================================

domains = {
    "D1":["9AM","10AM"],
    "D2":["10AM","11AM"],
    "D3":["9AM","11AM"]
}

def consistent(assign):
    values = list(assign.values())
    return len(values) == len(set(values))

def mrv(assign):
    unassigned = [v for v in domains if v not in assign]
    return min(unassigned, key=lambda x: len(domains[x]))

def lcv(var):
    return domains[var]

def backtrack(assign):

    if len(assign) == len(domains):
        return assign

    var = mrv(assign)

    for value in lcv(var):

        assign[var] = value

        if consistent(assign):

            result = backtrack(assign)

            if result:
                return result

        del assign[var]

    return None


# ==========================================================
# CO4 : MINIMAX
# ==========================================================

tree = {
    "A":["B","C"],
    "B":[3,5],
    "C":[2,9]
}

def minimax(node, maximizing):

    if isinstance(node, int):
        return node

    if maximizing:

        value = float("-inf")

        for child in tree[node]:
            value = max(
                value,
                minimax(child, False)
            )

        return value

    else:

        value = float("inf")

        for child in tree[node]:
            value = min(
                value,
                minimax(child, True)
            )

        return value


# ==========================================================
# CO4 : ALPHA BETA
# ==========================================================

def alphabeta(node, alpha, beta, maximizing):

    if isinstance(node, int):
        return node

    if maximizing:

        value = float("-inf")

        for child in tree[node]:

            value = max(
                value,
                alphabeta(
                    child,
                    alpha,
                    beta,
                    False
                )
            )

            alpha = max(alpha, value)

            if beta <= alpha:
                break

        return value

    else:

        value = float("inf")

        for child in tree[node]:

            value = min(
                value,
                alphabeta(
                    child,
                    alpha,
                    beta,
                    True
                )
            )

            beta = min(beta, value)

            if beta <= alpha:
                break

        return value


# ==========================================================
# CO5 : BAYESIAN + MARKOV
# ==========================================================

def predict_weather(current):

    transition = {

        "Sunny":{"Sunny":0.7,"Cloudy":0.3},
        "Cloudy":{"Sunny":0.6,"Rainy":0.4},
        "Rainy":{"Rainy":0.5,"Cloudy":0.5}

    }

    return max(
        transition[current],
        key=transition[current].get
    )

def bayes_success(weather):

    probs = {
        "Sunny":0.95,
        "Cloudy":0.80,
        "Rainy":0.50
    }

    return probs[weather]


# ==========================================================
# CO6 : VISUALIZATION
# ==========================================================

def draw_path(grid, path):

    board = [
        ["X" if c == 1 else "." for c in row]
        for row in grid
    ]

    for x, y in path:
        board[x][y] = "*"

    print("\nGRID VISUALIZATION\n")

    for row in board:
        print(" ".join(row))


# ==========================================================
# MAIN
# ==========================================================

def main():

    grid, start, goal, battery = create_environment()

    drone = DroneState(start, battery)

    logger.add("Drone initialized")

    print("\n========== CO2 SEARCH COMPARISON ==========")

    start_time = time.time()
    bfs_path = bfs(grid, start, goal)
    bfs_time = time.time() - start_time

    start_time = time.time()
    dfs_path = dfs(grid, start, goal)
    dfs_time = time.time() - start_time

    ucs_path, ucs_cost = ucs(grid, start, goal)
    greedy_path = greedy(grid, start, goal)
    astar_path, astar_cost = astar(grid, start, goal)

    print("BFS Length :", len(bfs_path))
    print("DFS Length :", len(dfs_path))
    print("UCS Cost   :", ucs_cost)
    print("Greedy Len :", len(greedy_path))
    print("A* Cost    :", astar_cost)

    print(f"BFS Time   : {bfs_time:.6f}")
    print(f"DFS Time   : {dfs_time:.6f}")

    logger.add("Search algorithms executed")

    print("\n========== CO3 CSP ==========")
    print(backtrack({}))
    logger.add("CSP solved using Backtracking + MRV + LCV")

    print("\n========== CO4 GAME SEARCH ==========")
    print("Minimax:", minimax("A", True))
    print("Alpha Beta:", alphabeta("A", float("-inf"), float("inf"), True))
    logger.add("Minimax and Alpha-Beta completed")

    weather = input("\nWeather (Sunny/Cloudy/Rainy): ")
    predicted = predict_weather(weather)
    success = bayes_success(predicted)

    print("\n========== CO5 UNCERTAINTY ==========")
    print("Predicted Weather:", predicted)
    print("Success Probability:", success * 100, "%")

    logger.add(f"Weather predicted as {predicted}")

    print("\n========== CO6 INTEGRATED PIPELINE ==========")

    if battery < len(astar_path) * 2:
        print("Battery Constraint Failed")
        return

    draw_path(grid, astar_path)

    drone.delivered = True

    logger.add("Optimal route selected using A*")
    logger.add("Package delivered successfully")

    logger.show()


if __name__ == "__main__":
    main()
