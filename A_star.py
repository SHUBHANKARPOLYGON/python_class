import heapq

# Define the graph
graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('D', 2), ('E', 5)],
    'C': [('F', 3)],
    'D': [],
    'E': [('F', 1)],
    'F': []
}

# Define the heuristic values (estimated cost to goal)
heuristic = {
    'A': 7,
    'B': 6,
    'C': 2,
    'D': 1,
    'E': 1,
    'F': 0  # Goal node has heuristic 0
}

# A* Algorithm
def a_star_search(start, goal):
    open_list = []
    heapq.heappush(open_list, (0 + heuristic[start], 0, start, [start]))  # (f = g + h, g, node, path)

    while open_list:
        f, g, current, path = heapq.heappop(open_list)
        print(f"Visiting Node: {current} with path cost {g} and estimated total cost {f}")

        if current == goal:
            print(f"\nGoal reached! Path: {path}")
            return path

        for neighbor, cost in graph[current]:
            heapq.heappush(open_list, (g + cost + heuristic[neighbor], g + cost, neighbor, path + [neighbor]))
    
    print("No path found!")
    return None

# Run A* Search
start_node = 'A'
goal_node = 'F'

print(f"Starting A* Search from {start_node} to {goal_node}:")
a_star_search(start_node, goal_node)
