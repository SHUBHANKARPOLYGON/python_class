# Define the graph as a dictionary
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

# DFS function
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()

    print(start, end=' ')
    visited.add(start)

    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

# Call DFS
print("Depth First Search starting from node A:")
dfs(graph, 'A')
