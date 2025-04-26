import networkx as nx
import matplotlib.pyplot as plt
from queue import PriorityQueue

# Create a simple graph
G = nx.Graph()
G.add_edge('A', 'B', weight=1)
G.add_edge('A', 'C', weight=3)
G.add_edge('B', 'D', weight=1)
G.add_edge('C', 'D', weight=1)

# Best First Search function
def best_first_search(graph, start, goal):
    visited = set()
    pq = PriorityQueue()
    pq.put((0, start))
    
    while not pq.empty():
        (priority, node) = pq.get()
        print(f"Visiting: {node}")
        if node == goal:
            print("Goal Reached!")
            return
        visited.add(node)
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                weight = graph.edges[node, neighbor]['weight']
                pq.put((weight, neighbor))

# Run Best First Search
best_first_search(G, 'A', 'D')

# Draw the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=500, font_size=10)
nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))
plt.show()
