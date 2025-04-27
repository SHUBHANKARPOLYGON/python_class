import networkx as nx
import matplotlib.pyplot as plt

# Basic Graph Coloring Function
def graph_coloring(graph):
    colors = {}
    available_colors = ['Red', 'Green', 'Blue', 'Yellow', 'Orange', 'Purple']

    for node in graph:
        used_colors = set(colors.get(neighbor) for neighbor in graph[node] if neighbor in colors)
        
        for color in available_colors:
            if color not in used_colors:
                colors[node] = color
                break

    return colors

# Example Graph (Adjacency List)
graph = {
    'A': ['B', 'C', 'D'],
    'B': ['A', 'D'],
    'C': ['A', 'D'],
    'D': ['A', 'B', 'C']
}

# Coloring the Graph
coloring = graph_coloring(graph)

# Print Coloring Output
print("\nGraph Coloring Output:")
for node in coloring:
    print(f"Node {node} ---> Color {coloring[node]}")

# Create a NetworkX Graph
G = nx.Graph()

# Add edges
for node in graph:
    for neighbor in graph[node]:
        G.add_edge(node, neighbor)

# Draw the Graph
pos = nx.spring_layout(G, seed=42)  # For consistent layout
color_map = [coloring[node].lower() for node in G.nodes()]  # Matplotlib needs lowercase colors

plt.figure(figsize=(8,6))
nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=800, edgecolors='black')
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos, font_size=14, font_color='white', font_weight='bold')

plt.title("Colored Graph with Nodes and Edges", fontsize=16)
plt.axis('off')
plt.show()
