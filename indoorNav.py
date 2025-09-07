import json
import heapq

# Load data
with open('hallway_graph.json', 'r') as f:
    data = json.load(f)

# Build adjacency list separately
adjacency = {}
for edge in data['edges']:
    adjacency.setdefault(edge['from'], []).append((
        edge['to'],
        edge['weight'],
        edge.get('instruction', [])
    ))

def dijkstra(graph_adj, start, end):
    queue = [(0, start, [])]
    visited = set()
    while queue:
        cost, node, path = heapq.heappop(queue)
        if node == end:
            return path + [node]
        if node in visited:
            continue
        visited.add(node)
        for neighbor, weight, _ in graph_adj.get(node, []):
            if neighbor not in visited:
                heapq.heappush(queue, (cost + weight, neighbor, path + [node]))
    return None

def find_edge_instruction(data, from_node, to_node):
    for edge in data.get('edges', []):
        if edge.get('from') == from_node and edge.get('to') == to_node:
            return edge.get('instruction', None)
    return None