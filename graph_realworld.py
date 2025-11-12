# graph_realworld.py
# Implements: Friend suggestion (BFS), Bellman-Ford, Dijkstra, MST (Kruskal/Prim)
# Profiling: time + memory (tracemalloc)
# Run demonstrations at bottom under `if __name__ == "__main__":`

import heapq
import collections
import time
import tracemalloc
from typing import Dict, List, Tuple, Any

# ------------------------------
# Utilities for timing & memory
# ------------------------------
def measure(func):
    """Decorator to measure time & memory of a function call. Returns (result, meta)."""
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        meta = {
            "time_s": t1 - t0,
            "mem_current_bytes": current,
            "mem_peak_bytes": peak
        }
        return result, meta
    return wrapper

# ------------------------------
# Problem 1: Friend Suggestion (BFS)
# ------------------------------
def build_undirected_adjlist(edges: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    g = collections.defaultdict(list)
    for u, v in edges:
        g[u].append(v)
        g[v].append(u)
    return dict(g)

@measure
def suggest_friends_bfs(adj: Dict[str, List[str]], user: str) -> List[Tuple[str,int]]:
    """
    Suggest friends-of-friends for `user` using BFS.
    Returns list of (suggested_user, num_mutual_friends) sorted by mutual count desc then name.
    """
    if user not in adj:
        return []

    # Direct friends
    friends = set(adj[user])

    # Count mutual friends among friends-of-friends
    mutual_counts = collections.Counter()
    visited = set([user])  # we don't include user
    q = collections.deque()
    # Start BFS from direct friends with depth = 1
    for f in friends:
        visited.add(f)
        q.append((f, 1))

    # BFS up to depth 2 to find friends-of-friends
    while q:
        node, depth = q.popleft()
        if depth >= 2:
            # neighbors are potential suggestions
            for nb in adj.get(node, []):
                if nb == user or nb in friends:  # skip self or existing friend
                    continue
                mutual_counts[nb] += 1
        else:
            # expand
            for nb in adj.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, depth + 1))

    # Sort suggestions: highest mutual friends first then lexicographically
    suggestions = sorted(mutual_counts.items(), key=lambda x: (-x[1], x[0]))
    return suggestions

# ------------------------------
# Problem 2: Bellman-Ford
# ------------------------------
@measure
def bellman_ford(vertices: List[Any], edges: List[Tuple[Any, Any, float]], src: Any):
    """
    vertices: list of vertex IDs
    edges: list of (u, v, weight) for directed edges
    Returns: (dist dict, predecessor dict, negative_cycle_nodes list(if any))
    """
    INF = float('inf')
    dist = {v: INF for v in vertices}
    pred = {v: None for v in vertices}
    dist[src] = 0

    V = len(vertices)
    # Relax edges V-1 times
    for i in range(V - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] != INF and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                pred[v] = u
                updated = True
        if not updated:
            break

    # Check for negative weight cycles
    neg_cycle_nodes = set()
    for u, v, w in edges:
        if dist[u] != INF and dist[u] + w < dist[v]:
            neg_cycle_nodes.add(v)
            neg_cycle_nodes.add(u)

    return dist, pred, list(neg_cycle_nodes)

# ------------------------------
# Problem 3: Dijkstra (min-heap)
# ------------------------------
@measure
def dijkstra(adj: Dict[Any, List[Tuple[Any, float]]], src: Any):
    """
    adj: adjacency list {u: [(v, weight), ...], ...} with all weights >= 0
    Returns: dist dict, predecessor dict
    """
    INF = float('inf')
    dist = {u: INF for u in adj.keys()}
    pred = {u: None for u in adj.keys()}
    dist[src] = 0
    heap = [(0, src)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in adj.get(u, []):
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                pred[v] = u
                heapq.heappush(heap, (nd, v))
    return dist, pred

# ------------------------------
# Problem 4: MST - Kruskal & Prim
# ------------------------------
class UnionFind:
    def __init__(self, nodes):
        self.parent = {n: n for n in nodes}
        self.rank = {n: 0 for n in nodes}
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        else:
            self.parent[rb] = ra
            if self.rank[ra] == self.rank[rb]:
                self.rank[ra] += 1
        return True

@measure
def kruskal_mst(nodes: List[Any], edges: List[Tuple[Any, Any, float]]):
    """
    edges: list of (u, v, weight) undirected (u,v) and (v,u) should not be duplicated.
    Returns: (mst_edges, total_cost)
    """
    uf = UnionFind(nodes)
    sorted_edges = sorted(edges, key=lambda e: e[2])
    mst_edges = []
    total = 0.0
    for u, v, w in sorted_edges:
        if uf.union(u, v):
            mst_edges.append((u, v, w))
            total += w
    # Check if all nodes connected:
    roots = set(uf.find(n) for n in nodes)
    if len(roots) != 1:
        raise ValueError("Graph is not connected; MST does not exist for all nodes.")
    return mst_edges, total

@measure
def prim_mst(adj: Dict[Any, List[Tuple[Any, float]]], start=None):
    """
    adj: adjacency list for undirected graph {u: [(v, w), ...], ...}
    Returns: (mst_edges, total_cost)
    """
    if not adj:
        return [], 0.0
    if start is None:
        start = next(iter(adj.keys()))
    visited = set([start])
    heap = []
    for v, w in adj[start]:
        heapq.heappush(heap, (w, start, v))
    mst_edges = []
    total = 0.0
    while heap and len(visited) < len(adj):
        w, u, v = heapq.heappop(heap)
        if v in visited:
            continue
        visited.add(v)
        mst_edges.append((u, v, w))
        total += w
        for nb, nw in adj[v]:
            if nb not in visited:
                heapq.heappush(heap, (nw, v, nb))
    if len(visited) != len(adj):
        raise ValueError("Graph not connected; Prim's MST incomplete.")
    return mst_edges, total

# ------------------------------
# Helper: Pretty path reconstruction
# ------------------------------
def reconstruct_path(pred: Dict[Any, Any], target):
    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = pred[cur]
    return list(reversed(path))

# ------------------------------
# Demo / Example usage
# ------------------------------
def demo_social_suggestion():
    edges = [
        ("A", "B"), ("A", "C"),
        ("B", "D"), ("C", "D"),
        ("C", "E"), ("D", "F"),
        ("E", "G"), ("F", "H"),
        ("G", "H"), ("B", "I"),
        ("I", "J")
    ]
    adj = build_undirected_adjlist(edges)
    (suggestions, meta) = suggest_friends_bfs(adj, "A")
    print("=== Social Network Friend Suggestions for A ===")
    print("Graph adjacency:", adj)
    print("Suggestions (user, mutual_count):", suggestions)
    print("Profiling meta:", meta)
    print()

def demo_bellman_ford():
    vertices = ["S", "A", "B", "C", "D"]
    edges = [
        ("S", "A", 4),
        ("S", "B", 5),
        ("A", "C", -3),
        ("B", "A", 6),
        ("C", "D", 4),
        ("D", "B", -10)   # creates negative cycle reachable? check
    ]
    (res, meta) = bellman_ford(vertices, edges, "S")
    dist, pred, neg_nodes = res
    print("=== Bellman-Ford ===")
    print("Distances:", dist)
    print("Predecessors:", pred)
    print("Negative cycle nodes (if any):", neg_nodes)
    print("Profiling meta:", meta)
    print()

def demo_dijkstra():
    adj = {
        "S": [("A", 1), ("B", 4)],
        "A": [("B", 2), ("C", 5)],
        "B": [("C", 1)],
        "C": []
    }
    (res, meta) = dijkstra(adj, "S")
    dist, pred = res
    print("=== Dijkstra ===")
    print("Adjacency:", adj)
    print("Distances from S:", dist)
    print("Predecessors:", pred)
    print("Profiling meta:", meta)
    print()

def demo_mst():
    # Undirected edges (u, v, w) — each edge only once
    nodes = ["A", "B", "C", "D", "E"]
    edges = [
        ("A", "B", 1),
        ("A", "C", 3),
        ("B", "C", 1),
        ("B", "D", 4),
        ("C", "D", 1),
        ("C", "E", 6),
        ("D", "E", 5)
    ]
    # Kruskal
    (kr_res, kr_meta) = kruskal_mst(nodes, edges)
    mst_edges, total = kr_res
    print("=== Kruskal MST ===")
    print("MST edges:", mst_edges)
    print("Total cost:", total)
    print("Profiling meta:", kr_meta)
    print()

    # Build adjacency for Prim
    adj = {n: [] for n in nodes}
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))
    (pr_res, pr_meta) = prim_mst(adj, start="A")
    p_mst_edges, p_total = pr_res
    print("=== Prim MST ===")
    print("MST edges:", p_mst_edges)
    print("Total cost:", p_total)
    print("Profiling meta:", pr_meta)
    print()

if __name__ == "__main__":
    print("Running demos for the 4 problems (with profiling)...\n")
    demo_social_suggestion()
    demo_bellman_ford()
    demo_dijkstra()
    demo_mst()

    print("Demo complete. Save this file as graph_realworld.py or paste into a notebook cell.")
    print("Notes:")
    print("- To generate plots of time vs input size, call measure-decorated functions on multiple input sizes and use matplotlib to plot meta['time_s'].")
    print("- If you prefer memory_profiler, install it and use @profile on functions (requires running with 'mprof' / 'python -m memory_profiler').")
