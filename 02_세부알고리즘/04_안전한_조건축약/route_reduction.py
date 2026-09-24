"""Linear graph preprocessing; no candidate route enumeration or local solving."""
from dataclasses import dataclass
from time import monotonic
from map_data import ResourceLimit


@dataclass(frozen=True)
class RouteReduction:
    fixed: dict
    bridges: frozenset


def reduce_route_domain(grid, *, deadline=None):
    adjacency = grid.adjacency()
    discovery = [-1]*grid.n
    low = [0]*grid.n
    parent_edge = [-1]*grid.n
    component = [-1]*grid.n
    required = set(grid.houses)
    houses = [int(i in required) for i in range(grid.n)]
    fixed, bridges = {}, set()
    clock = 0
    # Iterative DFS avoids recursion limits on long corridors.
    for root in range(grid.n):
        if discovery[root] != -1:
            continue
        discovery[root] = low[root] = clock
        clock += 1
        component[root] = root
        stack = [(root, iter(adjacency[root]))]
        while stack:
            if deadline is not None and monotonic() >= deadline:
                raise ResourceLimit('total_timeout')
            u, children = stack[-1]
            item = next(children, None)
            if item is not None:
                v, _, e = item
                if e == parent_edge[u]:
                    continue
                if discovery[v] == -1:
                    parent_edge[v] = e
                    component[v] = root
                    discovery[v] = low[v] = clock
                    clock += 1
                    stack.append((v, iter(adjacency[v])))
                else:
                    low[u] = min(low[u], discovery[v])
                continue
            stack.pop()
            if stack:
                parent = stack[-1][0]
                e = parent_edge[u]
                if low[u] > discovery[parent]:
                    bridges.add(e)
                    if root == 0 and houses[u]:
                        fixed[e] = 2
                low[parent] = min(low[parent], low[u])
                houses[parent] += houses[u]
    for e, edge in enumerate(grid.edges):
        if component[edge.u] != 0:
            fixed[e] = 0
    return RouteReduction(fixed, frozenset(bridges))
