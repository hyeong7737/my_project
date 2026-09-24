"""Weighted forest expansion of det(t I - H(I+B diag(g*m) B.T)).

Forests are algebraic terms of one shared polynomial, not candidate routes.
All arithmetic is exact. An incomplete expansion must never be used.
"""
from fractions import Fraction as F
from math import prod
from time import monotonic
from map_data import ResourceLimit


def forest_basis(grid, *, max_forests=10000, deadline=None):
    a = tuple(1+node.delay for node in grid.nodes)
    g = tuple(F(e.time, a[e.u]+a[e.v]) for e in grid.edges)
    basis = {}
    # Each state extends a forest by a larger edge ID; no route-vector loop.
    stack = [((), tuple(range(grid.n)), 0)]
    while stack:
        if deadline is not None and monotonic() >= deadline:
            raise ResourceLimit("total_timeout")
        subset, labels, next_edge = stack.pop()
        if len(basis) >= max_forests:
            raise ResourceLimit("forest_limit")
        components = {}
        for i, label in enumerate(labels):
            components.setdefault(label, []).append(i)
        polynomial = [F((-1)**len(subset))*prod(g[e] for e in subset)]
        for vertices in components.values():
            product_a = prod(a[i] for i in vertices)
            slope = sum(F(product_a, a[i]) for i in vertices)
            intercept = -len(vertices)*product_a
            updated = [F(0)]*(len(polynomial)+1)
            for j, coefficient in enumerate(polynomial):
                updated[j] += coefficient*slope
                updated[j+1] += coefficient*intercept
            polynomial = updated
        basis[subset] = tuple([F(0)]*len(subset)+polynomial)
        for j in range(grid.p-1, next_edge-1, -1):
            edge = grid.edges[j]
            u, v = labels[edge.u], labels[edge.v]
            if u != v:
                merged = tuple(u if label == v else label for label in labels)
                stack.append((subset+(j,), merged, j+1))
    return basis


def _mul(left, right):
    """Ascending polynomial coefficients, exact rationals."""
    result = [F(0)]*(len(left)+len(right)-1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            result[i+j] += a*b
    return result


def _fixed_tree_determinant(vertices, fixed_edges, a, deadline):
    """det(diag(sum(t/a_i-1)) - Laplacian(fixed weights)).

    Contracted fixed edges remain a forest because each was a bridge in the
    original graph. Division-free leaf elimination sums all their subsets
    without enumerating them. This is a determinant computation, not a route DP.
    """
    size = len(vertices)
    adjacency = [[] for _ in vertices]
    for u, v, weight in fixed_edges:
        assert u != v
        adjacency[u].append((v, weight))
        adjacency[v].append((u, weight))
    diagonal = [(-F(len(group))-sum((w for _,w in adjacency[i]), F(0)),
                 sum((F(1,a[v]) for v in group), F(0))) for i,group in enumerate(vertices)]
    parent, order, roots = [-2]*size, [], []
    for root in range(size):
        if parent[root] != -2:
            continue
        roots.append(root)
        parent[root] = -1
        stack = [root]
        while stack:
            u = stack.pop()
            order.append(u)
            for v, _ in adjacency[u]:
                if v == parent[u]:
                    continue
                assert parent[v] == -2, 'fixed edges must remain acyclic'
                parent[v] = u
                stack.append(v)
    full, deleted = {}, {}
    for u in reversed(order):
        if deadline is not None and monotonic() >= deadline:
            raise ResourceLimit('total_timeout')
        children = [(v,w) for v,w in adjacency[u] if parent[v] == u]
        prefix, suffix = [[F(1)]], [[F(1)] for _ in range(len(children)+1)]
        for v,_ in children:
            prefix.append(_mul(prefix[-1], full[v]))
        for j in range(len(children)-1,-1,-1):
            suffix[j] = _mul(full[children[j][0]], suffix[j+1])
        deleted[u] = prefix[-1]
        value = _mul(list(diagonal[u]), deleted[u])
        for j,(v,w) in enumerate(children):
            term = _mul(deleted[v], _mul(prefix[j], suffix[j+1]))
            for k,c in enumerate(term):
                value[k] -= w*w*c
        full[u] = value
    result = [F(1)]
    for root in roots:
        result = _mul(result, full[root])
    return result


def reduced_forest_basis(grid, reduction, *, max_forests=10000, deadline=None):
    """Expand only free-edge forests; absorb forced bridge values algebraically.

    Same full n-dimensional spectrum, including all unused grounded nodes.
    No candidate route eigenvalue/characteristic-polynomial evaluations.
    """
    if not reduction.fixed:
        return forest_basis(grid, max_forests=max_forests, deadline=deadline)
    a = tuple(1+node.delay for node in grid.nodes)
    g = tuple(F(e.time,a[e.u]+a[e.v]) for e in grid.edges)
    free = tuple(e for e in range(grid.p) if e not in reduction.fixed)
    fixed_on = tuple(e for e,m in reduction.fixed.items() if m == 2)
    assert all(e in reduction.bridges for e in fixed_on)
    scale = prod(a)
    basis = {}
    stack = [((), tuple(range(grid.n)), 0)]
    while stack:
        if deadline is not None and monotonic() >= deadline:
            raise ResourceLimit('total_timeout')
        subset, labels, next_index = stack.pop()
        if len(basis) >= max_forests:
            raise ResourceLimit('forest_limit')
        groups = {}
        for i,label in enumerate(labels):
            groups.setdefault(label, []).append(i)
        ids = {label:i for i,label in enumerate(groups)}
        links = [(ids[labels[grid.edges[e].u]], ids[labels[grid.edges[e].v]], 2*g[e]) for e in fixed_on]
        polynomial = _fixed_tree_determinant(list(groups.values()), links, a, deadline)
        factor = F((-1)**len(subset)*scale)*prod(g[e] for e in subset)
        basis[subset] = tuple([F(0)]*len(subset)+[factor*c for c in reversed(polynomial)])
        for j in range(len(free)-1,next_index-1,-1):
            e = free[j]
            edge = grid.edges[e]
            u,v = labels[edge.u], labels[edge.v]
            if u != v:
                merged = tuple(u if label == v else label for label in labels)
                stack.append((subset+(e,), merged, j+1))
    return basis
