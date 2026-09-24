"""Exact Euler-tour feasibility: parity, required visits, and support connectivity."""
import z3


def build_route_constraints(grid, *, reduction=None):
    n, p = grid.n, grid.p
    solver = z3.SolverFor("QF_LIA")
    fixed = reduction.fixed if reduction is not None else {}
    bridges = reduction.bridges if reduction is not None else ()
    counts = tuple(z3.IntVal(fixed[e]) if e in fixed else z3.Int(f"m_{e}") for e in range(p))
    incident = [[] for _ in range(n)]
    for j, edge in enumerate(grid.edges):
        if j not in fixed:
            solver.add(counts[j] >= 0, counts[j] <= 2)
            if j in bridges:
                solver.add(z3.Or(counts[j] == 0, counts[j] == 2))
        incident[edge.u].append((j, edge.v))
        incident[edge.v].append((j, edge.u))
    active = [z3.BoolVal(True) if i == 0 else z3.Or(*(counts[j] > 0 for j, _ in incident[i]))
              for i in range(n)]
    for i in range(n):
        half_degree = z3.Int(f"half_degree_{i}")
        solver.add(half_degree >= 0, half_degree <= len(incident[i]),
                   z3.Sum(*(counts[j] for j, _ in incident[i])) == 2*half_degree)
        if grid.nodes[i].house:
            solver.add(active[i])

    # Signed unit-demand flows certify all used vertices are depot-connected.
    flows = tuple(z3.Int(f"flow_{j}") for j in range(p))
    incoming = [[] for _ in range(n)]
    for j, edge in enumerate(grid.edges):
        solver.add(flows[j] >= -(n-1), flows[j] <= n-1,
                   z3.Implies(counts[j] == 0, flows[j] == 0))
        incoming[edge.u].append(-flows[j])
        incoming[edge.v].append(flows[j])
    for i in range(1, n):
        solver.add(z3.Sum(*incoming[i]) == z3.If(active[i], 1, 0))
    solver.add(z3.Sum(*incoming[0]) == -z3.Sum(*(z3.If(active[i], 1, 0) for i in range(1, n))))

    return solver, counts
