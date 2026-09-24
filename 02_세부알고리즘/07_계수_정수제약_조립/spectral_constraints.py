"""Existential route variables, projected characteristic-coefficient outputs."""
from dataclasses import dataclass
from math import lcm
from time import monotonic
import z3
from map_data import ResourceLimit
from route_constraints import build_route_constraints
from forest_coefficients import reduced_forest_basis
from route_reduction import reduce_route_domain


@dataclass
class SpectralProblem:
    solver: object
    counts: tuple
    outputs: tuple
    denominators: tuple
    statistics: dict


def build_spectral_problem(grid, *, max_forests=10000, deadline=None):
    reduction = reduce_route_domain(grid, deadline=deadline)
    basis = reduced_forest_basis(grid, reduction, max_forests=max_forests, deadline=deadline)
    solver, counts = build_route_constraints(grid, reduction=reduction)
    monomials = {(): z3.IntVal(1)}
    for subset in sorted(basis, key=lambda s: (len(s), s)):
        if deadline is not None and monotonic() >= deadline:
            raise ResourceLimit("total_timeout")
        if not subset:
            continue
        if len(subset) == 1:
            monomials[subset] = counts[subset[0]]
        else:
            value = z3.Int("product_"+"_".join(map(str, subset)))
            previous, last = monomials[subset[:-1]], counts[subset[-1]]
            solver.add(value == z3.If(last == 1, previous,
                                      z3.If(last == 2, 2*previous, 0)),
                       value >= 0, value <= 2**len(subset))
            monomials[subset] = value
    outputs, denominators = [], []
    for k in range(1, grid.n+1):
        if deadline is not None and monotonic() >= deadline:
            raise ResourceLimit("total_timeout")
        denominator = lcm(*(row[k].denominator for row in basis.values()))
        terms, upper = [], 0
        for subset, row in basis.items():
            coefficient = int((-1)**k*row[k]*denominator)
            assert coefficient >= 0
            if coefficient:
                terms.append(coefficient*monomials[subset])
                upper += coefficient*2**len(subset)
        lower = int((-1)**k*basis[()][k]*denominator)
        output = z3.Int(f"spectral_coefficient_{k}")
        solver.add(output == z3.Sum(*terms), output >= lower, output <= upper)
        outputs.append(output)
        denominators.append(denominator)
    return SpectralProblem(solver, counts, tuple(outputs), tuple(denominators),
                           {"structural_forest_terms": len(basis),
                            "coefficient_construction": "forced_bridge_absorption" if reduction.fixed else "weighted_forest",
                            "fixed_count_values": {str(e):m for e,m in sorted(reduction.fixed.items())},
                            "fixed_count_edge_ids_are_zero_based": True,
                            "fixed_zero_edges": sum(m == 0 for m in reduction.fixed.values()),
                            "fixed_two_edges": sum(m == 2 for m in reduction.fixed.values()),
                            "free_count_variables": grid.p-len(reduction.fixed),
                            "parity_restricted_free_bridges": len(reduction.bridges-set(reduction.fixed)),
                            "fixed_bridge_subsets_enumerated": 0,
                            "shared_product_variables": sum(len(s) > 1 for s in basis),
                            "coefficient_outputs": grid.n,
                            "constraint_builds": 1, "constraint_count": len(solver.assertions()),
                            "logic": "QF_LIA"})
