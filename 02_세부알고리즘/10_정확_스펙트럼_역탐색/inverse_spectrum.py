"""Part 1: structured IEP, alternating projection + complete integer certification.

No block decomposition or route optimization DP. Floating point operations only
propose candidates/order branches. All exclusions and accepted answers are exact.
"""
from fractions import Fraction as F
from math import gcd
import numpy as np
from lc_model import LCModel
from map_data import ValidationError, integer
from matrix_recovery import family_matrix, recover_matrix
from spectral_target import SpectralTarget


def _projection_proposals(model, target, caps, restarts, iterations, seed, accept):
    """Classical spectral projection paired with bounded LC-family projection."""
    report = {"method": "alternating_projection_relaxed_LC_family",
              "restarts_requested": restarts, "restarts_completed": 0,
              "iterations": 0, "accepted_new_solutions": 0,
              "certifies_completeness": False, "best_relative_family_gap": None}
    p, n = model.grid.p, model.grid.n
    if restarts == 0 or not any(caps):
        report["skipped"] = "disabled_or_no_free_edge"
        return None, report
    if n*n*p > 2_000_000:
        report["skipped"] = "dense_projection_memory_guard"
        return None, report
    try:
        from scipy.optimize import lsq_linear
    except ImportError:
        report["skipped"] = "scipy_unavailable_exact_search_still_runs"
        return None, report
    try:
        values = np.array(target.approximate_values())
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("target_float_range")
        h = np.diag(model.a).astype(float)
        columns = []
        for u, v, zu, zv in model.z_columns:
            z = np.zeros(n)
            z[u], z[v] = zu, zv
            columns.append(np.outer(z, z).reshape(-1))
        design = np.array(columns).T
        active = np.flatnonzero(np.array(caps) > 0)
        upper = np.array([float(caps[e]) for e in active])
        if not np.all(np.isfinite(design)) or not np.all(np.isfinite(upper)):
            raise ValueError("LC_float_range")
        rng = np.random.default_rng(seed)
        best_hint, best_gap = None, float("inf")
        tested = set()
        for _ in range(restarts):
            q, _ = np.linalg.qr(rng.normal(size=(n, n)))
            spectral = (q*values)@q.T
            for _ in range(iterations):
                fit = lsq_linear(design[:, active], (spectral-h).reshape(-1),
                                 bounds=(np.zeros(len(active)), upper), method="bvls", max_iter=200)
                x = np.zeros(p)
                x[active] = fit.x
                family = h+(design@x).reshape(n, n)
                _, vectors = np.linalg.eigh(family)
                new_spectral = (vectors*values)@vectors.T
                gap = float(np.linalg.norm(new_spectral-family)/max(1.0, np.linalg.norm(new_spectral)))
                if not np.isfinite(gap):
                    raise ValueError("nonfinite_projection_gap")
                if gap < best_gap:
                    best_hint, best_gap = x.copy(), gap
                rounded = tuple(min(caps[e], max(0, int(round(v)))) for e, v in enumerate(x))
                if rounded not in tested:
                    tested.add(rounded)
                    report["accepted_new_solutions"] += int(accept(rounded, "alternating_projection"))
                report["iterations"] += 1
                change = np.linalg.norm(new_spectral-spectral)
                spectral = new_spectral
                # A stopping condition is NOT an acceptance criterion.
                if gap < 1e-11 or change < 1e-12*max(1.0, np.linalg.norm(spectral)):
                    break
            report["restarts_completed"] += 1
        report["best_relative_family_gap"] = best_gap
        return best_hint, report
    except (ValueError, OverflowError, np.linalg.LinAlgError, FloatingPointError) as error:
        report["numerical_failure"] = str(error)
        return None, report


def solve_inverse(grid, target, *, max_count=None, max_states=200000,
                  max_solutions=10000, ap_restarts=4, ap_iterations=100, seed=0):
    """Find ALL compatible circuits within the reported domain if complete=True.

    By default caps come from the target trace, so also handles nonoptimal targets
    requiring more than two traversals. A caller may explicitly restrict counts.
    Limits yield a labeled partial result; they never masquerade as no solution.
    """
    if not isinstance(target, SpectralTarget):
        target = SpectralTarget.from_dict(target)
    if target.n != grid.n:
        raise ValidationError("목표 스펙트럼 차원과 지도 노드 수가 다릅니다.")
    if max_count is not None:
        integer(max_count, "max_count", 0)
    integer(max_states, "max_states", 1)
    integer(max_solutions, "max_solutions", 1)
    integer(ap_restarts, "ap_restarts", 0)
    integer(ap_iterations, "ap_iterations", 1)
    integer(seed, "seed", 0)
    model = LCModel(grid)
    budget = target.trace-sum(model.a)
    stats = {"states_visited": 0, "pruned_trace": 0, "pruned_parity": 0,
             "pruned_connectivity": 0, "pruned_second_moment": 0,
             "characteristic_polynomials_checked": 0}
    report = {"schema": "lc-inverse-result-v1", "map": grid.to_dict(),
              "target": target.to_dict(),
              "domain": {"mode": "all_target_time_routes" if max_count is None else "explicit_count_cap",
                         "max_count": max_count, "travel_budget_ticks": str(budget)},
              "complete": True, "status": "complete", "stop_reason": None,
              "statistics": stats, "solutions": [], "solution_count": 0, "existence": "no",
              "projection": {"skipped": "target_trace_infeasible"},
              "algorithm": "structured_IEP_projection_plus_exact_integer_search"}
    if budget < 0 or budget.denominator != 1:
        report["infeasibility_reason"] = "target_trace_requires_negative_or_noninteger_travel"
        stats["pruned_trace"] = 1
        return report
    budget = int(budget)
    caps = tuple(min(budget//e.time, max_count) if max_count is not None
                 else budget//e.time for e in grid.edges)
    report["domain"]["effective_edge_caps"] = list(caps)
    solutions = {}
    evaluated = set()

    def accept(counts, source):
        if counts in evaluated or len(solutions) >= max_solutions:
            return False
        if grid.candidate_errors(counts):
            return False
        if sum(m*e.time for m, e in zip(counts, grid.edges)) != budget:
            return False
        evaluated.add(counts)
        r = family_matrix(model, counts)
        stats["characteristic_polynomials_checked"] += 1
        if not target.matches(r):
            return False
        item = recover_matrix(grid, r, target, max_count=max_count)
        item["discovered_by"] = source
        solutions[counts] = item
        return True

    hint, projection = _projection_proposals(model, target, caps, ap_restarts,
                                             ap_iterations, seed, accept)
    report["projection"] = projection
    p, n = grid.p, grid.n
    # Exact upper/lower moment bounds for each unassigned suffix.
    max_cost = [0]*(p+1)
    weight_gcd = [0]*(p+1)
    available = [[0]*n for _ in range(p+1)]
    diag_add = [[F(0)]*n for _ in range(p+1)]
    off_add = [F(0)]*(p+1)
    for i in range(p-1, -1, -1):
        e, cap, g = grid.edges[i], caps[i], model.g[i]
        max_cost[i] = max_cost[i+1]+cap*e.time
        weight_gcd[i] = gcd(weight_gcd[i+1], e.time if cap else 0)
        available[i] = available[i+1].copy()
        diag_add[i] = diag_add[i+1].copy()
        available[i][e.u] += bool(cap)
        available[i][e.v] += bool(cap)
        diag_add[i][e.u] += model.a[e.u]*g*cap
        diag_add[i][e.v] += model.a[e.v]*g*cap
        off_add[i] = off_add[i+1]+model.a[e.u]*model.a[e.v]*(g*cap)**2

    def potentially_connected(prefix, degree, index):
        neighbors = [[] for _ in range(n)]
        for j, e in enumerate(grid.edges):
            if (j < index and prefix[j] > 0) or (j >= index and caps[j] > 0):
                neighbors[e.u].append(e.v)
                neighbors[e.v].append(e.u)
        seen, pending = {0}, [0]
        while pending:
            for v in neighbors[pending.pop()]:
                if v not in seen:
                    seen.add(v)
                    pending.append(v)
        return all(v in seen for v in grid.houses) and all(not d or v in seen for v, d in enumerate(degree))

    # Stack avoids Python recursion limits. Every unpruned discrete assignment
    # is visited once; numerical hints change traversal order only.
    stack = [((), budget, (0,)*n, tuple(map(F, model.a)), F(0))]
    while stack:
        if stats["states_visited"] >= max_states or len(solutions) >= max_solutions:
            report.update(complete=False, status="partial",
                          stop_reason="max_states" if stats["states_visited"] >= max_states else "max_solutions")
            break
        prefix, remaining, degree, diagonal, off = stack.pop()
        i = len(prefix)
        stats["states_visited"] += 1
        if remaining < 0 or remaining > max_cost[i] or (weight_gcd[i] and remaining % weight_gcd[i]):
            stats["pruned_trace"] += 1
            continue
        if any(d % 2 and available[i][v] == 0 for v, d in enumerate(degree)):
            stats["pruned_parity"] += 1
            continue
        if not potentially_connected(prefix, degree, i):
            stats["pruned_connectivity"] += 1
            continue
        lower = sum(d*d for d in diagonal)+2*off
        upper = sum((d+a)**2 for d, a in zip(diagonal, diag_add[i]))+2*(off+off_add[i])
        if not lower <= target.trace_square <= upper:
            stats["pruned_second_moment"] += 1
            continue
        if i == p:
            accept(prefix, "exact_integer_spectral_search")
            continue
        e, g = grid.edges[i], model.g[i]
        lo = max(0, (remaining-max_cost[i+1]+e.time-1)//e.time)
        hi = min(caps[i], remaining//e.time)
        if lo > hi:
            stats["pruned_trace"] += 1
            continue
        # Lazy-in-range expansion would save more memory for huge integer caps;
        # the current implementation enforces a pending-state resource guard.
        if len(stack)+(hi-lo+1) > max_states:
            report.update(complete=False, status="partial", stop_reason="pending_state_limit")
            break
        order = range(lo, hi+1)
        if hint is not None:
            order = sorted(order, key=lambda m: (abs(m-hint[i]), m))
        for m in reversed(order):
            new_degree = list(degree)
            new_diag = list(diagonal)
            new_degree[e.u] += m
            new_degree[e.v] += m
            new_diag[e.u] += model.a[e.u]*g*m
            new_diag[e.v] += model.a[e.v]*g*m
            new_off = off+model.a[e.u]*model.a[e.v]*(g*m)**2
            stack.append((prefix+(m,), remaining-m*e.time, tuple(new_degree), tuple(new_diag), new_off))
    report["solutions"] = [solutions[c] for c in sorted(solutions)]
    report["solution_count"] = len(solutions)
    report["existence"] = "yes" if solutions else ("no" if report["complete"] else "unknown")
    return report
