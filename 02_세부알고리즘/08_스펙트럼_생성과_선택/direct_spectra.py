"""Projected enumeration of feasible characteristic polynomials (one per spectrum)."""
from fractions import Fraction as F
from time import monotonic
import z3
from map_data import ValidationError, ResourceLimit, integer
from spectral_target import SpectralTarget
from whole_map_matrix import decode_whole_map, encode_whole_map
from spectral_constraints import build_spectral_problem


def generate_spectra(whole_map, *, max_spectra=1000, timeout_ms=10000,
                     total_timeout_ms=60000, max_forests=10000, include_roots=False):
    integer(max_spectra, "max_spectra", 1)
    integer(timeout_ms, "timeout_ms", 1)
    integer(total_timeout_ms, "total_timeout_ms", 1)
    integer(max_forests, "max_forests", 1)
    if type(include_roots) is not bool:
        raise ValidationError("include_roots는 bool이어야 합니다.")
    started = monotonic()
    grid = decode_whole_map(whole_map)
    stats = {"constraint_builds": 0, "solver_checks": 0, "spectra_emitted": 0,
             "whole_spectrum_exclusion_clauses": 0,
             "application_route_enumerations": 0, "per_route_charpoly_calls": 0,
             "numerical_matrix_eigendecompositions": 0, "root_isolations": 0,
             "internal_solver_search_not_counted_as_zero": True}
    report = {"schema": "lc-generated-spectra-v1",
              "whole_map": encode_whole_map(grid), "map": grid.to_dict(),
              "domain": {"counts_per_edge": [0, 1, 2], "preserves_all_optimal_routes": True,
                         "includes_all_unbounded_nonoptimal_walks": False},
              "method": "reduced_weighted_forest_projected_coefficients_QF_LIA",
              "complete": False, "status": "partial", "stop_reason": None,
              "spectra": [], "statistics": stats}
    problem = None
    try:
        problem = build_spectral_problem(grid, max_forests=max_forests,
                                         deadline=started+total_timeout_ms/1000)
    except ResourceLimit as error:
        report["stop_reason"] = str(error)
    if problem is not None:
        stats.update(problem.statistics)
        report["coefficient_denominators"] = list(map(str, problem.denominators))
        keys = set()
        while True:
            remaining_ms = total_timeout_ms-int(1000*(monotonic()-started))
            if remaining_ms <= 0:
                report["stop_reason"] = "total_timeout"
                break
            problem.solver.set(timeout=min(timeout_ms, remaining_ms))
            stats["solver_checks"] += 1
            result = problem.solver.check()
            if result == z3.unsat:
                report.update(complete=True, status="complete", stop_reason=None)
                break
            if result != z3.sat:
                report["stop_reason"] = "solver_unknown"
                report["solver_reason"] = problem.solver.reason_unknown()
                break
            # One extra query may prove completion exactly at the output cap.
            if len(report["spectra"]) >= max_spectra:
                report["stop_reason"] = "max_spectra"
                break
            model = problem.solver.model()
            key = tuple(model.eval(v, model_completion=True).as_long() for v in problem.outputs)
            if key in keys:
                raise AssertionError("스펙트럼 전체 제외 조건이 중복 출력을 막지 못했습니다.")
            keys.add(key)
            coefficients = (F(1),)+tuple(F((-1)**k*v, d) for k, (v, d) in
                                        enumerate(zip(key, problem.denominators), 1))
            ticks = -coefficients[1]-grid.n
            if ticks.denominator != 1:
                raise AssertionError("스펙트럼 시간 복원에서 정수성이 깨졌습니다.")
            # Only one existence witness for each NEW spectrum; no traversal of
            # all count vectors and no forward spectrum calculation here.
            witness = tuple(model.eval(v, model_completion=True).as_long() for v in problem.counts)
            grid.require_candidate(witness)
            if grid.time_ticks(witness) != ticks:
                raise AssertionError("스펙트럼 시간과 존재 증인 시간이 다릅니다.")
            item = {"characteristic_polynomial_coefficients": [str(c) for c in coefficients],
                    "integer_coefficient_key": [str(v) for v in key],
                    "time_ticks": int(ticks), "time_seconds": str(ticks*grid.tick_seconds),
                    "existence_witness_counts": list(witness),
                    "spectrum_representation": "all_roots_of_exact_characteristic_polynomial_with_multiplicity"}
            if include_roots:
                item["dimensionless_eigenvalues_approx"] = SpectralTarget(coefficients).approximate_values()
                stats["root_isolations"] += 1
            report["spectra"].append(item)
            # Excludes ALL routes having this spectrum, regardless of witness.
            problem.solver.add(z3.Or(*(v != observed for v, observed in zip(problem.outputs, key))))
            stats["spectra_emitted"] += 1
            stats["whole_spectrum_exclusion_clauses"] += 1
        stats["solver_statistics"] = {key: value for key, value in problem.solver.statistics()}
    report["spectra"].sort(key=lambda s: (s["time_ticks"],
                                        tuple(F(v) for v in s["characteristic_polynomial_coefficients"])))
    best = min((s["time_ticks"] for s in report["spectra"]), default=None)
    report["selection"] = {
        "best_observed_time_ticks": best,
        "minimum_certified": report["complete"],
        "optimal_spectrum_indices": [i for i, s in enumerate(report["spectra"]) if s["time_ticks"] == best],
        "indices_are_zero_based": True,
    }
    stats["elapsed_seconds"] = monotonic()-started
    return report
