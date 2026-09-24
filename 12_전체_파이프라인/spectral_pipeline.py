"""Connect whole-map input -> 5 generation -> 6 selection -> 7 LC recovery."""
from direct_spectra import generate_spectra
from whole_map_matrix import decode_whole_map
from inverse_spectrum import solve_inverse


def solve_spectral_map(whole_map, *, inverse_max_states=200000,
                       inverse_max_solutions=10000, **generation_options):
    generated = generate_spectra(whole_map, **generation_options)
    report = {"schema": "lc-spectral-pipeline-v1", "generation": generated,
              "complete": False, "optimal_time_certified": generated["complete"],
              "inverse_results": [], "optimal_route_counts": [],
              "all_optimal_routes_recovered": False}
    if not generated["complete"]:
        report["stop_reason"] = "generation_incomplete"
        return report
    grid = decode_whole_map(whole_map)
    recovered = set()
    for index in generated["selection"]["optimal_spectrum_indices"]:
        spectrum = generated["spectra"][index]
        result = solve_inverse(grid, spectrum, max_count=2, ap_restarts=0,
                               max_states=inverse_max_states,
                               max_solutions=inverse_max_solutions)
        report["inverse_results"].append({"spectrum_index": index, "result": result})
        recovered.update(tuple(item["counts"]) for item in result["solutions"])
    complete = all(item["result"]["complete"] for item in report["inverse_results"])
    report.update(complete=complete, all_optimal_routes_recovered=complete,
                  optimal_route_counts=[list(v) for v in sorted(recovered)],
                  stop_reason=None if complete else "inverse_incomplete")
    return report
