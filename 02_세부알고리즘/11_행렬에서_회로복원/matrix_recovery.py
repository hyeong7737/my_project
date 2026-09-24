"""Part 2: exact labeled matrix -> unique canonical LC circuit for a fixed map."""
from fractions import Fraction as F
import sympy as sp
from lc_model import LCModel, decode_circuit
from map_data import ValidationError, integer
from spectral_target import exact_fraction


def family_matrix(model, counts):
    """R = H Kbar = sqrt(H) D invsqrt(H), even for infeasible counts."""
    if len(counts) != model.grid.p or any(type(m) is not int or m < 0 for m in counts):
        raise ValidationError("행렬족의 매개변수는 도로 수와 같은 길이의 비음 정수 목록이어야 합니다.")
    n = model.grid.n
    k = sp.eye(n)
    for edge, m, g in zip(model.grid.edges, counts, model.g):
        q = sp.Rational(g.numerator, g.denominator)*m
        u, v = edge.u, edge.v
        k[u, u] += q
        k[v, v] += q
        k[u, v] -= q
        k[v, u] -= q
    return sp.diag(*model.a)*k


def parse_matrix(rows, n):
    if isinstance(rows, sp.MatrixBase):
        if rows.shape != (n, n):
            raise ValidationError("행렬 차원과 지도 노드 수가 다릅니다.")
        rows = rows.tolist()
    if not isinstance(rows, (list, tuple)) or len(rows) != n:
        raise ValidationError("행렬 차원과 지도 노드 수가 다릅니다.")
    if any(not isinstance(row, (list, tuple)) or len(row) != n for row in rows):
        raise ValidationError("정사각 행렬이 필요합니다.")
    return sp.Matrix([[sp.Rational(exact_fraction(x).numerator, exact_fraction(x).denominator)
                       for x in row] for row in rows])


def string_rows(matrix):
    return [[str(v) for v in row] for row in matrix.tolist()]


def recover_matrix(grid, rows, target=None, *, kind="rational", max_count=None):
    """Reject every off-family matrix. No rounding or spectral-only acceptance."""
    if max_count is not None:
        integer(max_count, "max_count", 0)
    model = LCModel(grid)
    h = sp.diag(*model.a)
    r = parse_matrix(rows, grid.n)
    if kind == "normalized_K":
        r = h*r
    elif kind != "rational":
        raise ValidationError("kind는 rational 또는 normalized_K여야 합니다.")
    allowed = {(e.u, e.v) for e in grid.edges}
    if any((i, j) not in allowed and (r[i, j] != 0 or r[j, i] != 0)
           for i in range(grid.n) for j in range(i+1, grid.n)):
        raise ValidationError("조건 2: 지도에 없는 도로 성분이 있습니다.")
    counts = []
    for edge, g in zip(grid.edges, model.g):
        m = exact_fraction(-r[edge.u, edge.v])/ (model.a[edge.u]*g)
        if m.denominator != 1 or m < 0 or (max_count is not None and m > max_count):
            raise ValidationError("조건 3: 허용된 정수 이용 횟수로 복원되지 않습니다.")
        counts.append(int(m))
    counts = tuple(counts)
    if r != family_matrix(model, counts):
        raise ValidationError("조건 6: 대각 성분·양방향 성분이 기존 L·C 배정 규칙과 다릅니다.")
    grid.require_candidate(counts)  # Conditions 4 and 5.
    if target is not None and (target.n != grid.n or not target.matches(r)):
        raise ValidationError("조건 1: 목표와 정확한 특성다항식이 다릅니다.")
    circuit = model.encode(counts)
    if decode_circuit(circuit) != (grid, counts):
        raise AssertionError("기존 LC 회로 왕복 변환 실패")
    kbar = h.inv()*r
    s = sp.diag(*(sp.sqrt(a) for a in model.a))
    d = s.inv()*r*s
    physical_c = sp.diag(*(sp.Rational(grid.C_star.numerator, grid.C_star.denominator)/a
                           for a in model.a))
    physical_k = kbar/sp.Rational(grid.L_star.numerator, grid.L_star.denominator)
    return {
        "counts": list(counts),
        "rational_matrix_similar_to_D": string_rows(r),
        "D_exact": string_rows(d),
        "normalized_K": string_rows(kbar),
        "physical_C": string_rows(physical_c),
        "physical_K_inverse_inductance": string_rows(physical_k),
        "characteristic_polynomial_coefficients": [str(x) for x in r.charpoly().all_coeffs()],
        "time_ticks": grid.time_ticks(counts),
        "time_seconds": str(grid.time_ticks(counts)*grid.tick_seconds),
        "circuit": circuit,
        "verification": {
            "target_spectrum_exact": True if target is not None else None,
            "map_edges_only": True, "integer_counts": True,
            "depot_connected_and_houses_visited": True, "even_degrees": True,
            "lc_assignment_exact": True, "circuit_map_roundtrip_exact": True,
        },
    }


def recover_symmetric_matrix(grid, matrix, target=None, *, max_count=None):
    """Python API for an exact SymPy D, possibly containing algebraic radicals."""
    if not isinstance(matrix, sp.MatrixBase) or matrix.shape != (grid.n, grid.n):
        raise ValidationError("정확한 SymPy 정사각 D 행렬이 필요합니다.")
    if matrix.has(sp.Float) or matrix != matrix.T:
        raise ValidationError("D는 근삿값이 없는 정확한 대칭행렬이어야 합니다.")
    s = sp.diag(*(sp.sqrt(1+n.delay) for n in grid.nodes))
    r = (s*matrix*s.inv()).applyfunc(sp.simplify)
    return recover_matrix(grid, r, target, max_count=max_count)
