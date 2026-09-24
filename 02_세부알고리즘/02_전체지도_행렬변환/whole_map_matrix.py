"""Exact all-roads-on LC matrix input; no valid-tour requirement on this matrix."""
from fractions import Fraction as F
from map_data import GridMap, Node, Edge, ValidationError, integer, positive_fraction
from spectral_target import exact_fraction


def encode_whole_map(grid):
    a = [1+node.delay for node in grid.nodes]
    r = [[F(a[i] if i == j else 0) for j in range(grid.n)] for i in range(grid.n)]
    for e in grid.edges:
        g = F(e.time, a[e.u]+a[e.v])
        for u, v in ((e.u, e.v), (e.v, e.u)):
            r[u][u] += a[u]*g
            r[u][v] -= a[u]*g
    return {"schema": "lc-whole-map-v1", "name": grid.name, "depot": 1,
            "matrix_kind": "rational_R_all_edges_one", "units": grid.to_dict()["units"],
            "nodes": [{"id": i+1, "x": node.x, "y": node.y, "house": node.house}
                      for i, node in enumerate(grid.nodes)],
            "R_map": [[str(v) for v in row] for row in r]}


def decode_whole_map(data):
    try:
        if data["schema"] != "lc-whole-map-v1" or data["matrix_kind"] != "rational_R_all_edges_one":
            raise ValidationError("lc-whole-map-v1 전체 지도 기준 행렬이 필요합니다.")
        if type(data["depot"]) is not int or data["depot"] != 1:
            raise ValidationError("가게는 노드 1이어야 합니다.")
        labels, raw = data["nodes"], data["R_map"]
        if not isinstance(labels, list) or not labels:
            raise ValidationError("노드 메타데이터가 필요합니다.")
        n = len(labels)
        if not isinstance(raw, list) or len(raw) != n or any(not isinstance(row, list) or len(row) != n for row in raw):
            raise ValidationError("R_map 차원과 노드 수가 다릅니다.")
        r = [[exact_fraction(v) for v in row] for row in raw]
        a = [sum(row) for row in r]
        if any(v.denominator != 1 or v < 1 for v in a):
            raise ValidationError("행 합은 1+정수 배달 지연이어야 합니다.")
        nodes = []
        for i, node in enumerate(labels):
            if integer(node["id"], "node id", 1) != i+1:
                raise ValidationError("노드 번호는 1부터 연속이어야 합니다.")
            nodes.append(Node(node["x"], node["y"], node["house"], int(a[i]-1)))
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                gu, gv = -r[u][v]/a[u], -r[v][u]/a[v]
                if gu != gv or gu < 0:
                    raise ValidationError("양방향 LC 성분의 비율 또는 부호가 잘못되었습니다.")
                if gu:
                    w = gu*(a[u]+a[v])
                    if w.denominator != 1 or w <= 0:
                        raise ValidationError("도로 이동 시간을 양의 정수로 복원할 수 없습니다.")
                    edges.append(Edge(u, v, int(w)))
        units = data["units"]
        grid = GridMap(data["name"], tuple(nodes), tuple(edges),
                       positive_fraction(units["tick_seconds"], "tick_seconds"),
                       positive_fraction(units["L_star_henry"], "L_star"),
                       positive_fraction(units["C_star_farad"], "C_star"))
        expected = encode_whole_map(grid)
        if [[exact_fraction(v) for v in row] for row in expected["R_map"]] != r:
            raise ValidationError("전체 지도 기준 LC 행렬을 정확히 복원할 수 없습니다.")
        return grid
    except (KeyError, TypeError, IndexError) as error:
        raise ValidationError(f"전체 지도 행렬 형식 오류: {error}") from error
