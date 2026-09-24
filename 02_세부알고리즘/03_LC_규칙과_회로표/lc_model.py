"""The v1 circuit encoding and normalized voltage-oscillation matrices."""
from fractions import Fraction as F
import numpy as np
from map_data import GridMap, Node, Edge, ValidationError, positive_fraction

DIRECTION = {(1, 0): 1, (0, 1): 2, (-1, 0): 3, (0, -1): 4}


class LCModel:
    def __init__(self, grid):
        self.grid = grid
        self.a = tuple(1+n.delay for n in grid.nodes)
        self.g = tuple(F(e.time, self.a[e.u]+self.a[e.v]) for e in grid.edges)
        self.ell = tuple(grid.L_star/g for g in self.g)
        # Sparse columns of Z: two stored entries per edge; no dense n-by-p cache.
        self.z_columns = tuple((e.u, e.v, float(self.a[e.u]*g)**0.5, -float(self.a[e.v]*g)**0.5)
                               for e, g in zip(grid.edges, self.g))

    def matrices(self, counts):
        self.grid.require_candidate(counts)
        n = self.grid.n
        k = [[F(int(i == j)) for j in range(n)] for i in range(n)]
        for edge, m, g in zip(self.grid.edges, counts, self.g):
            q, u, v = m*g, edge.u, edge.v
            k[u][u] += q
            k[v][v] += q
            k[u][v] -= q
            k[v][u] -= q
        rational = [[self.a[i]*k[i][j] for j in range(n)] for i in range(n)]
        scale = np.sqrt(np.array(self.a, dtype=float))
        dynamic = np.array(k, dtype=float)*scale[:, None]*scale[None, :]
        if not np.all(np.isfinite(dynamic)):
            raise ValidationError("현재 수치 행렬 범위를 넘는 비용입니다.")
        return k, rational, dynamic

    def encode(self, counts):
        self.grid.require_candidate(counts)
        nodes = [[i+1, n.x, n.y, int(n.house), 2, str(self.grid.C_star/self.a[i]), 1, str(self.grid.L_star)]
                 for i, n in enumerate(self.grid.nodes)]
        branches = []
        for i, (edge, ell, count) in enumerate(zip(self.grid.edges, self.ell, counts)):
            u, v = self.grid.nodes[edge.u], self.grid.nodes[edge.v]
            branches.append([i+1, edge.u+1, edge.v+1, DIRECTION[v.x-u.x, v.y-u.y], 1, str(ell), count])
        return {"schema": "lc-circuit-v1", "map_name": self.grid.name,
                "ground": 0, "depot": 1, "units": self.grid.to_dict()["units"],
                "node_columns": ["id", "x", "y", "house", "C_type", "C_farad", "ground_L_type", "ground_L_henry"],
                "branch_columns": ["id", "u", "v", "direction", "L_type", "base_L_henry", "count"],
                "nodes": nodes, "branches": branches}


def decode_circuit(data):
    try:
        if data["schema"] != "lc-circuit-v1" or data["ground"] != 0 or data["depot"] != 1:
            raise ValidationError("회로 schema/접지/가게 번호가 맞지 않습니다.")
        units = data["units"]
        lstar = positive_fraction(units["L_star_henry"], "L_star")
        cstar = positive_fraction(units["C_star_farad"], "C_star")
        tick = positive_fraction(units["tick_seconds"], "tick_seconds")
        nodes = []
        for i, row in enumerate(data["nodes"]):
            if len(row) != 8 or row[0] != i+1 or row[3] not in (0, 1) or row[4] != 2 or row[6] != 1:
                raise ValidationError("정점 회로 행 형식이 맞지 않습니다.")
            if positive_fraction(row[7], "ground L") != lstar:
                raise ValidationError("기준 접지 L이 일정해야 합니다.")
            delay = cstar/positive_fraction(row[5], "node C")-1
            if delay.denominator != 1 or delay < 0:
                raise ValidationError("C에서 정수 배달 지연을 복원할 수 없습니다.")
            nodes.append(Node(row[1], row[2], bool(row[3]), int(delay)))
        edges, counts = [], []
        for i, row in enumerate(data["branches"]):
            if len(row) != 7 or row[0] != i+1 or row[4] != 1:
                raise ValidationError("도로 회로 행 형식이 맞지 않습니다.")
            u, v = row[1]-1, row[2]-1
            if type(row[1]) is not int or type(row[2]) is not int or not 0 <= u < v < len(nodes):
                raise ValidationError("도로 끝점이 맞지 않습니다.")
            direction = DIRECTION.get((nodes[v].x-nodes[u].x, nodes[v].y-nodes[u].y))
            if row[3] != direction or direction is None:
                raise ValidationError("방향 코드가 좌표와 다릅니다.")
            w = lstar*(2+nodes[u].delay+nodes[v].delay)/positive_fraction(row[5], "base L")
            if w.denominator != 1 or w <= 0:
                raise ValidationError("L에서 양의 정수 이동 시간을 복원할 수 없습니다.")
            edges.append(Edge(u, v, int(w)))
            counts.append(row[6])
        grid = GridMap(data["map_name"], tuple(nodes), tuple(edges), tick, lstar, cstar)
        grid.require_candidate(counts)
        if LCModel(grid).encode(counts) != data:
            raise ValidationError("정규 회로 표현과 다릅니다. 생성한 표와 열 정의를 유지하세요.")
        return grid, tuple(counts)
    except (KeyError, TypeError, IndexError, ZeroDivisionError) as error:
        raise ValidationError(f"회로 데이터 형식 오류: {error}") from error
