"""Validated, labeled, undirected grid instances. Costs are integer time ticks."""
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import json
import random


class ValidationError(ValueError):
    pass


class ResourceLimit(RuntimeError):
    pass


def integer(value, name, minimum=None):
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValidationError(f"{name}: 정수" + (f" ≥ {minimum}" if minimum is not None else "") + "가 필요합니다.")
    return value


def positive_fraction(value, name):
    if isinstance(value, bool) or isinstance(value, float):
        raise ValidationError(f"{name}: 정확한 정수 또는 소수/분수 문자열을 사용하세요.")
    try:
        result = Fraction(value)
    except (ValueError, TypeError, ZeroDivisionError) as error:
        raise ValidationError(f"{name}: 유효한 수가 아닙니다.") from error
    if result <= 0:
        raise ValidationError(f"{name}: 양수여야 합니다.")
    return result


@dataclass(frozen=True)
class Node:
    x: int
    y: int
    house: bool
    delay: int


@dataclass(frozen=True)
class Edge:
    u: int
    v: int
    time: int


@dataclass(frozen=True)
class GridMap:
    name: str
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    tick_seconds: Fraction = Fraction(1)
    L_star: Fraction = Fraction(1, 1000)
    C_star: Fraction = Fraction(1, 1000000)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.nodes:
            raise ValidationError("이름과 한 개 이상의 노드가 필요합니다.")
        coords = []
        for i, node in enumerate(self.nodes):
            integer(node.x, f"node {i+1} x")
            integer(node.y, f"node {i+1} y")
            if type(node.house) is not bool:
                raise ValidationError("house는 true/false여야 합니다.")
            integer(node.delay, "delay", 0)
            if not node.house and node.delay:
                raise ValidationError("배달 지점이 아닌 노드의 지연은 0이어야 합니다.")
            coords.append((node.x, node.y))
        if len(set(coords)) != len(coords):
            raise ValidationError("중복된 노드 좌표입니다.")
        if coords[1:] != sorted(coords[1:], key=lambda xy: (xy[1], xy[0])):
            raise ValidationError("가게 1번 이후 노드는 (y,x) 순서로 번호를 붙이세요.")
        pairs = []
        for edge in self.edges:
            integer(edge.u, "u", 0)
            integer(edge.v, "v", 0)
            integer(edge.time, "time", 1)
            if not 0 <= edge.u < edge.v < len(self.nodes):
                raise ValidationError("도로는 범위 내 서로 다른 노드 u<v로 기록하세요.")
            u, v = self.nodes[edge.u], self.nodes[edge.v]
            if abs(u.x-v.x) + abs(u.y-v.y) != 1:
                raise ValidationError("도로는 상하좌우 한 칸이어야 합니다.")
            pairs.append((edge.u, edge.v))
        if pairs != sorted(set(pairs)):
            raise ValidationError("도로는 중복 없이 (u,v) 순서로 기록하세요.")
        for field in ("tick_seconds", "L_star", "C_star"):
            object.__setattr__(self, field, positive_fraction(getattr(self, field), field))
        seen, pending = {0}, [0]
        adjacency = self.adjacency()
        while pending:
            for v, _, _ in adjacency[pending.pop()]:
                if v not in seen:
                    seen.add(v)
                    pending.append(v)
        if any(node.house and i not in seen for i, node in enumerate(self.nodes)):
            raise ValidationError("가게에서 도달할 수 없는 필수 집이 있습니다.")

    @property
    def n(self):
        return len(self.nodes)

    @property
    def p(self):
        return len(self.edges)

    @property
    def service(self):
        return sum(node.delay for node in self.nodes)

    @property
    def houses(self):
        return tuple(i for i, node in enumerate(self.nodes) if node.house and i != 0)

    def adjacency(self):
        result = [[] for _ in self.nodes]
        for e, edge in enumerate(self.edges):
            result[edge.u].append((edge.v, edge.time, e))
            result[edge.v].append((edge.u, edge.time, e))
        return result

    def candidate_errors(self, counts):
        if len(counts) != self.p or any(type(m) is not int or m < 0 for m in counts):
            return ["이용 횟수는 도로 수와 같은 길이의 비음 정수 목록이어야 합니다."]
        degree, neighbors = [0]*self.n, [set() for _ in self.nodes]
        for edge, m in zip(self.edges, counts):
            if m:
                degree[edge.u] += m
                degree[edge.v] += m
                neighbors[edge.u].add(edge.v)
                neighbors[edge.v].add(edge.u)
        errors = []
        if any(d % 2 for d in degree):
            errors.append("사용 도로의 통행 차수에 홀수가 있습니다.")
        seen, pending = {0}, [0]
        while pending:
            for v in neighbors[pending.pop()]:
                if v not in seen:
                    seen.add(v)
                    pending.append(v)
        if any(node.house and i not in seen for i, node in enumerate(self.nodes)):
            errors.append("방문하지 않은 필수 집이 있습니다.")
        if any(degree[i] and i not in seen for i in range(self.n)):
            errors.append("가게와 떨어진 사용 도로 성분이 있습니다.")
        return errors

    def require_candidate(self, counts):
        errors = self.candidate_errors(counts)
        if errors:
            raise ValidationError(" ".join(errors))

    def time_ticks(self, counts):
        self.require_candidate(counts)
        return self.service + sum(edge.time*m for edge, m in zip(self.edges, counts))

    def to_dict(self):
        return {"schema": "lc-grid-v1", "name": self.name, "depot": 1,
                "units": {"tick_seconds": str(self.tick_seconds), "L_star_henry": str(self.L_star),
                          "C_star_farad": str(self.C_star)},
                "nodes": [{"id": i+1, "x": n.x, "y": n.y, "house": n.house, "delay": n.delay}
                          for i, n in enumerate(self.nodes)],
                "edges": [{"id": i+1, "u": e.u+1, "v": e.v+1, "time": e.time}
                          for i, e in enumerate(self.edges)]}

    @classmethod
    def from_dict(cls, data):
        try:
            if data["schema"] != "lc-grid-v1" or type(data["depot"]) is not int or data["depot"] != 1:
                raise ValidationError("schema=lc-grid-v1, depot=1이 필요합니다.")
            for key in ("nodes", "edges"):
                if any(type(row["id"]) is not int or row["id"] != i+1 for i, row in enumerate(data[key])):
                    raise ValidationError(f"{key}: 행 번호는 1부터 연속이어야 합니다.")
            nodes = tuple(Node(r["x"], r["y"], r["house"], r["delay"]) for r in data["nodes"])
            edges = tuple(Edge(integer(r["u"], "u", 1)-1, integer(r["v"], "v", 1)-1, r["time"])
                          for r in data["edges"])
            units = data["units"]
            return cls(data["name"], nodes, edges, positive_fraction(units["tick_seconds"], "tick_seconds"),
                       positive_fraction(units["L_star_henry"], "L_star"),
                       positive_fraction(units["C_star_farad"], "C_star"))
        except (KeyError, TypeError) as error:
            raise ValidationError(f"지도 필드 누락 또는 형식 오류: {error}") from error


def load_map(path):
    return GridMap.from_dict(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def diamond_chain(blocks=2):
    integer(blocks, "blocks", 1)
    coords, roads = set(), set()
    for i in range(blocks):
        u, a, b, v = (i, i), (i+1, i), (i, i+1), (i+1, i+1)
        coords.update((u, a, b, v))
        roads.update((x, y) for x, y in ((u, a), (u, b), (a, v), (b, v)))
    ordered = sorted(coords, key=lambda xy: (xy[1], xy[0]))
    ids = {xy: i for i, xy in enumerate(ordered)}
    nodes = tuple(Node(x, y, (x, y) == (blocks, blocks), int((x, y) == (blocks, blocks)))
                  for x, y in ordered)
    edges = tuple(Edge(u, v, 1) for u, v in sorted(tuple(sorted((ids[x], ids[y]))) for x, y in roads))
    return GridMap(f"diamond_chain_{blocks}", nodes, edges)


def random_grid(width=3, height=2, houses=3, seed=7, max_time=3):
    for name, value in (("width", width), ("height", height), ("max_time", max_time)):
        integer(value, name, 1)
    integer(houses, "houses", 0)
    if houses >= width*height:
        raise ValidationError("집 수는 가게를 제외한 노드 수 이하여야 합니다.")
    rng = random.Random(seed)
    required = set(rng.sample(range(1, width*height), houses))
    nodes = tuple(Node(x, y, y*width+x in required, rng.randint(1, max_time) if y*width+x in required else 0)
                  for y in range(height) for x in range(width))
    edges = []
    for u, node in enumerate(nodes):
        for dx, dy in ((1, 0), (0, 1)):
            x, y = node.x+dx, node.y+dy
            if x < width and y < height:
                edges.append(Edge(u, y*width+x, rng.randint(1, max_time)))
    return GridMap(f"grid_{width}x{height}_seed{seed}", nodes, tuple(sorted(edges, key=lambda e: (e.u, e.v))))
