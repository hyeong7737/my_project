"""Exact rational characteristic polynomials, including root multiplicities."""
from dataclasses import dataclass
from fractions import Fraction as F
import sympy as sp
from map_data import ValidationError


def exact_fraction(value):
    # Do not use sympify on user strings (it can evaluate Python expressions).
    if type(value) is int or isinstance(value, F):
        return F(value)
    if isinstance(value, sp.Rational):
        return F(int(value.p), int(value.q))
    if isinstance(value, str):
        try:
            return F(value)
        except (ValueError, ZeroDivisionError):
            pass
    raise ValidationError("정확한 정수·분수/소수 문자열이 필요합니다. 실수 근삿값은 허용하지 않습니다.")


@dataclass(frozen=True)
class SpectralTarget:
    coefficients: tuple

    def __post_init__(self):
        c = tuple(exact_fraction(x) for x in self.coefficients)
        if len(c) < 2 or c[0] != 1:
            raise ValidationError("차수 1 이상인 monic 특성다항식의 내림차순 계수가 필요합니다.")
        object.__setattr__(self, "coefficients", c)
        # Exact real-root isolation; multiplicities matter.
        intervals = self.poly.intervals()
        if sum(mult for _, mult in intervals) != self.n or self.poly.count_roots(-sp.oo, 0):
            raise ValidationError("현재 접지 LC 모델의 목표 고윳값은 모두 양의 실수여야 합니다.")

    @property
    def n(self):
        return len(self.coefficients)-1

    @property
    def poly(self):
        return sp.Poly.from_list([sp.Rational(c.numerator, c.denominator)
                                 for c in self.coefficients], sp.Symbol("lambda"))

    @property
    def trace(self):
        return -self.coefficients[1]

    @property
    def trace_square(self):
        c2 = self.coefficients[2] if self.n > 1 else F(0)
        return self.trace**2-2*c2

    def approximate_values(self):
        # Only guides the heuristic; never used to certify or prune exact search.
        values = []
        for (left, right), mult in self.poly.intervals(eps=sp.Rational(1, 10**14)):
            values.extend([float((left+right)/2)]*mult)
        return values

    def matches(self, matrix):
        return tuple(exact_fraction(c) for c in matrix.charpoly().all_coeffs()) == self.coefficients

    def to_dict(self):
        return {"schema": "lc-spectrum-target-v1",
                "characteristic_polynomial_coefficients": [str(c) for c in self.coefficients],
                "dimension": self.n, "trace": str(self.trace),
                "trace_square": str(self.trace_square)}

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValidationError("목표 스펙트럼은 JSON 객체여야 합니다.")
        data = data.get("spectrum", data.get("target", data))
        if not isinstance(data, dict):
            raise ValidationError("목표 스펙트럼 객체 형식이 잘못되었습니다.")
        if "characteristic_polynomial_coefficients" in data:
            c = data["characteristic_polynomial_coefficients"]
            if not isinstance(c, (list, tuple)):
                raise ValidationError("특성다항식 계수 목록이 필요합니다.")
            return cls(tuple(c))
        # Exact rational eigenvalues are a convenience. Irrational roots should
        # be supplied through their exact rational characteristic polynomial.
        if "exact_dimensionless_eigenvalues" in data:
            values = data["exact_dimensionless_eigenvalues"]
            if not isinstance(values, (list, tuple)) or not values:
                raise ValidationError("중복도를 포함한 정확한 고윳값 목록이 필요합니다.")
            coeff = [F(1)]
            for raw in values:
                v = exact_fraction(raw)
                new = [F(0)]*(len(coeff)+1)
                for i, c in enumerate(coeff):
                    new[i] += c
                    new[i+1] -= v*c
                coeff = new
            return cls(tuple(coeff))
        raise ValidationError("정확 특성다항식 또는 exact_dimensionless_eigenvalues가 필요합니다.")
