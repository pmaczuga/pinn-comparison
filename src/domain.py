from __future__ import annotations

from typing import Tuple

from src.struct import Params


class Domain1D:
    def __init__(self, low: float, high: float):
        self.l: float = low
        self.u: float = high

    def lower(self) -> float:
        return self.l

    def upper(self) -> float:
        return self.u

    def as_tuple(self) -> Tuple[float, float]:
        return self.l, self.u


class Domain:
    def __init__(self, x_domain: Domain1D, y_domain: Domain1D):
        self.x: Domain1D = x_domain
        self.t: Domain1D = y_domain

    def as_tuple(self) -> Tuple[Domain1D, Domain1D]:
        return self.x, self.t

    @classmethod
    def from_params(cls, params: Params) -> Domain:
        x_domain = Domain1D(0.0, params.length)
        t_domain = Domain1D(0.0, params.total_time)
        return cls(x_domain, t_domain)
    