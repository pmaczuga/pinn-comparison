from __future__ import annotations

from abc import ABC
from typing import Dict, Tuple, Type
import torch
import numpy as np
from src.domain import Domain, Domain1D

from src.struct import Params

class CPoints(ABC):
    """Collocation points"""
    def __init__(self, 
                 n_points_x: int, 
                 n_points_t: int, 
                 n_points_rand: int, 
                 n_points_init: int, 
                 n_points_boundary: int, 
                 domain: Domain, 
                 device: torch.device):
        self.n_points_x: int = n_points_x
        self.n_points_t: int = n_points_t
        self.n_points_rand: int = n_points_rand
        self.n_points_init: int = n_points_init
        self.n_points_boundary: int = n_points_boundary
        self.domain: Domain = domain
        self.device: torch.device = device

    def residual(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplemented

    def init(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplemented

    def boundary_left(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplemented

    def boundary_right(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplemented

    @classmethod
    def get_class(cls, string: str) -> Type[CPoints]:
        checker: Dict[str, Type[CPoints]] = {
            "const": EquispacedCPoints,
            "rand": RandomCPoints,
            "latin": LatinHypercubeCPoints
        }
        return checker[string]

    @classmethod
    def from_params(cls, params: Params, device = torch.device("cpu")) -> CPoints:
        concrete_class = cls.get_class(params.collocation_points)
        return concrete_class(
            params.n_points_x,
            params.n_points_t,
            params.n_points_rand,
            params.n_points_init,
            params.n_points_boundary,
            Domain.from_params(params),
            device
        )


class EquispacedCPoints(CPoints):
    def __init__(self, 
                 n_points_x: int, 
                 n_points_t: int, 
                 n_points_random: int, 
                 n_points_init: int, 
                 n_points_boundary: int, 
                 domain: Domain, 
                 device: torch.device):
        super().__init__(n_points_x, n_points_t, n_points_random, n_points_init, n_points_boundary, domain, device)

        x_raw = torch.linspace(domain.x.l, domain.x.u, steps=n_points_x, requires_grad=True)
        t_raw = torch.linspace(domain.t.l, domain.t.u, steps=n_points_t, requires_grad=True)
        grids = torch.meshgrid(x_raw, t_raw, indexing="ij")

        x = grids[0].flatten().reshape(-1, 1).to(device)
        t = grids[1].flatten().reshape(-1, 1).to(device)

        x_init_raw = torch.linspace(domain.x.l, domain.x.u, steps=n_points_init, requires_grad=True)
        x_init = x_init_raw.reshape(-1, 1).to(device)
        t_init = torch.full_like(x_init, domain.t.l)
        t_init.requires_grad = True

        t_boundary_raw = torch.linspace(domain.t.l, domain.t.u, steps=n_points_boundary, requires_grad=True)
        t_boundary_left = t_boundary_raw.reshape(-1, 1).to(device)
        t_boundary_right = t_boundary_raw.reshape(-1, 1).to(device)
        x_boundary_left = torch.full_like(t_boundary_left, domain.x.l)
        x_boundary_left.requires_grad = True
        x_boundary_right = torch.full_like(t_boundary_right, domain.x.u)
        x_boundary_right.requires_grad = True

        self._residual = (x, t)
        self._init = (x_init, t_init)
        self._boundary_left = (x_boundary_left, t_boundary_left)
        self._boundary_right = (x_boundary_right, t_boundary_right)

    def residual(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._residual

    def init(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._init

    def boundary_left(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._boundary_left

    def boundary_right(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._boundary_right


class RandomCPoints(CPoints):
    def residual(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.FloatTensor(self.n_points_rand) \
            .uniform_(self.domain.x.l, self.domain.x.u) \
            .reshape(-1, 1) \
            .to(self.device)
        t = torch.FloatTensor(self.n_points_rand) \
            .uniform_(self.domain.t.l, self.domain.t.u) \
            .reshape(-1, 1) \
            .to(self.device)
        x.requires_grad = True
        t.requires_grad = True
        return (x, t)

    def init(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.FloatTensor(self.n_points_init) \
            .uniform_(self.domain.x.l, self.domain.x.u) \
            .reshape(-1, 1) \
            .to(self.device)
        t = torch.full_like(x, self.domain.t.l)
        x.requires_grad = True
        t.requires_grad = True
        return (x, t)

    def boundary_left(self) -> Tuple[torch.Tensor, torch.Tensor]:
        t_left = torch.FloatTensor(self.n_points_boundary) \
            .uniform_(self.domain.t.l, self.domain.t.u) \
            .reshape(-1, 1) \
            .to(self.device)
        x_left = torch.full_like(t_left, self.domain.x.l)
        t_left.requires_grad = True
        x_left.requires_grad = True
        return (x_left, t_left)

    def boundary_right(self) -> Tuple[torch.Tensor, torch.Tensor]:
        t_right = torch.FloatTensor(self.n_points_boundary) \
            .uniform_(self.domain.t.l, self.domain.t.u) \
            .reshape(-1, 1) \
            .to(self.device)
        x_right = torch.full_like(t_right, self.domain.x.u)
        t_right.requires_grad = True
        x_right.requires_grad = True
        return (x_right, t_right)


from scipy.stats import qmc
class LatinHypercubeCPoints(CPoints):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampler_2d = qmc.LatinHypercube(d=2)
        self.sampler_1d = qmc.LatinHypercube(d=1)
        self.l_bounds = [self.domain.x.l, self.domain.t.l]
        self.u_bounds = [self.domain.x.u, self.domain.t.u]

    def residual(self) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.sampler_2d.random(n=self.n_points_rand)
        sample = qmc.scale(sample, self.l_bounds, self.u_bounds)
        sample = sample.astype("float32")
        x = torch.tensor(sample[:,0]) \
            .reshape(-1, 1) \
            .to(self.device)
        t = torch.tensor(sample[:,1]) \
            .reshape(-1, 1) \
            .to(self.device)
        x.requires_grad = True
        t.requires_grad = True
        return (x, t)

    def init(self) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.sampler_1d.random(n=self.n_points_init)
        sample = qmc.scale(sample, self.l_bounds[0], self.u_bounds[0])
        sample = sample.astype("float32")
        x = torch.tensor(sample) \
            .reshape(-1, 1) \
            .to(self.device)
        t = torch.full_like(x, self.domain.t.l)
        x.requires_grad = True
        t.requires_grad = True
        return (x, t)

    def boundary_left(self) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.sampler_1d.random(n=self.n_points_boundary)
        sample = qmc.scale(sample, self.l_bounds[1], self.u_bounds[1])
        sample = sample.astype("float32")
        t_left = torch.tensor(sample) \
            .reshape(-1, 1) \
            .to(self.device)
        x_left = torch.full_like(t_left, self.domain.x.l)
        t_left.requires_grad = True
        x_left.requires_grad = True
        return (x_left, t_left)

    def boundary_right(self) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.sampler_1d.random(n=self.n_points_boundary)
        sample = qmc.scale(sample, self.l_bounds[1], self.u_bounds[1])
        sample = sample.astype("float32")
        t_right = torch.tensor(sample) \
            .reshape(-1, 1) \
            .to(self.device)
        x_right = torch.full_like(t_right, self.domain.x.u)
        t_right.requires_grad = True
        x_right.requires_grad = True
        return (x_right, t_right)

def default_cpoints() -> CPoints:
    return EquispacedCPoints(100, 100, 10000, 100, 100, Domain(Domain1D(0, 1), Domain1D(0, 1)), torch.device("cpu"))

def get_cpoints(cpoints: str) -> Type[CPoints]:
    checker: Dict[str, Type[CPoints]] = {
        "const": EquispacedCPoints,
        "rand": RandomCPoints,
        "latin": LatinHypercubeCPoints
    }
    return checker[cpoints]
