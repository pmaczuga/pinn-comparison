from abc import ABC
from typing import Dict, Tuple, Type
import torch

class CPoints(ABC):
    """Collocation points"""
    def __init__(self, 
                 n_points_x: int, 
                 n_points_t: int, 
                 n_points_rand: int, 
                 n_points_init: int, 
                 n_points_boundary: int, 
                 x_domain: Tuple[float, float], 
                 t_domain: Tuple[float, float],
                 device: torch.device):
        self.x_domain = x_domain
        self.t_domain = t_domain
        self.n_points_x = n_points_x
        self.n_points_t = n_points_t
        self.n_points_rand = n_points_rand
        self.n_points_init = n_points_init
        self.n_points_boundary = n_points_boundary
        self.device = device

    def residual(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplemented

    def init(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplemented

    def boundary_left(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplemented

    def boundary_right(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplemented


class EquispacedCPoints(CPoints):
    def __init__(self, 
                 n_points_x: int, 
                 n_points_t: int, 
                 n_points_random: int, 
                 n_points_init: int, 
                 n_points_boundary: int, 
                 x_domain: Tuple[float, float], 
                 t_domain: Tuple[float, float],
                 device: torch.device):
        super().__init__(n_points_x, n_points_t, n_points_random, n_points_init, n_points_boundary, x_domain, t_domain, device)

        x_raw = torch.linspace(x_domain[0], x_domain[1], steps=n_points_x, requires_grad=True)
        t_raw = torch.linspace(t_domain[0], t_domain[1], steps=n_points_t, requires_grad=True)
        grids = torch.meshgrid(x_raw, t_raw, indexing="ij")

        x = grids[0].flatten().reshape(-1, 1).to(device)
        t = grids[1].flatten().reshape(-1, 1).to(device)

        x_init_raw = torch.linspace(x_domain[0], x_domain[1], steps=n_points_init, requires_grad=True)
        x_init = x_init_raw.reshape(-1, 1).to(device)
        t_init = torch.full_like(x_init, t_domain[0])
        t_init.requires_grad = True

        t_boundary_raw = torch.linspace(t_domain[0], t_domain[1], steps=n_points_boundary, requires_grad=True)
        t_boundary_left = t_boundary_raw.reshape(-1, 1).to(device)
        t_boundary_right = t_boundary_raw.reshape(-1, 1).to(device)
        x_boundary_left = torch.full_like(t_boundary_left, x_domain[0])
        x_boundary_left.requires_grad = True
        x_boundary_right = torch.full_like(t_boundary_right, x_domain[1])
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
            .uniform_(self.x_domain[0], self.x_domain[1]) \
            .reshape(-1, 1) \
            .to(self.device)
        t = torch.FloatTensor(self.n_points_rand) \
            .uniform_(self.t_domain[0], self.t_domain[1]) \
            .reshape(-1, 1) \
            .to(self.device)
        return (x, t)

    def init(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.FloatTensor(self.n_points_init) \
            .uniform_(self.x_domain[0], self.x_domain[1]) \
            .reshape(-1, 1) \
            .to(self.device)
        t = torch.full_like(x, self.t_domain[0])
        return (x, t)

    def boundary_left(self) -> Tuple[torch.Tensor, torch.Tensor]:
        t_left = torch.FloatTensor(self.n_points_boundary) \
            .uniform_(self.t_domain[0], self.t_domain[1]) \
            .reshape(-1, 1) \
            .to(self.device)
        x_left = torch.full_like(t_left, self.x_domain[0])
        return (x_left, t_left)

    def boundary_right(self) -> Tuple[torch.Tensor, torch.Tensor]:
        t_right = torch.FloatTensor(self.n_points_boundary) \
            .uniform_(self.t_domain[0], self.t_domain[1]) \
            .reshape(-1, 1) \
            .to(self.device)
        x_right = torch.full_like(t_right, self.x_domain[1])
        return (x_right, t_right)


# TODO
class LatinHypercubeCPoints(CPoints):
    pass

def default_cpoints() -> CPoints:
    return EquispacedCPoints(100, 100, 10000, 100, 100, (0,1), (0,1), torch.device("cpu"))

def get_cpoints(cpoints: str) -> Type[CPoints]:
    checker: Dict[str, Type[CPoints]] = {
        "const": EquispacedCPoints,
        "rand": RandomCPoints,
        "latin": LatinHypercubeCPoints
    }
    return checker[cpoints]
