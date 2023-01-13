from __future__ import annotations

import torch

from src.struct import Params

class Exact:
    """
    Exact solution of PINN. Instance can be called as function.

    For 'zero' boundary condition:
        f(x, t) = a * sin(phi * pi * x) * sin(c * phi * pi * t)
    For 'reflective' boundary condition:
        f(x, t) = a * cos(phi * pi * x) * cos(c * phi * pi * t)
    
    Parameters:
        - bc - boundary condition: 'zero' or 'reflective'
        - a - amplitude
        - phi - period = 2 / phi
        - c - parameter

    Examples:
        >>> exact = Exact('reflective', 1.0, 1.0, 1.0)
        >>> x = torch.randn(16)
        >>> exact(x)
    """
    def __init__(self, bc: str, a: float = 1.0, phi: float = 1.0, c: float = 1.0):
        self.a = a
        self.phi = phi
        self.c = c
        if bc == 'zero':
            self.tri = torch.sin
        elif bc == 'reflective':
            self.tri = torch.cos 
        else:
            raise ValueError(f"Wrong argument for bc: {bc}")

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        tri = self.tri
        a = self.a
        phi = self.phi
        c = self.c
        return a * tri(phi * torch.pi * x) * torch.cos(c * phi * torch.pi * t)

    @classmethod
    def from_params(cls, params: Params) -> Exact:
        return cls(params.boundary_condition, params.a, params.phi, params.c)


class InitialCondition:
    """
    Initial condition of PINN. Instance can be called as function.
    
    Parameters:
        - equation - 'linear', 'nonlinear'
        - bc - boundary condition: 'zero' or 'reflective'
        - a - amplitude
        - phi - period = 2 / phi

    Examples:
        >>> init = InitialCondition('linear', 'reflective', 1.0, 1.0)
        >>> x = torch.randn(16)
        >>> init(x)
    """
    def __init__(self, equation: str = 'zero', bc: str = 'zero', a: float = 1.0, phi: float = 1.0):
        if equation == 'zero':
            self.eq = lambda x: torch.zeros_like(x)
        elif equation == 'linear' and bc == 'reflective':
            self.eq = lambda x: a * torch.cos( phi * torch.pi * x)
        elif equation == 'linear' and bc == 'zero':
            self.eq = lambda x: a * torch.sin( phi * torch.pi * x)
        elif equation == 'nonlinear':
            self.eq = lambda x: a * torch.exp(-((x-0.5)*10)**2)
        else:
            raise ValueError(f"Wrong combination of arguments: equation ({equation}) and bc ({bc})")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.eq(x)

    @classmethod
    def from_params(cls, params: Params) -> InitialCondition:
        return cls(params.equation, params.boundary_condition, params.a, params.phi)
