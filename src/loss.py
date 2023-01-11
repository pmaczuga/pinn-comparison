from typing import Tuple
from src.cpoints import CPoints
from src.func import InitialCondition
from src.pinn import PINN, dfdx, dfdt, f

import torch

class Loss:
    def __init__(self, 
                 cpoints: CPoints,
                 initial_condition: InitialCondition,
                 equation: str = 'linear', 
                 boundary_condition: str = 'reflective',
                 c: float = 1,
                 weight_f: float = 1.0, 
                 weight_b: float = 1.0, 
                 weight_i: float = 1.0,
                 adapt_weights: bool = False):
        self.cpoints = cpoints
        self.initial_condition = initial_condition
        self.boundary_condition = boundary_condition
        self.c = c
        self.equation = equation
        self.weight_f = weight_f
        self.weight_b = weight_b
        self.weight_i = weight_i
        self.adapt_weights = adapt_weights

    def __call__(self, pinn: PINN) -> torch.Tensor:
        return self.verbose(pinn)[0]

    def verbose(self, pinn: PINN) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual_loss = self._residual_loss(pinn)
        initial_loss = self._initial_loss(pinn)
        boundary_loss = self._boundary_loss(pinn)

        weight_f, weight_i, weight_b = self.weight_f, self.weight_i, self.weight_b
        if self.adapt_weights:
            loss = torch.tensor([residual_loss, initial_loss, boundary_loss])
            minimum = torch.min(loss)
            weight_f, weight_i, weight_b = loss / minimum

        final_loss = \
            weight_f * residual_loss + \
            weight_i * initial_loss + \
            weight_b * boundary_loss

        return (
            final_loss, 
            weight_f * residual_loss, 
            weight_i * initial_loss, 
            weight_b * boundary_loss
        )

    def _residual_loss(self, pinn: PINN) -> torch.Tensor:
        x, t = self.cpoints.residual()
        c = self.c
        if self.equation == 'linear':
            loss = dfdt(pinn, x, t, order=2) - c**2 * dfdx(pinn, x, t, order=2)
        else: 
            loss = dfdt(pinn, x, t, order=2) - c**2 * (dfdx(pinn, x, t, order=2) * f(pinn, x, t) + dfdx(pinn, x, t, order=1).pow(2))
        return loss.pow(2).mean()

    def _initial_loss(self, pinn: PINN):
        x, t = self.cpoints.init()
        f_initial: torch.Tensor = self.initial_condition(x)

        loss_f = f(pinn, x, t) - f_initial 
        loss_df = dfdt(pinn, x, t, order=1)

        return loss_f.pow(2).mean() + loss_df.pow(2).mean()

    def _boundary_loss(self, pinn: PINN):
        x_left, t_left = self.cpoints.boundary_left()
        x_right, t_right = self.cpoints.boundary_right()

        if self.boundary_condition == 'periodic':
            return (f(pinn, x_left, t_left) - f(pinn, x_right, t_right)).pow(2).mean()

        if self.boundary_condition == 'reflective':
            return dfdx(pinn, x_left, t_left).pow(2).mean() + dfdx(pinn, x_right, t_right).pow(2).mean()

        if self.boundary_condition == 'zero':
            return f(pinn, x_left, t_left).pow(2).mean() + f(pinn, x_right, t_right).pow(2).mean()

        raise ValueError(f"Wrong boundary_condition: {self.boundary_condition}")
