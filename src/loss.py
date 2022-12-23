from typing import Tuple
from src.pinn import PINN, dfdx, dfdt, f
from config import *

import torch


def interior_loss(pinn: PINN, x: torch.Tensor, t: torch.tensor, equation):
    if equation == 'linear':
        loss = dfdt(pinn, x, t, order=2) - C**2 * dfdx(pinn, x, t, order=2)
    else: 
        loss = dfdt(pinn, x, t, order=2) - C**2 * (dfdx(pinn, x, t, order=2) * f(pinn, x, t) + dfdx(pinn, x, t, order=1).pow(2))
    return loss.pow(2).mean()

def initial_loss(pinn: PINN, x: torch.Tensor, t_domain: Tuple[float, float], initial_condition):
    f_initial = initial_condition(x)
    t_initial = torch.ones_like(x) * t_domain[0]
    t_initial.requires_grad = True

    initial_loss_f = f(pinn, x, t_initial) - f_initial 
    initial_loss_df = dfdt(pinn, x, t_initial, order=1)

    return initial_loss_f.pow(2).mean() + initial_loss_df.pow(2).mean()

def boundary_loss(pinn: PINN, t: torch.tensor, x_domain: Tuple[float, float], boundary_condition: str):
    boundary_left = torch.ones_like(t, requires_grad=True) * x_domain[0]
    boundary_loss_left = dfdx(pinn, boundary_left, t)
    
    boundary_right = torch.ones_like(t, requires_grad=True) * x_domain[1]
    boundary_loss_right = dfdx(pinn, boundary_right, t)

    if boundary_condition == 'periodic':
        return (f(pinn, boundary_left, t) - f(pinn, boundary_right, t)).pow(2).mean()

    if boundary_condition == 'reflective':
        return boundary_loss_left.pow(2).mean() + boundary_loss_right.pow(2).mean()

    if boundary_condition == 'zero':
        return f(pinn, boundary_left, t).pow(2).mean() + f(pinn, boundary_right, t).pow(2).mean()

def compute_loss(
    pinn: PINN, 
    x: torch.Tensor = None, 
    t: torch.Tensor = None, 
    x_init: torch.Tensor = None, 
    initial_condition = None,
    t_boundary : torch.Tensor = None,
    x_domain: Tuple[float, float] = [0., 1.], 
    t_domain: Tuple[float, float] = [0., 1.],
    n_points: int = 10**4, 
    n_points_init: int = 150, 
    n_points_boundary: int = 150,
    weight_f = 1.0, 
    weight_b = 1.0, 
    weight_i = 1.0, 
    equation: str = 'linear', 
    boundary_condition: str = 'reflective',
    verbose = False,
    random = False,
    device = 'cpu'
) -> torch.float:

    if random:
        x = torch.rand(n_points, 1) * (x_domain[1] - x_domain[0]) + x_domain[0]
        x.requires_grad = True
        x = x.to(device)
        t = torch.rand(n_points, 1) * (t_domain[1] - t_domain[0]) + t_domain[0]
        t.requires_grad = True
        t = t.to(device)
    if random:
        x_init = torch.rand(n_points_init, 1) * (x_domain[1] - x_domain[0]) + x_domain[0]
        x_init.requires_grad = True
        x_init = x_init.to(device)
    if random:
        t_boundary = torch.rand(n_points_boundary, 1) * (t_domain[1] - t_domain[0]) + t_domain[0]
        t_boundary.requires_grad = True
        t_boundary = t_boundary.to(device)

    final_loss = \
        weight_f * interior_loss(pinn, x, t, equation) + \
        weight_i * initial_loss(pinn, x_init, t_domain, initial_condition) + \
        weight_b * boundary_loss(pinn, t_boundary, x_domain, boundary_condition)

    if not verbose:
        return final_loss
    else:
        return (
            final_loss, 
            interior_loss(pinn, x, t, equation), 
            initial_loss(pinn, x_init, t_domain), 
            boundary_loss(pinn, t_boundary, x_domain, boundary_condition)
        )