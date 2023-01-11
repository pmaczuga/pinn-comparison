import time
from typing import Tuple
from src.func import Exact
from src.pinn import PINN
from src.struct import Params
from config import *

import torch
from torch import linalg as LA

def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    return device

def time_as_string():
    time.strftime("%y%m%d-%H%M%S")

def fname(tag, name, format):
    return f"results/{tag}/{tag}_{name}.{format}"

def get_domains(params: Params) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    x_domain = (0.0, params.length)
    t_domain = (0.0, params.total_time)
    return x_domain, t_domain

def get_points(x_domain: Tuple[float, float], 
               t_domain: Tuple[float, float],
               n_points_x: int = N_POINTS_PLOT,
               n_points_t: int = N_POINTS_PLOT):
    x_raw = torch.linspace(x_domain[0], x_domain[1], steps=n_points_x).detach()
    t_raw = torch.linspace(t_domain[0], t_domain[1], steps=n_points_t).detach()
    grids = torch.meshgrid(x_raw, t_raw, indexing="ij")
    x = grids[0].flatten().reshape(-1, 1)
    t = grids[1].flatten().reshape(-1, 1)
    return x_raw, t_raw, x, t

def l2_error(pinn: PINN, 
            exact: Exact,
            x_domain: Tuple[float, float], 
            t_domain: Tuple[float, float],
            n_points_x: int = N_POINTS_PLOT,
            n_points_t: int = N_POINTS_PLOT):
    x_raw, t_raw, x, t = get_points(x_domain, t_domain, n_points_x, n_points_t)
    pinn_sol = pinn(x, t).reshape(n_points_x, n_points_t).detach()
    exact_sol = exact(x, t).reshape(n_points_x, n_points_t)
    diff = pinn_sol - exact_sol
    return LA.norm(diff, ord=2)

def l2_error_init(pinn: PINN,
                  exact: Exact,
                  x_domain: Tuple[float, float], 
                  n_points_x: int = N_POINTS_PLOT):
    x_init_raw = torch.linspace(x_domain[0], x_domain[1], steps=n_points_x)
    x_init = x_init_raw.reshape(-1, 1)
    pinn_init = pinn(x_init, torch.zeros_like(x_init)).flatten().detach()
    exact_init = exact(x_init, torch.zeros_like(x_init)).flatten()
    diff = pinn_init - exact_init
    return LA.norm(diff, ord=2)
