import torch
import torch.nn as nn
import os
import datetime
from src.cpoints import get_cpoints

from src.struct import Params
from src.pinn import PINN
from src.utils import get_device, l2_error, l2_error_init
from src.func import Exact, InitialCondition
from src.plot import save_anim, save_solution_plot, save_initial_plot, save_loss_plot
from src.train import train_pinn_from_params
from src.io import *
from src.loss import Loss

import os

from src.io import *

device = torch.device("cpu")

filter = ("dir", "neu")

def cond(filename):
    return os.path.basename(filename).startswith(filter)

dirnames = [dirname[0] for dirname in os.walk("results") if cond(dirname[0])]
print(f"Number of results: {len(dirnames)}")
for dirname in dirnames:
    tag = os.path.basename(dirname)
    result_filename = os.path.join(dirname, f"{tag}_result.csv")
    pinn_filename = os.path.join(dirname, f"{tag}_state.pth")
    params, result = load_result(result_filename)
    pinn = load_pinn(params, pinn_filename)
    x_domain = (0.0, params.length)
    t_domain = (0.0, params.total_time)
    cpoints_class = get_cpoints(params.collocation_points)
    cpoints = cpoints_class(
        params.n_points_x, 
        params.n_points_t, 
        params.n_points_rand, 
        params.n_points_init, 
        params.n_points_boundary, 
        x_domain, 
        t_domain,
        device
    )
    initial_condition = InitialCondition.from_params(params)
    exact = Exact.from_params(params)

    loss_fn = Loss(
        cpoints,
        initial_condition,
        params.equation,
        params.boundary_condition,
        params.c,
        params.weight_residual,
        params.weight_boundary,
        params.weight_initial,
        adapt_weights=params.adapt_weights
    )
    losses = loss_fn.verbose(pinn)

    l2 = l2_error(pinn, exact, x_domain, t_domain)
    l2_init = l2_error_init(pinn, exact, x_domain)
    
    losses = list(map(lambda x: x.item(), losses))
    result = Result(tag, losses[0], losses[1], losses[2], losses[3], l2, l2_init)
    save_result(params, result)
