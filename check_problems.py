import torch
import torch.nn as nn
import os
import datetime

from src.struct import Params
from src.pinn import PINN
from src.utils import get_device
from src.func import Exact
from src.plot import save_anim, save_solution_plot, save_initial_plot
from src.train import train_pinn_from_params
from src.io import *

params_bc = ['zero', 'reflective']
params_bc_abbr = ["dir", "neu"]

params_a = [0.5, 1.0, 2.0]
params_a_abbr = ["05", "1", "2"]

params_c = [0.3, 0.5, 1.0, 2.0, 3.0]
params_c_abbr = ["03", "05", "1", "2", "3"]

params_phi = [2.0, 4.0, 6.0]
params_phi_abbr = ["2", "4", "6"]

device = get_device()
paramss = []
results = []
for bc, bc_abbr in zip(params_bc, params_bc_abbr):
    for a, a_abbr in zip(params_a, params_a_abbr):
        for c, c_abbr in zip(params_c, params_c_abbr):
            for phi, phi_abbr in zip(params_phi, params_phi_abbr):
                tag = f"{bc_abbr}_c{c_abbr}_A{a_abbr}_phi{phi_abbr}"
                os.mkdir(f"results/{tag}")
                params = Params(boundary_condition=bc, a=a, c=c, phi=phi)
                x_domain = (0, params.length)
                t_domain = (0, params.total_time)
                # pinn, loss, result = train_pinn_from_params(params, tag, device)
                result = Result(tag, 1,2,3,4,5,6)
                paramss.append(params)
                results.append(result)
                save_result(params, result)
                # save_pinn(pinn, tag)
                # save_loss(loss, tag)
                # save_solution_plot(pinn, exact, tag, x_domain, t_domain)
                # save_initial_plot(pinn, exact, tag, x_domain)
                # save_anim(pinn, tag, x_domain, t_domain)

save_results(paramss, results, "checked_problems")
