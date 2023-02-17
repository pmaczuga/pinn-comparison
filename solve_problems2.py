import torch
import torch.nn as nn
import os
import datetime
from src.domain import Domain

from src.struct import Params
from src.pinn import PINN
from src.utils import get_device
from src.func import Exact
from src.plot import save_anim, save_loss_plot, save_solution_plot, save_initial_plot
from src.train import train_pinn_from_params
from src.io import *

import matplotlib as mpl
mpl.use("Agg")

params_eq = [
    {"boundary_condition": "zero", "a": 1.0, "c": 2.0, "phi": 4.0},
    {"boundary_condition": "reflective", "a": 2.0, "c": 1.0, "phi": 4.0},
    {"boundary_condition": "reflective", "a": 1.0, "c": 2.0, "phi": 4.0},
]
params_eq_abbr = ["eq4", "eq5", "eq6"]

params_act = ["tanh", "atanh", "sin", "swish", "sigmoid"]
params_act_abbr = ["tanh", "atanh", "sin", "swish", "sig"]

device = get_device()
paramss = []
results = []
i = 1
all = 120
for eq, eq_abbr in zip(params_eq, params_eq_abbr):
    for act, act_abbr in zip(params_act, params_act_abbr):
        tag = f"{eq_abbr}_{act_abbr}"
        print(f"Training PINN: {i}/{all}")
        os.mkdir(f"results/{tag}")
        params = Params(activation=act, **eq)
        domain = Domain.from_params(params)
        pinn, loss, result = train_pinn_from_params(params, tag, device, print_each=None)
        exact = Exact.from_params(params)
        # result = Result(tag, 1,2,3,4,5,6)
        paramss.append(params)
        results.append(result)
        save_result(params, result)
        save_pinn(pinn, tag)
        save_loss(loss, tag)
        save_solution_plot(pinn, exact, tag, domain)
        save_initial_plot(pinn, exact, tag, domain.x)
        save_loss_plot(loss, tag)
        save_anim(pinn, tag, domain)
        torch.cuda.empty_cache()
        i = i + 1

save_results(paramss, results, "solved_problems")
