import torch
import torch.nn as nn
import os
import datetime

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
    {"boundary_condition": "zero", "a": 1.0, "c": 3.0, "phi": 4.0},
    {"boundary_condition": "reflective", "a": 1.0, "c": 3.0, "phi": 4.0},
    {"boundary_condition": "reflective", "a": 1.0, "c": 3.0, "phi": 6.0},
]
params_eq_abbr = ["eq1", "eq2", "eq3"]

params_act = ["tanh", "atanh", "sin", "swish", "sigmoid"]
params_act_abbr = ["tanh", "atanh", "sin", "swish", "sig"]

params_hard = [False, True]
params_hard_abbr = ["soft", "hard"]

params_weights = [False, True]
params_weights_abbr = ["cweight", "aweight"]

params_cpoints = [False, True]
params_cpoints_abbr = ["pconst", "platin"]

device = get_device()
paramss = []
results = []
i = 1
all = 120
for eq, eq_abbr in zip(params_eq, params_eq_abbr):
    for act, act_abbr in zip(params_act, params_act_abbr):
        for hard, hard_abbr in zip(params_hard, params_hard_abbr):
            for weights, weights_abbr in zip(params_weights, params_weights_abbr):
                for cpoints, cpoints_abbr in zip(params_cpoints, params_cpoints_abbr):
                    tag = f"{eq_abbr}_{act_abbr}_{hard_abbr}_{weights_abbr}_{cpoints_abbr}"
                    print(f"Training PINN: {i}/{all}")
                    os.mkdir(f"results/{tag}")
                    params = Params(activation=act, hard_constraint=hard, adapt_weights=weights, collocation_points=cpoints, **eq)
                    x_domain = (0, params.length)
                    t_domain = (0, params.total_time)
                    pinn, loss, result = train_pinn_from_params(params, tag, device, print_each=None)
                    exact = Exact.from_params(params)
                    # result = Result(tag, 1,2,3,4,5,6)
                    paramss.append(params)
                    results.append(result)
                    save_result(params, result)
                    save_pinn(pinn, tag)
                    save_loss(loss, tag)
                    save_solution_plot(pinn, exact, tag, x_domain, t_domain)
                    save_initial_plot(pinn, exact, tag, x_domain)
                    save_loss_plot(loss, tag)
                    save_anim(pinn, tag, x_domain, t_domain)
                    torch.cuda.empty_cache()
                    i = i + 1

save_results(paramss, results, "solved_problems")
