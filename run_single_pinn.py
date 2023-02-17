import torch
import torch.nn as nn
import os
import datetime
import time
from src.domain import Domain

from src.struct import Params
from src.pinn import PINN
from src.utils import get_device, time_as_string
from src.func import Exact
from src.plot import save_anim, save_loss_plot, save_solution_plot, save_initial_plot
from src.train import train_pinn_from_params
from src.io import *

device = get_device()
tag = f"{time_as_string()}"
os.mkdir(f"results/{tag}")
params = Params()
params, result = load_result("results/neu_c2_A1_phi4/neu_c2_A1_phi4_result.csv")
params.activation = "atanh"
params.collocation_points = 'const'
params.hard_constraint = True
params.epochs = 80000
params.learning_rate = 0.002
params.layers = 4
params.neurons_per_layer = 80
params.c = 1.
params.phi = 6
domain = Domain.from_params(params)
x_domain = (0, params.length)
t_domain = (0, params.total_time)

t = time.time()
pinn, loss, result = train_pinn_from_params(params, tag, device)
print(f"It took: {time.time() - t} seconds")

exact = Exact.from_params(params)
save_result(params, result)
save_pinn(pinn, tag)
save_loss(loss, tag)
save_solution_plot(pinn, exact, tag, domain)
save_initial_plot(pinn, exact, tag, domain.x)
save_loss_plot(loss, tag)
save_anim(pinn, tag, domain)
