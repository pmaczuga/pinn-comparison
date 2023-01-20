import torch
import torch.nn as nn
import os
import datetime
import time

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
params, result = load_result("results/dir_c2_A1_phi4/dir_c2_A1_phi4_result.csv")
params.activation = "tanh"
params.collocation_points = 'latin'
params.epochs = 60000
x_domain = (0, params.length)
t_domain = (0, params.total_time)

t = time.time()
pinn, loss, result = train_pinn_from_params(params, tag, device)
print(f"It took: {time.time() - t} seconds")

exact = Exact.from_params(params)
save_result(params, result)
save_pinn(pinn, tag)
save_loss(loss, tag)
save_solution_plot(pinn, exact, tag, x_domain, t_domain)
save_initial_plot(pinn, exact, tag, x_domain)
save_loss_plot(loss, tag)
save_anim(pinn, tag, x_domain, t_domain)
