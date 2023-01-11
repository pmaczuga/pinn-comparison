import torch
import torch.nn as nn
import os
import datetime

from src.struct import Params
from src.pinn import PINN
from src.utils import get_device
from src.func import Exact
from src.plot import save_anim, save_solution_plot, save_initial_plot, save_loss_plot
from src.train import train_pinn_from_params
from src.io import *

device = get_device()
params, result = load_result("resources/sample-result.csv")
pinn = load_pinn(params, "resources/sample-state.pth")
loss = load_loss("resources/sample-loss.csv")
tag = result.tag
os.mkdir(f"results/{tag}")
exact = Exact.from_params(params)
x_domain = (0.0, params.length)
t_domain = (0.0, params.total_time)

save_result(params, result)
save_pinn(pinn, tag)
save_loss(loss, tag)
save_solution_plot(pinn, exact, tag, x_domain, t_domain)
save_initial_plot(pinn, exact, tag, x_domain)
save_loss_plot(loss, tag)
save_anim(pinn, tag, x_domain, t_domain)
