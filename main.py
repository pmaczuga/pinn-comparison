from functools import partial
import torch
from torch import nn
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from config import *
from src.loss import compute_loss
from src.pinn import PINN
from src.plot import running_average
from src.train import train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

bc = {0: 'zero', 1: 'reflective', 2: 'periodic'}[BOUNDARY_CONDITION]
equation = 'linear' if EQUATION == 0 else 'nonlinear'
filename = equation + "-" + time.strftime("%y%m%d-%H%M%S")
os.mkdir(filename)

def initial_condition(x) -> torch.Tensor:
    # res = 0.2 * torch.exp(-((x-0.5)*10)**2)
    # new_x = (x-0.5)*10
    # mapped_x = 1.0 - torch.floor(torch.abs(new_x))
    # res = torch.cos((x-0.5)*10) if x >= (x-0.5)*10 >= -torch.pi/2.0 and (x-0.5)*10 <= torch.pi/2.0 else torch.zeros_like(x)
    # res = torch.sin( 2*torch.pi * x)
    res = A * torch.cos( PHI * np.pi * x)
    # res1 = -torch.sign(x-0.2)/2 + 0.5
    # res2 = -torch.cos(2*torch.pi * x*5) / 10 + 0.1
    # res = res1 * res2
    # res = torch.zeros_like(x)
    return res

def exact(x, t):
    return A * torch.cos(PHI * torch.pi * x) * torch.cos( C * PHI * torch.pi * t)

def train_network():
    # Prepare collocation points
    x_domain = [0.0, LENGTH]
    t_domain = [0.0, TOTAL_TIME]

    x_raw = torch.linspace(x_domain[0], x_domain[1], steps=N_POINTS_X, requires_grad=True)
    t_raw = torch.linspace(t_domain[0], t_domain[1], steps=N_POINTS_T, requires_grad=True)
    grids = torch.meshgrid(x_raw, t_raw, indexing="ij")

    x = grids[0].flatten().reshape(-1, 1).to(device)
    t = grids[1].flatten().reshape(-1, 1).to(device)

    x_init_raw = torch.linspace(x_domain[0], x_domain[1], steps=N_POINTS_INIT, requires_grad=True)
    x_init = x_init_raw.reshape(-1, 1).to(device)
    # x_init = 0.5*((x_init-0.5*LENGTH)*2)**3 + 0.5

    t_boundary_raw = torch.linspace(t_domain[0], t_domain[1], steps=N_POINTS_BOUNDARY, requires_grad=True)
    t_boundary = t_boundary_raw.reshape(-1, 1).to(device)

    # Plot initial condition
    u_init = initial_condition(x_init_raw)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.set_title("Initial condition points")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.scatter(x_init_raw.detach(), u_init.detach(), s=2)

    if SAVE_TO_FILE:
        fig.savefig(filename + "/" + filename + "-initial.png")

    # Train the PINN
    pinn = PINN(LAYERS, NEURONS_PER_LAYER, act=nn.Tanh()).to(device)
    loss_fn = partial(
        compute_loss, 
        x = x, 
        t = t, 
        x_init = x_init,
        t_boundary = t_boundary,
        x_domain = x_domain,
        t_domain = t_domain,
        initial_condition = initial_condition,
        n_points = N_POINTS_X * N_POINTS_T,
        n_points_init = N_POINTS_INIT,
        n_points_boundary = N_POINTS_BOUNDARY,
        weight_f=WEIGHT_INTERIOR, 
        weight_i=WEIGHT_INTERIOR, 
        weight_b=WEIGHT_BOUNDARY,
        equation = equation,
        boundary_condition = bc,
        device = device,
        random = RANDOM
    )

    pinn_trained, loss_values = train_model(
        pinn, loss_fn=loss_fn, learning_rate=LEARNING_RATE, max_epochs=EPOCHS)

    losses = compute_loss(pinn.to(device), x=x, t=t, x_init=x_init, t_boundary=t_boundary, x_domain=x_domain, t_domain=t_domain, equation=equation, verbose=True, initial_condition=initial_condition)
    print(f'Total loss: \t{losses[0]:.5f}    ({losses[0]:.3E})')
    print(f'Interior loss: \t{losses[1]:.5f}    ({losses[1]:.3E})')
    print(f'Initial loss: \t{losses[2]:.5f}    ({losses[2]:.3E})')
    print(f'Boundary loss: \t{losses[3]:.5f}    ({losses[3]:.3E})')


    average_loss = running_average(loss_values, window=100)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.set_title("Loss function (runnig average)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.plot(average_loss)
    ax.set_yscale('log')

    if SAVE_TO_FILE:
        fig.savefig(filename + "/" + filename + "-loss.png")

def main():
    train_network()

if __name__ == "__main__":
    main()
