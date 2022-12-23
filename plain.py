from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial
import time
import os
from typing import Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

LENGTH = 1.                     # Domain size in x axis. Always starts at 0
TOTAL_TIME = 1.                 # Domain size in t axis. Always starts at 0
N_POINTS_X = 150                # Number of points in x axis, where the PINN will calculate loss in each epoch
N_POINTS_T = 150                # Number of points in t axis
N_POINTS_INIT = 150             # Number of points, where the PINN will calculate initial loss
N_POINTS_BOUNDARY = 150         # Number of points, where the PINN will calculate boundary loss
WEIGHT_INTERIOR = 1.0           # Weight of interior part of loss function
WEIGHT_INITIAL = 1.0            # Weight of initial part of loss function
WEIGHT_BOUNDARY = 1.0           # Weight of boundary part of loss function
LAYERS = 4
NEURONS_PER_LAYER = 40
EPOCHS = 20_000
LEARNING_RATE = 0.005
C = 1.                          # Equation constant
A = 0.5                         # Amplitude
EQUATION = 0                    # Equation to be used. 0 - linear, 1 - nonlinear
BOUNDARY_CONDITION = 0          # 0 - zero, 1 - reflectice, 2 - peridoic
EXACT = True                    # Whether to compare result to exact solution. Must be implemented below
RANDOM = False                  # Whether to choose points randomly
SAVE_TO_FILE = True             # Save plots and download them in zip file
TAG = ""

def initial_condition(x) -> torch.Tensor:
    # res = 0.2 * torch.exp(-((x-0.5)*10)**2)
    # new_x = (x-0.5)*10
    # mapped_x = 1.0 - torch.floor(torch.abs(new_x))
    # res = torch.cos((x-0.5)*10) if x >= (x-0.5)*10 >= -torch.pi/2.0 and (x-0.5)*10 <= torch.pi/2.0 else torch.zeros_like(x)
    res = 0.5*torch.sin( 2*torch.pi * x)
    # res = A * torch.cos( 2*np.pi * x)
    # res1 = -torch.sign(x-0.2)/2 + 0.5
    # res2 = -torch.cos(2*torch.pi * x*5) / 10 + 0.1
    # res = res1 * res2
    # res = torch.zeros_like(x)
    return res

ix = torch.linspace(0, 1, 200)
iu = initial_condition(ix)
plt.plot(ix, iu)

def exact(x, t):
    return A * torch.sin(2*torch.pi*x) * torch.sin(C*2*torch.pi*t)

bc = {0: 'zero', 1: 'reflective', 2: 'periodic'}[BOUNDARY_CONDITION]
equation = 'linear' if EQUATION == 0 else 'nonlinear'
filename = equation + "-" + time.strftime("%y%m%d-%H%M%S")
os.mkdir(filename)

if SAVE_TO_FILE:
    file = open(filename + "/" + filename + ".txt", "w")
    file.write(f"equation\t{equation}\n")
    file.write(f"boundary condition\t{bc}\n")
    file.write(f"length\t{LENGTH}\n")
    file.write(f"time\t{TOTAL_TIME}\n")
    file.write(f"n points x\t{N_POINTS_X}\n")
    file.write(f"n points t\t{N_POINTS_T}\n")
    file.write(f"n points init\t{N_POINTS_INIT}\n")
    file.write(f"n points boundary\t{N_POINTS_BOUNDARY}\n")
    file.write(f"weight interior\t{WEIGHT_INTERIOR}\n")
    file.write(f"weight initial\t{WEIGHT_INITIAL}\n")
    file.write(f"weight boundary\t{WEIGHT_BOUNDARY}\n")
    file.write(f"layers\t{LAYERS}\n")
    file.write(f"neurons per layer\t{NEURONS_PER_LAYER}\n")
    file.write(f"epochs\t{EPOCHS}\n")
    file.write(f"learning rate\t{LEARNING_RATE}\n")
    file.write(f"c\t{C}\n")
    file.write(f"random\t{RANDOM}\n")

class PINN(nn.Module):
    """Simple neural network accepting two features as input and returning a single output
    
    In the context of PINNs, the neural network is used as universal function approximator
    to approximate the solution of the differential equation
    """
    def __init__(self, num_hidden: int, dim_hidden: int, act=nn.Tanh()):

        super().__init__()

        self.layer_in = nn.Linear(2, dim_hidden)
        self.layer_out = nn.Linear(dim_hidden, 1)

        num_middle = num_hidden - 1
        self.middle_layers = nn.ModuleList(
            [nn.Linear(dim_hidden, dim_hidden) for _ in range(num_middle)]
        )
        self.act = act

    def forward(self, x, t):

        x_stack = torch.cat([x, t], dim=1)        
        out = self.act(self.layer_in(x_stack))
        for layer in self.middle_layers:
            out = self.act(layer(out))
        logits = self.layer_out(out)


        
        # logits = logits * t + initial_condition(x)
        
        return logits

def f(pinn: PINN, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Compute the value of the approximate solution from the NN model"""
    return pinn(x, t)


def df(output: torch.Tensor, input: torch.Tensor, order: int = 1) -> torch.Tensor:
    """Compute neural network derivative with respect to input features using PyTorch autograd engine"""
    df_value = output
    for _ in range(order):
        df_value = torch.autograd.grad(
            df_value,
            input,
            grad_outputs=torch.ones_like(input),
            create_graph=True,
            retain_graph=True,
        )[0]

    return df_value


def dfdt(pinn: PINN, x: torch.Tensor, t: torch.Tensor, order: int = 1):
    """Derivative with respect to the time variable of arbitrary order"""
    f_value = f(pinn, x, t)
    return df(f_value, t, order=order)


def dfdx(pinn: PINN, x: torch.Tensor, t: torch.Tensor, order: int = 1):
    """Derivative with respect to the spatial variable of arbitrary order"""
    f_value = f(pinn, x, t)
    return df(f_value, x, order=order)

def interior_loss(pinn: PINN, x: torch.Tensor, t: torch.tensor, equation):
    if equation == 'linear':
        loss = dfdt(pinn, x, t, order=2) - C**2 * dfdx(pinn, x, t, order=2)
    else: 
        loss = dfdt(pinn, x, t, order=2) - C**2 * (dfdx(pinn, x, t, order=2) * f(pinn, x, t) + dfdx(pinn, x, t, order=1).pow(2))
    return loss.pow(2).mean()

def initial_loss(pinn: PINN, x: torch.Tensor, t_domain: Tuple[float, float]):
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
        weight_i * initial_loss(pinn, x_init, t_domain) + \
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

def train_model(
    nn_approximator: PINN,
    loss_fn: Callable,
    learning_rate: int = 0.01,
    max_epochs: int = 1_000
) -> PINN:

    optimizer = torch.optim.Adam(nn_approximator.parameters(), lr=learning_rate)
    loss_values = []
    for epoch in range(max_epochs):

        try:

            loss: torch.Tensor = loss_fn(nn_approximator)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_values.append(loss.item())
            if (epoch + 1) % 1000 == 0:
                print(f"Epoch: {epoch + 1} - Loss: {float(loss):>7f}")

        except KeyboardInterrupt:
            break

    return nn_approximator, np.array(loss_values)


def check_gradient(nn_approximator: PINN, x: torch.Tensor, t: torch.Tensor) -> bool:

    eps = 1e-4
    
    dfdx_fd = (f(nn_approximator, x + eps, t) - f(nn_approximator, x - eps, t)) / (2 * eps)
    dfdx_autodiff = dfdx(nn_approximator, x, t, order=1)
    is_matching_x = torch.allclose(dfdx_fd.T, dfdx_autodiff.T, atol=1e-2, rtol=1e-2)

    dfdt_fd = (f(nn_approximator, x, t + eps) - f(nn_approximator, x, t - eps)) / (2 * eps)
    dfdt_autodiff = dfdt(nn_approximator, x, t, order=1)
    is_matching_t = torch.allclose(dfdt_fd.T, dfdt_autodiff.T, atol=1e-2, rtol=1e-2)
    
    eps = 1e-2

    d2fdx2_fd = (f(nn_approximator, x + eps, t) - 2 * f(nn_approximator, x, t) + f(nn_approximator, x - eps, t)) / (eps ** 2)
    d2fdx2_autodiff = dfdx(nn_approximator, x, t, order=2)
    is_matching_x2 = torch.allclose(d2fdx2_fd.T, d2fdx2_autodiff.T, atol=1e-2, rtol=1e-2)

    d2fdt2_fd = (f(nn_approximator, x, t + eps) - 2 * f(nn_approximator, x, t) + f(nn_approximator, x, t - eps)) / (eps ** 2)
    d2fdt2_autodiff = dfdt(nn_approximator, x, t, order=2)
    is_matching_t2 = torch.allclose(d2fdt2_fd.T, d2fdt2_autodiff.T, atol=1e-2, rtol=1e-2)
    
    return is_matching_x and is_matching_t and is_matching_x2 and is_matching_t2

def plot_solution(pinn: PINN, x: torch.Tensor, t: torch.Tensor, figsize=(8, 6), dpi=100):

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    x_raw = torch.unique(x).reshape(-1, 1)
    t_raw = torch.unique(t)
        
    def animate(i):

        if not i % 10 == 0:
            t_partial = torch.ones_like(x_raw) * t_raw[i]
            f_final = f(pinn, x_raw, t_partial)
            ax.clear()
            ax.plot(
                x_raw.detach().numpy(), f_final.detach().numpy(), label=f"Time {float(t[i])}"
            )
            ax.set_ylim(-1, 1)
            ax.legend()

    n_frames = t_raw.shape[0]
    return FuncAnimation(fig, animate, frames=n_frames, interval=100, repeat=False)

def plot_color(z: torch.Tensor, x: torch.Tensor, t: torch.Tensor, n_points_x, n_points_t, title, figsize=(8, 6), dpi=100, cmap="viridis"):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    z_raw = z.detach().cpu().numpy()
    x_raw = x.detach().cpu().numpy()
    t_raw = t.detach().cpu().numpy()
    X = x_raw.reshape(n_points_x, n_points_t)
    T = t_raw.reshape(n_points_x, n_points_t)
    Z = z_raw.reshape(n_points_x, n_points_t)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("x")
    ax.set_ylabel("x")
    c = ax.pcolormesh(T, X, Z, cmap=cmap)
    fig.colorbar(c, ax=ax)

    return fig

def running_average(y, window=100):
    cumsum = np.cumsum(np.insert(y, 0, 0)) 
    return (cumsum[window:] - cumsum[:-window]) / float(window)

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

u_init = initial_condition(x_init_raw)

fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
ax.set_title("Initial condition points")
ax.set_xlabel("x")
ax.set_ylabel("u")
ax.scatter(x_init_raw.detach(), u_init.detach(), s=2)

if SAVE_TO_FILE:
    fig.savefig(filename + "/" + filename + "-initial.png")

pinn = PINN(LAYERS, NEURONS_PER_LAYER, act=nn.Tanh()).to(device)
# assert check_gradient(nn_approximator, x, t)

# train the PINN
loss_fn = partial(
    compute_loss, 
    x = x, 
    t = t, 
    x_init = x_init,
    t_boundary = t_boundary,
    x_domain = x_domain,
    t_domain = t_domain,
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

if RANDOM:
    x_test = torch.rand(N_POINTS_X * N_POINTS_T, 1) * (x_domain[1] - x_domain[0]) + x_domain[0]
    t_test = torch.rand(N_POINTS_X * N_POINTS_T, 1) * (t_domain[1] - t_domain[0]) + t_domain[0]
    x_init_test = torch.rand(N_POINTS_INIT, 1) * (x_domain[1] - x_domain[0]) + x_domain[0]
    t_init_test = torch.zeros_like(x_init_test)
    t_boundary_test = torch.rand(N_POINTS_BOUNDARY, 1) * (t_domain[1] - t_domain[0]) + t_domain[0]
    x_boundary_left = torch.ones_like(t_boundary_test) * x_domain[0]
    x_boundary_right = torch.ones_like(t_boundary_test) * x_domain[1]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.scatter(x_test, t_test, s=0.2)
    ax.scatter(x_init_test, t_init_test, s=0.5, c='red')
    ax.scatter(torch.cat([x_boundary_left, x_boundary_right], axis=1), torch.cat([t_boundary_test, t_boundary_test], axis=1), s=0.5, c='lime')
    ax.legend(["Interior", "Initial", "Boundary"])
    ax.set_title("Random points distribution")

losses = compute_loss(pinn.to(device), x=x, t=t, x_init=x_init, t_boundary=t_boundary, x_domain=x_domain, t_domain=t_domain, equation=equation, verbose=True)
print(f'Total loss: \t{losses[0]:.5f}    ({losses[0]:.3E})')
print(f'Interior loss: \t{losses[1]:.5f}    ({losses[1]:.3E})')
print(f'Initial loss: \t{losses[2]:.5f}    ({losses[2]:.3E})')
print(f'Boundary loss: \t{losses[3]:.5f}    ({losses[3]:.3E})')

if SAVE_TO_FILE:
    file.write(f'total loss\t{losses[0]:.4E}\n')
    file.write(f'interior loss\t{losses[1]:.4E}\n')
    file.write(f'initial loss\t{losses[2]:.4E}\n')
    file.write(f'boundary loss\t{losses[3]:.4E}\n')
    file.close()

average_loss = running_average(loss_values, window=100)
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
ax.set_title("Loss function (running average)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.plot(average_loss)
ax.set_yscale('log')

if SAVE_TO_FILE:
    fig.savefig(filename + "/" + filename + "-loss.png")

pinn_init = f(pinn.to(device), x_init, torch.zeros_like(x_init))
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
ax.set_title("Initial condition difference")
ax.set_xlabel("x")
ax.set_ylabel("u")
ax.plot(x_init_raw.detach(), u_init.detach(), label="Initial condition")
ax.plot(x_init_raw.detach(), pinn_init.flatten().detach().cpu(), '--' ,label="PINN solution")
ax.legend()

if SAVE_TO_FILE:
    fig.savefig(filename + "/" + filename + "-diff.png")

z = f(pinn.to(device), x, t)
color = plot_color(z.cpu(), x.cpu(), t.cpu(), N_POINTS_X, N_POINTS_T, title="PINN solution (" + equation + " equation)")

if SAVE_TO_FILE:
    color.savefig(filename + "/" + filename + "-solution.png")

if EXACT:
    color = plot_color(exact(x, t).cpu() , x.cpu(), t.cpu(), N_POINTS_X, N_POINTS_T, title="Exact solution (" + equation + " equation)")
    if SAVE_TO_FILE:
        color.savefig(filename + "/" + filename + "-exact.png")

if EXACT:
    error = torch.abs(z - exact(x, t)).cpu()
    color = plot_color(error , x.cpu(), t.cpu(), N_POINTS_X, N_POINTS_T, title="Error (" + equation + " equation)", cmap="Wistia")

    if SAVE_TO_FILE:
        color.savefig(filename + "/" + filename + "-error.png")

import matplotlib.animation as animation
ani = plot_solution(pinn_trained.cpu(), x.cpu(), t.cpu())
writervideo = animation.FFMpegWriter(fps=60)
if SAVE_TO_FILE:
    ani.save(filename + "/" + filename + '-anim.mp4', writer=writervideo)
