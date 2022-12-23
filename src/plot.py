from typing import Callable, List
from src.pinn import PINN, f
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

DPI = 300

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
    c = ax.pcolormesh(T, X, Z, cmap=cmap)
    fig.colorbar(c, ax=ax)
    return fig

def running_average(y, window=100):
    cumsum = np.cumsum(np.insert(y, 0, 0)) 
    return (cumsum[window:] - cumsum[:-window]) / float(window)

def plot_initial(pinn: PINN,
                 x: torch.Tensor,
                 initial_condition: function):
    u = f(pinn, x, torch.zeros_like(x))
    return plot_initial(x, u, initial_condition)

def plot_initial(x: torch.Tensor,
                 u: torch.Tensor,
                 initial_condition: Callable[[torch.Tensor], torch.Tensor]
                ):
    init = initial_condition(x)
    fig, ax = plt.subplots(dpi=DPI)
    ax.set_title("Initial condition difference")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.plot(x.detach(), init.detach(), label="Initial condition")
    ax.plot(x.detach(), u.detach(), '--' ,label="PINN solution")
    ax.legend()
    return fig

def plot_loss(loss: List[float], window: int = 100):
    average_loss = running_average(loss, window=window)
    fig, ax = plt.subplots(dpi=DPI)
    ax.set_title("Loss function (running average)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.plot(average_loss)
    ax.set_yscale('log')
    return fig 

def plot_solution(x: torch.Tensor, 
                  t: torch.Tensor, 
                  n_points_x: int, 
                  n_points_t: int):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, dpi=DPI)
    color = plot_color(z.cpu(), x.cpu(), t.cpu(), N_POINTS_X, N_POINTS_T, title="PINN solution (" + equation + " equation)")
