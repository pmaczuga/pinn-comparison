from typing import Tuple
from src.domain import Domain, Domain1D
from src.func import Exact
from src.pinn import PINN, f
from src.utils import get_points, fname
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import numpy as np

from config import *

# -----------------------------------------------------------------------------
# -----------------------------SOLUTION-COLOR-MAP------------------------------
# -----------------------------------------------------------------------------

def plot_color(x, t, sol, title, cmap="viridis") -> Figure:
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    c = ax.pcolormesh(t, x, sol, cmap=cmap)
    ax.axes.set_aspect('equal')
    fig.colorbar(c, ax=ax)
    return fig

def solution_plot(pinn: PINN, 
                  exact: Exact, 
                  tag: str, 
                  domain: Domain,
                  n_points_x: int = N_POINTS_PLOT,
                  n_points_t: int = N_POINTS_PLOT) -> Tuple[Figure, Figure, Figure]:
    x_raw, t_raw, x, t = get_points(domain, n_points_x, n_points_t)
    pinn_sol = pinn(x, t).reshape(n_points_x, n_points_t).detach()
    exact_sol = exact(x, t).reshape(n_points_x, n_points_t)
    diff = torch.abs(pinn_sol - exact_sol)
    pinn_fig = plot_color(x_raw, t_raw, pinn_sol, "PINN solution", cmap=CMAP_SOL)
    exact_fig = plot_color(x_raw, t_raw, exact_sol, "Exact solution", cmap=CMAP_SOL)
    diff_fig = plot_color(x_raw, t_raw, diff, "Difference", cmap=CMAP_DIFF)
    return pinn_fig, exact_fig, diff_fig

def save_solution_plot(pinn: PINN, 
                  exact: Exact, 
                  tag: str, 
                  domain: Domain,
                  n_points_x: int = N_POINTS_PLOT,
                  n_points_t: int = N_POINTS_PLOT,
                  format: str = FORMAT) -> Tuple[Figure, Figure, Figure]:
    pinn_fig, exact_fig, diff_fig = solution_plot(pinn, exact, tag, domain, n_points_x, n_points_t)

    pinn_fig.savefig(fname(tag, "pinn", format), format=format, bbox_inches='tight', dpi=DPI)
    exact_fig.savefig(fname(tag, "exact", format), format=format, bbox_inches='tight', dpi=DPI)
    diff_fig.savefig(fname(tag, "diff", format), format=format, bbox_inches='tight', dpi=DPI)

    return pinn_fig, exact_fig, diff_fig

# -----------------------------------------------------------------------------
# ------------------------------------LOSS-------------------------------------
# -----------------------------------------------------------------------------

def running_average(y, window=RUNNING_AVG_WINDOW) -> np.ndarray:
    cumsum = np.cumsum(np.insert(y, 0, 0)) 
    return (cumsum[window:] - cumsum[:-window]) / float(window)

def loss_plot(loss_values: torch.Tensor, tag: str) -> Figure:
    average_loss = running_average(loss_values)
    fig, ax = plt.subplots()
    ax.set_title("Loss function (runnig average)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.plot(average_loss)
    ax.set_yscale('log')
    return fig

def save_loss_plot(loss_values: torch.Tensor, tag: str) -> Figure:
    fig = loss_plot(loss_values, tag)
    fig.savefig(fname(tag, "loss", FORMAT), format=FORMAT, bbox_inches='tight', dpi=DPI)
    return fig

# -----------------------------------------------------------------------------
# ----------------------------------ANIMATION----------------------------------
# -----------------------------------------------------------------------------

def plot_anim(pinn: PINN, x: torch.Tensor, t: torch.Tensor, dpi=DPI) -> FuncAnimation:

    fig, ax = plt.subplots()
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


def save_anim(pinn: PINN, 
              tag: str,
              domain: Domain,
              n_points_x: int = N_POINTS_PLOT,
              n_points_t: int = N_POINTS_PLOT,
              fps: int = FPS) -> FuncAnimation:
    import matplotlib.animation as animation
    x_raw, t_raw, x, t = get_points(domain, n_points_x, n_points_t)
    ani = plot_anim(pinn, x, t)
    writer = animation.FFMpegWriter(fps=fps)
    name = fname(tag, "anim", "mp4")
    ani.save(name, writer=writer)
    return ani

# -----------------------------------------------------------------------------
# ------------------------------INITIAL-CONDITION------------------------------
# -----------------------------------------------------------------------------

def initial_plot(pinn: PINN, 
                 exact: Exact, 
                 tag: str,
                 x_domain: Domain1D, 
                 n_points_x: int = N_POINTS_PLOT) -> Figure:
    x_init_raw = torch.linspace(x_domain.l, x_domain.u, steps=n_points_x)
    x_init = x_init_raw.reshape(-1, 1)

    pinn_init = pinn(x_init, torch.zeros_like(x_init)).flatten().detach()
    exact_init = exact(x_init, torch.zeros_like(x_init)).flatten()

    fig, ax = plt.subplots()
    ax.set_title("Initial condition difference")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.plot(x_init_raw, exact_init, label="Initial condition")
    ax.plot(x_init_raw, pinn_init, '--' ,label="PINN solution")
    ax.legend()
    return fig

def save_initial_plot(pinn: PINN, 
                 exact: Exact, 
                 tag: str,
                 x_domain: Domain1D, 
                 n_points_x: int = N_POINTS_PLOT,
                 dpi: int = DPI,
                 format: str = FORMAT) -> Figure:
    fig = initial_plot(pinn, exact, tag, x_domain, n_points_x)

    name = fname(tag, "init", format)
    fig.savefig(name, dpi=dpi, format=format, bbox_inches='tight')
    return fig
