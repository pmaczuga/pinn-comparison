from __future__ import annotations

import torch
from torch import nn
from src.activation import get_activation

from src.func import InitialCondition
from src.struct import Params

class PINN(nn.Module):
    """Simple neural network accepting two features as input and returning a 
    single output
    
    In the context of PINNs, the neural network is used as universal function 
    approximator to approximate the solution of the differential equation
    """
    def __init__(self, 
                 num_hidden: int, 
                 dim_hidden: int, 
                 act: nn.Module = nn.Tanh(), 
                 initial_condition: InitialCondition = InitialCondition(),
                 hard_constraint=False):
        super().__init__()

        self.layer_in = nn.Linear(2, dim_hidden)
        self.layer_out = nn.Linear(dim_hidden, 1)

        num_middle = num_hidden - 1
        self.middle_layers = nn.ModuleList(
            [nn.Linear(dim_hidden, dim_hidden) for _ in range(num_middle)]
        )
        self.act = act
        self.initial_condition = initial_condition
        self.hard_constraint = hard_constraint

    def forward(self, x, t):

        x_stack = torch.cat([x, t], dim=1)        
        out = self.act(self.layer_in(x_stack))
        for layer in self.middle_layers:
            out = self.act(layer(out))
        logits = self.layer_out(out)


        if self.hard_constraint:
            logits = logits * t + self.initial_condition(x)
        
        return logits

    @classmethod
    def from_params(cls, params: Params) -> PINN:
        activation = get_activation(params.activation)
        initial_condition = InitialCondition.from_params(params)
        return cls(params.layers, params.neurons_per_layer, activation, initial_condition, params.hard_constraint)

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


def dfdt(pinn: PINN, x: torch.Tensor, t: torch.Tensor, order: int = 1) -> torch.Tensor:
    """Derivative with respect to the time variable of arbitrary order"""
    f_value = f(pinn, x, t)
    return df(f_value, t, order=order)


def dfdx(pinn: PINN, x: torch.Tensor, t: torch.Tensor, order: int = 1) -> torch.Tensor:
    """Derivative with respect to the spatial variable of arbitrary order"""
    f_value = f(pinn, x, t)
    return df(f_value, x, order=order)
