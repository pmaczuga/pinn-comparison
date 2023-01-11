from typing import Dict, Type
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

Tanh = nn.Tanh
Sigmoid = nn.Sigmoid

class Sin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(input)

class Swish(nn.Module):
    '''
    Implementation of Swish activation.

        swish(x) = x * sigmoid(beta * x)

    Shape:
        - Input: (N, *) where * means, any number of additional dimensions
        - Output: (N, *), same shape as the input

    Parameters:
        - beta - trainable parameter

    References:
        - See related paper:
        https://arxiv.org/pdf/1710.05941v2.pdf

    Examples:
        >>> act = Swish()
        >>> x = torch.randn(16)
        >>> act(x)
    '''

    def __init__(self, beta = None):
        '''
        Initialization.
        INPUT:
            - beta: trainable parameter
            beta is initialized with 1.0 value by default
        '''
        super().__init__()

        # initialize beta
        if beta == None:
            self.beta = Parameter(torch.tensor(1.0))
        else:
            self.beta = Parameter(torch.tensor(beta))
            
        self.beta.requires_grad = True 

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

class AdaptiveTanh(nn.Module):
    '''
    Implementation of adaptive Tanh activation.

        atanh(x) = tanh(alpha * x)

    Shape:
        - Input: (N, *) where * means, any number of additional dimensions
        - Output: (N, *), same shape as the input

    Parameters:
        - alpha - trainable parameter

    References:
        - See related paper:
        https://arxiv.org/pdf/1710.05941v2.pdf

    Examples:
        >>> act = AdaptiveTanh()
        >>> x = torch.randn(16)
        >>> act(x)
    '''

    def __init__(self, alpha = None):
        '''
        Initialization.
        INPUT:
            - alpha: trainable parameter
            alpha is initialized with 1.0 value by default
        '''
        super().__init__()

        # initialize alpha
        if alpha == None:
            self.alpha = Parameter(torch.tensor(1.0))
        else:
            self.alpha = Parameter(torch.tensor(alpha))
            
        self.alpha.requires_grad = True 

    def forward(self, x):
        return torch.tanh(self.alpha * x)

def get_activation(act: str) -> nn.Module:
    checker: Dict[str, nn.Module] = {
        "tanh": Tanh(),
        "sigmoid": Sigmoid(),
        "sin": Sin(),
        "swish": Swish(),
        "atanh": AdaptiveTanh(),
    }
    return checker[act]
