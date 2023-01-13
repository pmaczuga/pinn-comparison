from typing import Tuple, Union
from src.cpoints import get_cpoints
from src.func import Exact, InitialCondition
from src.loss import Loss
from src.pinn import PINN
import torch

from src.struct import Params, Result
from src.utils import l2_error, l2_error_init

def train_model(
    pinn: PINN,
    loss_fn: Loss,
    learning_rate: float = 0.01,
    max_epochs: int = 1_000,
    print_each: Union[int, None] = 1_000
) -> torch.Tensor:

    optimizer = torch.optim.Adam(pinn.parameters(), lr=learning_rate)
    loss_values = []
    for epoch in range(max_epochs):

        try:
            loss: torch.Tensor = loss_fn(pinn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_values.append(loss.item())
            if print_each and(epoch + 1) % print_each == 0:
                print(f"Epoch: {epoch + 1} - Loss: {float(loss):>7f}")

        except KeyboardInterrupt:
            break

    return torch.tensor(loss_values)
    

def train_pinn_from_params(params: Params, 
                           tag: str, 
                           device: torch.device, 
                           print_each: Union[int, None] = 1_000
) -> Tuple[PINN, torch.Tensor, Result]:
    print(f"Training network {tag}")

    # Prepare collocation points
    x_domain = (0.0, params.length)
    t_domain = (0.0, params.total_time)
    cpoints_class = get_cpoints(params.collocation_points)
    cpoints = cpoints_class(
        params.n_points_x, 
        params.n_points_t, 
        params.n_points_rand, 
        params.n_points_init, 
        params.n_points_boundary, 
        x_domain, 
        t_domain,
        device
    )
    initial_condition = InitialCondition.from_params(params)
    exact = Exact.from_params(params)

    # Create PINN instance
    pinn = PINN.from_params(params).to(device)
    loss_fn = Loss(
        cpoints,
        initial_condition,
        params.equation,
        params.boundary_condition,
        params.c,
        params.weight_residual,
        params.weight_boundary,
        params.weight_initial
    )

    # Train the PINN
    loss_values = train_model(
        pinn, loss_fn=loss_fn, learning_rate=params.learning_rate, max_epochs=params.epochs, print_each=print_each)


    # Result
    losses = loss_fn.verbose(pinn)

    # Move to cpu. All tensors from here on will be on cpu and we dont need 
    # extra speed, since training is over
    pinn = pinn.cpu()
    l2 = l2_error(pinn, exact, x_domain, t_domain)
    l2_init = l2_error_init(pinn, exact, x_domain)
    
    losses = list(map(lambda x: x.item(), losses))
    result = Result(tag, losses[0], losses[1], losses[2], losses[3], l2, l2_init)
    print(f'Total loss: \t{losses[0]:.5f}    ({losses[0]:.3E})')
    print(f'Residual loss: \t{losses[1]:.5f}    ({losses[1]:.3E})')
    print(f'Initial loss: \t{losses[2]:.5f}    ({losses[2]:.3E})')
    print(f'Boundary loss: \t{losses[3]:.5f}    ({losses[3]:.3E})')
    print(f'L2 error: \t{result.l2_error:.5f}    ({result.l2_error:.3E})')
    print(f'L2 error init: \t{result.l2_error_init:.5f}    ({result.l2_error_init:.3E})')
    
    return pinn, loss_values, result
