from typing import List, Tuple
import pandas as pd
import torch
from src.pinn import PINN

from src.utils import fname
from src.struct import Params, Result, to_pandas

def save_loss(loss: torch.Tensor, tag: str):
    filename = fname(tag, 'loss', 'csv')
    loss_df = pd.DataFrame(loss.numpy())
    loss_df.to_csv(filename, index=False)

def load_loss(filename) -> torch.Tensor:
    loss_df = pd.read_csv(filename)
    return torch.tensor(loss_df.to_numpy())

def save_result(params: Params, result: Result, tag=None):
    tag = result.tag if tag == None else tag
    filename = fname(result.tag, 'result', 'csv')
    df = to_pandas([params], [result])
    df.to_csv(filename, index=False)

def save_results(params: List[Params], results: List[Result], tag: str):
    filename = f"results/{tag}.csv"
    df = to_pandas(params, results)
    df.to_csv(filename, index=False)

def load_results(filename: str) -> Tuple[List[Params], List[Result]]:
    df = pd.read_csv(filename)
    dicts = df.to_dict('records')
    params = []
    results = []
    for d in dicts:
        str_d = {str(key):value for (key,value) in d.items()}
        params.append(Params(**str_d))
        results.append(Result(**str_d))
    return params, results

def load_result(filename: str) -> Tuple[Params, Result]:
    ps, rs = load_results(filename)
    return ps[0], rs[0]

def save_pinn(pinn: PINN, tag: str):
    filename = fname(tag, 'state', 'pth')
    torch.save(pinn.state_dict(), filename)

def load_pinn(params: Params, filename) -> PINN:
    pinn = PINN.from_params(params)
    pinn.load_state_dict(torch.load(filename))
    return pinn

