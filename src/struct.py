from typing import List
from config import *
import pandas as pd

class Struct:
    def to_pandas(self):
        # d = self.__dict__
        # new_d = {k: d[k] for k in set(list(d.keys())) - set(["to_pandas"])}
        return pd.DataFrame(self.__dict__, index=[0])

class Params(Struct):
    def __init__(self, **kwargs):
        self.length: float = kwargs.get("length", LENGTH)
        self.total_time: float = kwargs.get("total_time", TOTAL_TIME)
        self.n_points_x = kwargs.get("n_points_x", N_POINTS_X)
        self.n_points_t = kwargs.get("n_points_t", N_POINTS_T)
        self.n_points_rand = kwargs.get("n_points_rand", N_POINTS_RAND)
        self.n_points_init = kwargs.get("n_points_init", N_POINTS_INIT)
        self.n_points_boundary = kwargs.get("n_points_boundary", N_POINTS_BOUNDARY)
        self.weight_residual = kwargs.get("weight_residual", WEIGHT_RESIDUAL)
        self.weight_initial = kwargs.get("weight_initial", WEIGHT_INITIAL)
        self.weight_boundary = kwargs.get("weight_boundary", WEIGHT_BOUNDARY)
        self.layers = kwargs.get("layers", LAYERS)
        self.neurons_per_layer = kwargs.get("neurons_per_layer", NEURONS_PER_LAYER)
        self.epochs = kwargs.get("epochs", EPOCHS)
        self.learning_rate = kwargs.get("learning_rate", LEARNING_RATE)
        self.activation = kwargs.get("activation", ACTIVATION)
        self.c = kwargs.get("c", C)
        self.a = kwargs.get("a", A)
        self.phi = kwargs.get("phi", PHI)
        self.equation = kwargs.get("equation", EQUATION)
        self.boundary_condition = kwargs.get("boundary_condition", BOUNDARY_CONDITION)
        self.hard_constraint = kwargs.get("hard_constraint", HARD_CONSTRAINT)
        self.exact = kwargs.get("exact", EXACT)
        self.collocation_points = kwargs.get("collocation_points", COLLOCATION_POINTS)

class Result(Struct):
    def __init__(self,
                 tag: str,
                 loss: float,
                 loss_residual: float,
                 loss_initial: float,
                 loss_boundary: float,
                 l2_error: float,
                 l2_error_init: float,
                 **kwargs):
        self.tag = tag
        self.loss = loss
        self.loss_residual = loss_residual
        self.loss_initial = loss_initial
        self.loss_boundary = loss_boundary
        self.l2_error = l2_error
        self.l2_error_init = l2_error_init

def to_pandas(params: List[Params], result: List[Result]):
    df_params = pd.concat([p.to_pandas() for p in params])
    df_result = pd.concat([r.to_pandas() for r in result])
    df = pd.concat([df_params, df_result], axis=1)
    return df.reset_index(drop=True)
