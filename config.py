# -----------------------------------------------------------------------------
# ------------------------------------PINN-------------------------------------
# -----------------------------------------------------------------------------

LENGTH = 1.                         # Domain size in x axis. Always starts at 0
TOTAL_TIME = 1.                     # Domain size in t axis. Always starts at 0
N_POINTS_X = 150                    # Number of points in x axis, where the PINN will calculate loss in each epoch
N_POINTS_T = 150                    # Number of points in t axis
N_POINTS_RAND = 22500               # Number of points when sampling randomly
N_POINTS_INIT = 150                 # Number of points, where the PINN will calculate initial loss
N_POINTS_BOUNDARY = 150             # Number of points, where the PINN will calculate boundary loss
WEIGHT_RESIDUAL = 1.0               # Weight of residual part of loss function. Only used when ADAPT_WEIGHTS is False
WEIGHT_INITIAL = 1.0                # Weight of initial part of loss function. Only used when ADAPT_WEIGHTS is False
WEIGHT_BOUNDARY = 1.0               # Weight of boundary part of loss function. Only used when ADAPT_WEIGHTS is False
ADAPT_WEIGHTS = False               # Whether to use adaptive weights. 
LAYERS = 4
NEURONS_PER_LAYER = 80
EPOCHS = 60_000
LEARNING_RATE = 0.002
ACTIVATION = 'tanh'                 # 'tanh', 'sin', 'sigmoid', 'swish', 'atanh'
C = 1.                              # Equation constant
A = 0.5                             # Amplitude
PHI = 2                             # Solution constant
EQUATION = 'linear'                 # Equation to be used. 'linear', 'nonlinear' or 'zero' (dont' use last)
BOUNDARY_CONDITION = 'reflective'   # 'zero', 'reflective' or 'periodic'
HARD_CONSTRAINT = False
EXACT = True                        # Whether to compare result to exact solution. Must be implemented below
COLLOCATION_POINTS = 'const'        # 'const', 'random', 'latin'

# -----------------------------------------------------------------------------
# ----------------------------------PLOTTING-----------------------------------
# -----------------------------------------------------------------------------

N_POINTS_PLOT = 300
DPI = 300
FPS = 60
RUNNING_AVG_WINDOW = 300
CMAP_SOL = "viridis"
CMAP_DIFF = "Wistia"
FORMAT = 'png'