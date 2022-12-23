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
NEURONS_PER_LAYER = 80
EPOCHS = 20_000
LEARNING_RATE = 0.005
C = 1.                          # Equation constant
A = 0.5                         # Amplitude
PHI = 2                         # Solution constant
EQUATION = 0                    # Equation to be used. 0 - linear, 1 - nonlinear
BOUNDARY_CONDITION = 1          # 0 - zero, 1 - reflectice, 1 - peridoic
EXACT = True                    # Whether to compare result to exact solution. Must be implemented below
RANDOM = False                  # Whether to choose points randomly
SAVE_TO_FILE = True            # Save plots and download them in zip file
TAG = ""