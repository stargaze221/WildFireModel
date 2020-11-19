'''
Hard coded parameter set
'''
import torch


OBSERVTAION_MATRIX = torch.Tensor([[0.98, 0.40, 0.01],[0.01, 0.59, 0.01],[0.01, 0.01, 0.98]])



TRANSITION_CNN_LAYER_WT = {}

param = torch.zeros(3,3,3,3)
# Probability to be normal state (output: 0) depends on the preivous state of the grid and its neighbors
ALPHA0 = 0.9999
BETA0 = 1 - ALPHA0
GAMMA0 = 0.1
param[0][0][:][:] = torch.Tensor([[GAMMA0*ALPHA0, GAMMA0*ALPHA0, GAMMA0*ALPHA0],[GAMMA0*ALPHA0, ALPHA0, GAMMA0*ALPHA0],[GAMMA0*ALPHA0, GAMMA0*ALPHA0, GAMMA0*ALPHA0]])
param[0][1][:][:] = torch.Tensor([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]])
param[0][2][:][:] = torch.Tensor([[GAMMA0*BETA0, GAMMA0*BETA0, GAMMA0*BETA0],[GAMMA0*BETA0, BETA0, GAMMA0*BETA0],[GAMMA0*BETA0, GAMMA0*BETA0, GAMMA0*BETA0]])

# Probability to be latent state (output: 1)
ALPHA0 = 0.99992
BETA0 = 1 - ALPHA0
GAMMA0 = 0.1
param[1][0][:][:] = torch.Tensor([[GAMMA0*BETA0, GAMMA0*BETA0, GAMMA0*BETA0],[GAMMA0*BETA0, BETA0, GAMMA0*BETA0],[GAMMA0*BETA0, GAMMA0*BETA0, GAMMA0*BETA0]])
param[1][1][:][:] = torch.Tensor([[GAMMA0*ALPHA0, GAMMA0*ALPHA0, GAMMA0*ALPHA0],[GAMMA0*ALPHA0, ALPHA0, GAMMA0*ALPHA0],[GAMMA0*ALPHA0, GAMMA0*ALPHA0, GAMMA0*ALPHA0]])
param[1][2][:][:] = torch.Tensor([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]])

# Probability to be fire state (output: 2)
ALPHA0 = 0.999
BETA0 = 1 - ALPHA0
GAMMA0 = 0.1
param[2][0][:][:] = torch.Tensor([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]])
param[2][1][:][:] = torch.Tensor([[GAMMA0*BETA0, GAMMA0*BETA0, GAMMA0*BETA0],[GAMMA0*BETA0, BETA0, GAMMA0*BETA0],[GAMMA0*BETA0, GAMMA0*BETA0, GAMMA0*BETA0]])
param[2][2][:][:] = torch.Tensor([[1.2*GAMMA0*ALPHA0, 1.2*GAMMA0*ALPHA0, 1.2*GAMMA0*ALPHA0],[GAMMA0*ALPHA0, ALPHA0, GAMMA0*ALPHA0],[0.8*GAMMA0*ALPHA0, 0.8*GAMMA0*ALPHA0, 0.8*GAMMA0*ALPHA0]])

TRANSITION_CNN_LAYER_WT.update({'weight':param})


#DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

ACTION_SET = [[-1, 1],[0, 1],[1, 1],
              [-1, 0],[0, 0],[1, 0],
              [-1,-1],[0,-1],[1,-1]]




