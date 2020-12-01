import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import random

from params import TRANSITION_CNN_LAYER_WT, DEVICE, ACTION_SET, OBSERVTAION_MATRIX



LR_ESTIMATOR = 0.001
BETAS = (0.5, 0.9)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class HMMEstimator(nn.Module):

    def __init__(self, grid_size, n_kernel, n_obs, n_state):
        super(HMMEstimator, self).__init__()

        self.grid_size = grid_size # (w, h)
        self.n_kernel = n_kernel
        self.n_obs = n_obs
        self.n_state = n_state

        # Model Network
        self.transition_cov2d_layer = nn.Conv2d(n_state, n_state, n_kernel, stride=1, padding=1).to(DEVICE)
        self.observation_lin_layer = nn.Linear(n_obs, n_state, bias=False).to(DEVICE)
         
        # State Predictor
        self.u_km1 = (torch.ones(grid_size[0], grid_size[1], n_state, 1)/n_state).to(DEVICE)
        self.y_km1 = None

        # Optimizer
        params = list(self.transition_cov2d_layer.parameters()) + list(self.observation_lin_layer.parameters())
        self.optimizer = torch.optim.Adam(params, LR_ESTIMATOR, BETAS)

        # Aux
        self.n_iteration = 0

    def update(self, obs):
        '''
        At current time step k, I can calcualte b(y_k).
        Also, I can use y_km1 and u_km1 to update and calculate u_k for state predictor
        '''

        if self.n_iteration > 0:

            ### Step 1: Bayesian Update for time of k-1 ###
            '''
            B u_km1
            --------
            b u_km1
            '''
            y_km1 = self.y_km1.detach()
            I_obs = torch.eye(self.n_obs).to(DEVICE)
            O = F.softmax(self.observation_lin_layer(I_obs), 0)  # rows: observation, column: state
            O_bat = O.T.unsqueeze(0).unsqueeze(0).repeat(self.grid_size[0], self.grid_size[1], 1, 1)
            b_km1 = torch.matmul(O_bat, y_km1) # Size : w x h x n_state x 1

            u_km1 = self.u_km1.detach() # Size : w x h x n_state x 1
            Bu_km1 = b_km1*u_km1 # Size : w x h x n_state x 1
            bu_km1 =  torch.sum(Bu_km1, 2).unsqueeze(-1)
            Bu_over_bu_km1 = Bu_km1 / bu_km1 # Size : w x h x n_state x 1

            ### Step 2: State Predictor ###
            '''       [  B u_km1  ]
            u_k = Phi [-----------]
                    [  b u_km1  ]
            '''
            u_k = F.softmax(self.transition_cov2d_layer(Bu_over_bu_km1.permute(3, 2, 0, 1)), 1).permute(2, 3, 1, 0)

            ### Step 3: Likelihood given y_k ###
            y_k = obs.unsqueeze(-1)
            I_obs = torch.eye(self.n_obs).to(DEVICE)
            O = F.softmax(self.observation_lin_layer(I_obs), 0)  # rows: observation, column: state
            O_bat = O.T.unsqueeze(0).unsqueeze(0).repeat(self.grid_size[0], self.grid_size[1], 1, 1)
            b_k = torch.matmul(O_bat, y_k) # Size : w x h x n_state x 1

            ### Step 4: 
            # Optimize Online Likeilhood ###
            '''
            S = log [b(y_k) u_k]
            '''
            S = torch.mean(torch.log(b_k*u_k))
            loss = -S

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            #########################
            ### Update the memory ###
            #########################
            self.y_km1 = y_k
            self.u_km1 = u_k
            self.n_iteration += 1
            loss_val = loss.item()

        else:
            print('Need to fill in memory!')
            #########################
            ### Update the memory ###
            #########################
            self.y_km1 = obs.unsqueeze(-1)
            self.n_iteration += 1
            loss_val = 0


        return loss_val




if __name__ == "__main__":
    from environment import FireEnvironment
    
    env = FireEnvironment(50, 50)
    hmm_estimator = HMMEstimator(grid_size = (env.map_width, env.map_height), n_kernel=3, n_obs=3, n_state=3)
    
    obs, state = env.reset()
    hmm_estimator.update(obs)
    
    for i in range(5000):
        print(i)

        act = random.randrange(len(ACTION_SET))
        obs, state = env.step(ACTION_SET[act])
        loss_val = hmm_estimator.update(obs)
        print(i, loss_val)






