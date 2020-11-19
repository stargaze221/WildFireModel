import torch
from torch.distributions.categorical import Categorical
from torch import nn
from torch.autograd import Variable
import random

import numpy as np
import cv2

import seaborn as sns

import matplotlib.pyplot as plt

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ACTION_SET = [[-1, 1],[0, 1],[1, 1],
              [-1, 0],[0, 0],[1, 0],
              [-1,-1],[0,-1],[1,-1]]

N_ACTION = len(ACTION_SET)


class FireMap:

    def __init__(self):

        self.map_width = 100
        self.map_height = 100
        self.N_state = 3

        ### Drone State ###
        self.drone_pos = [25,25]
        self.drone_obs_windows = torch.zeros((3,3,3))
        self.drone_state_windows = torch.zeros((3,3,3))


        ### State Variable ###
        self.init_prob_dist = torch.zeros(self.N_state, self.map_width, self.map_height).to(DEVICE)
        self.init_prob_dist[0][:][:] = 0.99
        self.init_prob_dist[1][:][:] = 0.01
        self.init_prob_dist[2][:][:] = 0.00

        ### State Observation Model ###
        self.observation_matrix = torch.Tensor([[0.98, 0.98, 0.01],[0.01, 0.01, 0.01],[0.01, 0.01, 0.98]]).to(DEVICE)
        self.observation_matrix = self.observation_matrix.repeat(self.map_width*self.map_height, 1, 1)
        

        ### State Transition Model ###
        self.transition = nn.Conv2d(3, 3, 3, stride=1, padding=1).to(DEVICE)
        '''
        Kernel weight size (n_ch_out, n_ch_in, kernel_width, kernel_height)
        Kernel bias size (n_ch_out)        
        '''

        for name, param in self.transition.named_parameters():
            if name == 'weight':
                print(name)
                param.data = torch.zeros_like(param.data)

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

            if name == 'bias':
                print(name)
                param.data = torch.ones_like(param.data)*0

        self.realization_state = self.sample_gridmap_from_fire_dist(self.init_prob_dist)
        self.observed_state = self.observe()


    def sample_gridmap_from_fire_dist(self, fire_dist):
        '''
        fire_dist : (3 x Width X Height)
        '''
        fire_dist = fire_dist.clone().detach().permute(1, 2, 0)
        sampler = Categorical(fire_dist.reshape(-1, 3))
        grid_map = sampler.sample()
        grid_map = grid_map.reshape(self.map_width, self.map_width, 1)
        onehot_grid_map = torch.FloatTensor(self.map_width, self.map_height, 3).to(DEVICE)
        onehot_grid_map.zero_()
        onehot_grid_map.scatter_(2, grid_map, 1)
        return onehot_grid_map.permute(2, 0, 1)

    def render(self):
        img_state = self.realization_state.data.permute(1,2,0).cpu().numpy().squeeze()
        
        img_obs = self.observed_state.data.cpu().numpy().squeeze()
        
        blank = np.zeros((self.map_height, int(self.map_width/20), 3))
        img = np.concatenate((img_state, blank, img_obs), axis=1)
        
        scale = 4
        dim = (int(self.map_width*(2.3))*scale, self.map_height*scale)
        img_resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow("image", img_resized)
        cv2.waitKey(50)


    def reset(self):
        ### Resample using the initial distribution ###
        self.realization_state = self.sample_gridmap_from_fire_dist(self.init_prob_dist)
        self.observed_state = self.observe()
        return self.realization_state

    def step(self, move):
        state = torch.unsqueeze(self.realization_state, 0).to(DEVICE)
        prob_dist = self.transition(state).squeeze()
        prob_dist = prob_dist / torch.sum(prob_dist, 0)
        prob_to_sample = prob_dist.clone().detach()
        self.realization_state = self.sample_gridmap_from_fire_dist(prob_to_sample)
        self.observed_state = self.observe()

        self.drone_pos[0] = min(max(self.drone_pos[0] + move[0], 1), self.map_width-2)
        self.drone_pos[1] = min(max(self.drone_pos[1] + move[1], 1), self.map_height-2)

        return self.realization_state

    def observe(self):
        prob_to_sample = torch.unsqueeze(self.realization_state.clone().detach().permute(1, 2, 0).reshape(-1, 3),2)
        #print(prob_to_sample.size(), self.observation_matrix.size()) 
        prob_to_sample = torch.bmm(self.observation_matrix, prob_to_sample)
        #print(prob_to_sample.size())
        sampler = Categorical(prob_to_sample.squeeze())
        observed_map = sampler.sample()
        observed_map = observed_map.reshape(self.map_width, self.map_width, 1)
        onehot_obs_map = torch.FloatTensor(self.map_width, self.map_height, 3).to(DEVICE)
        onehot_obs_map.zero_()
        onehot_obs_map.scatter_(2, observed_map, 1)

        x_drone = self.drone_pos[0]
        y_drone = self.drone_pos[1]

        state_map = self.realization_state.clone().detach().permute(1, 2, 0)
        
        for i in range(-1,2):
            for j in range(-1,2):
                self.drone_obs_windows[i][j]= onehot_obs_map[x_drone+i][y_drone+j]
                self.drone_state_windows[i][j]= state_map[x_drone+i][y_drone+j]

        return onehot_obs_map
        


if __name__ == "__main__":
    env = FireMap()
    onehot_grid_map = env.reset()
    #print(onehot_grid_map)
    env.render()
    #for i in range(10000):

    while True:
        act = random.randrange(N_ACTION)
        onehot_grid_map = env.step(ACTION_SET[act])
        #print(onehot_grid_map)
        env.render()
        #break