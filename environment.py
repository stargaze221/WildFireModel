import torch
from torch.distributions.categorical import Categorical
from torch import nn
from torch.autograd import Variable


import numpy as np
import cv2

import seaborn as sns

import matplotlib.pyplot as plt

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FireMap:

    def __init__(self):

        self.map_width = 20
        self.map_height = 20
        self.N_state = 3

        ### State Variable ###
        self.init_prob_dist = torch.zeros(self.N_state, self.map_width, self.map_height).to(DEVICE)
        self.init_prob_dist[0][:][:] = 0.9
        self.init_prob_dist[1][:][:] = 0.01
        self.init_prob_dist[2][:][:] = 0.00

        self.realization_state = self.sample_gridmap_from_fire_dist(self.init_prob_dist)

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
                ALPHA0 = 0.999
                BETA0 = 0.1
                GAMMA0 = 0.01
                param[0][0][:][:] = torch.Tensor([[GAMMA0*ALPHA0, GAMMA0*ALPHA0, GAMMA0*ALPHA0],[GAMMA0*ALPHA0, ALPHA0, GAMMA0*ALPHA0],[GAMMA0*ALPHA0, GAMMA0*ALPHA0, GAMMA0*ALPHA0]])
                param[0][1][:][:] = torch.Tensor([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]])
                param[0][2][:][:] = torch.Tensor([[GAMMA0*BETA0, GAMMA0*BETA0, GAMMA0*BETA0],[GAMMA0*BETA0, BETA0, GAMMA0*BETA0],[GAMMA0*BETA0, GAMMA0*BETA0, GAMMA0*BETA0]])
                
                # Probability to be latent state (output: 1)
                ALPHA0 = 0.99
                BETA0 = 0.1
                GAMMA0 = 0.001
                param[1][0][:][:] = torch.Tensor([[GAMMA0*BETA0, GAMMA0*BETA0, GAMMA0*BETA0],[GAMMA0*BETA0, BETA0, GAMMA0*BETA0],[GAMMA0*BETA0, GAMMA0*BETA0, GAMMA0*BETA0]])
                param[1][1][:][:] = torch.Tensor([[GAMMA0*ALPHA0, GAMMA0*ALPHA0, GAMMA0*ALPHA0],[GAMMA0*ALPHA0, ALPHA0, GAMMA0*ALPHA0],[GAMMA0*ALPHA0, GAMMA0*ALPHA0, GAMMA0*ALPHA0]])
                param[1][2][:][:] = torch.Tensor([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]])

                # Probability to be fire state (output: 2)
                ALPHA0 = 0.9
                BETA0 = 0.1
                GAMMA0 = 0.01
                param[2][0][:][:] = torch.Tensor([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]])
                param[2][1][:][:] = torch.Tensor([[GAMMA0*BETA0, GAMMA0*BETA0, GAMMA0*BETA0],[GAMMA0*BETA0, BETA0, GAMMA0*BETA0],[GAMMA0*BETA0, GAMMA0*BETA0, GAMMA0*BETA0]])
                param[2][2][:][:] = torch.Tensor([[GAMMA0*ALPHA0, GAMMA0*ALPHA0, GAMMA0*ALPHA0],[GAMMA0*ALPHA0, ALPHA0, GAMMA0*ALPHA0],[GAMMA0*ALPHA0, GAMMA0*ALPHA0, GAMMA0*ALPHA0]])

            if name == 'bias':
                print(name)
                param.data = torch.ones_like(param.data)*0

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
        img = self.realization_state.data.permute(1,2,0).cpu().numpy().squeeze()
        print(img.shape)
        scale = 20
        dim = (self.map_width*scale, self.map_height*scale)
        img_resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow("image", img_resized)
        cv2.waitKey(100)

    def reset(self):
        ### Resample using the initial distribution ###
        self.realization_state = self.sample_gridmap_from_fire_dist(self.init_prob_dist)
        return self.realization_state


    def step(self):
        state = torch.unsqueeze(self.realization_state, 0).to(DEVICE)
        print(state)
        prob_dist = self.transition(state).squeeze()
        print(prob_dist)
        prob_dist = prob_dist / torch.sum(prob_dist, 0)
        print(prob_dist)
        prob_to_sample = prob_dist.clone().detach()
        self.realization_state = self.sample_gridmap_from_fire_dist(prob_to_sample)
        return self.realization_state

    
        
        

        


        


if __name__ == "__main__":
    env = FireMap()
    onehot_grid_map = env.reset()
    #print(onehot_grid_map)
    env.render()
    for i in range(100):    
        onehot_grid_map = env.step()
        #print(onehot_grid_map)
        env.render()