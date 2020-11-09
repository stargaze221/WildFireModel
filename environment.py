import torch
from torch.distributions.categorical import Categorical
from torch import nn

import numpy as np
import cv2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FireMap:

    def __init__(self):

        self.map_width = 20
        self.map_height = 20
        self.N_state = 3

        ### State Variable ###
        self.prob_dist = torch.zeros(self.N_state, self.map_width, self.map_height).to(DEVICE)
        self.prob_dist[0][:][:] = 0.9
        self.prob_dist[1][:][:] = 0.1
        self.prob_dist[2][:][:] = 0.01


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
                # Probability to be normal state (output: 0) depends
                param[0][0][:][:] = torch.Tensor([[0.5, 0.5, 0.5],[0.5, 0.9, 0.5],[0.5, 0.5, 0.5]]) # on the preivous state (0:normal) of the grid and its neighbors
                param[0][1][:][:] = torch.Tensor([[0.2, 0.2, 0.2],[0.2, 0.1, 0.2],[0.2, 0.2, 0.2]]) # on the preivous state (1:latent) of the grid and its neighbors
                param[0][2][:][:] = torch.Tensor([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]]) # on the preivous state (2:fire) of the grid and its neighbors

                # Probability to be latent state (output: 1) depends
                param[1][0][:][:] = torch.Tensor([[0.1, 0.1, 0.1],[0.1, 0.1, 0.1],[0.1, 0.1, 0.1]]) # on the preivous state (0:normal) of the grid and its neighbors
                param[1][1][:][:] = torch.Tensor([[0.5, 0.5, 0.5],[0.5, 0.9, 0.5],[0.5, 0.5, 0.5]]) # on the preivous state (1:latent) of the grid and its neighbors
                param[1][2][:][:] = torch.Tensor([[0.2, 0.2, 0.2],[0.2, 0.1, 0.2],[0.2, 0.2, 0.2]]) # on the preivous state (2:fire) of the grid and its neighbors

                # Probability to be latent state (output: 1) depends
                param[2][0][:][:] = torch.Tensor([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]]) # on the preivous state (0:normal) of the grid and its neighbors
                param[2][1][:][:] = torch.Tensor([[0.1, 0.1, 0.1],[0.1, 0.1, 0.1],[0.1, 0.1, 0.1]]) # on the preivous state (1:latent) of the grid and its neighbors
                param[2][2][:][:] = torch.Tensor([[0.5, 0.5, 0.5],[0.5, 0.9, 0.5],[0.5, 0.5, 0.5]]) # on the preivous state (2:fire) of the grid and its neighbors

            if name == 'bias':
                print(name)
                param.data = torch.ones_like(param.data)*0.05


    def reset(self):
        ### Initialize the state variable values ###
        self.prob_dist[0][:][:] = 0.9
        self.prob_dist[1][:][:] = 0.1
        self.prob_dist[2][:][:] = 0.01

        prob_to_sample = self.prob_dist.clone().detach()
        prob_to_sample = prob_to_sample.permute(1, 2, 0)
        print(prob_to_sample.size())
        sampler = Categorical(torch.tensor(prob_to_sample.view(-1, 3)))
        grid_map = sampler.sample()
        print(grid_map)

        '''

        print(self.prob_dist.size())

        sampler = Categorical(torch.tensor(self.prob_dist.view(-1, 3).detach()))
        grid_map = sampler.sample()

        print(self.prob_dist.view(-1, 3))

        print(grid_map)
        '''
         

        '''
        N_fire_starts = 3 # Choose three grid points to set latent fire randomly.
        latent_fire_i_s = np.random.randint(self.map_width, size=(N_fire_starts))
        latent_fire_j_s = np.random.randint(self.map_height, size=(N_fire_starts))
        for i, j in zip(latent_fire_i_s, latent_fire_j_s):
            self.grid_tensor[2][i][j] = 1
        '''

    def step(self):
        prob_tensor = torch.unsqueeze(self.prob_dist, 0).to(DEVICE)
        prob_tensor = self.transition(prob_tensor).squeeze()
        prob_tensor = prob_tensor / torch.sum(prob_tensor, 0)
        self.prob_dist = prob_tensor

        return prob_tensor

    def render(self):
        
        img = self.grid_tensor.data.cpu().numpy().squeeze().transpose((1,2,0))
        print(img.shape)
        scale = 20
        dim = (self.map_width*scale, self.map_height*scale)
        img_resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        
        cv2.imshow("image", img_resized)
        cv2.waitKey(1000)

        '''
        img = np.zeros([self.map_width, self.map_height,3])
        img[:,:,2] = self.grid_tensor.data.cpu().numpy()   #Blue, Green, Red
        img[:,:,1] = self.grid_tensor.data.cpu().numpy()   #Blue, Green, Red
        scale = 20
        dim = (self.map_width*scale, self.map_height*scale)
        img_resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        
        cv2.imshow("image", img_resized)
        cv2.waitKey(1000)
        '''

        


        


if __name__ == "__main__":
    env = FireMap()
    env.reset()
    #print(env.prob_dist)

    #env.step()
    #print(env.prob_dist)
    
    #env.reset()
    #env.render()
    #print(env.grid_tensor)
    #env.step()
    #print(env.grid_tensor)
    #env.render()
    