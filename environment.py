'''
environment.py
author: Hyung Jin
date: 01/13/2020
'''


#modules to import
import torch
from torch.distributions.categorical import Categorical
from torch import nn
from torch.autograd import Variable

torch.manual_seed(1234)


import numpy as np
import cv2, tqdm

import seaborn as sns

import matplotlib.pyplot as plt

from params import ACTION_SET, TRANSITION_CNN_LAYER_WT, DEVICE, OBSERVTAION_MATRIX


N_ACTION = len(ACTION_SET)

from memory import SingleTrajectoryBuffer

T_EVAL = 30000 + 1000




'''
#Notes: 
'''


'''
class: FireEnvironment
inherits: None 
Description: 
'''
class FireEnvironment:

    
    '''
    Function: Constructor for FireEnvironment 
    Params:
        w: map width
        h: map height 
    
    return: none 
    
    1. Set up state variables 
    2. Set up state observation and transition models 
    '''
    def __init__(self, w, h, for_eval=False):

        self.map_width = w
        self.map_height = h
        self.N_state = 3

        ### State Variable ###
        self.init_prob_dist = torch.zeros(self.N_state, self.map_width, self.map_height).to(DEVICE)
        self.init_prob_dist[0][:][:] = 0.99
        self.init_prob_dist[1][:][:] = 0.01
        self.init_prob_dist[2][:][:] = 0.00

        ### State Observation Model ###
        self.observation_matrix = OBSERVTAION_MATRIX.to(DEVICE)
        self.observation_matrix = self.observation_matrix.repeat(self.map_width*self.map_height, 1, 1)
        

        ### State Transition Model ###
        self.transition = nn.Conv2d(3, 3, 3, stride=1, padding=1).to(DEVICE)
        '''
        Kernel weight size (n_ch_out, n_ch_in, kernel_width, kernel_height)
        Kernel bias size (n_ch_out)        
        '''

        for name, param in self.transition.named_parameters():
            if name == 'weight':
                #print(name)
                param.data=TRANSITION_CNN_LAYER_WT['weight'].to(DEVICE)
                
            if name == 'bias':
                #print(name)
                param.data = torch.ones_like(param.data)*0

        self.realization_state = self.sample_gridmap_from_fire_dist(self.init_prob_dist)
        self.observed_state = self.observe()
        self.masked_observation = torch.zeros_like(self.observed_state)

        ### Previous mask for reward calculation ###
        self.prev_state = torch.zeros_like(self.realization_state)

        ### Save trajectory for evaluation ###
        self.t = 0
        if for_eval:
            self.if_use_saved_eval = False
            self.memory_evaluation_state = []
            self.memory_evaluation_observation = []
            self.save_evaluation_memory()
            self.t = 0
        self.if_use_saved_eval = for_eval



    '''
    Function: sample_gridmap_from_fire_dist
    Params:
        fire_dist: fire matrix
    
    return: tensor 
    
    1. Create a categorial distributionand sample the grid map from the 
        fire distribution 
    '''
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


    '''
    Function: render
    Params: none

    return: none

    1. display/render the image to the screen
    '''
    def render(self):
        img_state = self.realization_state.data.permute(1,2,0).cpu().numpy().squeeze()
        img_obs = self.observed_state.data.cpu().numpy().squeeze()
        img_masked_obs = self.masked_observation.data.cpu().numpy().squeeze()
        
        blank = np.zeros((self.map_height, int(self.map_width/20), 3))
        img = np.concatenate((img_state, blank, img_obs, blank, img_masked_obs), axis=1)
        
        scale = 2
        dim = (int(self.map_width*(3.3))*scale, self.map_height*scale)
        img_resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow("image", img_resized)
        cv2.waitKey(1)


    '''
    Function: output_image
    Params:
        size: size of the image 

    return: the image state
    
    1. returns the image of the states
    '''
    def output_image(self, size=(1200,400)):
        img_state = self.realization_state.data.permute(1,2,0).cpu().numpy().squeeze()
        img_obs = self.observed_state.data.cpu().numpy().squeeze()
        img_masked_obs = self.masked_observation.data.cpu().numpy().squeeze()
        
        blank = np.zeros((self.map_height, int(self.map_width/20), 3))
        img = np.concatenate((img_state, blank, img_obs, blank, img_masked_obs), axis=1)
        
        dim = size
        img_resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        return img_resized


    '''
    Function: reset
    Params: none

    return: the reset observations and states
    
    1. Set the observations and state to be reset to either a saved value 
        or initial distribution 
    '''
    def reset(self):
        self.t = 0
        if self.if_use_saved_eval:

            self.masked_observation = torch.zeros_like(self.observed_state)
            self.realization_state = self.memory_evaluation_state[0]
            self.observed_state = self.memory_evaluation_observation[0]

        else:
            ### Resample using the initial distribution ###
            self.realization_state = self.sample_gridmap_from_fire_dist(self.init_prob_dist)
            self.observed_state = self.observe()
            self.masked_observation = torch.zeros_like(self.observed_state)


        return self.masked_observation, self.observed_state, self.realization_state


    '''
    Function: step
    Params:
        obs: observation
        mask: mask to use 

    return: state estimation grids, reward and info
    
    1. Estimate the state 
    2. Run the RNN to get the state estimate grid with the likelihood
    '''
    def step(self, obs_mask=None):
        with torch.no_grad():
            if self.if_use_saved_eval:
                self.realization_state = torch.Tensor(self.memory_evaluation_state[self.t]).to(DEVICE)
                self.observed_state = torch.Tensor(self.memory_evaluation_observation[self.t]).to(DEVICE)

            else:
                state = torch.unsqueeze(self.realization_state, 0).to(DEVICE)
                prob_dist = self.transition(state).squeeze()
                prob_dist = prob_dist / torch.sum(prob_dist, 0)
                prob_to_sample = prob_dist.clone().detach()
                self.realization_state = self.sample_gridmap_from_fire_dist(prob_to_sample)
                self.observed_state = self.observe()
                del state, prob_dist, prob_to_sample

            ## Maks observation 
            #obs_mask = torch.FloatTensor(obs_mask).unsqueeze(-1).to(DEVICE)
            if obs_mask != None:
                obs_mask = obs_mask.to(DEVICE)
                self.masked_observation = obs_mask.unsqueeze(-1) * self.observed_state
            else:
                self.masked_observation = self.observed_state

            # New fire grid
            #new_fire = torch.clamp(self.realization_state[2] - self.prev_state[2], 0, 1)
            new_fire = self.realization_state[2]  #- self.prev_state[2], 0, 1)

            '''
            print('realization_state[2]:', self.realization_state[2].size())
            print('new_fire:', new_fire.size(), new_fire.max(), new_fire.min())        
            print('obs_mask:', obs_mask.size())
            '''

            # Count the sum of visit to the new red grid
            #reward = torch.sum(obs_mask * self.realization_state[2])/torch.sum(obs_mask)
            if obs_mask != None:
                reward = torch.sum(obs_mask * new_fire).item()
            else:
                reward = None

            # Update the previous state
            self.prev_state = self.realization_state

            info ={}
            info.update({'new_fire_count': torch.sum(new_fire).item()})

            #print('reward:', reward, 'new_fire_count', info['new_fire_count'])

            del new_fire
            torch.cuda.empty_cache()

        self.t += 1

        return self.masked_observation, self.observed_state, self.realization_state, reward, info


    '''
    Function: Observe
    Params: none

    return: the observed onehot map 
    
    1. Create a categorial distributionand sample the realization state and
        observation matrix 
    '''
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
        
        return onehot_obs_map


    '''
    Function: save_evaluation_memory
    Params: none

    return: none
    
    1. call the step function to run the RNN for state estimate 
    2. set the memory evaluations nad observations 
    '''
    def save_evaluation_memory(self):

        for t in tqdm.tqdm(range(T_EVAL)):
            _, observation, state, _, _ = self.step()
            self.memory_evaluation_observation.append(observation.data.cpu().numpy())
            self.memory_evaluation_state.append(state.data.cpu().numpy())


def main():
    env = FireEnvironment(64,64, True)

    for t in range(1000):
        env.step()
        env.render()    


if __name__ == "__main__":
    main()