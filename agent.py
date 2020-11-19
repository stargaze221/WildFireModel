import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import random

from params import TRANSITION_CNN_LAYER_WT, DEVICE, ACTION_SET, OBSERVTAION_MATRIX

IF_PRINT = False

class POMDPAgent(nn.Module):

    def __init__(self, grid_size, n_obs, n_state):
        super(POMDPAgent, self).__init__()

        self.grid_size = grid_size # (w, h)

        ### Model Parameters ###
        # State Transition
        self.transition = nn.Conv2d(3, 3, 3, stride=1, padding=1).to(DEVICE)
        for name, param in self.transition.named_parameters():
            if name == 'weight':
                print(name)
                param.data=TRANSITION_CNN_LAYER_WT['weight'].to(DEVICE)
            if name == 'bias':
                print(name)
                param.data = torch.ones_like(param.data)*0
        # State Observation
        self.observation_matrix = OBSERVTAION_MATRIX.to(DEVICE)
        self.obs_bmm_matrix = (self.observation_matrix.T).repeat(grid_size[0]*grid_size[1], 1, 1).to(DEVICE)

        # State Estimate
        self.state_est = (torch.ones(grid_size[0], grid_size[1], 3)/3).to(DEVICE)
        

    def output_image(self, size=(1200,400)):
        img = self.state_est.data.cpu().numpy().squeeze()

        img_state0 = img[:,:,0]
        img_state1 = img[:,:,1]
        img_state2 = img[:,:,2]

        w, h = self.grid_size
        blank = np.zeros((h, int(w/20)))
        #img = np.concatenate((img_state0, blank, np.clip(img_state1 - img_state0, 0, 1), blank, img_state2), axis=1)
        img = np.concatenate((img_state0, blank, img_state1, blank, img_state2), axis=1)

        dim = size
        
        img_resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        return img_resized

    def Bayesian_update(self, obs):
        '''
        Beysian recursive update using the following formula

                   B u_k
        u_k+1 = P --------
                   b u_k

        where
        obs: 
        '''

        ### Bayesian Posterior ###
        if IF_PRINT:    
            print('obs:', obs.size())
            print('state_est:', self.state_est.size())

        u_k = self.state_est
        u_k = u_k.reshape(-1, 3, 1) # reshape for batch matrix multiplication (BMM) 
        obs = obs.reshape(-1, 3, 1) # reshape for batch matrix multiplication (BMM)

        if IF_PRINT:    
            print('u_k', u_k.size())
            print('obs', obs.size())
            print('bmm_mat', self.obs_bmm_matrix.size())

        b = torch.bmm(self.obs_bmm_matrix, obs)
        Bu_k = b*u_k
        bu_k =  torch.sum(Bu_k, 1).unsqueeze(-1)
        Bu_over_bu = Bu_k / bu_k
        Bu_over_bu = Bu_over_bu.reshape(self.grid_size[0], self.grid_size[1], 3).permute(2, 0, 1).unsqueeze(0)

        ### State Transitoin ###
        prob_dist = self.transition(Bu_over_bu).squeeze()
        prob_dist = prob_dist / torch.sum(prob_dist, 0)
        prob_dist = prob_dist.permute(1,2,0)

        if IF_PRINT:
            print('Bu_over_bu', Bu_over_bu.size())
            print('prob_dist', prob_dist.size())
            print(prob_dist)

        self.state_est = prob_dist

        return self.state_est

def render (window_name, image):
    cv2.imshow(window_name, image)
    cv2.waitKey(50)




if __name__ == "__main__":
    from environment import FireEnvironment

    env = FireEnvironment(50, 50)
    agent = POMDPAgent(grid_size = (env.map_width, env.map_height), n_obs=3, n_state=3)
    
    obs, state = env.reset()

    for i in range(5000):
        print(i)
        img_env   = env.output_image()
        img_agent = agent.output_image()

        render('env', img_env)
        render('est', img_agent)

        act = random.randrange(len(ACTION_SET))
        obs, state = env.step(ACTION_SET[act])

        state_est = agent.Bayesian_update(obs)

    






