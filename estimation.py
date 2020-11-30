import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import random

#DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

from torch.autograd.functional import jacobian

class HMMEstimator(nn.Module):

    def __init__(self, grid_size, n_obs, n_state):
        super(HMMEstimator, self).__init__()

        self.grid_size = grid_size # (w, h)

        ### Model Parameters ###
        self.params = []
        # State Transition
        print('state transition')
        self.transition = nn.Conv2d(3, 3, 3, stride=1, padding=1).to(DEVICE)
        for name, param in self.transition.named_parameters():
            print(name, param.size())
            self.params.append(torch.nn.utils.parameters_to_vector(param))

        # Observation
        print('observation')
        self.observation_linear = nn.Linear(n_obs, n_state).to(DEVICE)
        for name, param in self.observation_linear.named_parameters():
            print(name, param.size())
            self.params.append(torch.nn.utils.parameters_to_vector(param))
        # List the parameters
        self.params = torch.cat(self.params)
        
        # State Estimate
        self.u_k = (torch.ones(grid_size[0], grid_size[1], 3)/3).to(DEVICE)

    def update(self, obs):
        '''
        Step1: Bu/bu
        '''
        b = self.observation_linear(obs)
        m = nn.Softmax(dim=2)
        b = m(b)
        Bu_k = b*self.u_k
        bu_k =  torch.sum(Bu_k, 2).unsqueeze(-1)
        Bu_over_bu = Bu_k / bu_k

        

        '''
        Step2:  Phi [ Bu/bu ] --> u_k
        '''
        u_kp1 = self.transition(Bu_over_bu.permute(2, 0, 1).unsqueeze(0))
        u_kp1 = u_kp1.squeeze().permute(1,2,0)
        m = nn.Softmax(dim=2)
        u_kp1 = m(u_kp1)

        def Phi(Z):
            phi = self.transition(Z.permute(2, 0, 1).unsqueeze(0))
            phi = phi.squeeze().permute(1,2,0)
            m = nn.Softmax(dim=2)
            return m(phi)

        tmp = jacobian(Phi, Bu_over_bu)
        print(tmp.size())
        print(tmp)

        


        

        '''
        Step2_REV:  Phi [ Bu/bu ] --> u_k
        Let us make it as a matrix multiplication.
        '''
        '''
        P = []
        unit1 = torch.zeros_like(self.u_k)
        unit1[:,:,0] = 1
        P_unit1 = self.transition(unit1.permute(2, 0, 1).unsqueeze(0))
        P.append(P_unit1)
        unit2 = torch.zeros_like(self.u_k)
        unit2[:,:,1] = 1
        P_unit2 = self.transition(unit2.permute(2, 0, 1).unsqueeze(0))
        P.append(P_unit2)
        unit3 = torch.zeros_like(self.u_k)
        unit3[:,:,2] = 1
        P_unit3 = self.transition(unit3.permute(2, 0, 1).unsqueeze(0))
        P.append(P_unit3)
        P = torch.stack(P).squeeze()
        m = nn.Softmax(dim=1)
        P = m(P)
        P = P.permute(3,2,0,1)

        print('Bu_over_bu', Bu_over_bu.size())

        u = Bu_over_bu.unsqueeze(-1)
        u_kp1_try = torch.matmul(P, u)
        u00 = u[0][0]
        M = P[0][0]
        print(M)
        print(u00)
        print(torch.matmul(M.T, u00))
        print(torch.sum(torch.matmul(M.T, u00)))
        '''






        





if __name__ == "__main__":
    from environment import FireEnvironment
    
    env = FireEnvironment(20, 20)
    hmm_estimator = HMMEstimator(grid_size = (env.map_width, env.map_height), n_obs=3, n_state=3)
    
    obs, state = env.reset()
    hmm_estimator.update(obs)


    






