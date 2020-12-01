import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import random



DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device("cpu")
from torch.autograd.functional import jacobian

class HMMEstimator(nn.Module):

    def __init__(self, grid_size, n_obs, n_state):
        super(HMMEstimator, self).__init__()

        self.grid_size = grid_size # (w, h)
        self.n_obs = n_obs
        self.n_state = n_state

        ### Model Parameters ###
        self.params = []
        # State Transition
        print('state transition')
        self.transition = nn.Conv2d(3, 3, 3, stride=1, padding=1).to(DEVICE)
        for name, param in self.transition.named_parameters():
            print(name, param.size())
            self.params.append(param)

        # State Observation
        self.obs_param_matrix = torch.ones(n_state, n_obs).to(DEVICE)
        self.obs_param_matrix = self.obs_param_matrix / torch.sum(self.obs_param_matrix, 1)
         
        # State Predictor
        self.u_k = (torch.ones(grid_size[0], grid_size[1], n_state, 1)/n_state).to(DEVICE)

        # Deriv of State Predictor (Denoted as Omega in the Paper)
        self.w_k_O = torch.zeros([grid_size[0], grid_size[1], n_state, 1]+list(self.obs_param_matrix.size())) # Size: w x h x n_state x 1 x n_state x n_obs
        self.w_k_T_weight = torch.zeros([grid_size[0], grid_size[1], n_state, 1]+list(self.params[0].size())) # Size: w x h x n_state x 1 x n_state x n_kernel (3) x n_kernel (3) x n_state
        self.w_k_T_bias = torch.zeros([grid_size[0], grid_size[1], n_state, 1]+list(self.params[1].size())) # Size: w x h x n_state x 1 x n_state


    def update(self, obs):
        #######################
        ### State Predictor ###
        #######################

        ### Step1 ###
        '''
          B u_k
        --------
          b u_k
        '''
        # Calculate the likelihood vector, b
        obs_bmm_matrix = (self.obs_param_matrix.T).repeat(self.grid_size[0]*self.grid_size[1], 1, 1).to(DEVICE)
        obs_bmm_matrix = obs_bmm_matrix.reshape(self.grid_size[0], self.grid_size[1], self.n_state, self.n_state)
        b = torch.matmul(obs_bmm_matrix, obs.unsqueeze(-1)) # Size : w x h x n_state x 1

        # Derivative of b respect to the observation matrix
        def b_likelihood_vector(obs_matrix):
            obs_bmm_matrix = (self.obs_param_matrix.T).repeat(self.grid_size[0]*self.grid_size[1], 1, 1).to(DEVICE)
            obs_bmm_matrix = obs_bmm_matrix.reshape(self.grid_size[0], self.grid_size[1], self.n_state, self.n_state)
            b = torch.matmul(obs_bmm_matrix, obs.unsqueeze(-1))
            return b
        deriv_b_resp_obs_param = jacobian(b_likelihood_vector, self.obs_param_matrix) # Size: w x h x n_state x 1 x n_state x n_obs
                
        u_k = self.u_k # Size : w x h x n_state x 1
        Bu_k = b*self.u_k # Size : w x h x n_state x 1
        bu_k =  torch.sum(Bu_k, 2).unsqueeze(-1)
        Bu_over_bu = Bu_k / bu_k # Size : w x h x n_state x 1

        ### Step2 ###
        '''
        Phi [ Bu/bu ] --> u_k
        '''
        # Calculate u_kp1
        u_kp1 = self.transition(Bu_over_bu.permute(3, 2, 0, 1))
        u_kp1 = u_kp1.squeeze().permute(1,2,0)
        m = nn.Softmax(dim=2)
        u_kp1 = m(u_kp1).unsqueeze(-1) # Size : w x h x n_state x 1

        # Derivative of Phi [ Bu/bu ] respect to Z
        def Phi(Z):
            phi = self.transition(Z.permute(3, 2, 0, 1))
            phi = phi.squeeze().permute(1,2,0)
            m = nn.Softmax(dim=2)
            return m(phi).unsqueeze(-1)

        deriv_Phi_resp_Z = jacobian(Phi, Bu_over_bu) # Size : (w x h x n_state x 1) x (w x h x n_state x 1)

        ####################################################
        ### Recursive Updating Deriv. of State Predictor ###
        ####################################################

        ### Part 1: Omega for Observation Param ###
        # self.w_k_O
        # Size: (w x h x n_state x 1) x n_state x n_obs
        # ------------------------
        # deriv_b_resp_obs_param
        # Size: (w x h x n_state x 1) x n_state x n_obs

        # term1: (rho_B_over_rho_theta u_k + B w_k_O) / b u_k
        term1 = (deriv_b_resp_obs_param * self.u_k.unsqueeze(-1).unsqueeze(-1))
        term1 += b.unsqueeze(-1).unsqueeze(-1)*self.w_k_O 
        term1 = term1*bu_k.unsqueeze(-1).unsqueeze(-1) 

        # term2: rho_b_over_rho_theta^T u_k + b^T w_k_O
        term2 = torch.sum((deriv_b_resp_obs_param * self.u_k.unsqueeze(-1).unsqueeze(-1)), 2)
        term2 += torch.sum(b.unsqueeze(-1).unsqueeze(-1)*self.w_k_O, 2)
        term2 = term2.unsqueeze(3)
        term2 = term2 * Bu_k.unsqueeze(-1).unsqueeze(-1)
        
        # w_kp1 = deriv_Phi_resp_Z (term1 - term2)/(b u_k)^2
        w_kp1_O = (term1 - term2)/(bu_k.unsqueeze(-1).unsqueeze(-1)**2)
        J = deriv_Phi_resp_Z.reshape(self.grid_size[0]*self.grid_size[1]*self.n_state,-1) # A Big matrix of the Jacobian for deriv_Phi_resp_Z
        w =  w_kp1_O.reshape(self.grid_size[0]*self.grid_size[1]*self.n_state, -1) # A Big vector of (term1 - term2)/(b u_k)^2
        w_kp1_O = torch.matmul(J, w)
        w_kp1_O = w_kp1_O.reshape(self.grid_size[0], self.grid_size[1], self.n_state, 1, self.n_state, self.n_obs)

        ### Part 2: Omega for State Transition ###

        



        
        
        


        
        






        





if __name__ == "__main__":
    from environment import FireEnvironment
    
    env = FireEnvironment(20, 20)
    hmm_estimator = HMMEstimator(grid_size = (env.map_width, env.map_height), n_obs=3, n_state=3)
    
    obs, state = env.reset()
    hmm_estimator.update(obs)


    






