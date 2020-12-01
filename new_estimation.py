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

    def __init__(self, grid_size, n_obs, n_state, n_kernel):
        super(HMMEstimator, self).__init__()

        self.grid_size = grid_size # (w, h)
        self.n_obs = n_obs
        self.n_state = n_state

        ### Model Parameters ###
        # State Transition
        print('state transition')
        self.transition = nn.Conv2d(n_state, n_state, n_kernel, stride=1, padding=1).to(DEVICE)
        self.param_trans_weight = torch.rand(n_state, n_kernel, n_kernel, n_state)
        self.param_trans_bias = torch.rand(n_state)
        for name, param in self.transition.named_parameters():
            if name == 'weight':
                param = self.param_trans_weight
            else:
                param = self.param_trans_bias

        # State Observation
        self.param_obs = torch.rand(n_state, n_obs).to(DEVICE)


        ### State of Recursive Estimator ###
        # State Predictor
        self.u_k = (torch.ones(grid_size[0], grid_size[1], n_state, 1)/n_state).to(DEVICE)

        # Derivatives of State Predictor (Denoted as Omega in the Paper)
        self.w_k_O = torch.zeros(list(self.param_obs.size())+[grid_size[0], grid_size[1], n_state, 1]) # Size: w x h x n_state x 1 x n_state x n_obs
        self.w_k_T_weight = torch.zeros(list(self.param_trans_weight.size())+[grid_size[0], grid_size[1], n_state, 1]) # Size: w x h x n_state x 1 x n_state x n_kernel (3) x n_kernel (3) x n_state
        self.w_k_T_bias = torch.zeros(list(self.param_trans_bias.size())+[grid_size[0], grid_size[1], n_state, 1]) # Size: w x h x n_state x 1 x n_state

    def update(self, obs):
        #######################
        ### State Predictor ###
        #######################
        Y = obs.unsqueeze(-1) # Size : w x h x n_obs x 1

        # Calculate the likelihood vector, b(Y)
        obs_matrix = F.softmax(self.param_obs, dim=1)
        obs_bmm_matrix = (obs_matrix).repeat(self.grid_size[0]*self.grid_size[1], 1, 1).to(DEVICE)
        obs_bmm_matrix = obs_bmm_matrix.reshape(self.grid_size[0], self.grid_size[1], self.n_state, self.n_state)
        b = torch.matmul(obs_bmm_matrix, Y) # Size : w x h x n_state x 1

        # Bayesian Update
        '''
          B u_k
        --------
          b u_k
        '''
        u_k = self.u_k # Size : w x h x n_state x 1
        Bu_k = b*self.u_k # Size : w x h x n_state x 1
        bu_k =  torch.sum(Bu_k, 2).unsqueeze(-1)
        Bu_over_bu = Bu_k / bu_k # Size : w x h x n_state x 1

        # Calculate P matrix using the convolution layer
        basis0 = torch.zeros((1, self.n_state, self.grid_size[0], self.grid_size[1]))
        basis0[:, 0, :, :] = 1
        basis1 = torch.zeros_like(basis0)
        basis1[:, 1, :, :] = 1
        basis2 = torch.zeros_like(basis0)
        basis2[:, 2, :, :] = 1
        m = nn.Softmax(dim=1)
        P0 = m(self.transition(basis0))
        P1 = m(self.transition(basis1))
        P2 = m(self.transition(basis2))
        P = torch.stack([P0, P1, P2]).squeeze() # Size: n_state x n_state x height x width
        P = P.permute(2,3,1,0)

        # State Prediction
        '''
                   B u_k
        u_kp1 = P --------
                   b u_k
        '''
        u_kp1 = torch.matmul(P, Bu_over_bu) # Size : w x h x n_state x 1

        ####################################################
        ### Recursive Updating Deriv. of State Predictor ###
        ####################################################

        # Derivative of b respect to the observation matrix
        def b_likelihood_vector(param_obs):
            obs_matrix = F.softmax(param_obs, dim=1)
            obs_bmm_matrix = (obs_matrix).repeat(self.grid_size[0]*self.grid_size[1], 1, 1).to(DEVICE)
            obs_bmm_matrix = obs_bmm_matrix.reshape(self.grid_size[0], self.grid_size[1], self.n_state, self.n_state)
            b = torch.matmul(obs_bmm_matrix, Y) # Size : w x h x n_state x 1
            return b
        deriv_b_resp_param_obs = jacobian(b_likelihood_vector, self.param_obs)
        # Size: w x h x n_state x 1 x n_state x n_obs

        # Derivative of P respect to weight
        def cal_P_given_weight(weight):
            for name, param in self.transition.named_parameters():
                if name == 'weight':
                    param = weight
                else:
                    param = self.param_trans_bias
    
            m = nn.Softmax(dim=1)
            P0 = m(self.transition(basis0))
            P1 = m(self.transition(basis1))
            P2 = m(self.transition(basis2))
            P = torch.stack([P0, P1, P2]).squeeze()            
            return P.permute(2,3,1,0)

        deriv_P_resp_weight = jacobian(cal_P_given_weight, self.param_trans_weight)
        # Size: (height x width x n_state x n_state) x  n_state x n_kernel x n_kernel x n_state

        # Derivative of P respect to bias
        def cal_P_given_bias(bias):
            for name, param in self.transition.named_parameters():
                if name == 'bias':
                    param = bias
                else:
                    param = self.param_trans_weight
    
            m = nn.Softmax(dim=1)
            P0 = m(self.transition(basis0))
            P1 = m(self.transition(basis1))
            P2 = m(self.transition(basis2))
            P = torch.stack([P0, P1, P2]).squeeze()            
            return P.permute(2,3,1,0)

        deriv_P_resp_bias = jacobian(cal_P_given_bias, self.param_trans_bias)
        # Size: (height x width x n_state x n_state) x  n_state

        ### Phi := P B / b^T u (I -  u b^T / b^T u)
        B = torch.diag_embed(b.squeeze())
        PB_over_bu = torch.matmul(P,B)/bu_k
        I = torch.ones(self.grid_size[0], self.grid_size[1], self.n_state)
        I = torch.diag_embed(I)
        bT = b.permute(0,1,3,2)
        u_bT_over_bu = torch.matmul(u_k, bT)/bu_k
        Phi = torch.matmul(PB_over_bu, I - u_bT_over_bu)

        ### partial f over partial [theta]_l
        # P ( I - B u e^T / b^T u) (partial B / partial [theta]_l) u /  b^T u
        # + (partial P / partial [theta]_l) B u / b^T u
        
        # w_kp1 = Phi w_k + partial f over partial param_obs
        Bu = torch.matmul(B, u_k)
        eT = torch.ones_like(Bu).permute(0,1,3,2)
        BueT_over_bu = torch.matmul(Bu,eT)/bu_k
        temp1 = torch.matmul(P, I - BueT_over_bu)
        temp1_batch = torch.stack([temp1 for i in range(self.n_obs)])
        temp1_batch = torch.stack([temp1_batch for i in range(self.n_state)])
        rho_B_over_rho_param_obs = deriv_b_resp_param_obs.permute(4, 5, 0, 1, 2, 3)
        u_k_over_bu_batch = torch.stack([u_k/bu_k for i in range(self.n_obs)])
        u_k_over_bu_batch = torch.stack([u_k_over_bu_batch for i in range(self.n_state)])
        rho_f_over_rho_param_obs = torch.matmul(temp1_batch, rho_B_over_rho_param_obs*u_k_over_bu_batch)

        w_k_O = self.w_k_O
        Phi_batch = torch.stack([Phi for i in range(self.n_obs)])
        Phi_batch = torch.stack([Phi_batch for i in range(self.n_state)])
        w_kp1_O = torch.matmul(Phi_batch, w_k_O) + rho_f_over_rho_param_obs

        # w_kp1 = Phi w_k + partial f over partial param_weight
        rho_P_over_rho_param_weight = deriv_P_resp_weight.permute(4,5,6,7,0,1,2,3)
        Bu_over_bu_batch = Bu_over_bu.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        Bu_over_bu_batch = Bu_over_bu_batch.repeat(3,3,3,3,1,1,1,1)
        rho_f_over_rho_param_weight = torch.matmul(rho_P_over_rho_param_weight, Bu_over_bu_batch) 

        w_k_T_weight = self.w_k_T_weight
        Phi_batch = Phi.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        Phi_batch = Phi_batch.repeat(3,3,3,3,1,1,1,1)
        w_kp1_T_weight = torch.matmul(Phi_batch, w_k_T_weight) + rho_f_over_rho_param_weight

        # w_kp1 = Phi w_k + partial f over partial param_bias
        rho_P_over_rho_param_bias = deriv_P_resp_bias.permute(4,0,1,2,3)
        Bu_over_bu_batch = Bu_over_bu.unsqueeze(0)
        Bu_over_bu_batch = Bu_over_bu_batch.repeat(3,1,1,1,1)
        rho_f_over_rho_param_bias = torch.matmul(rho_P_over_rho_param_bias, Bu_over_bu_batch) 

        w_k_T_bias = self.w_k_T_bias
        Phi_batch = Phi.unsqueeze(0)
        Phi_batch = Phi_batch.repeat(3,1,1,1,1)
        w_kp1_T_bias = torch.matmul(Phi_batch, w_k_T_bias) + rho_f_over_rho_param_bias











        return 0





if __name__ == "__main__":
    from environment import FireEnvironment
    
    env = FireEnvironment(20, 20)
    hmm_estimator = HMMEstimator(grid_size = (env.map_width, env.map_height), n_obs=3, n_state=3, n_kernel=3)
    
    obs, state = env.reset()
    hmm_estimator.update(obs)


    






