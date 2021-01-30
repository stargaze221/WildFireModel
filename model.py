import torch
import torch.nn as nn



class DynamicAutoEncoderNetwork(nn.Module):

    def __init__(self, grid_size, n_state, n_obs, encoding_dim, gru_hidden_dim):
        super(DynamicAutoEncoderNetwork, self).__init__()

        ### Likelihood Params ###
        self.W_obs_param = torch.nn.Parameter(torch.randn(n_state, n_obs))
        
        ### Encoding Likelihood ###
        self.encoder = nn.Sequential(
            nn.Conv2d(n_state, 32, 8, stride=4),
            nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, 4, stride=2),
            nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 8, 3, stride=1),
            nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Flatten(),
            nn.Linear(128, encoding_dim)  #<--- 32 is hard-coded as dependent on 448 x 448 x 3.
            )

        ### Reccurent Neural Network ###
        self.rnn_layer = nn.GRU(input_size= encoding_dim, hidden_size=gru_hidden_dim, batch_first=True)

        ngf = 16 # filter size for generator
        nz = gru_hidden_dim
        nc = n_state

        ### Generate State Estimate Grid Map ###
        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.Dropout2d(0.5),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.Dropout2d(0.5),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.Dropout2d(0.5),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.Dropout2d(0.5),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False)
            #nn.Sigmoid()
            # state size. (nc) x 64 x 64
        ) 




# https://github.com/xkiwilabs/DQN-using-PyTorch-and-ML-Agents/blob/master/model.py
"""
Example Neural Network Model for Vector Observation DQN Agent
DQN Model for Unity ML-Agents Environments using PyTorch
Example Developed By:
Michael Richardson, 2018
Project for Udacity Danaodgree in Deep Reinforcement Learning (DRL)
Code expanded and adapted from code examples provided by Udacity DRL Team, 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):

    """
    #################################################
    Initialize neural network model 
    Initialize parameters and build model.
    """
    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128):
        """
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)


    """
    ###################################################
    Build a network that maps state -> action values.
    """
    def forward(self, state):
        
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)