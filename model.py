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
            #nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, 4, stride=2),
            #nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 8, 3, stride=1),
            #nn.Dropout2d(0.5),
            nn.ReLU(),
            #nn.BatchNorm2d(8),
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
            #nn.Dropout2d(0.5),
            #nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            #nn.Dropout2d(0.5),
            #nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.Dropout2d(0.5),
            #nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            #nn.Dropout2d(0.5),
            #nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False)
            #nn.Sigmoid()
            # state size. (nc) x 64 x 64
        ) 
