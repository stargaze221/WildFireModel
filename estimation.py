'''
Computation Graphs for Dynamic Autoencoder, Actor-Critic
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import cv2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DynamicAutoEncoderNetwork(nn.Module):

    def __init__(self, grid_size, n_state, n_obs, encoding_dim, gru_hidden_dim):
        super(DynamicAutoEncoderNetwork, self).__init__()

        ### Likelihood Params ###
        self.W_obs_param = torch.nn.Parameter(torch.randn(n_state, n_obs))

        ### Encoding Likelihood ###
        self.encoder = nn.Sequential(
            nn.Conv2d(n_state, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 16, 4, stride=2), nn.ReLU(),
            nn.Conv2d(16, 8, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32, encoding_dim)  #<--- 32 is hard-coded as dependent on 448 x 448 x 3.
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
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        ) 



LR_ESTIMATOR = 0.001
BETAS = (0.5, 0.9)
EPS = 1e-10



class DynamicAutoEncoder:

    def __init__(self, grid_size, n_state, n_obs, encoding_dim, gru_hidden_dim):
        self.model = DynamicAutoEncoderNetwork(grid_size, n_state, n_obs, encoding_dim, gru_hidden_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), LR_ESTIMATOR, BETAS)

        self.encoding_dim = encoding_dim
        self.grid_size = grid_size
        self.n_state = n_state
        self.n_obs = n_obs
        self.gru_hidden_dim = gru_hidden_dim

        ### States ###
        self.u_k = F.softmax(torch.rand(self.grid_size[0], self.grid_size[1], n_state, 1),2).to(DEVICE)
        self.h_k = torch.zeros(1, 1, self.gru_hidden_dim).to(DEVICE)

    def step(self, obs):
        self.model.eval()

        ### Likelihood ###
        O_T = F.softmax(self.model.W_obs_param,0).T # Transpose of the observation matrix
        O_T = O_T.unsqueeze(0).unsqueeze(0).repeat(self.grid_size[0], self.grid_size[1], 1, 1)
        y = obs.unsqueeze(-1).detach()
        b = torch.matmul(O_T, y)


        ### State Estimate ###
        Bu = b*self.u_k # Size : w x h x n_state x 1
        bu =  torch.sum(Bu, 2).unsqueeze(-1)
        Bu_over_bu = Bu / bu # Size : w x h x n_state x 1
        state_est_grid = Bu_over_bu

        ### Encoding the Likelihood ###
        x = self.model.encoder(b.permute(3,2,0,1))

        ### RNN Step ###
        h0 = self.h_k
        output, h_n = self.model.rnn_layer(x.unsqueeze(0), h0)
        self.h_k = output

        ### Decoding ###
        output = output[0].unsqueeze(-1).unsqueeze(-1)
        pred_state_est = self.model.decoder(output)
        pred_state_est = pred_state_est[:, :, :self.grid_size[0], :self.grid_size[1]] # Crop Image
        pred_state_est = F.softmax(pred_state_est, 1)
        pred_state_est = pred_state_est.permute(2, 3, 1, 0)
        self.u_k = pred_state_est

        return pred_state_est
    
    def output_image(self, state_est_grid, size=(1200,400)):
        img = state_est_grid.data.cpu().numpy().squeeze()

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
        
        

    def calculate_likelihood_tensor(self, obs):
        O_T = F.softmax(self.model.W_obs_param,0).T # Transpose of the observation matrix
        O_T = O_T.unsqueeze(0).unsqueeze(0).repeat(self.grid_size[0], self.grid_size[1], 1, 1)
        O_T_bat = O_T.unsqueeze(0).repeat(obs.size()[0], 1, 1, 1, 1)
        y = obs.unsqueeze(-1).detach()
        b = torch.matmul(O_T_bat, y)
        return b # Size: h x w x n_state x 1
    
    def update(self, memory):
        self.model.train()

        batch_obs_stream, batch_state_stream = memory.sample(N_BATCH, N_SAMPLE_WINDOW)
        batch_pred_state_est_stream = []
        batch_b_stream = []

        ### Foward the observation through the model ###
        for i in range(N_BATCH):
            ### Encoding and Likelihood ###
            obs_stream = torch.stack(batch_obs_stream[i])
            b_stream = self.calculate_likelihood_tensor(obs_stream)
            b_stream = b_stream.squeeze().permute(0, 3, 1, 2).contiguous()
            x_stream = self.model.encoder(b_stream)
            batch_b_stream.append(b_stream)

            ### RNN State Predictor ###
            h0 = torch.zeros(1, 1, self.gru_hidden_dim).to(DEVICE)
            output, h_n = self.model.rnn_layer(x_stream.unsqueeze(0), h0)

            ### Decoding ###
            output = output.squeeze().unsqueeze(-1).unsqueeze(-1)
            pred_state_est_stream = self.model.decoder(output)
            pred_state_est_stream = pred_state_est_stream[:, :, :self.grid_size[0], :self.grid_size[1]] # Crop Image
            pred_state_est_stream = F.softmax(pred_state_est_stream, 2)
            batch_pred_state_est_stream.append(pred_state_est_stream)

        batch_pred_state_est_stream = torch.stack(batch_pred_state_est_stream)
        batch_b_stream = torch.stack(batch_b_stream)
        

        #### Translate one step the target for calculating loss in prediction
        batch_b_ks = batch_b_stream[:, 1:, :, :, :]
        batch_u_ks = batch_pred_state_est_stream[:, :-1, :, :, :]  # n_batch x n_window x n_state x grid_size[0] x grid_size[1]


        
        #### Cross Entropy Loss ###
        loss = torch.mean(-(batch_b_ks*torch.log(batch_u_ks + EPS)+(1-batch_b_ks)*torch.log(1-batch_u_ks + EPS)))
        



        '''
        ### Calculate the entropy ###
        p = batch_u_ks
        log_p = torch.log(p)
        entropy = -torch.mean(p*log_p)
        '''


        ### Calculate the loss function using E [Log b u]
        #S = torch.mean(torch.log(batch_b_ks) + torch.log(batch_u_ks))
        #loss = -S
        

        ### Update Model ###
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_val = loss.item()

        del batch_obs_stream, batch_state_stream, batch_pred_state_est_stream, batch_b_stream, loss
        torch.cuda.empty_cache()

        return loss_val

def render (window_name, image):
    cv2.imshow(window_name, image)
    cv2.waitKey(1)


        
    
if __name__ == "__main__":
    from environment import FireEnvironment
    from memory import SingleTrajectoryBuffer
    import tqdm, random
    from params import ACTION_SET

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()

    N_MEMORY_SIZE = 20000
    N_SAMPLE_WINDOW = 50
    N_BATCH = 4
    N_TOTAL_TIME_STEPS = 20000

    env = FireEnvironment(50, 50)
    dyn_autoencoder = DynamicAutoEncoder(grid_size = (env.map_width, env.map_height), n_state=3, n_obs=3, encoding_dim=16, gru_hidden_dim=16)
    memory = SingleTrajectoryBuffer(N_MEMORY_SIZE)
    
    obs, state = env.reset()

    list_loss = []

    for i in tqdm.tqdm(range(N_TOTAL_TIME_STEPS)):
        act = random.randrange(len(ACTION_SET))
        obs, state = env.step(ACTION_SET[act])

        state_est_grid = dyn_autoencoder.step(obs)

        if i%5 == 0:
            img_env   = env.output_image()
            img_agent = dyn_autoencoder.output_image(state_est_grid)

            render('env', img_env)
            render('est', img_agent)


        memory.add(obs, state)
        if i > int(N_MEMORY_SIZE/10):
            loss_val =  dyn_autoencoder.update(memory)
            list_loss.append(loss_val)

            if i%100 == 0:
                list_loss = np.array(list_loss)
                avg_loss = np.mean(list_loss)
                writer.add_scalar('/dynautoenc/loss', avg_loss, i)
                list_loss = []
                print(i, 'th loss:', avg_loss)






    




#self.observation_lin_layer = nn.Linear(n_obs, n_state).to(DEVICE)


### Encoding Likelihood ###



'''

ngf = 16 # filter size for generator
nc = 3 # n color chennal (RGB)

### Image Encoder ###
self.encoder = nn.Sequential(
    nn.Conv2d(3, 32, 12, stride=5), nn.BatchNorm2d(32), nn.ReLU(),
    nn.Conv2d(32, 64, 8, stride=4), nn.BatchNorm2d(64), nn.ReLU(),
    nn.Conv2d(64, 32, 4, stride=2), nn.BatchNorm2d(32), nn.ReLU(),
    nn.Conv2d(32, 16, 3, stride=1), nn.BatchNorm2d(16), nn.ReLU(),
    nn.Flatten(),
    nn.Linear(784, self.encoding_dim)  #<--- 784 is hard-coded as dependent on 448 x 448 x 3.
)

### State Predictor Given Prvious State and Current Encoded Image and Action ###
self.gru_hidden_dim = self.encoding_dim
self.rnn_layer = nn.GRU(input_size=self.encoding_dim + self.action_dim, hidden_size=self.gru_hidden_dim, batch_first=True) 

### Image Reconstructed from the State Predictors ###
self.decoder = nn.Sequential(
    # input is Z, going into a convolutionc
    nn.ConvTranspose2d( self.gru_hidden_dim, ngf * 8, 4, 1, 0, bias=False),
    nn.BatchNorm2d(ngf * 8), nn.ReLU(True),
    nn.ConvTranspose2d(ngf * 8, ngf * 8, 5, 2, 1, bias=False),
    nn.BatchNorm2d(ngf * 8), nn.ReLU(True),
    nn.ConvTranspose2d(ngf * 8, ngf * 4, 5, 2, 1, bias=False),
    nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
    nn.ConvTranspose2d( ngf * 4, ngf * 4, 4, 2, 1, bias=False),
    nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
    nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
    nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
    nn.ConvTranspose2d( ngf * 2, ngf, 7, 3, 1, bias=False),
    nn.BatchNorm2d(ngf), nn.ReLU(True),
    nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
    nn.Tanh()
)
'''

