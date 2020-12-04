import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import random
import os

from params import TRANSITION_CNN_LAYER_WT, DEVICE, ACTION_SET, OBSERVTAION_MATRIX

IF_PRINT = False
EPS = 1e-10

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

from model import DynamicAutoEncoderNetwork


class DynamicAutoEncoder:

    def __init__(self, setting, grid_size, n_state, n_obs, encoding_dim, gru_hidden_dim):


        self.model = DynamicAutoEncoderNetwork(grid_size, n_state, n_obs, encoding_dim, gru_hidden_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), setting['lr_optim_dynautoenc'], setting['betas_optim_dynautoenc'])

        self.encoding_dim = encoding_dim
        self.grid_size = grid_size
        self.n_state = n_state
        self.n_obs = n_obs
        self.gru_hidden_dim = gru_hidden_dim

        ### States ###
        self.u_k = F.softmax(torch.rand(self.grid_size[0], self.grid_size[1], n_state, 1),2).to(DEVICE)
        self.h_k = torch.rand(1, 1, self.gru_hidden_dim).to(DEVICE)

        

    def save_the_model(self, iteration):
        if not os.path.exists('./save/dynautoenc/'):
            os.makedirs('./save/dynautoenc/')
        f_name = 'dynautoenc_network_param_' +  str(iteration) + '_model.pth'
        torch.save(self.model.state_dict(), './save/dynautoenc/'+f_name)
        print('Model Saved')


    def load_the_model(self, iteration):
        f_path = './save/dynautoenc/dynautoenc_network_param_' +  str(iteration) + '_model.pth'
        self.model.load_state_dict(torch.load(f_path))
        print('Model Loaded')


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
        obs = obs.unsqueeze(-1).detach()
        x = self.model.encoder(obs.permute(3,2,0,1).contiguous())

        ### RNN Step ###
        h0 = self.h_k
        output, h_n = self.model.rnn_layer(x.unsqueeze(0), h0)
        self.h_k = output.detach()

        ### Decoding ###
        output = output[0].unsqueeze(-1).unsqueeze(-1)
        pred_state_est = self.model.decoder(output)
        pred_state_est = pred_state_est[:, :, :self.grid_size[0], :self.grid_size[1]] # Crop Image
        pred_state_est = F.softmax(pred_state_est, 1)
        pred_state_est = pred_state_est.permute(2, 3, 1, 0).contiguous()
        self.u_k = pred_state_est.detach()
        
        return state_est_grid
    
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

    
    def update(self, memory, n_batch, n_window):
        self.model.train()

        batch_obs_stream, batch_state_stream = memory.sample(n_batch, n_window)
        batch_pred_obs_stream = []
        batch_tgt_obs_stream = []

        O = F.softmax(self.model.W_obs_param,0)
        O_np_val = O.data.cpu().numpy()
        O = O.unsqueeze(0).unsqueeze(0).repeat(self.grid_size[0], self.grid_size[1], 1, 1)
        O_bat = O.unsqueeze(0).repeat(n_window, 1, 1, 1, 1)

        ### Foward the observation through the model ###
        for i in range(n_batch):
            ### Encoding and Likelihood ###
            obs_stream = torch.stack(batch_obs_stream[i])
            batch_tgt_obs_stream.append(obs_stream.unsqueeze(-1))
            obs_stream = obs_stream.squeeze().permute(0, 3, 1, 2).contiguous()
            x_stream = self.model.encoder(obs_stream)
            

            ### RNN State Predictor ###
            h0 = torch.rand(1, 1, self.gru_hidden_dim).to(DEVICE)
            output, h_n = self.model.rnn_layer(x_stream.unsqueeze(0), h0)

            ### Decoding ###
            output = output.squeeze().unsqueeze(-1).unsqueeze(-1)
            pred_state_est_stream = self.model.decoder(output)
            pred_state_est_stream = pred_state_est_stream[:, :, :self.grid_size[0], :self.grid_size[1]] # Crop Image
            pred_state_est_stream = F.softmax(pred_state_est_stream, 1)
            pred_state_est_stream = pred_state_est_stream.permute(0,2,3,1).contiguous().unsqueeze(-1)
            pred_obs_stream = torch.matmul(O_bat, pred_state_est_stream)
            batch_pred_obs_stream.append(pred_obs_stream)

        batch_pred_obs_stream = torch.stack(batch_pred_obs_stream)
        batch_tgt_obs_stream = torch.stack(batch_tgt_obs_stream)

        #### Translate one step the target for calculating loss in prediction
        tgt_grid_obs = batch_tgt_obs_stream[:, 1:, :, :, :]
        pred_grid_obs = batch_pred_obs_stream[:, :-1, :, :, :]

        #### Cross Entropy Loss ###
        loss1 = torch.mean(-(tgt_grid_obs*torch.log(pred_grid_obs + EPS)+(1-tgt_grid_obs)*torch.log(1-pred_grid_obs + EPS)))

        ### Shannon entropy loss ###
        p = pred_grid_obs
        log_p = torch.log(p)
        loss2 = -torch.mean(p*log_p)

        loss = loss1 #+ loss2

        ### loss3 ###
        #loss3 = torch.mean(self.model.W_obs_param**2)

        #loss = loss + 0.001*loss3

        ### Update Model ###
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        loss_val = loss.item()
        loss1_val = loss1.item()
        loss2_val = loss2.item()

        del batch_obs_stream, batch_state_stream, batch_pred_obs_stream, batch_tgt_obs_stream, loss, loss1, loss2
        torch.cuda.empty_cache()

        return loss_val, loss1_val, loss2_val, O_np_val



def render (window_name, image, wait_time):
    cv2.imshow(window_name, image)
    cv2.waitKey(wait_time)














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

    






