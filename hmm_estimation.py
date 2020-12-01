import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import random

from params import TRANSITION_CNN_LAYER_WT, DEVICE, ACTION_SET, OBSERVTAION_MATRIX



LR_ESTIMATOR = 0.001
BETAS = (0.5, 0.9)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class HMMEstimator(nn.Module):

    def __init__(self, grid_size, n_kernel, n_obs, n_state):
        super(HMMEstimator, self).__init__()

        self.grid_size = grid_size # (w, h)
        self.n_kernel = n_kernel
        self.n_obs = n_obs
        self.n_state = n_state

        # Model Network
        self.transition_cov2d_layer = nn.Conv2d(n_state, n_state, n_kernel, stride=1, padding=1, bias = False).to(DEVICE)
        self.observation_lin_layer = nn.Linear(n_obs, n_state, bias=False).to(DEVICE)
         
        # State Predictor
        self.u_km1 = (torch.rand(grid_size[0], grid_size[1], n_state, 1)).to(DEVICE)
        self.y_km1 = None

        # Optimizer
        params = list(self.transition_cov2d_layer.parameters()) + list(self.observation_lin_layer.parameters())
        self.optimizer = torch.optim.Adam(params, LR_ESTIMATOR, BETAS)

        # Aux
        self.n_iteration = 0

    def output_image(self, size=(1200,400)):
        img = self.u_km1.data.cpu().numpy().squeeze()

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


    def update(self, obs):
        '''
        At current time step k, I can calcualte b(y_k).
        Also, I can use y_km1 and u_km1 to update and calculate u_k for state predictor
        '''

        if self.n_iteration > 0:

            ### Step 1: Bayesian Update for time of k-1 ###
            '''
            B u_km1
            --------
            b u_km1
            '''
            y_km1 = self.y_km1.detach()
            I_obs = torch.eye(self.n_obs).to(DEVICE)
            O = F.softmax(self.observation_lin_layer(I_obs), 0)  # rows: observation, column: state
            #O_bat = O.T.unsqueeze(0).unsqueeze(0).repeat(self.grid_size[0], self.grid_size[1], 1, 1)
            O_bat = O.unsqueeze(0).unsqueeze(0).repeat(self.grid_size[0], self.grid_size[1], 1, 1)
            b_km1 = torch.matmul(O_bat, y_km1) # Size : w x h x n_state x 1

            u_km1 = self.u_km1.detach() # Size : w x h x n_state x 1
            Bu_km1 = b_km1*u_km1 # Size : w x h x n_state x 1
            bu_km1 =  torch.sum(Bu_km1, 2).unsqueeze(-1)
            Bu_over_bu_km1 = Bu_km1 / bu_km1 # Size : w x h x n_state x 1

            ### Step 2: State Predictor ###
            '''       [  B u_km1  ]
            u_k = Phi [-----------]
                      [  b u_km1  ]
            '''
            u_k = F.softmax(self.transition_cov2d_layer(Bu_over_bu_km1.permute(3, 2, 0, 1)), 1).permute(2, 3, 1, 0)

            ### Step 3: Likelihood given y_k ###
            y_k = obs.unsqueeze(-1)
            I_obs = torch.eye(self.n_obs).to(DEVICE)
            O = F.softmax(self.observation_lin_layer(I_obs), 0)  # rows: observation, column: state
            #O_bat = O.T.unsqueeze(0).unsqueeze(0).repeat(self.grid_size[0], self.grid_size[1], 1, 1)
            O_bat = O.unsqueeze(0).unsqueeze(0).repeat(self.grid_size[0], self.grid_size[1], 1, 1)
            b_k = torch.matmul(O_bat, y_k) # Size : w x h x n_state x 1

            ### Step 4: 
            # Optimize Online Likeilhood ###
            '''
            S = log [b(y_k) u_k]
            '''
            S = torch.mean(torch.log(b_k*u_k))
            loss = -S

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            #########################
            ### Update the memory ###
            #########################
            self.y_km1 = y_k
            self.u_km1 = u_k
            self.n_iteration += 1
            loss_val = loss.item()

        else:
            print('Need to fill in memory!')
            #########################
            ### Update the memory ###
            #########################
            self.y_km1 = obs.unsqueeze(-1)
            self.n_iteration += 1
            loss_val = 0


        return loss_val


def render (window_name, image):
    cv2.imshow(window_name, image)
    cv2.waitKey(1)




if __name__ == "__main__":
    from environment import FireEnvironment
    from torch.utils.tensorboard import SummaryWriter
    import tqdm


    writer = SummaryWriter()
    
    env = FireEnvironment(50, 50)
    hmm_estimator = HMMEstimator(grid_size = (env.map_width, env.map_height), n_kernel=3, n_obs=3, n_state=3)
    
    obs, state = env.reset()
    hmm_estimator.update(obs)
    
    list_loss_val = []
    for i in tqdm.tqdm(range(200000)):

        
        
        
        act = random.randrange(len(ACTION_SET))
        obs, state = env.step(ACTION_SET[act])
        loss_val = hmm_estimator.update(obs)
        list_loss_val.append(loss_val)

        ### Rendering

        if i > 2000:
            img_env   = env.output_image()
            img_agent = hmm_estimator.output_image()

            render('env', img_env)
            render('est', img_agent)


        ### Monitoring ###

        if i%100 == 0:

            

            # Average loss
            avg_loss = np.mean(np.array(list_loss_val))
            list_loss_val = []


            
            total_norm = 0
            for p in hmm_estimator.transition_cov2d_layer.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)

            writer.add_scalar('param/trans_grad_norm', total_norm, i+1)

            total_norm = 0
            for p in hmm_estimator.observation_lin_layer.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)


            writer.add_scalar('loss/log_bp', avg_loss, i+1)

            writer.add_scalar('param/obs_grad_norm', total_norm, i+1)

            print(i, avg_loss, total_norm)

            



        






