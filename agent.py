'''
# agent.py
# author: Hyung Jin
#date: 01/13/2020

List of Agents:
- BaysianEstimator: Model based state estimation (that needs the true model parameters).
- 
'''


#Modules to import

import numpy as np
import cv2, os, random

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1234)


from params import TRANSITION_CNN_LAYER_WT, DEVICE, ACTION_SET, OBSERVTAION_MATRIX
from model import DynamicAutoEncoderNetwork



IF_PRINT = False
EPS = 1e-10 # To avoid NAN in Log function.


'''
#Notes: 
'''


'''
class: BayesianEstimator
inherits: nn.Module 
Description: 
'''
class BaysianEstimator(nn.Module):

    '''
    Function: Constructor for BayesianEstimator 
    Params:
        grid_size: size of the grid
        n_obs: number of observations 
        n_state: number of states
    
    return: none 
    
    1. Inherits from nn.module 
    2. Set up the grid size, state transition, observation and estimation 
        matrices 
    '''
    def __init__(self, grid_size, n_obs, n_state):
        super(BaysianEstimator, self).__init__()

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
        

    '''
    Function: output_image
    Params:
        size: size of the image 

    return: the image state
    
    1. returns the image of the states
    '''
    def output_image(self, size=(1200,400)):
        img = self.state_est.data.cpu().numpy().squeeze()

        img_state0 = img[:,:,0]
        img_state1 = img[:,:,1]
        img_state2 = img[:,:,2]

        blank = np.zeros((self.grid_size[1], int(self.grid_size[0]/20)))
        img = np.concatenate((img_state0, blank, img_state1, blank, img_state2), axis=1)
        img_resized = cv2.resize(img, size, interpolation = cv2.INTER_AREA)

        return img_resized



    '''
    Function: Bayesian_update 
    Params:
        obs: observation 
        
    return: state estimation 
    

    1. Bayesian recursive update using the formula: 
                   B u_k
        u_k+1 = P --------
                   b u_k
                      
    '''

    def Bayesian_update(self, obs):
        '''
        Beysian recursive update using the following formula
                   B u_k
        u_k+1 = P --------
                   b u_k
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



'''
class: DynamicAutoEncoder
inherits: none 
Description: 
'''
class DynamicAutoEncoder:

    '''
    Function: Constructor for DynamicAutoEncoder
    Params:
        setting: settings for torch
        grid_size: size of the grid
        n_obs: number of observations 
        n_state: number of states
        encoding_dim: dimensions of the encoder 
        gru_hidden_dim: dimensions of the hidden layer
    
    return: none 
    
    1. sets up the appropriate objects and their values for the class
        matrices 
    '''
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

        
    '''
    Function: save_the_model
    Params:
        iteration: iteration 
        f_name: name of the file to be appended to save as

    return: none
    
    1. Saves the model into a file
    '''
    def save_the_model(self, iteration, f_name):
        if not os.path.exists('./save/dynautoenc/'):
            os.makedirs('./save/dynautoenc/')
        f_name = 'dynautoenc_network_param_' +  str(iteration) + '_' + f_name + '_model.pth'
        torch.save(self.model.state_dict(), './save/dynautoenc/'+f_name)
        print('Model Saved')



    '''
    Function: load_the_model
    Params:
        iteration: iteration 
        f_name: name of the file to be appended to load model

    return: none
    
    1. loads the model from a file
    '''
    def load_the_model(self, iteration, f_name):
        f_path = './save/dynautoenc/dynautoenc_network_param_' +  str(iteration) + '_' + f_name + '_model.pth'
        self.model.load_state_dict(torch.load(f_path))
        print('Model Loaded')


    '''
    Function: step
    Params:
        obs: observation
        mask: mask to use 

    return: state estimation grid
    
    1. Estimate the state 
    2. Run the RNN to get the state estimate grid with the likelihood
    '''
    def step(self, obs, mask):
        with torch.no_grad():
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
            Bu_over_bu[Bu_over_bu != Bu_over_bu] = 0 

            ### Masking ###
            mask = mask.unsqueeze(-1).unsqueeze(-1)
            state_est_grid = self.u_k*(1-mask)
            state_est_grid = state_est_grid + Bu_over_bu*mask

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
    
     
    '''
    Function: output_image
    Params:
        state_est_grid: size of the state estimation grid

    return: the state estimation grid image 
    
    1. returns the image of the state estimation grid
    '''
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



    '''
    Function: update
    Params:
        memory: list of observations and state 
        n_batch: sequence batch for the memory sample
        n_window: length of the sample window

    return: (loss_val, loss1_val, loss2_val, O_np_val)
        loss_val:
        loss1_val: 
        loss2_val:
        O_np_val: 
    
    '''    
    def update(self, memory, n_batch, n_window, update=True):
        

        batch_obs_stream, batch_state_stream, batch_mask_stream = memory.sample(n_batch, n_window)
        batch_pred_obs_stream = []
        batch_tgt_obs_stream = []
        batch_tgt_mask_stream = []

        O = F.softmax(self.model.W_obs_param,0)
        O_np_val = O.data.cpu().numpy()
        O = O.unsqueeze(0).unsqueeze(0).repeat(self.grid_size[0], self.grid_size[1], 1, 1)
        O_bat = O.unsqueeze(0).repeat(n_window, 1, 1, 1, 1)

        if update:
            ### Foward the observation through the model ###
            self.model.train()
            for i in range(n_batch):
                ### Encoding and Likelihood ###
                obs_stream = torch.stack(batch_obs_stream[i]).float()
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

                ### Mask ###
                mask_stream = torch.stack(batch_mask_stream[i])
                batch_tgt_mask_stream.append(mask_stream.unsqueeze(-1).unsqueeze(-1))

            batch_pred_obs_stream = torch.stack(batch_pred_obs_stream)
            batch_tgt_obs_stream = torch.stack(batch_tgt_obs_stream)
            batch_tgt_mask_stream = torch.stack(batch_tgt_mask_stream)

            #### Translate one step the target for calculating loss in prediction
            tgt_mask_obs = batch_tgt_mask_stream[:, 1:, :, :, :]
            tgt_grid_obs = batch_tgt_obs_stream[:, 1:, :, :, :]
            pred_grid_obs = batch_pred_obs_stream[:, :-1, :, :, :]

            mask_tgt_grid_obs = tgt_grid_obs*tgt_mask_obs
            mask_pred_grid_obs = pred_grid_obs*tgt_mask_obs
            

            #### Cross Entropy Loss ###
            loss1 = torch.sum(-(mask_tgt_grid_obs*torch.log(mask_pred_grid_obs + EPS)+(1-mask_tgt_grid_obs)*torch.log(1-mask_pred_grid_obs + EPS)))/torch.sum(tgt_mask_obs)

            ### Ordering property loss ###
            #O = O[0][0]
            #loss3 = torch.sign(O[0][1] -  O[0][0]) + torch.sign(O[0][2] -  O[2][2])

            ### Shannon entropy loss ###
            p = pred_grid_obs
            log_p = torch.log(p)
            loss2 = -torch.mean(p*log_p)

            loss = loss1 # + 0.001*loss3

            

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

        return 0, 0, 0, 0



'''
class: ImageStreamWriter
inherits: none 
Description: 
'''
class ImageStreamWriter:

    '''
    Function: Constructor for ImageStreamWriter
    Params:
        f_path: video file name to write to
        fps: fps of the video file
        image_size: image size of each frame in the video
    
    return: none 
    
    1. sets up the video writer to be able to write the sequence of images  as
        a video file
    '''
    def __init__(self, f_path, fps, image_size):
        self.writer = cv2.VideoWriter(f_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, image_size)


    '''
    Function: write_image_frame
    Params:
        image_frame: image frame to write to the video file

    return: none
    
    1. writes the image to the video file
    '''
    def write_image_frame(self, image_frame):
        self.writer.write(image_frame)


    '''
    Function: close
    Params: none

    return: none
    
    1. releases/closes the video writer
    '''
    def close(self):
        self.writer.release()



'''
Function: render
Params: 
    window_name: name of the window to display the images to 
    image: current image to display 
    wait_time: time for next block of code to execute/image to be displayed for

return: none

1. display/render the image to the screen
'''
def render(window_name, image, wait_time):
    cv2.imshow(window_name, image)
    cv2.waitKey(wait_time)




'''
class: Vehicle
inherits: none 
Description: 
'''
class Vehicle:
    
    '''
    Function: Constructor for Vehicle
    Params:
        n_time_windows: number of time windows
        grid_size: grid size for the state
        planner_type: the planner type
    
    return: none 
    
    1. sets up the appropriate objects and their values for the Vehicle class
    '''    
    def __init__(self, n_time_windows=128, grid_size=(64, 64), planner_type='Default'):

        self.grid_size = grid_size

        self.n_time_windows = n_time_windows
        self.position_state = np.zeros(2) #torch.zeros(2,1)

        self.action_set = [[0, 1],[-1, 0],[1, 0],[0,-1]]
        self.n_action = len(self.action_set)

        ### Planenr type ###
        self.planner_type = planner_type

    '''
    Function: full_mask
    Params: none
    
    return: 
        mask: the mask created
        img_resized: resized image with area interpolation
    
    1. generates the mask and image
    '''  
    def full_mask(self):
        mask = np.ones(self.grid_size)

        dim = (400, 400)
        img_resized = cv2.resize(mask, dim, interpolation = cv2.INTER_AREA)
        mask = torch.FloatTensor(mask).to(DEVICE)

        return mask, img_resized


    '''
    Function: generate_a_random_trajectory
    Params: 
        state_est_map: state estimation map
    
    return: 
        mask: the modified mask 
        img_resized: resized image with area interpolation
    
    1. Generate a random trajectory based on the action sets 
    '''  
    def generate_a_random_trajectory(self, stat_est_map=None):

        ACTION_SET = self.action_set
        mask = np.zeros(self.grid_size)

        for t in range(self.n_time_windows):
            act = random.randrange(len(ACTION_SET))
            self.position_state += np.array(ACTION_SET[act])
            self.position_state = np.clip(self.position_state, 0, self.grid_size[0]-1)

            pos_i = int(self.position_state[0])
            pos_j = int(self.position_state[1])
            mask[pos_i][pos_j] = 1

        dim = (400, 400)
        img_resized = cv2.resize(mask, dim, interpolation = cv2.INTER_AREA)
        mask = torch.FloatTensor(mask).to(DEVICE)

        return mask, img_resized


    '''
    Function: plan_a_trajectory
    Params: 
        state_est_map: state estimation map
        n_sample: number of samples to sample trajectory off of
        action_param: action params for the trajectory
    
    return: 
        mask: the modified mask 
        img_resized: resized image with area interpolation
    
    1. Generate a trajectory based on state map and action sample set
    '''  
    def plan_a_trajectory(self, stat_est_map, n_sample, action_param):
        with torch.no_grad():

            #u_basis = (torch.LongTensor(self.action_set).T).to(DEVICE).detach()
            u_basis = torch.LongTensor([[1,0],[-1,0],[0,1],[0,-1]]).T.to(DEVICE).detach()
            position_state = torch.LongTensor(self.position_state).unsqueeze(-1).to(DEVICE).detach()

            ### Random Action Stream ###
            action_samples = torch.randint(low=0, high=self.n_action, size=(n_sample, self.n_time_windows)).to(DEVICE).long().detach()
            action_samples_onehot = torch.nn.functional.one_hot(action_samples, num_classes=self.n_action).unsqueeze(-1)
            u_basis_repeat = u_basis.unsqueeze(0).unsqueeze(0)  
            u_basis_repeat = u_basis_repeat.repeat(n_sample, self.n_time_windows, 1, 1)
            del action_samples, u_basis

            ### Integrate the Action Stream and Add it to the State ###
            action_sum = torch.matmul(u_basis_repeat.float(), action_samples_onehot.float()).detach()
            action_sum = torch.cumsum(action_sum, 1)
            trajectories = position_state + action_sum
            trajectories = torch.clamp(trajectories, 0, self.grid_size[0]-1)
            trajectories = trajectories.permute(1,0,2,3).long().to(DEVICE).detach() # n_time x n_sample x n_coord x 1
            terminal_positions = trajectories[-1]

            ### Cacaluate Visit Counter Map ###
            map_visted_binary = torch.zeros((n_sample, self.grid_size[0], self.grid_size[1])).to(DEVICE).long()
            T = trajectories.size()[0]
            for t in range(T):
                indice = trajectories[t]
                i_indice_onehot = torch.nn.functional.one_hot(indice[:,0], num_classes=self.grid_size[0]).repeat(1, self.grid_size[1], 1).int().detach()
                j_indice_onehot = torch.nn.functional.one_hot(indice[:,1], num_classes=self.grid_size[1]).repeat(1, self.grid_size[0], 1).permute(0, 2, 1).int().detach()
                positions_onehot = i_indice_onehot * j_indice_onehot
                map_visted_binary += positions_onehot
            map_visted_binary = torch.clamp(map_visted_binary, 0, 1).float()

            del trajectories, position_state, action_sum, i_indice_onehot, j_indice_onehot, positions_onehot
            torch.cuda.empty_cache()

            '''
            i_indice_onehot = torch.nn.functional.one_hot(trajectories[:,:,0], num_classes=self.grid_size[0]).repeat(1, 1, self.grid_size[1], 1).int().detach()
            j_indice_onehot = torch.nn.functional.one_hot(trajectories[:,:,1], num_classes=self.grid_size[1]).repeat(1, 1, self.grid_size[0], 1).permute(0, 1, 3, 2).int().detach()
            positions_onehot = i_indice_onehot * j_indice_onehot
            del trajectories, position_state, action_sum, i_indice_onehot, j_indice_onehot
            torch.cuda.empty_cache()
            map_visted_binary = torch.clamp(torch.sum(positions_onehot, dim=0), 0, 1).float()
            '''
            
            ### Calcualte the Reward to Maximize given Map Visted Counter and State Estimate ###
            '''
            Uncertain grid visit reward
            '''
            # Uncertainty Reward - Shannon entropy
            p = stat_est_map.squeeze().detach()
            log_p = torch.log(p).detach()
            uncertainty_grid = -torch.mean(p*log_p, 2).detach()
            uncertainty_grid = uncertainty_grid.unsqueeze(0).repeat(n_sample,1,1)
            uncertainty_reward = torch.sum(uncertainty_grid*map_visted_binary, dim=(1,2))


            ### Rewards based on the action and settings
            rewards = torch.sum(map_visted_binary, dim=(1,2)) 

            if self.planner_type == 'Random':
                rewards += 0

            elif self.planner_type == 'VisitingGrayArea':
                rewards += uncertainty_reward

            else:    
                if action_param == 0:
                    '''
                    Reward for visiting state estimate of 0 grid
                    '''
                    state0_prob = stat_est_map[:,:,0].squeeze().unsqueeze(0).repeat(n_sample,1,1).detach()
                    rewards += torch.sum(state0_prob*map_visted_binary, dim=(1,2))
                    del state0_prob

                elif action_param == 1:
                    '''
                    Reward for visiting state estimate of 1 grid
                    '''
                    state1_prob = stat_est_map[:,:,1].squeeze().unsqueeze(0).repeat(n_sample,1,1).detach()
                    rewards += torch.sum(state1_prob*map_visted_binary, dim=(1,2))
                    del state1_prob

                elif action_param == 2:
                    '''
                    Reward for visiting state estimate of 2 grid
                    '''
                    state2_prob = stat_est_map[:,:,2].squeeze().unsqueeze(0).repeat(n_sample,1,1).detach()
                    rewards +=  torch.sum(state2_prob*map_visted_binary, dim=(1,2))
                    del state2_prob

                else:
                    '''
                    Uncertain grid visit reward
                    '''
                    rewards += uncertainty_reward
        
            
            indice = torch.argmax(rewards)
            max_val = rewards[indice]

            ### Return the best one out of the random search ###
            mask = map_visted_binary[indice].float().detach()
            dim = (400, 400)
            img_resized = cv2.resize(mask.cpu().data.numpy(), dim, interpolation = cv2.INTER_AREA)

            ### Update the initial position for the next iteration ###
            self.position_state = terminal_positions[indice].squeeze().cpu().data.numpy()

            ### Delete Torch Tensors being Accumulated ###
            del terminal_positions, rewards, map_visted_binary, indice, uncertainty_reward, stat_est_map
            torch.cuda.empty_cache()

        return mask, img_resized


            

'''
Function: main
Params: none

return: none

1. used for testing
2. new_tracjectory is not a valid function of vehicle (NOTE, NEEDS TO BE FIXED)
'''
def main():
    vehicle = Vehicle()
    n_sample = 100
    action_param = 3

    stat_est_map = F.softmax(torch.rand(64, 64, 3, 1),2).to(DEVICE)

    vehicle.new_trajectory(stat_est_map, n_sample, action_param)   


if __name__ == "__main__":
    main()