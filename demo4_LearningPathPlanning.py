'''
Here, now, I am predicting observation instead of increasing the likelihood. It is kind of equivalent.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import cv2, os

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from agent import DynamicAutoEncoder, render, Vehicle, ImageStreamWriter
from dqn_agent import Agent as DQN_Agent
from environment import FireEnvironment
from memory import SingleTrajectoryBuffer
import tqdm, random
from torch.utils.tensorboard import SummaryWriter
 

LR_ESTIMATOR = 0.001
BETAS = (0.5, 0.9)

SETTING = {}
SETTING.update({'lr_optim_dynautoenc':LR_ESTIMATOR})
SETTING.update({'betas_optim_dynautoenc':BETAS})

N_TRAIN_WINDOW = 1000
N_TRAIN_BATCH = 1
N_TRAIN_WAIT = 1000
N_TOTAL_TIME_STEPS =  30000
N_MEMORY_SIZE = 5000

N_LOGGING_PERIOD = 200
N_SAVING_PERIOD = 5000
N_RENDER_PERIOD = 1

FPS=10

def demo4_LearningPathPlanning(setting):

    n_sample = 100

    # Environment
    env = FireEnvironment(64, 64)
    # Vehicle to generate observation mask
    vehicle = Vehicle(n_time_windows=512, grid_size=(64,64), planner_type='Default')
    # Trainer and Estimator
    dyn_autoencoder = DynamicAutoEncoder(SETTING, grid_size = (env.map_width, env.map_height), n_state=3, n_obs=3, encoding_dim=16, gru_hidden_dim=16)
    ### DQN agent  
    dqn_agent = DQN_Agent(state_size=16, action_size=4, replay_memory_size=1000, batch_size=64, gamma=0.99, learning_rate=0.01, target_tau=0.01, update_rate=1, seed=0)
    # Train Data Buffer
    memory = SingleTrajectoryBuffer(N_MEMORY_SIZE)
    # Train Iteration Logger

    writer = SummaryWriter()
    # Video Writier
    video_writer1 = ImageStreamWriter('LearningPlanner.avi', FPS, image_size=(1200,820))

    # Add concat. text
    setting_text = ''
    for k,v in setting.items():
        setting_text += k
        setting_text += ':'
        setting_text += str(v)
        setting_text += '\t'
    writer.add_text('setting', setting_text)

    ########################################
    ### Interacting with the Environment ###
    ########################################
    mask_obs, obs, state = env.reset()
    state_est_grid = dyn_autoencoder.u_k

    ### Loss Monitors ###
    list_loss = []
    list_cross_entropy_loss = []
    list_entropy_loss = []
    list_rewards = []
    list_new_fire_count = []
    list_action = []
    
    ### Filling the Data Buffer ###
    for i in tqdm.tqdm(range(N_TRAIN_WAIT)):         
        map_visit_mask, img_resized =  vehicle.full_mask()
        mask_obs, obs, state, reward, info = env.step(map_visit_mask)
        memory.add(mask_obs.detach().long(), state.detach().long(), map_visit_mask.detach().long())

    for i in tqdm.tqdm(range(N_TOTAL_TIME_STEPS)):

        # determine epsilon-greedy action from current sate
        h_k = dyn_autoencoder.h_k.squeeze().data.cpu().numpy()
        epsilon = 0.1
        action = dqn_agent.act(h_k, epsilon)
        list_action.append(action)   
        
        ### Collect Data from the Env. ###
        map_visit_mask, img_resized = vehicle.plan_a_trajectory(state_est_grid, n_sample, action)
        mask_obs, obs, state, reward, info = env.step(map_visit_mask)
        memory.add(mask_obs.detach().long(), state.detach().long(), map_visit_mask.detach().long())

        ### Run the Estimator ###
        state_est_grid = dyn_autoencoder.step(mask_obs, map_visit_mask)
        h_kp1 = dyn_autoencoder.h_k.squeeze().data.cpu().numpy()

        #### Update the reinforcement learning agent ###
        dqn_agent.step(h_k, action, reward, h_kp1, done=False)

        list_rewards.append(reward)
        list_new_fire_count.append(info['new_fire_count'])

        ################################
        ### Rendering and Save Video ###
        ################################        
        img_env   = env.output_image()
        img_agent = dyn_autoencoder.output_image(state_est_grid)

        # State Est
        #blank = np.zeros((400, 200, 3))
        img_top = img_env  #np.concatenate((blank, img_env[:,:800], blank), axis=1)
        blank = np.zeros((20, 1200, 3))
        img_top = np.concatenate((img_top, blank), axis=0)
        img_top = (img_top*255).astype('uint8')

        img_state_est_grid_uint8 = (img_agent*255).astype('uint8')
        backtorgb = cv2.cvtColor(img_state_est_grid_uint8, cv2.COLOR_GRAY2RGB)
        img_bayes_uint8 = np.concatenate((img_top, backtorgb), axis=0) #<-- to be saved
        render('Dynamic Auto Encoder', img_bayes_uint8, 1)

        # Save video #
        video_writer1.write_image_frame(img_bayes_uint8)

        ### Training ###
        loss_val, loss_val_cross, loss_val_ent, O_np_val =  dyn_autoencoder.update(memory, N_TRAIN_BATCH, N_TRAIN_WINDOW)
        list_loss.append(loss_val)
        list_cross_entropy_loss.append(loss_val_cross)
        list_entropy_loss.append(loss_val_ent)

        if i%N_LOGGING_PERIOD == 0:
            avg_loss = np.mean(np.array(list_loss))
            list_loss = []
            writer.add_scalar('dynautoenc/loss', avg_loss, i)

            avg_loss_cross = np.mean(np.array(list_cross_entropy_loss))
            list_cross_entropy_loss = []
            writer.add_scalar('dynautoenc/crossentropy', avg_loss_cross, i)

            avg_loss_entropy = np.mean(np.array(list_entropy_loss))
            list_entropy_loss = []
            writer.add_scalar('dynautoenc/shannonentropy', avg_loss_entropy, i)

            avg_reward = np.mean(np.array(list_rewards))
            list_rewards = []
            writer.add_scalar('perform/rewards', avg_reward, i)

            avg_new_fire_count = np.mean(np.array(list_new_fire_count))
            list_new_fire_count = []
            writer.add_scalar('perform/new_fire_counts', avg_new_fire_count, i)

            writer.add_scalar('perform/pc_coverd_new_fire', avg_reward/avg_new_fire_count, i)

            action_0_count = list_action.count(0)
            action_1_count = list_action.count(1)
            action_2_count = list_action.count(2)
            action_3_count = list_action.count(3)
            

            writer.add_scalar('action_count/0', action_0_count/len(list_action), i)
            writer.add_scalar('action_count/1', action_1_count/len(list_action), i)
            writer.add_scalar('action_count/2', action_2_count/len(list_action), i)
            writer.add_scalar('action_count/3', action_3_count/len(list_action), i)
            list_action = []

            writer.add_scalar('obs_state0/o00', O_np_val[0][0], i)
            writer.add_scalar('obs_state1/o01', O_np_val[0][1], i)
            writer.add_scalar('obs_state2/o02', O_np_val[0][2], i)
            writer.add_scalar('obs_state0/o10', O_np_val[1][0], i)
            writer.add_scalar('obs_state1/o11', O_np_val[1][1], i)
            writer.add_scalar('obs_state2/o12', O_np_val[1][2], i)
            writer.add_scalar('obs_state0/o20', O_np_val[2][0], i)
            writer.add_scalar('obs_state1/o21', O_np_val[2][1], i)
            writer.add_scalar('obs_state2/o22', O_np_val[2][2], i)

            print('losses at iteration: %d, losses: total %.3f, cross %.3f, shannon %.3f' % (i, avg_loss, avg_loss_cross, avg_loss_entropy))
            print('memory size at iteration: %d, size: %d' % (i, len(memory.obs_memory)))

        if (i+1)%N_SAVING_PERIOD==0:
            f_name = setting['name']
            dyn_autoencoder.save_the_model(i, f_name)
            dqn_agent.save_the_model(i, f_name)

    video_writer1.close()



def main():
    setting = {}
    setting.update({'name':'demo4'})
    demo4_LearningPathPlanning(setting)


    
if __name__ == "__main__":
    main()