'''
Here, now, I am predicting observation instead of increasing the likelihood. It is kind of equivalent.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import cv2, os

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from agent import DynamicAutoEncoder, render
from environment import FireEnvironment
from memory import SingleTrajectoryBuffer

import tqdm, random
from params import ACTION_SET

from agent import Vehicle

from dqn_agent import Agent as DQN_Agent


LR_ESTIMATOR = 0.001
BETAS = (0.5, 0.9)

SETTING = {}
SETTING.update({'lr_optim_dynautoenc':LR_ESTIMATOR})
SETTING.update({'betas_optim_dynautoenc':BETAS})

N_TRAIN_WINDOW = 1000
N_TRAIN_BATCH = 1
N_TRAIN_WAIT = 1000
N_TOTAL_TIME_STEPS = 30000
N_MEMORY_SIZE = 5000

N_LOGGING_PERIOD = 200
N_SAVING_PERIOD = 5000
N_RENDER_PERIOD = 1

def train(fullcover, name, setting):

    n_sample = 20

    # Environment
    env = FireEnvironment(64, 64)

    # Vehicle to generate observation mask
    vehicle = Vehicle(n_time_windows=1000, grid_size=(64,64), planner_type=setting['planner_type'])

    # Trainer and Estimator
    dyn_autoencoder = DynamicAutoEncoder(SETTING, grid_size = (env.map_width, env.map_height), n_state=3, n_obs=3, encoding_dim=16, gru_hidden_dim=16)

    # Train Data Buffer
    memory = SingleTrajectoryBuffer(N_MEMORY_SIZE)

    ### DQN agent
    dqn_agent = DQN_Agent(state_size=16, action_size=4, replay_memory_size=1000, batch_size=64, gamma=0.99, learning_rate=0.01, target_tau=0.01, update_rate=1, seed=0)

    # Train Iteration Logger
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()

    # Add concat. text
    setting_text = ''
    for k,v in setting.items():
        setting_text += k
        setting_text += str(v)
        setting_text += '\t'
    writer.add_text('setting', setting_text)


    ########################################
    ### Interacting with the Environment ###
    ########################################
    mask_obs, obs, state = env.reset()
    map_visit_mask, img_resized = vehicle.full_mask()
    state_est_grid = dyn_autoencoder.u_k

    ### Loss Monitors ###
    list_loss = []
    list_cross_entropy_loss = []
    list_entropy_loss = []
    list_rewards = []
    list_count_fire_visit = []
    list_count_all_fire = []
    list_action = []

    ### Filling the Data Buffer ###
    for i in tqdm.tqdm(range(N_TRAIN_WAIT)):         
        if fullcover:
            map_visit_mask, img_resized = vehicle.plan_a_trajectory(state_est_grid, n_sample, action)
        else:
            map_visit_mask, img_resized = vehicle.full_mask()

        mask_obs, obs, state, reward = env.step(map_visit_mask)
        memory.add(mask_obs, state, map_visit_mask)
        


    for i in tqdm.tqdm(range(N_TOTAL_TIME_STEPS)):

        # determine epsilon-greedy action from current sate
        h_k = dyn_autoencoder.h_k.squeeze().data.cpu().numpy()
        epsilon = 0.1
        action = dqn_agent.act(h_k, epsilon)
        list_action.append(action)    

        ### Collect Data from the Env. ###
        if fullcover:
            map_visit_mask, img_resized = vehicle.full_mask()
        else:
            map_visit_mask, img_resized = vehicle.plan_a_trajectory(state_est_grid, n_sample, action)
            
        
        mask_obs, obs, state, reward = env.step(map_visit_mask)
        memory.add(mask_obs, state, map_visit_mask)

        ### Run the Estimator ###
        state_est_grid = dyn_autoencoder.step(mask_obs, map_visit_mask)
        h_kp1 = dyn_autoencoder.h_k.squeeze().data.cpu().numpy()

        #### Update the reinforcement learning agent ###
        dqn_agent.step(h_k, action, reward, h_kp1, done=False)

        list_rewards.append(reward)
        fire_count = (torch.sum(state[2])).item()
        fire_visit = (torch.sum(mask_obs.permute(2,0,1) * state[2].unsqueeze(0))).item()

        if fire_count < 1:
            print('no fire')
        else:
            list_count_fire_visit.append(fire_visit)
            list_count_all_fire.append(fire_count)

        ### Render the Env. and the Est. ###
        if i % N_RENDER_PERIOD == 0:
            img_env   = env.output_image()
            img_state_est_grid = dyn_autoencoder.output_image(state_est_grid)
            
            render('env', img_env, 1)
            render('img_state_est_grid', img_state_est_grid, 1)            


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

            avg_count_fire_visit = np.mean(np.array(list_count_fire_visit))
            list_count_fire_visit = []
            writer.add_scalar('perform/avg_count_fire_visit', avg_count_fire_visit, i)

            avg_count_all_fire = np.mean(np.array(list_count_all_fire))
            list_count_all_fire = []
            writer.add_scalar('perform/avg_count_all_fire', avg_count_all_fire, i)


            action_0_count = list_action.count(0)
            action_1_count = list_action.count(1)
            action_2_count = list_action.count(2)
            action_3_count = list_action.count(3)
            list_action = []

            if setting['planner_type'] == 'Default':
                writer.add_scalar('action_count/0', action_0_count, i)
                writer.add_scalar('action_count/1', action_1_count, i)
                writer.add_scalar('action_count/2', action_2_count, i)
                writer.add_scalar('action_count/3', action_3_count, i)


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
            f_name = name
            dyn_autoencoder.save_the_model(i, f_name)



def use_the_model(name, omega, n_iteration, fullcover=False):

    writer = cv2.VideoWriter("output.avi",cv2.VideoWriter_fourcc(*"MJPG"), 30,(1200,800))

    # Environment
    env = FireEnvironment(64, 64)

    # Vehicle to generate observation mask
    vehicle = Vehicle(n_time_windows=1024, grid_size=(64,64))

    # Load the model
    dyn_autoencoder = DynamicAutoEncoder(SETTING, grid_size = (env.map_width, env.map_height), n_state=3, n_obs=3, encoding_dim=16, gru_hidden_dim=16)
    dyn_autoencoder.load_the_model(name, omega, n_iteration)

    ########################################
    ### Interacting with the Environment ###
    ########################################
    mask_obs, obs, state = env.reset()
    map_visit_mask, img_resized = vehicle.full_mask()
    state_est_grid = dyn_autoencoder.u_k

    for i in tqdm.tqdm(range(2000)):
        ### Collect Data from the Env. ###
        if fullcover:
            map_visit_mask, img_resized = vehicle.full_mask()
        else:
            map_visit_mask, img_resized = vehicle.generate_a_random_trajectory(state_est_grid)

        mask_obs, obs, state, reward = env.step(map_visit_mask)

        ### Run the Estimator ###
        state_est_grid = dyn_autoencoder.step(mask_obs, map_visit_mask)

        ### Render the Env. and the Est. ###
        img_env   = env.output_image()
        img_state_est_grid = dyn_autoencoder.output_image(state_est_grid)
        
        render('env', img_env, 10)
        render('img_state_est_grid', img_state_est_grid, 10)

        ### Save the video
        img_env_uint8 = (img_env*255).astype('uint8')
        img_state_est_grid_uint8 = (img_state_est_grid*255).astype('uint8')
        backtorgb = cv2.cvtColor(img_state_est_grid_uint8,cv2.COLOR_GRAY2RGB)
        img = np.concatenate((img_env_uint8, backtorgb), axis=0)        
        writer.write(img)
    
    writer.release()


def use_the_model_with_a_planner(name, omega, n_iteration):

    writer = cv2.VideoWriter("output.avi",cv2.VideoWriter_fourcc(*"MJPG"), 30,(1200,800))

    # Environment
    env = FireEnvironment(64, 64)

    # Vehicle to generate observation mask
    vehicle = Vehicle(n_time_windows=512, grid_size=(64,64))

    # Load the model
    dyn_autoencoder = DynamicAutoEncoder(SETTING, grid_size = (env.map_width, env.map_height), n_state=3, n_obs=3, encoding_dim=16, gru_hidden_dim=16)
    dyn_autoencoder.load_the_model(name, omega, n_iteration)

    ########################################
    ### Interacting with the Environment ###
    ########################################
    mask_obs, obs, state = env.reset()
    map_visit_mask, img_resized = vehicle.full_mask()
    state_est_grid = dyn_autoencoder.u_k

    for i in tqdm.tqdm(range(2000)):
        ### Collect Data from the Env. ###
        #map_visit_mask, img_resized = vehicle.generate_a_random_trajectory(state_est_grid)
        map_visit_mask, img_resized = vehicle.plan_a_trajectory(state_est_grid, n_sample=100, omega=0.0)
        mask_obs, obs, state, reward = env.step(map_visit_mask)

        ### Run the Estimator ###
        state_est_grid = dyn_autoencoder.step(mask_obs, map_visit_mask)

        ### Render the Env. and the Est. ###
        img_env   = env.output_image()
        img_state_est_grid = dyn_autoencoder.output_image(state_est_grid)
        
        render('env', img_env, 1)
        render('img_state_est_grid', img_state_est_grid, 1)

        ### Save the video
        img_env_uint8 = (img_env*255).astype('uint8')
        img_state_est_grid_uint8 = (img_state_est_grid*255).astype('uint8')
        backtorgb = cv2.cvtColor(img_state_est_grid_uint8,cv2.COLOR_GRAY2RGB)
        img = np.concatenate((img_env_uint8, backtorgb), axis=0)        
        writer.write(img)

    writer.release()

    render('env', img_env, 1)
    render('img_state_est_grid', img_state_est_grid, 1)


    
if __name__ == "__main__":

    import names

    setting = {}
    planner_list = ['Random', 'Default', 'VisitingGrayArea']
    for i in range(3):
        for planner_type in planner_list:
            
            last_name = names.get_last_name()
            setting.update({'planner_type':planner_type})
            setting.update({'name':last_name})
            train(False, last_name, setting)
