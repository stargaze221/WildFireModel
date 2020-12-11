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



LR_ESTIMATOR = 0.001
BETAS = (0.5, 0.9)

SETTING = {}
SETTING.update({'lr_optim_dynautoenc':LR_ESTIMATOR})
SETTING.update({'betas_optim_dynautoenc':BETAS})

N_TRAIN_WINDOW = 1000
N_TRAIN_BATCH = 1
N_TRAIN_WAIT = 1000
N_TOTAL_TIME_STEPS = 50000
N_MEMORY_SIZE = 10000

N_LOGGING_PERIOD = 200
N_SAVING_PERIOD = 5000
N_RENDER_PERIOD = 1

def train(fullcover, name, omega):

    n_sample = 20

    # Environment
    env = FireEnvironment(64, 64)

    # Vehicle to generate observation mask
    vehicle = Vehicle(n_time_windows=1024, grid_size=(64,64))

    # Trainer and Estimator
    dyn_autoencoder = DynamicAutoEncoder(SETTING, grid_size = (env.map_width, env.map_height), n_state=3, n_obs=3, encoding_dim=16, gru_hidden_dim=16)

    # Train Data Buffer
    memory = SingleTrajectoryBuffer(N_MEMORY_SIZE)

    # Train Iteration Logger
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()


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

    ### Filling the Data Buffer ###
    for i in tqdm.tqdm(range(N_TRAIN_WAIT)):

        if fullcover:
            map_visit_mask, img_resized = vehicle.plan_a_trajectory(state_est_grid, n_sample, omega)
        else:
            map_visit_mask, img_resized = vehicle.full_mask()

        mask_obs, obs, state = env.step(map_visit_mask)
        memory.add(mask_obs, state, map_visit_mask)
        


    for i in tqdm.tqdm(range(N_TOTAL_TIME_STEPS)):

        ### Collect Data from the Env. ###
        if fullcover:
            map_visit_mask, img_resized = vehicle.full_mask()
        else:
            map_visit_mask, img_resized = vehicle.plan_a_trajectory(state_est_grid, n_sample, omega)
            
        
        mask_obs, obs, state = env.step(map_visit_mask)
        memory.add(mask_obs, state, map_visit_mask)

        ### Run the Estimator ###
        state_est_grid = dyn_autoencoder.step(mask_obs, map_visit_mask)

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
            f_name = name+str(omega)
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

        mask_obs, obs, state = env.step(map_visit_mask)

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
        mask_obs, obs, state = env.step(map_visit_mask)

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

    render('env', img_env, -1)
    render('img_state_est_grid', img_state_est_grid, -1)


    
if __name__ == "__main__":

    name = "Cork"
    omega = 1.0
    n_iteration = 49999

    use_the_model_with_a_planner(name, omega, n_iteration)

    '''

    import names
    
    
    last_name = names.get_last_name()

    

    #use_the_model(44999)

    for i in range(5):
        last_name = names.get_last_name()
        omega = 1.0
        train(False, last_name, omega)
    '''

    
