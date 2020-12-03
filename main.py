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



LR_ESTIMATOR = 0.001
BETAS = (0.5, 0.9)

SETTING = {}
SETTING.update({'lr_optim_dynautoenc':LR_ESTIMATOR})
SETTING.update({'betas_optim_dynautoenc':BETAS})

N_TRAIN_WINDOW = 500
N_TRAIN_BATCH = 1
N_TRAIN_WAIT = 1000
N_TOTAL_TIME_STEPS = 50000
N_MEMORY_SIZE = 10000

N_LOGGING_PERIOD = 200
N_SAVING_PERIOD = 5000
N_RENDER_PERIOD = 10


def train():
    # Environment
    env = FireEnvironment(64, 64)

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
    obs, state = env.reset()

    ### Loss Monitors ###
    list_loss = []
    list_cross_entropy_loss = []
    list_entropy_loss = []

    ### Filling the Data Buffer ###
    for i in tqdm.tqdm(range(N_TRAIN_WAIT)):
        memory.add(obs, state)
        act = random.randrange(len(ACTION_SET))
        obs, state = env.step(ACTION_SET[act])


    for i in tqdm.tqdm(range(N_TOTAL_TIME_STEPS)):

        ### Collect Data from the Env. ###
        memory.add(obs, state)
        act = random.randrange(len(ACTION_SET))
        obs, state = env.step(ACTION_SET[act])

        ### Run the Estimator ###
        state_est_grid = dyn_autoencoder.step(obs)

        ### Render the Env. and the Est. ###
        if i % N_RENDER_PERIOD == 0:
            img_env   = env.output_image()
            img_agent = dyn_autoencoder.output_image(state_est_grid)
            render('env', img_env)
            render('est', img_agent)


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
            dyn_autoencoder.save_the_model(i)
    
if __name__ == "__main__":
    train()