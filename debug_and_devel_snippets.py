import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import random

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ACTION_SET = [[-1, 1],[0, 1],[1, 1],
              [-1, 0],[0, 0],[1, 0],
              [-1,-1],[0,-1],[1,-1]]

N_ACTION = len(ACTION_SET)


class POMDPAgent(nn.Module):

    def __init__(self, grid_size, n_obs, n_state):
        super(POMDPAgent, self).__init__()

        self.grid_size = grid_size

        self.transition = nn.Conv2d(3, 3, 3, stride=1, padding=1).to(DEVICE)

        self.observation_matrix = torch.Tensor([[0.98, 0.98, 0.01],[0.01, 0.01, 0.01],[0.01, 0.01, 0.98]]).to(DEVICE)
        #self.observation_matrix = self.observation_matrix.repeat(self.map_width*self.map_height, 1, 1)


def render(state_grid):
    img = state_grid.data.permute(1,2,0).cpu().numpy().squeeze()
    scale = 4

    dim = (600, 600)
    img_resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("image", img_resized)
    cv2.waitKey(-1)


if __name__ == "__main__":
    from environment import FireEnvironment

    env = FireEnvironment(20, 20)
    obs, state = env.reset()

    for i in range(10):
        act = random.randrange(N_ACTION)
        obs, state = env.step(ACTION_SET[act])

    agent = POMDPAgent(grid_size = (env.map_width, env.map_height), n_obs=3, n_state=3)

    ### The following operation needs to be done in tensor ####
    '''
                B u_k
    u_k+1 = P --------
                b u_k
    '''

    u = torch.ones_like(state)/3
    u_next = torch.ones_like(state)/3

    print(obs.size()) 

    obs_reshape = obs.reshape(-1, 3, 1)

    print(obs_reshape.size())

    obs_bmm_matrix = (agent.observation_matrix.T).repeat(env.map_width*env.map_height, 1, 1)

    b = torch.bmm(obs_bmm_matrix, obs_reshape)

    u_reshape = u.reshape(-1, 3, 1)

    print('b:', b.size())
    print('u_reshape:', u_reshape.size())

    Bu = b*u_reshape

    print('Bu:', Bu.size())

    temp =  torch.sum(Bu, 1).unsqueeze(-1)
    print('temp', temp.size())

    Bu_over_bu = Bu / temp

    print(Bu_over_bu.size())
    print(Bu_over_bu[0])

    test1 = Bu_over_bu.reshape(env.map_width, env.map_height, 3)
    print(test1.size())




    

    IF_PRINT = False

    if True:

        for i in range(env.map_width):
            for j in range(env.map_height):

                u_ij = u[:,i,j]
                if IF_PRINT: print('state est.:', u_ij)

                obs_ij = obs[i][j]
                if IF_PRINT: print('observation:', obs_ij)

                if IF_PRINT:
                    print('observation matrix')
                    print(agent.observation_matrix)

                b_given_obs_ij = torch.matmul(agent.observation_matrix.T, obs_ij)
                if IF_PRINT:
                    print(b_given_obs_ij)
                    print('likelihood given obs_ij for states')
                
                B = torch.diag(b_given_obs_ij)
                if IF_PRINT:
                    print('B')
                    print(B)
                
                Bu = torch.matmul(B, u_ij)
                if IF_PRINT: print('Bu:', Bu)

                bu = torch.sum(Bu)
                Bu_over_bu = Bu/bu
                if IF_PRINT:
                    print(' Bu')
                    print('----')
                    print(' bu')
                    print(Bu_over_bu)

                u_next[:, i, j] = Bu_over_bu

        print('Beysian Update')
        #print(u_next)

        #render(u_next)
    print('u_next:', u_next.size())

    error = u_next.permute(1,2,0) - test1
    print('error:', torch.max(error), torch.min(error))
    print(error)

    render(test1.permute(2,0,1))

    render(u_next)
    




