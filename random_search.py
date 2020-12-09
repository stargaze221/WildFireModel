import numpy as np
import torch
import cv2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SimpleTrajectoryPlanner:

    def __init__(self, n_time_windows=500, grid_size=(64, 64)):

        self.grid_size = grid_size
        self.n_time_windows = n_time_windows
        self.position_state = torch.zeros(2,1)+32
        self.n_action = 4
        self.u_basis = torch.LongTensor([[1,0],[-1,0],[0,1],[0,-1]]).T

        # Memories
        self.position_state.to(DEVICE)
        self.u_basis.to(DEVICE)


    def sample_trajectories(self, n_sample=50):

        action_samples = torch.randint(low=0, high=self.n_action, size=(n_sample, self.n_time_windows))
        action_samples_onehot = torch.nn.functional.one_hot(action_samples, num_classes=self.n_action).unsqueeze(-1)

        u_basis_repeat = self.u_basis.unsqueeze(0).unsqueeze(0)  
        u_basis_repeat = u_basis_repeat.repeat(n_sample, self.n_time_windows, 1, 1)

        action_sum = torch.matmul(u_basis_repeat, action_samples_onehot)
        action_sum = torch.cumsum(action_sum, 1)

        trajectories = self.position_state + action_sum
        trajectories = torch.clamp(trajectories, 0, self.grid_size[0]-1)
        trajectories = trajectories.permute(1,0,2,3).long().to(DEVICE) # n_time x n_sample x n_coord x 1

        i_indice_onehot = torch.nn.functional.one_hot(trajectories[:,:,0], num_classes=self.grid_size[0]).repeat(1, 1, self.grid_size[1], 1)
        j_indice_onehot = torch.nn.functional.one_hot(trajectories[:,:,1], num_classes=self.grid_size[1]).repeat(1, 1, self.grid_size[0], 1).permute(0, 1, 3, 2)
        positions_onehot = i_indice_onehot * j_indice_onehot
        del i_indice_onehot, j_indice_onehot

        map_visted_counter = torch.clamp(torch.sum(positions_onehot, dim=0), 0, 1).float()
        

        

        

        




        

        

if __name__ == "__main__":

    planner = SimpleTrajectoryPlanner()

    for i in range(100):
        print(i)
        planner.sample_trajectories()
    

    '''
    batch_size=10
    n_classes=5
    target = torch.randint(high=5, size=(1,10)) # set size (2,10) for MHE
    print(target)
    y = torch.zeros(batch_size, n_classes)
    y[range(y.shape[0]), target]=1
    print(y)
    '''





