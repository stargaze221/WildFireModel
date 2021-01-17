'''
memory.py
author: Hyung Jin
date: 01/13/2020
'''

#modules to import
import numpy as np
import random
from collections import namedtuple, deque
import itertools



'''
#Notes: 
'''


'''
class: SingleTrajectoryBuffer
inherits: None 
Description: 
'''
class SingleTrajectoryBuffer:

    
    '''
    Function: Constructor for SingleTrajectoryBuffer 
    Params:
        n_window_size = window size of the buffer
    
    return: none 
    
    1. Set up object variables
    '''
    def __init__(self, n_window_size):
        self.obs_memory = deque(maxlen=n_window_size)
        self.state_memory = deque(maxlen=n_window_size)
        self.mask_memory = deque(maxlen=n_window_size)

        self.t = 0


    '''
    Function: add
    Params:
        obs_grid_map: observed grid map to append/add
        state_grid_map: the state grid map to append/add
        mask_map: mask map to append/add
    
    return: none 
    
    1. Append/add the passed parameters to their respective arrays/lists
    '''
    def add(self, obs_grid_map, state_grid_map, mask_map):
        self.obs_memory.append(obs_grid_map)
        self.state_memory.append(state_grid_map)
        self.mask_memory.append(mask_map)


    '''
    Function: sample
    Params:
        n_sample: number of samples to samples
        n_windows: number of windows to sample from
    
    return: the observed streams, state stream and mask stream from the sample
    
    1. randomly sample the start and stop indices 
    2. returns the randomly sampled batches
    '''
    def sample(self, n_sample, n_windows):

        batch_obs_stream = []
        batch_state_stream = []
        batch_mask_stream = []

        length = len(self.obs_memory)
        for i in range(n_sample):
            start = np.random.choice(length-n_windows)
            stop = min(start + n_windows, length)
            batch_obs_stream.append(list(itertools.islice(self.obs_memory, start, stop)))
            batch_state_stream.append(list(itertools.islice(self.state_memory, start, stop)))
            batch_mask_stream.append(list(itertools.islice(self.mask_memory, start, stop)))

        return batch_obs_stream, batch_state_stream, batch_mask_stream


### From pytorch tuorial ###
### https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


'''
#Notes: 
'''


'''
class: ReplayMemory
inherits: None 
Description: 
'''
class ReplayMemory(object):

        '''
    Function: Constructor for SingleTrajectoryBuffer 
    Params:
        capacity = capacity to set for the object
    
    return: none 
    
    1. Set up object variables  
    '''
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0


    '''
    Function: push
    Params:
        *args: 
    
    return: none 
    
    1. set the memory and position for the replay
    '''
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    '''
    Function: sample
    Params:
        batch_size: batch size to randomly sample from 
    
    return: randomply sampled batch_size from the memory 
    
    1. randomly sample from the memory of batch size and return it
    '''
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


    '''
    Function: __len__
    Params: none
    
    return: length of the memory list
    
    1. calculate and return length of the memory list
    '''
    def __len__(self):
        return len(self.memory)
