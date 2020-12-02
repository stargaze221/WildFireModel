import numpy as np
import random
from collections import namedtuple, deque
import itertools


class SingleTrajectoryBuffer:

    def __init__(self, n_window_size):
        self.obs_memory = deque(maxlen=n_window_size)
        self.state_memory = deque(maxlen=n_window_size)

    def add(self, obs_grid_map, state_grid_map):
        self.obs_memory.append(obs_grid_map)
        self.state_memory.append(state_grid_map)

    def sample(self, n_sample, n_windows):

        batch_obs_stream = []
        batch_state_stream = []

        length = len(self.obs_memory)
        for i in range(n_sample):
            start = np.random.choice(length-n_windows)
            stop = min(start + n_windows, length)
            batch_obs_stream.append(list(itertools.islice(self.obs_memory, start, stop)))
            batch_state_stream.append(list(itertools.islice(self.state_memory, start, stop)))

        return batch_obs_stream, batch_state_stream



if __name__ == "__main__":
    from environment import FireEnvironment
    from params import ACTION_SET
    import tqdm
    
    env = FireEnvironment(50, 50)
    memory = SingleTrajectoryBuffer(1000)
    
    obs, state = env.reset()
    
    for i in tqdm.tqdm(range(1000)):
        act = random.randrange(len(ACTION_SET))
        obs, state = env.step(ACTION_SET[act])
        memory.add(obs, state)
        if i > 100:
            batch_obs_stream, batch_state_stream = memory.sample(2, 10)

    

    
        
        