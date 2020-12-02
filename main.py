from environment import 

if __name__ == "__main__":
    env = FireEnvironment()
    onehot_grid_map = env.reset()
    #print(onehot_grid_map)
    env.render()
    #for i in range(10000):

    while True:
        act = random.randrange(N_ACTION)
        onehot_grid_map = env.step(ACTION_SET[act])
        #print(onehot_grid_map)
        env.render()
        #break

        env = FireEnvironment(50, 50)
        hmm_estimator = HMMEstimator(grid_size = (env.map_width, env.map_height), n_kernel=3, n_obs=3, n_state=3)
    
        obs, state = env.reset()
        hmm_estimator.update(obs)