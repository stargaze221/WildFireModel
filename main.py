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