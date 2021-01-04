from agent import BaysianEstimator, render, ImageStreamWriter
from environment import FireEnvironment
import cv2
import numpy as np

FPS = 10

def model_based_recursive_estimation():
    
    env = FireEnvironment(64, 64)
    agent = BaysianEstimator(grid_size = (env.map_width, env.map_height), n_obs=3, n_state=3)

    video_writer0 = ImageStreamWriter('ModelDemo.avi', FPS, image_size=(795,400))
    video_writer1 = ImageStreamWriter('BayesEstDemo.avi', FPS, image_size=(1200,820))

    _, obs, state = env.reset()

    for i in range(5000):

        img_env   = env.output_image()
        img_agent = agent.output_image()

        ################################
        ### Rendering and Save Video ###
        ################################

        # Model Image Crop 
        img_model_uint8 = cv2.normalize(img_env[:,:795], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        #img_model_uint8 = (img_model*255).astype('uint8') #<-- to be saved
        render('Model', img_model_uint8, 10)

        # Model Bayes State Est
        blank = np.zeros((400, 200, 3))
        img_top = np.concatenate((blank, img_env[:,:800], blank), axis=1)
        blank = np.zeros((20, 1200, 3))
        img_top = np.concatenate((img_top, blank), axis=0)
        img_top = (img_top*255).astype('uint8')

        img_state_est_grid_uint8 = (img_agent*255).astype('uint8')
        backtorgb = cv2.cvtColor(img_state_est_grid_uint8, cv2.COLOR_GRAY2RGB)
        img_bayes_uint8 = np.concatenate((img_top, backtorgb), axis=0) #<-- to be saved
        render('Bayes Est.', img_bayes_uint8, 10)

        # Save video #
        video_writer0.write_image_frame(img_model_uint8)
        video_writer1.write_image_frame(img_bayes_uint8)


        ######################################
        ### Step Env. and Update Estimator ###
        ######################################
        mask_obs, obs, state, reward = env.step()
        state_est = agent.Bayesian_update(obs)

    video_writer0.close()
    video_writer1.close()

if __name__ == "__main__":
    model_based_recursive_estimation()