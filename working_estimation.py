'''
Here, now, I am predicting observation instead of increasing the likelihood. It is kind of equivalent.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import cv2, os

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DynamicAutoEncoderNetwork(nn.Module):

    def __init__(self, grid_size, n_state, n_obs, encoding_dim, gru_hidden_dim):
        super(DynamicAutoEncoderNetwork, self).__init__()

        ### Likelihood Params ###
        self.W_obs_param = torch.nn.Parameter(torch.randn(n_state, n_obs))
        
        ### Encoding Likelihood ###
        self.encoder = nn.Sequential(
            nn.Conv2d(n_state, 32, 8, stride=4),
            #nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, 4, stride=2),
            #nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 8, 3, stride=1),
            #nn.Dropout2d(0.5),
            nn.ReLU(),
            #nn.BatchNorm2d(8),
            nn.Flatten(),
            nn.Linear(128, encoding_dim)  #<--- 32 is hard-coded as dependent on 448 x 448 x 3.
            )

        ### Reccurent Neural Network ###
        self.rnn_layer = nn.GRU(input_size= encoding_dim, hidden_size=gru_hidden_dim, batch_first=True)

        ngf = 16 # filter size for generator
        nz = gru_hidden_dim
        nc = n_state

        ### Generate State Estimate Grid Map ###
        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            #nn.Dropout2d(0.5),
            #nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            #nn.Dropout2d(0.5),
            #nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.Dropout2d(0.5),
            #nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            #nn.Dropout2d(0.5),
            #nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False)
            #nn.Sigmoid()
            # state size. (nc) x 64 x 64
        ) 



LR_ESTIMATOR = 0.001
BETAS = (0.5, 0.9)
EPS = 1e-10
N_SAMPLE_WINDOW = 500


class DynamicAutoEncoder:

    def __init__(self, grid_size, n_state, n_obs, encoding_dim, gru_hidden_dim):
        self.model = DynamicAutoEncoderNetwork(grid_size, n_state, n_obs, encoding_dim, gru_hidden_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), LR_ESTIMATOR, BETAS)

        self.encoding_dim = encoding_dim
        self.grid_size = grid_size
        self.n_state = n_state
        self.n_obs = n_obs
        self.gru_hidden_dim = gru_hidden_dim

        ### States ###
        self.u_k = F.softmax(torch.rand(self.grid_size[0], self.grid_size[1], n_state, 1),2).to(DEVICE)
        self.h_k = torch.rand(1, 1, self.gru_hidden_dim).to(DEVICE)

        

    def save_the_model(self, iteration):
        if not os.path.exists('./save/dynautoenc/'):
            os.makedirs('./save/dynautoenc/')
        f_name = 'dynautoenc_network_param_' +  str(iteration) + '_model.pth'
        torch.save(self.model.state_dict(), './save/dynautoenc/'+f_name)
        print('Model Saved')


    def load_the_model(self, f_path):
        self.nn_model.load_state_dict(torch.load(f_path))
        print('Model Loaded')


    def step(self, obs):
        self.model.eval()

        ### Likelihood ###
        O_T = F.softmax(self.model.W_obs_param,0).T # Transpose of the observation matrix
        O_T = O_T.unsqueeze(0).unsqueeze(0).repeat(self.grid_size[0], self.grid_size[1], 1, 1)
        y = obs.unsqueeze(-1).detach()
        b = torch.matmul(O_T, y)

        ### State Estimate ###
        Bu = b*self.u_k # Size : w x h x n_state x 1
        bu =  torch.sum(Bu, 2).unsqueeze(-1)
        Bu_over_bu = Bu / bu # Size : w x h x n_state x 1
        state_est_grid = Bu_over_bu


        ### Encoding the Likelihood ###
        obs = obs.unsqueeze(-1).detach()
        x = self.model.encoder(obs.permute(3,2,0,1).contiguous())

        ### RNN Step ###
        h0 = self.h_k
        output, h_n = self.model.rnn_layer(x.unsqueeze(0), h0)
        self.h_k = output

        ### Decoding ###
        output = output[0].unsqueeze(-1).unsqueeze(-1)
        pred_state_est = self.model.decoder(output)
        pred_state_est = pred_state_est[:, :, :self.grid_size[0], :self.grid_size[1]] # Crop Image
        pred_state_est = F.softmax(pred_state_est, 1)
        pred_state_est = pred_state_est.permute(2, 3, 1, 0).contiguous()
        self.u_k = pred_state_est
        #print(torch.mean(pred_state_est), torch.max(pred_state_est), torch.min(pred_state_est))

        return pred_state_est # pred_state_est or state_est_grid
    
    def output_image(self, state_est_grid, size=(1200,400)):
        img = state_est_grid.data.cpu().numpy().squeeze()

        img_state0 = img[:,:,0]
        img_state1 = img[:,:,1]
        img_state2 = img[:,:,2]

        w, h = self.grid_size
        blank = np.zeros((h, int(w/20)))
        #img = np.concatenate((img_state0, blank, np.clip(img_state1 - img_state0, 0, 1), blank, img_state2), axis=1)
        img = np.concatenate((img_state0, blank, img_state1, blank, img_state2), axis=1)
        dim = size
    
        img_resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        return img_resized

    
    def update(self, memory):
        self.model.train()

        batch_obs_stream, batch_state_stream = memory.sample(N_BATCH, N_SAMPLE_WINDOW)
        batch_pred_obs_stream = []
        batch_tgt_obs_stream = []

        O = F.softmax(self.model.W_obs_param,0)
        O_np_val = O.data.cpu().numpy()
        O = O.unsqueeze(0).unsqueeze(0).repeat(self.grid_size[0], self.grid_size[1], 1, 1)
        O_bat = O.unsqueeze(0).repeat(N_SAMPLE_WINDOW, 1, 1, 1, 1)

        ### Foward the observation through the model ###
        for i in range(N_BATCH):
            ### Encoding and Likelihood ###
            obs_stream = torch.stack(batch_obs_stream[i])
            batch_tgt_obs_stream.append(obs_stream.unsqueeze(-1))
            obs_stream = obs_stream.squeeze().permute(0, 3, 1, 2).contiguous()
            x_stream = self.model.encoder(obs_stream)
            

            ### RNN State Predictor ###
            h0 = torch.rand(1, 1, self.gru_hidden_dim).to(DEVICE)
            output, h_n = self.model.rnn_layer(x_stream.unsqueeze(0), h0)

            ### Decoding ###
            output = output.squeeze().unsqueeze(-1).unsqueeze(-1)
            pred_state_est_stream = self.model.decoder(output)
            pred_state_est_stream = pred_state_est_stream[:, :, :self.grid_size[0], :self.grid_size[1]] # Crop Image
            pred_state_est_stream = F.softmax(pred_state_est_stream, 1)
            pred_state_est_stream = pred_state_est_stream.permute(0,2,3,1).contiguous().unsqueeze(-1)
            pred_obs_stream = torch.matmul(O_bat, pred_state_est_stream)
            batch_pred_obs_stream.append(pred_obs_stream)

        batch_pred_obs_stream = torch.stack(batch_pred_obs_stream)
        batch_tgt_obs_stream = torch.stack(batch_tgt_obs_stream)

        #### Translate one step the target for calculating loss in prediction
        tgt_grid_obs = batch_tgt_obs_stream[:, 1:, :, :, :]
        pred_grid_obs = batch_pred_obs_stream[:, :-1, :, :, :]

        #### Cross Entropy Loss ###
        loss1 = torch.mean(-(tgt_grid_obs*torch.log(pred_grid_obs + EPS)+(1-tgt_grid_obs)*torch.log(1-pred_grid_obs + EPS)))

        ### Shannon entropy loss ###
        p = pred_grid_obs
        log_p = torch.log(p)
        loss2 = -torch.mean(p*log_p)

        loss = loss1 #+ loss2

        ### loss3 ###
        #loss3 = torch.mean(self.model.W_obs_param**2)

        #loss = loss + 0.001*loss3

        ### Update Model ###
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        loss_val = loss.item()
        loss1_val = loss1.item()
        loss2_val = loss2.item()

        del batch_obs_stream, batch_state_stream, batch_pred_obs_stream, batch_tgt_obs_stream, loss, loss1, loss2
        torch.cuda.empty_cache()

        return loss_val, loss1_val, loss2_val, O_np_val

def render (window_name, image):
    cv2.imshow(window_name, image)
    cv2.waitKey(1)


        
    
if __name__ == "__main__":
    from environment import FireEnvironment
    from memory import SingleTrajectoryBuffer
    import tqdm, random
    from params import ACTION_SET

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()

    EPOCH = 1000

    N_MEMORY_SIZE = 20000
    #N_SAMPLE_WINDOW = 50
    N_BATCH = 1
    N_TOTAL_TIME_STEPS = int(EPOCH*N_MEMORY_SIZE/N_SAMPLE_WINDOW/N_BATCH)

    EPOCH_N_PERIOD = int(N_MEMORY_SIZE/N_SAMPLE_WINDOW/N_BATCH)

    env = FireEnvironment(64, 64)
    dyn_autoencoder = DynamicAutoEncoder(grid_size = (env.map_width, env.map_height), n_state=3, n_obs=3, encoding_dim=16, gru_hidden_dim=16)
    memory = SingleTrajectoryBuffer(N_MEMORY_SIZE)

    
    
    obs, state = env.reset()

    list_loss = []
    list_cross_entropy_loss = []
    list_entropy_loss = []

    for i in tqdm.tqdm(range(N_TOTAL_TIME_STEPS)):
        act = random.randrange(len(ACTION_SET))
        obs, state = env.step(ACTION_SET[act])

    
        state_est_grid = dyn_autoencoder.step(obs)

        if i%5 == 0:
            img_env   = env.output_image()
            img_agent = dyn_autoencoder.output_image(state_est_grid)

            render('env', img_env)
            render('est', img_agent)


        memory.add(obs, state)
        if i > 2000:
            loss_val, loss_val_cross, loss_val_ent, O_np_val =  dyn_autoencoder.update(memory)
            list_loss.append(loss_val)
            list_cross_entropy_loss.append(loss_val_cross)
            list_entropy_loss.append(loss_val_ent)

            if i%EPOCH_N_PERIOD == 0:
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

        if i%int(EPOCH_N_PERIOD*10)==0:
            #dyn_autoencoder.save_the_model(i)
            print('ok')
    