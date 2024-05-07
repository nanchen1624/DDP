# DDP nn implementation: step 1
import numpy as np
import torch 
import torch.nn as nn
import argparse
import time
from random import seed, shuffle
from torch.distributions.multivariate_normal import MultivariateNormal


torch.backends.cudnn.benchmark = True 

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

#----- Input argparse -----#
parser = argparse.ArgumentParser(description="dnn")
parser.add_argument('-shuffled', '--shuffled', action='store_true', default=False, help='shuffle data')
parser.add_argument('-epoch', '--epoch', nargs=1, type=int, default=100, help='training epochs')
parser.add_argument('-sample', '--sample', nargs=1, type=int, default=1, help='number of samples')
parser.add_argument('-seed', '--seed', nargs=1, type=int, default=38, help='random seed')
parser.add_argument('-step', '--step', nargs=1, type=int, default=100, help='training steps in each epoch')
parser.add_argument('-batch', '--batch', nargs=1, type=int, default=100, help='training batch size')
parser.add_argument('-T', '--T', nargs=1, type=int, default=20, help='time horizon')
parser.add_argument('-lr', '--lr', nargs=1, type=float, default=0.01, help='training learning rate')
parser.add_argument('-con', '--con', nargs=1, type=float, default=200, help='lagrange multiplier')
args = parser.parse_args()
#----- Input argparse -----#

#------ Hyperparameters -----#
# problem setting
T = args.T[0] if type(args.T) == list else args.T  # time horizon
dim_stocks = 3
dim_impact = 2
theta = [[30., 7., 3.], [7., 25., -5.], [3., -5., 20.]]
gamma = [[50., 20.], [30., 20.], [10., 40.]]
rho = [[0.8, 0.1], [0.2, 0.6]]
delta = 1.0 
d_value = 20.
rho = torch.tensor(rho, device=device)
matrix_A = torch.tensor(theta, device=device) 
matrix_B = torch.tensor(gamma, device=device)
matrix_C = rho * delta
matrix_D = torch.eye(dim_stocks, device=device) * d_value

eta_0 = torch.tensor([0.0, 0.0], device=device)
matrix_sigma_eta = torch.tensor([[10., 2.], [2., 8.]], device=device)
mean_eta = torch.tensor([0, 0.], device=device)

X_0 = torch.tensor([0., 0.], device=device)            # initial information dynamics: X[t] = rho * X[t-1] + eta[t]
P_0 = torch.tensor([0.0, 0.0, 0.0], device=device)        # initial price dynamics: P[t] = P[t-1] + theta * S[t] + gamma * X[t] + epsilon[t]
S_bar_0 = 100.
S_bar = torch.tensor([S_bar_0, S_bar_0, S_bar_0], device=device)         # target shares

# For networks
STEPS = args.step[0] if type(args.step) == list else args.step # 7812.5, 3125*5, 3125*10, 3125*20
BATCH_SIZE = args.batch[0] if type(args.batch) == list else args.batch
N_EPOCHS = args.epoch[0] if type(args.epoch) == list else args.epoch
N_training = BATCH_SIZE*STEPS # Size of the out training set

#------ Hyperparameters ------#
SEED = args.seed[0] if type(args.seed) == list else args.seed
np.random.seed(SEED)
torch.manual_seed(SEED)
#------ networks ------#
lr_a = args.lr[0] if type(args.lr) == list else args.lr
k = 100 # hidden units
dim_states = dim_stocks + dim_impact
CON_LAM = args.con[0] if type(args.con) == list else args.con

class Multi_action_net(nn.Module):
    
    def __init__(self, hidden_size, is_train, device):
        super(Multi_action_net, self).__init__()
        self.is_train = is_train
        self.device = device
        self.sub_action_nets = nn.ModuleList([Sub_net(dim_states, dim_stocks, hidden_size) for _ in range(T)]).to(self.device) # only use list will not output parameters

    def forward(self, z):
        eta = z[0]
        batch_size = eta.shape[0]
        X = torch.zeros([batch_size, dim_impact, T+1]).to(self.device)
        dif_P = torch.zeros([batch_size, dim_stocks, T+1], device=device)
        trade_P = torch.zeros([batch_size, dim_stocks, T+1], device=device)
        W = torch.zeros([batch_size, dim_stocks, T+2]).to(self.device)

        X[:, :, 0] = X_0
        W[:, :, 0] = S_bar
        W[:, :, T+1] = 0

        for t in range(1, T+1):
            X[:, :, t] = torch.bmm(rho.unsqueeze(0).expand(batch_size, dim_impact, -1), X[:, :, t-1].unsqueeze(2)).squeeze(2) + eta[:, :, t]

        opt_action = torch.zeros(batch_size, dim_stocks, T+1).to(self.device)
        for t in range(1, T+1):
            W[:, :, t] = W[:, :, t-1] - opt_action[:, :, t-1]
            state_data = torch.cat((X[:, :, t], W[:, :, t]), 1)
            opt_action[:, :, t] = self.sub_action_nets[t-1](state_data)
            dif_P[:, :, t] = torch.bmm(matrix_A.unsqueeze(0).expand(batch_size, dim_stocks, -1), opt_action[:, :, t].unsqueeze(2)).squeeze(2) + torch.bmm(matrix_B.unsqueeze(0).expand(batch_size, dim_stocks, -1), X[:, :, t].unsqueeze(2)).squeeze(2)
            trade_P[:, :, t] = d_value * torch.sqrt(opt_action[:, :, t])  # D*sqrt(S_t)

        loss = torch.mean(torch.sum(torch.sum(trade_P[:,:,1:] * opt_action[:,:,1:] + dif_P[:,:,1:] * W[:,:,1:T+1], 2),1))

        con_loss2 = CON_LAM * torch.mean(torch.sum(torch.pow(S_bar - torch.sum(opt_action[:,:,1:], 2), 2), 1)) # constraint loss

        total_loss = loss + con_loss2
        return X, opt_action, total_loss
    
    def __len__(self):
        return self.sub_action_nets[0].__len__()

class Sub_net(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Sub_net, self).__init__()
        self.seq = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.BatchNorm1d(num_features=hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, output_size),
                    nn.BatchNorm1d(num_features=output_size),
                    nn.ReLU(),
                    )
    def forward(self, x):
        action = torch.minimum(self.seq(x), S_bar)
        return action
    
    def __len__(self):
        return self.seq.__len__()
#------ networks ------#

#------ utils ------#
# dynamics
def x_t(Xt, eta_t):
    if len(Xt.shape) == 1:
        data_len = 1
        return torch.mm(rho, Xt.unsqueeze(1)).squeeze(1) + eta_t
    else:
        data_len = Xt.shape[0]
        return torch.bmm(rho.unsqueeze(0).expand(data_len, dim_impact, -1), Xt.unsqueeze(2)).squeeze(2) + eta_t

def w_t(Wt, St):
    return Wt - St

def p_t(Pt_1, St, Xt, epsilon_t):
    data_len = Pt_1.shape[0]
    return Pt_1 + torch.bmm(matrix_A.unsqueeze(0).expand(data_len, dim_stocks, -1), St.unsqueeze(2)).squeeze(2) + torch.bmm(matrix_B.unsqueeze(0).expand(data_len, dim_stocks, -1), Xt.unsqueeze(2)).squeeze(2) + epsilon_t

def weights_init(m):
    if isinstance(m, nn.Linear):
        # torch.nn.init.xavier_uniform_(m.weight)
        # torch.nn.init.kaiming_uniform_(m.weight)
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)

def inplace_activation(m):
    classname = m.__class__.__name__
    if classname.find('Softmax') != -1:
        m.inplace=True
    if classname.find('ReLU') != -1:
        m.inplace=True

def generate_data(data_size):     # get training data
    # uniform distribution
    # epsilon = torch.Tensor(data_size, dim_stocks, T+1).uniform_(-1, 1) * sigma_epsilon/2
    # eta = torch.Tensor(data_size, dim_impact, T+1).uniform_(-1, 1) * sigma_eta/2
    # normal distribution
    eta = torch.Tensor(data_size, dim_impact, T+1)
    m_eta = MultivariateNormal(loc=mean_eta, covariance_matrix=matrix_sigma_eta)
    for t in range(T):     
        eta[:,:,t+1] = m_eta.rsample(sample_shape=(data_size,))
    eta[:, :, 0] = torch.zeros(dim_impact)
    eta[:, :, 1] = eta_0
    return eta.to(device)

def opt_abcd():
    delta_w = torch.zeros(dim_stocks, dim_stocks, T, device=device)
    delta_x = torch.zeros(dim_stocks, dim_impact, T, device=device)
    A = torch.zeros(dim_stocks, dim_stocks, T, device=device)
    B = torch.zeros(dim_impact, dim_stocks, T, device=device)
    C = torch.zeros(dim_impact, dim_impact, T, device=device)
    d_value = torch.zeros(T, device=device)
    delta_w[:, :, 0] = torch.eye(dim_stocks, device=device)
    delta_x[:, :, 0] = torch.zeros(dim_stocks, dim_impact, device=device)
    A[:, :, 0] = matrix_A
    B[:, :, 0] = matrix_B.t()
    C[:, :, 0] = torch.zeros(dim_impact, dim_impact, device=device) 
    d_value[0] = 0
    for k in range(1, T):
        F_temp = torch.inverse(A[:, :, k-1])
        A[:,:,k] = matrix_A - torch.mm(torch.mm(matrix_A, F_temp), matrix_A.t()) / 4
        B[:,:,k] = torch.mm(torch.mm(matrix_C.t(), B[:,:,k-1]), torch.mm(F_temp, matrix_A.t())) / 2 + matrix_B.t()
        C[:,:,k] = torch.mm(torch.mm(matrix_C.t(), C[:,:,k-1]), matrix_C) - torch.mm(torch.mm(matrix_C.t(), B[:,:,k-1]), torch.mm(F_temp.t(), torch.mm(B[:,:,k-1].t(), matrix_C))) / 4

        d_value[k] = d_value[k-1] + torch.sum(torch.sum(C[:,:,k-1]*matrix_sigma_eta))

        delta_w[:,:,k] = torch.eye(dim_stocks, device=device) - torch.mm(F_temp, matrix_A.t()) / 2
        delta_x[:,:,k] = torch.mm(F_temp, torch.mm(B[:,:,k-1].t(), matrix_C)) / 2 
    return delta_w, delta_x, A, B, C, d_value

def optimal_value(x0):
    x1, w1 = x0[0], x0[1]
    _, _, A, B, C, d_value = opt_abcd()
    opt_value = torch.mm(torch.mm(w1.unsqueeze(0), A[:, :, -1]), w1.unsqueeze(1)).squeeze(1) + torch.mm(torch.mm(x1.unsqueeze(0), B[:, :, -1]), w1.unsqueeze(1)).squeeze(1) + torch.mm(torch.mm(x1.unsqueeze(0), C[:, :, -1]), x1.unsqueeze(1)).squeeze(1) + d_value[-1]
    return opt_value

def sample_optimal_value(x0, sample_data):
    eta_0, S_bar = x0[0], x0[1]
    eta = sample_data

    trade_P = torch.zeros([N_training, dim_stocks, T+1], device=device)
    dif_P = torch.zeros([N_training, dim_stocks, T+1], device=device)
    X = torch.zeros([N_training, dim_impact, T+1], device=device)
    W = torch.zeros([N_training, dim_stocks, T+2], device=device)

    X[:, :, 0] = X_0
    W[:, :, 0] = S_bar
    W[:, :, T+1] = torch.zeros(dim_stocks, device=device)
    X[:, :, 1] = eta_0
    W[:, :, 1] = S_bar

    for t in range(1, T+1):
        X[:, :, t] = torch.bmm(rho.unsqueeze(0).expand(N_training, dim_impact, -1), X[:, :, t-1].unsqueeze(2)).squeeze(2) + eta[:, :, t]

    delta_w, delta_x, _, _, _, _ = opt_abcd()
    opt_action = torch.zeros(N_training, dim_stocks, T+1, device=device)
    for t in range(1, T+1):
        W[:, :, t] = W[:, :, t-1] - opt_action[:, :, t-1]
        # no caonstraint
        # opt_action[:, :, t] = torch.bmm(delta_w[:, :, T-t].unsqueeze(0).expand(N_training, dim_stocks, -1), W[:, :, t].unsqueeze(2)).squeeze(2) + torch.bmm(delta_x[:, :, T-t].unsqueeze(0).expand(N_training, dim_stocks, -1), X[:, :, t].unsqueeze(2)).squeeze(2)

        # positive caonstraint
        if t == T:
            opt_action[:, :, t] = W[:, :, t]
        else:
            opt_action[:, :, t] = torch.min(torch.max(torch.bmm(delta_w[:, :, T-t].unsqueeze(0).expand(N_training, dim_stocks, -1), W[:, :, t].unsqueeze(2)).squeeze(2) + torch.bmm(delta_x[:, :, T-t].unsqueeze(0).expand(N_training, dim_stocks, -1), X[:, :, t].unsqueeze(2)).squeeze(2), torch.zeros(N_training, dim_stocks, device=device)), W[:, :, t])
        dif_P[:, :, t] = torch.bmm(matrix_A.unsqueeze(0).expand(N_training, dim_stocks, -1), opt_action[:, :, t].unsqueeze(2)).squeeze(2) + torch.bmm(matrix_B.unsqueeze(0).expand(N_training, dim_stocks, -1), X[:, :, t].unsqueeze(2)).squeeze(2)
        trade_P[:, :, t] = d_value * torch.sqrt(torch.abs(opt_action[:, :, t]))

    sample_value = torch.mean(torch.sum(torch.sum(trade_P[:,:,1:] * opt_action[:,:,1:] + dif_P[:,:,1:] * W[:,:,1:T+1], 2),1))  
    # sample_value_std = torch.std(torch.sum(torch.sum(trade_P[:,:,1:] * opt_action[:,:,1:] + dif_P[:,:,1:] * W[:,:,1:T+1], 2),1))  
    # print('sample_value', sample_value.item())
    # print('sample_value_std', sample_value_std.item())
    return sample_value.item()

#------ utils ------#

if __name__ == '__main__':
    #----- training process -----#
    # generate training data
    train_data = generate_data(data_size=N_training)
    eta_all = train_data.to(device) 

    W1 = S_bar
    X1 = x_t(X_0, eta_0)
    x0 = (X1, W1)
    Vstar = optimal_value(x0)
    print("optimal value: {:.4f}".format(Vstar.item()))
    true_in_loss = sample_optimal_value(x0, train_data)
    print("MC sample value: {:.4f}".format(true_in_loss))

    # data need to be stored
    loss_list = []

    hidden_size = k
    action_net = Multi_action_net(hidden_size, True, device) # action network initialization

    hidden_len = 0
    print("action net structure, hidden_len, {}, hidden_size, {}".format(hidden_len, hidden_size))

    # define optimizer and initial learning rate
    opt = torch.optim.Adam(action_net.parameters(), lr=lr_a)

    # initial parameter assigned
    action_net.apply(weights_init)

    # train epoch
    tick1 = time.time()
    for epoch in range(N_EPOCHS):
        # shuffle the data
        if args.shuffled:
            ind_list = list(range(N_training))
            shuffle(ind_list)
            eta_all = eta_all[ind_list]
        
        for step in range(STEPS):
            # split training set
            if step == STEPS-1:
                eta = eta_all[step*BATCH_SIZE:, :]
                LAST_BATCH_SIZE = eta.shape[0]
            else:
                eta = eta_all[step*BATCH_SIZE:(step+1)*BATCH_SIZE, :]

            if step == 0 and epoch==0:
                action_net.eval() # evaluate the action net
                noise_data = [eta_all]
                _, _, action_loss = action_net(noise_data)
                print("epoch:", epoch, "step:", step, "training value function: {:.4f}".format(action_loss))

            action_net.train() # train the action net
            noise_data = [eta]
            _, _, action_loss = action_net(noise_data)
            opt.zero_grad()
            action_loss.backward()
            opt.step()
            
            train_time = time.time()
            
            if step == STEPS-1:
                if epoch % 10 == 0 or epoch == N_EPOCHS-1:
                    action_net.eval() # evaluate the action net
                    noise_data = [eta_all]
                    _, _, action_loss = action_net(noise_data)

                    loss_list.append((epoch, step, action_loss))
                    print("epoch:", epoch, "step:", step, "training value function: {:.4f}".format(action_loss), "eval time: {:.4f}".format(train_time-tick1))
                    if epoch % 50 == 0 or epoch == N_EPOCHS-1:
                        torch.save(action_net.state_dict(), "results/action_nets/dnn_adam_action_net_seed_"+str(SEED)+"_T_"+str(T)+"_epoch_"+str(N_EPOCHS)+"_current_epoch_"+str(epoch)+"_batch_"+str(BATCH_SIZE)+"_datasize_"+str(N_training)+"_hidden_size_"+str(hidden_size)+".pkl")

    tock1 = time.time()
    # calculate the time used in N_EPOCHS training epochs
    print("training time: {:.4f} mins".format((tock1-tick1)/60))

    



    






















