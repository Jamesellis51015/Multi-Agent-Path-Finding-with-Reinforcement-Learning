import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym import spaces



class CNN_Base_Old(nn.Module):
    def __init__(self, observation_space, h_dim = 120):
        super().__init__()
        #For strides of 1:
        (channels, d1, d2) =  observation_space.shape
        k1 = 2
        self.c1 = nn.Conv2d(channels, 3*channels, k1)
        d1c1 = d1- (k1 - 1) 
        d2c1 = d2 -(k1 - 1)

        k2 = k1
        self.c2 = nn.Conv2d(3*channels, 2*channels, k2)
        d1c2 = d1c1- (k2 - 1) 
        d2c2 = d2c1 -(k2 - 1)
        self.fc1 = nn.Linear(2*channels*(d1c2)*(d2c2), h_dim)


    
    def forward(self, x):
        batch_size = x.size(0)
        x = self.c1(x)
        x = torch.relu(x)
        x = self.c2(x)
        x = torch.relu(x)
        x = x.reshape((batch_size,-1))
        x = self.fc1(x)
        x = torch.relu(x)
        return x

class CNN_Base_New(nn.Module):
    def __init__(self, observation_space, h_dim = 120):
        super().__init__()
       
        (channels, d1, d2) =  observation_space.shape
        k1 = 3
        self.c1 = nn.Conv2d(channels, 6*channels, k1)
        d1c1 = d1- (k1 - 1) 
        d2c1 = d2 -(k1 - 1)
        k2 = k1
        self.c2 = nn.Conv2d(6*channels, 4*channels, k2)
        d1c2 = d1c1- (k2 - 1) 
        d2c2 = d2c1 -(k2 - 1)
        self.fc1 = nn.Linear(4*channels*(d1c2)*(d2c2), h_dim)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = self.c1(x)
        x = torch.relu(x)
        x = self.c2(x)
        x = torch.relu(x)
        x = x.reshape((batch_size,-1))
        x = self.fc1(x)
        x = torch.relu(x)
        return x

class Mlp_Base(nn.Module):
    def __init__(self, observation_space, h_dim = 120):
        super().__init__()
        obs_shape = observation_space.shape
        flat_dim = 1
        for i in obs_shape:
            flat_dim *= i
        self.fc1 = nn.Linear(flat_dim, h_dim)
    def forward(self, x):
        batch_size = x.size()[0]

        x = self.fc1(x.reshape(batch_size, -1))
        x = F.relu(x)
        return x

# class ActorMlp(Mlp_Base):
#     def __init__(self, observation_space, action_space, h_dim = 120):
#         super(ActorMlp, self).__init__(observation_space, h_dim)
#         n_actions = action_space.n
#         self.action_layer = nn.Linear(h_dim, n_actions)
#     def forward(self, x):
#         out = super(ActorMlp, self).forward(x)
#         probs = F.softmax(out, dim=1)
#         return probs

# class ActorCNN_Old(CNN_Base_Old):
#     def __init__(self, observation_space, action_space, h_dim = 120):
#         super(ActorCNN_Old, self).__init__(observation_space, h_dim)
#         n_actions = action_space.n
#         self.action_layer = nn.Linear(h_dim, n_actions)
#     def forward(self, x):
#         out = super(ActorCNN_Old, self).forward(x)
#         out = self.action_layer(out)
#         probs = F.softmax(out, dim=1)
#         return probs

# class ActorCNN_New(CNN_Base_New):
#     def __init__(self, observation_space, action_space, h_dim = 120):
#         super(ActorCNN_New, self).__init__(observation_space, h_dim)
#         n_actions = action_space.n
#         self.action_layer = nn.Linear(h_dim, n_actions)
#     def forward(self, x):
#         out = super(ActorCNN_New, self).forward(x)
#         out = self.action_layer(out)
#         probs = F.softmax(out, dim=1)
#         return probs

def make_policies(policy_type):
    if policy_type == "mlp":
        base_policy = Mlp_Base
    elif policy_type == "cnn_old":
        base_policy = CNN_Base_Old
    elif policy_type == "cnn_new":
        base_policy = CNN_Base_New
    else:
        raise Exception("Base policy type not implemented")
    
    class Actor(base_policy):
        def __init__(self, observation_space, action_space, comm_channels, lr, h_dim = 120):
            super(Actor, self).__init__(observation_space, h_dim)
            n_actions = action_space.n
            self.fc_in = nn.Linear(h_dim + comm_channels, h_dim)
            self.out_layer = nn.Linear(h_dim, n_actions)
            self.optimizer = optim.Adam(self.parameters(), lr =lr)
        def forward(self, x, comm_in):
            out = super(Actor, self).forward(x)
            out = self.fc_in(torch.cat([out, comm_in], dim = -1))
            out = F.relu(out)
            out = self.out_layer(out)
            probs = F.softmax(out, dim=1)
            return probs
        def take_action(self, obs, comm, greedy = False):
            a_prob = self.forward(obs, comm)
            if greedy:
                a = torch.argmax(a_prob, dim=-1, keepdim=True)
            else:
                a = torch.multinomial(a_prob, 1)
            # a_prob_chosen = torch.gather(a_prob, -1, a)
            return (a , a_prob)

    class Critic(base_policy):
        def __init__(self, observation_space, lr, h_dim = 120):
            super(Critic, self).__init__(observation_space, h_dim)
            self.fc_in = nn.Linear(h_dim , h_dim)
            self.out_layer = nn.Linear(h_dim, 1)
            self.optimizer = optim.Adam(self.parameters(), lr =lr)
        def forward(self, x):
            out = super(Critic, self).forward(x)
            out = self.fc_in(out)
            out = F.relu(out)
            out = self.out_layer(out)
            return out

    class CommNet(base_policy):
        def __init__(self, observation_space, comm_in, comm_out, lr, h_dim = 120):
            super(CommNet, self).__init__(observation_space, h_dim)
            # if comm_out == 0:
            #     self.zero_comm = True
            #     comm_out = 1
            # else:
            #     self.zero_comm = False
            self.comm_out= comm_out
            self.fc_in = nn.Linear(h_dim + comm_in, h_dim)
            self.out_layer = nn.Linear(h_dim, comm_out)
            self.optimizer = optim.Adam(self.parameters(), lr =lr)
        def forward(self, x, comm_in):
            out = super(CommNet, self).forward(x)
            out = self.fc_in(torch.cat([out, comm_in], dim = -1))
            out = F.relu(out)
            out = self.out_layer(out)
            # if self.zero_comm:
            #     z = torch.zeros(out.size())
            #     out = out*z
            return out
        def init_message(self):
            init_val = 0.0
            return torch.tensor([[init_val for _ in range(self.comm_out)]], requires_grad=True)
    return [Actor, Critic, CommNet]
