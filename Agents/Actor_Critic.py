import numpy as np
import gym 
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple
import random
import itertools
#from rl_lib.policyDQN_2 import ReplayBuff
obser = namedtuple('observations', 's a r next_s ter')
import config

#random.seed(1)
#np.random.seed(1)
#torch.manual_seed(1)

print("Using device {}".format(config.device))

class ReplayBuff():
    def __init__(self, buffer_size,buffer_start_size):
        self.buffer_size = buffer_size
        self.observations = []
        self.start_size = buffer_start_size

    def add(self, obs): #(state, a , r, next_state,ter):
        """ """
        #TODO: Make so that can accept both obs type as well as tuple of observation
        if len(self.observations) >= self.buffer_size:
            self.observations.pop(0)
            self.observations.append(obs)
        else:
            self.observations.append(obs)
    
    def mini_batch(self, batch_size):
        assert (len(self.observations) != 0), "NO OBSERVATIONS ADDED"
        if len(self.observations) < batch_size:
            return random.sample(self.observations, k=len(self.observations))
        else:
            return random.sample(self.observations, k=batch_size)
    def get_obs_len(self):
        return len(self.observations)
    def get_obs(self):
        return self.observations
    def is_start(self):
        """Returns a boolean value to indicate whether or not 
        the start size of the buffer has been reached """
        if len(self.observations) >= self.start_size:
            return True
        else:
            return False

class Baseline_Value_Function(nn.Module, ReplayBuff):
    def __init__(self, in_sz, hid_sz, buffer_size,buffer_start_size, batch_size, gamma = 1.0):
        '''Output size is fixed to be one '''
        self.gamma = gamma
        nn.Module.__init__(self)
        ReplayBuff.__init__(self, buffer_size = buffer_size, buffer_start_size= buffer_start_size)
        self.batch_size = batch_size
        self.fc1 = nn.Linear(in_sz, hid_sz)
        self.fc0 = nn.Linear(2,12) #for magnitude and angle
        self.fc2 = nn.Linear(hid_sz+12, hid_sz)
        self.fc3 = nn.Linear(hid_sz, 1)
        self.optimizer = optim.Adam(self.parameters(), lr = 0.001)
        self.loss_f = nn.MSELoss()

    def forward(self, x, y):
        """NB input batch of states (2D) """
        x = self.fc1(x)
        x = F.relu(x)
        y = self.fc0(y)
        y = F.relu(y)
        x = torch.cat((x,y), 1)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

    def update(self, batch_size= None):
        if not self.is_start(): 
            #print("Buffer not full...")
            return
        if batch_size == None: batch_size = self.batch_size
        batch = self.mini_batch(batch_size)
        batch = obser(*zip(*batch))
        #hldr = torch.cat(batch.s, dim =0)
        obs_view = [s[0] for s in batch.s]
        obs_vec = [s[1] for s in batch.s]
        v_s =  self.forward(torch.cat(obs_view, dim =0).to(config.device), torch.cat(obs_vec, dim=0).to(config.device)).squeeze()

        obs_view = [s[0] for s in batch.next_s]
        obs_vec = [s[1] for s in batch.next_s]
        v_s_prime =  self.forward(torch.cat(obs_view, dim =0).to(config.device), torch.cat(obs_vec, dim=0).to(config.device)).squeeze()
        #hldr = torch.tensor(batch.r) + v_s_prime
        targets = torch.tensor(batch.r) + self.gamma*v_s_prime.to("cpu")
        loss = self.loss_f(v_s, targets.to(config.device))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()



class Reinforce(nn.Module):
    def __init__(self, in_sz, nr_actions, hid_sz):
        super().__init__()
        #self.l1 = nn.Conv2d(in_sz_len, in_sz_len, 16, stride = 1, padding=0)
        #self.l2 = nn.Conv2d(in_sz_len-1, in_sz_len - 1, 16, stride = 1, padding=0)
        self.fc1 = nn.Linear(in_sz, hid_sz)
        self.fc0 = nn.Linear(2,12) #for magnitude and angle
        self.fc2 = nn.Linear(hid_sz+12, hid_sz)
        self.fc3 = nn.Linear(hid_sz, nr_actions)
        self.optimizer = optim.Adam(self.parameters(), lr = 0.001)

    def forward(self, x, y):
        """NB input batch of states (2D) """
        x = self.fc1(x)
        x = F.relu(x)
        y = self.fc0(y)
        y = F.relu(y)
        x = torch.cat((x,y), 1)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

    def update(self, states, actions, targets, entropy_coef = 0.01):
        '''Update Actor
           targets: The return corresponding to a particular state under the current on policy
        This is updating a batch of states and targets 
              states: (batch, state) 2D
              actions: (action*batch)  1D 
              targets: (return*batch) 1D'''
        probabilities = self.forward(states[0].to(config.device), states[1].to(config.device))
        m = torch.distributions.Categorical(probabilities)

        loss = -m.log_prob(actions.to(config.device))
        loss *= targets.to(config.device)
        loss -= entropy_coef * m.entropy()
        loss = loss.sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def get_action(self, action_probabilities):
        return torch.multinomial(action_probabilities , 1)

    def change_lr(self,lr):
        self.optimizer = optim.Adam(self.parameters(), lr = lr)

class AC():
    def __init__(self, in_sz, nr_actions, hid_sz, buffer_size,buffer_start_size, gamma = 1.0, c = 70, batch_size = 32):
        """Actor and critic is two seperate NN. Critic gives 
        value function and updated by sampling from replay buffer. """
        self.actor = Reinforce(in_sz, nr_actions, hid_sz).to(config.device)
        #ReplayBuff.__init__(buffer_size, buffer_start_size)

        self.baseline = Baseline_Value_Function(in_sz, hid_sz, buffer_size,buffer_start_size, batch_size).to(config.device)

class AC_shared(nn.Module):
    def __init__(self, in_size, action_size, hidden_size, learning_rate = 0.001):
        super().__init__()
        self.fc_shared0 = nn.Linear(in_size, hidden_size)
        self.fc_shared1 = nn.Linear(2,12) #for magnitude and angle
        self.critic_hidden = nn.Linear(hidden_size + 12, hidden_size)
        self.actor_hidden = nn.Linear(hidden_size + 12, hidden_size)
        self.critic = nn.Linear(hidden_size, 1)
        self.actor = nn.Linear(hidden_size, action_size)

        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)

        #print(" AC  np   {};    Torch:  {}".format(np.random.normal(), torch.rand(1)))

    def forward(self, x, y):
        x = self.fc_shared0(x)
        x = F.relu(x)
        y = self.fc_shared1(y)
        y = F.relu(y)
        x = torch.cat((x,y), 1)
        critic_out = self.critic_hidden(x)
        critic_out = F.relu(critic_out)
        actor_out = self.actor_hidden(x)
        actor_out = F.relu(actor_out)
        critic_out = self.critic(critic_out)
        actor_out = self.actor(actor_out)
        actor_out = F.softmax(actor_out, dim = 1)
        return actor_out, critic_out

    def update(self, s, a, targets, c1, c2): 
        a_prob, v = self.forward(s[0].to(config.device), s[1].to(config.device))
        targets = targets.to(config.device)

        m = torch.distributions.Categorical(a_prob)

        loss = -m.log_prob(a.to(config.device))
        loss *= targets
        loss += c1 * (v -(v.detach() + targets))**2 #Targets are general advantage estimates of the form r + V(s') - V(s)
        loss -= c2 * m.entropy()
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss



    def get_action(self, action_probabilities):
        return torch.multinomial(action_probabilities , 1)
  
    def save(self, path):
        torch.save(self.state_dict(), path)



