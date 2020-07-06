import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym import spaces
from Agents.general_utils.policy import make_base_policy
from Agents.PPO.env_wrappers import SubprocVecEnv, DummyVecEnv
from Env.make_env import make_env


def make_parallel_env(args,seed, n_rollout_threads):
    def get_env_fn(rank):
        def init_env():
            env = make_env(args)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        #return DummyVecEnv([get_env_fn(0), get_env_fn(2)])
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])



def make_actor_net(base_policy_type,double_obs_space = False, recurrent = False, blocking = False):
    BasePoliy = make_base_policy(base_policy_type,double_obs_space)
    class Actor(BasePoliy):
        def __init__(self, observation_space, hidden_dim, action_dim, lr = 0.001, nonlin = F.leaky_relu):
            if double_obs_space:
                dim1_shape = observation_space[0].shape
                dim2 = observation_space[1].shape[0]
                super(Actor, self).__init__(dim1_shape, dim2, nonlin= nonlin)
            else:
                super(Actor, self).__init__(observation_space, hidden_dim, nonlin= nonlin)
            self.recurrent = recurrent
            self.blocking = blocking
            self.nonlin = nonlin
            
            self.fc_out = nn.Linear(hidden_dim, action_dim)
            if recurrent:
                self.lstm = nn.LSTMCell(self.hidden_dim, hidden_dim)
            else:
                self.fc_mid = nn.Linear(self.hidden_dim, hidden_dim)

            if self.blocking:
                self.fc_block_mid = nn.Linear(hidden_dim, hidden_dim)
                self.fc_block_out = nn.Linear(hidden_dim, 2)
            self.optimizer = optim.Adam(self.parameters(), lr =lr)
            
        def forward(self, x, hid_cell_state = None):
            x = super(Actor, self).forward(x)
            hx, cx = None,None
            if self.recurrent:
                x, cx = self.lstm(x, hid_cell_state)
                hx = x
                hx, cx = hx.clone(), cx.clone()
                x = self.nonlin(x)
            else:
                x = self.nonlin(self.fc_mid(x))
            x_act = F.softmax(self.fc_out(x))
            if self.blocking:
                x_block = self.nonlin(self.fc_block_mid(x))
                x_block = self.fc_block_out(x_block) #softmax removed since incuded in loss f
            else:
                x_block = None
            return (x_act, hx, cx, x_block)

        def take_action(self, obs, greedy = False, hid_cell_state = None, valid_act = None):
            (a_probs, hx, cx, x_block) = self.forward(obs, hid_cell_state)
            if greedy:
                a_select = torch.argmax(a_probs, dim=-1, keepdim= True)
            else:
                a_probs_new = a_probs.clone()
                if not valid_act is None:
                    assert a_probs.size(0) == 1
                    non_valid = set([i for i in range(a_probs.size(-1))]) - set(valid_act)
                    if len(non_valid) != 0 and len(non_valid) != a_probs.size(0):
                        idx = torch.tensor(list(non_valid))
                        a_probs_new[0,idx] = 0
                if a_probs_new.sum() == 0:
                    print("a_prob_new is zero")
                    a_probs_new = a_probs.clone()
                a_select = torch.multinomial(a_probs_new, 1)
            return a_probs, a_select, (hx, cx), x_block

        def loss(self, a_prob, a_taken, adv, entropy_coeff):
            a_prob_select = torch.gather(a_prob, dim= -1, index = a_taken)
            entropy = torch.distributions.Categorical(a_prob).entropy() * entropy_coeff
            loss = -torch.log(a_prob_select) * adv
            loss -= entropy.reshape(-1,1)
            loss = loss.mean()
            return loss
        def update(self, loss):
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    return Actor

def make_value_net(base_policy_type,double_obs_space = False, recurrent = False):
    BasePoliy = make_base_policy(base_policy_type, double_obs_space)
    class Critic(BasePoliy):
        def __init__(self, observation_space, hidden_dim, lr = 0.001, nonlin = F.leaky_relu):
            if double_obs_space:
                dim1_shape = observation_space[0].shape
                dim2 = observation_space[1].shape[0]
                super(Critic, self).__init__(dim1_shape, dim2, nonlin= nonlin)
            else:
                super(Critic, self).__init__(observation_space, hidden_dim, nonlin= nonlin)
           # super(Critic, self).__init__(observation_space, hidden_dim, nonlin= nonlin)
            self.recurrent = recurrent
            self.nonlin = nonlin
            self.fc_mid = nn.Linear(self.hidden_dim, hidden_dim)
            self.fc_out = nn.Linear(hidden_dim, 1)
            if recurrent:
                self.lstm = nn.LSTMCell(self.hidden_dim, hidden_dim)
            self.optimizer = optim.Adam(self.parameters(), lr =lr)
        def forward(self, x, hid_cell_state = None):
            x = super(Critic, self).forward(x)
            hx, cx = None,None
            if self.recurrent:
                x, cx = self.lstm(x, hid_cell_state)
                hx = x
                hx, cx = hx.clone(), cx.clone()
                x = self.nonlin(x)
            else:
                x = self.nonlin(self.fc_mid(x))
            x = self.fc_out(x)
            return x, (hx, cx)
        def update(self, loss):
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    return Critic





def advantages(policy,agent_id, r, v, ter_mask, dones_mask, next_obs, hc_actr_next, hc_cr_next, gamma, lambda_):
    '''
    Calculate advantages for a single rollout
        r-pytorch tensor 
        v-pytorch tensor
        ter-mask- list of True false ter values'''
    batch_len = len(ter_mask)
    v = v.cpu()
    v = v.detach()
    d_mask = torch.ones_like(v) 
    d_mask[dones_mask] = 0 
    t_mask = torch.ones_like(v) 
    t_mask[ter_mask] = 0 

    # if not hc_actr_next is None:
    #     hc_actr_next = (hc_actr_next[0].cpu(), hc_actr_next[1].cpu())
    #     hc_cr_next = (hc_cr_next[0].cpu(), hc_cr_next[1].cpu())

    add_vn = torch.zeros_like(v)
    select_n_obs_mask = t_mask.eq(0) * d_mask #where terminating but not done
    if not hc_actr_next is None:
        for ind, (select, nob, hc_a, hc_c) in enumerate(zip(select_n_obs_mask, next_obs, hc_actr_next, hc_cr_next)):
            if select:
               # nob = nob.cpu()
                hc_a = (hc_a[0].cpu(), hc_a[1].cpu()) 
                hc_c = (hc_c[0].cpu(), hc_c[1].cpu()) 
                (_, _, vnext, _,_,_ ) = policy.forward({agent_id:nob},{agent_id:hc_a}, {agent_id:hc_c})
                add_vn[ind] = vnext[agent_id].detach()
    else:
        for ind, (select, nob) in enumerate(zip(select_n_obs_mask, next_obs)):
            if select:
              #  nob = nob.cpu()
                (_, _, vnext, _,_, _ )= policy.forward({agent_id:nob})
                add_vn[ind] = vnext[agent_id].detach()

    vn = v.clone()
    vn = vn.roll(-1) 
    vn = vn * t_mask
    vn = vn + add_vn
    delta = r + gamma*vn - v
    adv = []
   # hldr = torch.zeros(1,1)
    for ind in range(batch_len-1, 0-1, -1):
        if ind == batch_len-1 or ter_mask[ind]:
            hldr = delta[ind]
        else:
            hldr = delta[ind] + gamma*lambda_*hldr
        adv.append(hldr)
    adv.reverse()
    return torch.stack(adv) 






