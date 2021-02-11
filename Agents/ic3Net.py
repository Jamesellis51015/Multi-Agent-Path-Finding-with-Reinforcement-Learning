'''
Code modified from: https://github.com/IC3Net/IC3Net/blob/master/comm.py 
'''
from collections import namedtuple
import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym import spaces
import config
import copy
from Agents.general_utils.policy import make_base_policy
import math
Transition = namedtuple('Transition', ('state', 'action_taken', 'action_prob', 'value', 'episode_mask', 'episode_mini_mask', 'next_state',
                                       'reward', 'misc'))

#Functions from https://github.com/IC3Net/IC3Net/blob/master/comm.py 
def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)
    
def multinomials_log_density(actions, log_probs):
    log_prob = 0
    for i in range(len(log_probs)):
        log_prob += log_probs[i].gather(1, actions[:, i].long().unsqueeze(1))
    return log_prob

def multinomials_log_densities(actions, log_probs):
    log_prob = [0] * len(log_probs)
    for i in range(len(log_probs)):
        log_prob[i] += log_probs[i].gather(1, actions[:, i].long().unsqueeze(1))
    log_prob = torch.cat(log_prob, dim=-1)
    return log_prob

class IC3Net(nn.Module):
    def __init__(self, args, observation_space, action_space):
        ''' num_inputs is observation '''
        super(IC3Net, self).__init__()
        self.args = args
        self.nagents = args.n_agents
        self.hid_size = args.hid_size
        self.comm_passes = args.comm_passes
        self.recurrent = args.recurrent
        self.n_actions = action_space[0].n
        self.base_policy_type = args.ic3_base_policy_type
        args.naction_heads = [self.n_actions, 2]
        self.heads = nn.ModuleList([nn.Linear(args.hid_size, o).double()
                                        for o in args.naction_heads])

        self.make_state_encoder(observation_space[0])

        
        if args.recurrent:
            #self.hidden_enc = nn.Linear(self.hid_size, self.hid_size)
            self.f_module = nn.LSTMCell(self.hid_size, self.hid_size).double()
        else:
            if args.share_weights:
                self.f_module = nn.Linear(args.hid_size, args.hid_size).double()
                self.f_modules = nn.ModuleList([self.f_module
                                                for _ in range(self.comm_passes)])
            else:
                self.f_modules = nn.ModuleList([nn.Linear(args.hid_size, args.hid_size).double()
                                                for _ in range(self.comm_passes)])
        if args.share_weights:
            self.C_module = nn.Linear(args.hid_size, args.hid_size).double()
            self.C_modules = nn.ModuleList([self.C_module
                                            for _ in range(self.comm_passes)])
        else:
            self.C_modules = nn.ModuleList([nn.Linear(args.hid_size, args.hid_size).double()
                                            for _ in range(self.comm_passes)])

        self.value_head = nn.Linear(self.hid_size, 1).double()

        if self.args.comm_mask_zero:
            self.comm_mask = torch.zeros(self.nagents, self.nagents)
        else:
            self.comm_mask = torch.ones(self.nagents, self.nagents) \
                            - torch.eye(self.nagents, self.nagents)

        hldr = self.parameters()
        self.optimizer = optim.Adam(self.parameters(), lr =0.001, weight_decay=0.0)

        self.to(config.device)


    
    def make_state_encoder(self, observation_space):
        if type(observation_space) == tuple:
            raise NotImplementedError
        elif isinstance(observation_space, spaces.Box):
            BasePolicy = make_base_policy(self.base_policy_type)
            self.enc = BasePolicy(observation_space.shape, self.hid_size, nonlin = F.leaky_relu).double()

    def forward_state_encoder(self, x):
        '''x = obs, (hidden_state, cell_state) '''
        hidden_state, cell_state = None, None

        if self.args.recurrent:
            raise Exception("Recurrent policy not implemented")
            x, extras = x
            hidden_state, cell_state = extras
        else:
            x = self._process_obs(x)
            bat, n, channels,d1,d2 = x.size()
            x = x.view(bat*n, channels,d1,d2)
            x = self.enc.forward(x)
            hidden_state = x.view(bat,n, -1)
        return x, hidden_state, cell_state


    def get_agent_mask(self, batch_size, info):
        n = self.nagents
        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask'])
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(n)
            num_agents_alive = n

        agent_mask = agent_mask.view(1, 1, n)
        agent_mask = agent_mask.expand(batch_size, n, n).unsqueeze(-1)

        return num_agents_alive, agent_mask

    def forward(self, x, info={}, greedy = False):
        # NB: This function taken from: https://github.com/IC3Net/IC3Net/blob/master/comm.py
        x, hidden_state, cell_state = self.forward_state_encoder(x)
        batch_size = 1
        n = self.nagents
        x = x.reshape(batch_size, n, self.hid_size)
        num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)
        agent_mask = agent_mask.clone()
        agent_mask = agent_mask.to(config.device)
        # Communication action:
        if self.args.hard_attn:
            comm_action = torch.tensor(info['comm_action']).to(config.device)
            comm_action_mask = comm_action.expand(batch_size, n, n).unsqueeze(-1)
            
            agent_mask *= comm_action_mask.double()

        agent_mask_transpose = agent_mask.transpose(1, 2)

        for i in range(self.comm_passes):
            comm = hidden_state.view(batch_size, n, self.hid_size) if self.args.recurrent else hidden_state

            comm = comm.unsqueeze(-2).expand(-1, n, n, self.hid_size)
            mask = self.comm_mask.view(1, n, n).to(config.device)
            mask = mask.expand(comm.shape[0], n, n)
            mask = mask.unsqueeze(-1)

            mask = mask.expand_as(comm)
            comm = comm * mask

            if hasattr(self.args, 'comm_mode') and self.args.comm_mode == 'avg' \
                and num_agents_alive > 1:
                comm = comm / (num_agents_alive - 1)
            comm = comm * agent_mask
            comm = comm * agent_mask_transpose

            comm_sum = comm.sum(dim=1)
            c = self.C_modules[i](comm_sum)


            if self.args.recurrent:
                inp = x + c

                inp = inp.view(batch_size * n, self.hid_size)

                output = self.f_module(inp, (hidden_state, cell_state))

                hidden_state = output[0]
                cell_state = output[1]

            else: 
                hidden_state = sum([x, self.f_modules[i](hidden_state), c])
                hidden_state = F.tanh(hidden_state)
        
        value_head = self.value_head(hidden_state)
        h = hidden_state.view(batch_size, n, self.hid_size)

        action_prob = [F.log_softmax(head(h), dim=-1) for head in self.heads]

        action = self.select_action(action_prob, greedy)
        action = [a.reshape(-1).cpu().numpy() for a in action[:]]
        action_dict = {i:a.item() for i,a in enumerate(action[0])}#self.make_action_dict(action)

        if self.args.recurrent:
            return action_dict, action, action_prob, value_head, (hidden_state.clone(), cell_state.clone())

        else:
            return action_dict, action, action_prob, value_head


    def _process_obs(self, obs):
        '''Convert [dict] of obs to tensor and returns list [obs_tens, hidden_tens] #No need to modify hid_tens yet '''
        n = []
        for key, val in obs.items():
            n.append(torch.from_numpy(val))
        #bat.append(n)
        obs = torch.stack(n).unsqueeze(0).to(config.device)
        return obs
        
    def _process_batch(self,batch):
        '''expected input a Transition named tuple:
            batch.state: NA
            action_taken: [ lst(act_d1_tens, act_d2_tens) ]
            action_prob: [ lst(act_d1_tens, act_d2_tens) ] 
            value: [ tens[bat*n,1] ] 

            Expected output:
            action_taken: a list of list([act_dim1 for each agent ; act_dim2 for each agent]) size: [[3,3] ...batch]
            action_prob: a list of lists list([act_prob_dim1 ; act_prob_dim2])
            episode_mask: a list of ep_mask arrays
            ep_mini_mask: a list of ep_mini_mask arrays
            reward: list of reward arrays, each array contains of length n_agents contains rewards for each agent.
            values: lsit of tensors containing the output value of each agent. tensor shape: (3,1)
            '''
        r_arr = [list(r.values()) for r in batch.reward]

        return Transition(batch.state, batch.action_taken, batch.action_prob, batch.value, batch.episode_mask, batch.episode_mini_mask, batch.next_state, r_arr, batch.misc)
        
    def take_action(self, observations, info, greedy=False):
        return self.forward(observations, info, greedy)

    def update(self, transitions_batch, per_agent = True):
        '''Perform update for each agent, since each agent will have its own batch of observations
           ALTERNATIVELY: Can combine all obervations into single batch and update [per-agent = false]
           transition batch => list of agent_id: value dictionaries'''
        #TAKEN FROM compute_grad FUNCTION IN  https://github.com/IC3Net/IC3Net/blob/master/comm.py

        stat = dict()
        num_actions = [self.n_actions,2]
        dim_actions = 2
        self.optimizer.zero_grad()
        n = self.nagents
        batch = self._process_batch(transitions_batch) #Transition batch make one large batch with all agents observations etc. 
        batch_size = len(batch.state)

        rewards = torch.Tensor(batch.reward)
        episode_masks = torch.Tensor(batch.episode_mask)
        episode_mini_masks = torch.Tensor(batch.episode_mini_mask)
        actions = torch.Tensor(batch.action_taken).to(config.device)
        actions = actions.transpose(1, 2).view(-1, n, dim_actions)

        # can't do batch forward.
        values = torch.cat(batch.value, dim=0)
        action_out = list(zip(*batch.action_prob))
        action_out = [torch.cat(a, dim=0).to(config.device) for a in action_out]

        alive_masks = torch.Tensor(np.concatenate([item['alive_mask'] for item in batch.misc])).view(-1).to(config.device)

       # coop_returns = torch.Tensor(batch_size, n)
        ncoop_returns = torch.Tensor(batch_size, n)
        returns = torch.Tensor(batch_size, n)
       # deltas = torch.Tensor(batch_size, n)
        advantages = torch.Tensor(batch_size, n)
        values = values.view(batch_size, n).cpu()

        prev_coop_return = 0
        prev_ncoop_return = 0
        prev_value = 0
        prev_advantage = 0

        for i in reversed(range(rewards.size(0))):
            ncoop_returns[i] = rewards[i] + self.args.discount * prev_ncoop_return * episode_masks[i] * episode_mini_masks[i]

            prev_ncoop_return = ncoop_returns[i].clone()
            returns[i] = ncoop_returns[i]

        for i in reversed(range(rewards.size(0))):
            advantages[i] = returns[i] - values.data[i]

        log_p_a = [action_out[i].view(-1, num_actions[i]) for i in range(dim_actions)]
        actions = actions.contiguous().view(-1, dim_actions)

        log_prob = multinomials_log_density(actions, log_p_a)

        entropy = 0

        ent_coeff = torch.tensor(self.args.entropy_coeff).to(config.device)
        #Policy action entropy
        entropy -= (log_p_a[0] * log_p_a[0].exp()).sum() * ent_coeff
        #Communication action entropy
        entropy -= (log_p_a[1] * log_p_a[1].exp()).sum() * ent_coeff*0.05

        #entropy = entropy * torch.tensor(self.args.entropy_coeff).to(config.device)
        action_loss = -advantages.to(config.device).view(-1) * log_prob.squeeze()
        action_loss *= alive_masks

        action_loss = action_loss.sum()
        action_loss -= entropy
        stat['action_loss'] = action_loss.item()

        # value loss term
        targets = returns.to(config.device)
        value_loss = (values.to(config.device) - targets).pow(2).view(-1)
        value_loss *= alive_masks
        value_loss = value_loss.sum()

        stat['value_loss'] = value_loss.item()
        loss = action_loss + torch.tensor(self.args.value_coeff).to(config.device) * value_loss

        loss /= torch.tensor(batch_size).to(config.device)
        
        loss.backward()
        self.optimizer.step()

        return stat['value_loss'], stat['action_loss'] 

    def init_hidden(self, batch_size):
        return tuple(( torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True).double().to(config.device),
                       torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True).double().to(config.device)))

    def select_action(self, a_log_prob, greedy = False):
        ret = None
        if type(a_log_prob) == list:
            log_p_a = a_log_prob
            p_a = [[z.exp() for z in x] for x in log_p_a]
            if greedy == False:
                ret = torch.stack([torch.stack([torch.multinomial(x, 1).detach() for x in p]) for p in p_a])
            else:
                ret = torch.stack([torch.stack([torch.argmax(x, dim = -1, keepdim=True).detach() for x in p]) for p in p_a])
        else:
            raise NotImplementedError
        return ret
    
    def make_action_dict(self, actions):
        '''Actions is a numpy array or tensor of shape (batch, n, 1) or [arr1, arr2] with arr being previous shape.
            returns either a list of dict or a single dict if batch == 1. dict is agent: action key-val pairs '''
        if type(actions) != list:
            actions = [actions]
        ret = []
        for a in actions:
            if type(a) == torch.Tensor:
                size = a.size()
            elif type(a) == np.ndarray:
                size = np.shape
            else:
                raise Exception('Type of action input passed to make_action_dict function not recognized')

            if size[0] ==1: #The batch dimension
                out = dict()
                for i in range(size[1]): #The agent dimension
                    out[i] = a[0, i, 0].item()
            else:
                out = []
                for bat in a:
                    act = dict()
                    for i,n in enumerate(bat):
                        act[i] = n.item()
                    out.append(act)
            ret.append(out)
        if len(ret) == 1: ret = ret[0]

        return ret
            
    def summary(self):
        return "Not implemented"
    
    def save(self, path):
        torch.save(self.state_dict(), path)





                



