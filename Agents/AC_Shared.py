import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym import spaces
import config

from Agents.get_returns import general_advantage_estimate

from collections import namedtuple


Transition = namedtuple('Transition', ('state', 'action_taken', 'action_prob', 'value', 'episode_mask', 'episode_mini_mask', 'next_state',
                                       'reward', 'misc'))

class AC_Shared(nn.Module):
    '''Reinforce with baseline (one network) '''
    def __init__(self, args, observation_space, action_space):
        self.args =  args
        self.cooperative_returns = args.cooperative_returns
        self.entropy_coefficient = self.args.c2
        self.value_ceofficient = self.args.c1
        self.discount = self.args.discount

        super().__init__()
        

       
        if type(observation_space) == tuple:
            raise NotImplementedError
        elif isinstance(observation_space, spaces.Box):
            (channels, d1, d2) =  observation_space.shape
            self.c1 = nn.Conv2d(channels, 3*channels, 2).double()
            self.c2 = nn.Conv2d(3*channels, 2*channels, 2).double()
            self.fc1 = nn.Linear(2*channels*(d1-2)*(d2-2), 120).double()
            self.fc2 = nn.Linear(120, 120).double()
            self.action_out = nn.Linear(120, action_space.n).double()
            self.value = nn.Linear(120, 1).double()

        else:
            raise Exception 
        
        self.optimizer = optim.Adam(self.parameters(), lr =0.001, weight_decay=0.0)

    def forward(self, x):
        batch_size = x.size(0)
        print("Does x require grad: {}".format(x.requires_grad))
        x = self.c1(x)
        x = torch.relu(x)
        x = self.c2(x)
        x = torch.relu(x)
        x = x.reshape((batch_size,-1))
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        y = self.action_out(x)
        a_prob = F.log_softmax(y, dim = -1)
        value = self.value(x)
        print("Does a_prob require grad: {}".format(a_prob.requires_grad))
        return value, a_prob

    def update(self, transitions_batch, per_agent = True):

        '''Perform update for each agent, since each agent will have its own batch of observations
           ALTERNATIVELY: Can combine all obervations into single batch and update [per-agent = false]
           transition batch => list of agent_id: value dictionaries'''


        processed_batch = self._process_batch(transitions_batch)
        self.optimizer.zero_grad()

            #Updating as if its one agent (since on-policy)... Therefore effective batch size is individual_batch-size * n_agents 
        total_loss = torch.tensor(0).double().to(config.device)
        for agent_id, batch in processed_batch.items():

            time_steps = torch.tensor(batch.action_prob.size(0)).double().to(config.device)
            actions_taken = batch.action_taken.to(config.device).unsqueeze(-1).long()
            a_log_prob = batch.action_prob.to(config.device)
            select_a_log_prob = torch.gather(a_log_prob,-1, actions_taken).squeeze(-1)
            action_loss = -select_a_log_prob *(batch.reward.detach().to(config.device) - batch.value.detach()) #return == reward as stored in namedtuple
            #action_loss -= batch.value.detach() #.to(config.device)  
            entropy = torch.distributions.Categorical(probs = a_log_prob.exp()).entropy() * torch.tensor(self.entropy_coefficient).double().to(config.device)
            action_loss -= entropy
            action_loss = action_loss.sum() / time_steps

            value_loss = (batch.value - batch.reward.detach().to(config.device))**2
            value_loss = value_loss.sum() / time_steps

            loss = action_loss + torch.tensor(self.value_ceofficient).to(config.device) * value_loss

            total_loss +=loss
        
        total_loss.backward()
        self.optimizer.step()

        return 0, loss.item()

    def take_action(self, observations):
        '''Observation is dict of agent_id: np_array pairs
         Need to return dict(agent_id: int_action), dict(agent_id: prob_action), dict(agent_id: value)'''
        actions = {}
        action_prob = {}
        value = {}
        self.to(config.device)

        for agent_id, obs in observations.items():
            obs = self._process_obs(obs)#torch.tensor(obs).unsqueeze(-1).to(config.device)
            val, a_log_prob = self.forward(obs)
            a = torch.multinomial(a_log_prob.exp(), 1)

            actions[agent_id] = a.item()
            action_prob[agent_id] = a_log_prob
            value[agent_id] = val

        return actions, action_prob, value



    def _process_batch(self,batch):
        '''Makes batches of each agents obs, r, a, next_obs, values
        returns a dictionary of agent_id: Transition(state ...), with state ..etc
        being a tensor '''

        n_agents = len(batch.state[0]) #Length of an dictionary item

        processed_batch = {}
        for agent_id in range(n_agents):
            obs = [torch.tensor(x[agent_id]).double() for x in batch.state]
            obs = torch.stack(obs)

            action_taken = [torch.tensor(x[agent_id]).double() for x in batch.action_taken]
            

            action_prob = [x[agent_id] for x in batch.action_prob]
            action_prob = torch.cat(action_prob, dim=0).double().unsqueeze(0)

            value = [x[agent_id] for x in batch.value]
            

            reward = [x[agent_id] for x in batch.reward]
            

            episode_mask = [mask for mask in batch.episode_mask]
            episode_mini_mask = [1 if done[agent_id] == False else 0 for done in batch.episode_mini_mask]

            #returns = self.calculate_return(episode_mask, episode_mini_mask, action_taken, value, reward, self.args.discount)
            returns = general_advantage_estimate(episode_mask, episode_mini_mask, action_taken, value, reward, self.args.discount, self.args.lambda_)

            episode_mask = torch.tensor(episode_mask).unsqueeze(0).double()
            episode_mini_mask = torch.tensor(episode_mini_mask).unsqueeze(0).double()
            action_taken = torch.stack(action_taken).unsqueeze(0).double()
            value = torch.cat(value, dim=0).reshape((1, -1))
            

            #Neglect: next_state and misc
            next_state = None
            misc = None

            processed_batch[agent_id] = Transition(obs, action_taken, action_prob, value, episode_mask, episode_mini_mask, next_state, returns, misc)

        return processed_batch
        #('state', 'action_taken', 'action_prob', 'value', 'episode_mask', 'episode_mini_mask', 'next_state',
        #                               'reward', 'misc')

    def _process_obs(self, obs):
        '''Converts sigle agent's observation to tensor '''
        obs = torch.tensor(obs, requires_grad=True).unsqueeze(0).to(config.device)
        return obs

    # def calculate_return(self, episode_mask, episode_mini_mask, actions_taken, value, reward, discount):
    #     """Calculates the montecarlo return for a particular agent"""
    #    # begin_flag = True
    #     size = len(reward)
    #     G = np.zeros(shape = (size,), dtype=float)
    #     for i, hldr in enumerate(reversed([a for a in zip(reward, value, actions_taken, episode_mini_mask, episode_mask)])):
    #         #NB: episode_mini_mask ignored ...episode does not finish until all agents finish
    #         if i == 0:
    #             if hldr[-1] == 0:
    #                 G[size - i - 1] = hldr[0] #Reward in terminal state
    #             else:
    #                 G[size - i - 1] = hldr[1] #The value [expected return] of being in that state (don't know the value of the next state)
    #         else:
    #             if hldr[-1]==1:
    #                 G[size - i -1] = hldr[0] + discount*G[size - i]
    #             else:
    #                 G[size - i -1] = hldr[0]       
    #     return torch.tensor([G]).double()
    
    def summary(self):
        return "Not implemented"

    def save(self, path):
        torch.save(self.state_dict(), path)