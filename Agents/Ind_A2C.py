import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym import spaces
from Agents.general_utils.policy import make_base_policy
from Env.make_env import make_env
from trainer import Trainer
from utils.wrappers import flat_np_lst, lst_to_dict
from utils.logger import Logger
import itertools
import copy
MSE_Loss = torch.nn.MSELoss()
#import config

#from Agents.get_returns import general_advantage_estimate

from collections import namedtuple
Transition = namedtuple('Transition', ('obs', 'a_select', 'a_prob', 'reward', 'value', 'next_state',"info"))


#Helper functions:
torch_obs = lambda x, dev: torch.from_numpy(flat_np_lst(x, flat=False)).type(torch.float32).to(dev)


def make_actor(base_policy_type):
    BasePoliy = make_base_policy(base_policy_type)
    class Actor(BasePoliy):
        def __init__(self, observation_space, hidden_dim, action_dim, lr = 0.001, nonlin = F.leaky_relu):
            super(Actor, self).__init__(observation_space, hidden_dim, nonlin= nonlin)
            self.nonlin = nonlin
            self.fc_mid = nn.Linear(hidden_dim, hidden_dim)
            self.fc_out = nn.Linear(hidden_dim, action_dim)
            self.optimizer = optim.Adam(self.parameters(), lr =lr)
        def forward(self, x):
            x = super(Actor, self).forward(x)
            x = self.nonlin(self.fc_mid(x))
            x = F.softmax(self.fc_out(x))
            return x
        def take_action(self, obs, greedy = False):
            a_probs = self.forward(obs)
            if greedy:
                a_select = torch.argmax(a_probs, dim=-1, keepdim= True)
            else:
                a_select = torch.multinomial(a_probs, 1)
            return a_probs, a_select
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

def make_value(base_policy_type):
    BasePoliy = make_base_policy(base_policy_type)
    class Critic(BasePoliy):
        def __init__(self, observation_space, hidden_dim, lr = 0.001, nonlin = F.leaky_relu):
            super(Critic, self).__init__(observation_space, hidden_dim, nonlin= nonlin)
            self.nonlin = nonlin
            self.fc_mid = nn.Linear(hidden_dim, hidden_dim)
            self.fc_out = nn.Linear(hidden_dim, 1)
            self.optimizer = optim.Adam(self.parameters(), lr =lr)
        def forward(self, x):
            x = super(Critic, self).forward(x)
            x = self.nonlin(self.fc_mid(x))
            x = self.fc_out(x)
            return x
        def loss(self, values, targets, value_coeff = 1.0):
            loss = MSE_Loss(values, targets.detach())
            #loss = ((targets.detach() - values)**2).mean()
            loss *= value_coeff
            return loss
        def update(self, loss):
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    return Critic

def get_batch(env, actor_lst, critic_lst, batch_size,dev, render = 0):
    n_agents = len(env.observation_space)
    batch = {i:[] for i in range(n_agents)}
    render_frames = []
    for ep in itertools.count():
        obs = env.reset()
        if render>0: render_frames.append(env.render('rgb_array'))
        obs = torch_obs(obs, dev)
        for t in itertools.count():
            a_select = []
            a_prob = []
            vals = []
            for i,(o, actor, critic) in enumerate(zip(obs, actor_lst, critic_lst)):
                a_p, a = actor.take_action(o.unsqueeze(0))
                a_select.append(a.item())
                a_prob.append(a_p)
                v = critic.forward(o.unsqueeze(0))
                vals.append(v)
            a_dict = lst_to_dict(a_select)
            obs_next, rewards, dones, info = env.step(a_dict)
            if render>0: render_frames.append(env.render('rgb_array'))
            obs_next = torch_obs(obs_next, dev)
            for i in range(n_agents):
                tran = Transition(obs[i], a_select[i], a_prob[i], rewards[i], vals[i], obs_next[i], copy.deepcopy(info))
                batch[i].append(tran)
            obs = obs_next
            if info["terminate"]:
                break
        render -= 1
        if len(batch[0]) >= batch_size:
            break
    return batch, render_frames

 

def mc_return(episode_mask, value, reward, discount):
    """Calculates the montecarlo return for a particular agent"""
    size = len(reward)
    G = np.zeros(shape = (size,), dtype=float)
    for i, hldr in enumerate(reversed([a for a in zip(reward, value, episode_mask)])):
        #NB: episode_mini_mask ignored ...episode does not finish until all agents finish
        if i == 0:
            if hldr[-1] == 0:
                G[size - i - 1] = hldr[0] #Reward in terminal state
            else:
                G[size - i - 1] = hldr[1] #The value [expected return] of being in that state (don't know the value of the next state)
        else:
            if hldr[-1]==1:
                G[size - i -1] = hldr[0] + discount*G[size - i]
            else:
                G[size - i -1] = hldr[0]       
    return torch.tensor([G], dtype=torch.float32)

def run_a2c(args):
    device = torch.device('cuda' if args.a2c_use_gpu else 'cpu')
    env = make_env(args)
    logger = Logger(args, env.summary(), "no policy summary", None)
    obs_space = env.observation_space[0].shape #assume agents are homogeneous
    n_actions = env.action_space[0].n
    n_agents = len(env.observation_space)
    gamma = args.a2c_discount
    
    if args.a2c_share_actor:
        Ac = make_actor(args.a2c_base_policy_type)
        ac = Ac(obs_space, args.a2c_hidden_dim, n_actions).to(device)
        actors = [ac for _ in range(n_agents)]
    else:
        Ac = make_actor(args.a2c_base_policy_type)
        actors = [Ac(obs_space, args.a2c_hidden_dim, n_actions).to(device) \
            for _ in range(n_agents)]
    if args.a2c_share_critic:
        Cr = make_value(args.a2c_base_policy_type)
        cr = Cr(obs_space, args.a2c_hidden_dim).to(device)
        critics = [cr for _ in range(n_agents)]
    else:
        Cr = make_value(args.a2c_base_policy_type)
        critics = [Cr(obs_space, args.a2c_hidden_dim).to(device) \
            for _ in range(n_agents)]
    stats = {}
    #stats["num_episodes"] = 0
    for it in range(args.iterations):
        print("Iteration: {}".format(it))
       # for a in actors: a.to(device)
       # for c in critics: c.to(device)
        batch, render_frames = get_batch(env, actors, critics, args.a2c_batch_size, device, render = logger.should_render())

        #convert dict of list of transitions to dict of transitonsion lists:
        for key in batch.keys():
            batch[key] = Transition(*zip(*batch[key]))
        
        #R = r + gamma*v(s')
        R = {}
        adv = {}
        val = {}
        for i in range(n_agents):
            v = batch[i].value
            v = torch.cat(v)
            val[i] = v
            v_lst = [vi.item() for vi in v]
            inf = batch[i].info

            ter_mask = torch.tensor([inf_["terminate"] for inf_ in inf])
            v_n_mask = torch.ones_like(v)
            v_n_mask[ter_mask] = 0
            g = mc_return(v_n_mask.squeeze().tolist(), v_lst, batch[i].reward, gamma)
            adv[i] = g.squeeze() - v.squeeze().detach()
            #hldr = g.reshape(-1,1)
            R[i] = g.reshape(-1,1)



            # v = batch[i].value
            # v = torch.cat(v)
            # val[i] = v
            # inf = batch[i].info
            # v_n = v.roll(-1)
            # ter_mask = torch.tensor([inf_["terminate"] for inf_ in inf])
            # v_n_mask = torch.ones_like(v)
            # v_n_mask[ter_mask] = 0
            # v_n = v_n*v_n_mask
            # v_n[-1] = 0 
            # r = torch.tensor(batch[i].reward).reshape(-1,1)
            # R[i] = r + gamma*v_n
            # adv[i] = R[i] - v
        
        #Critics loss:
        all_critic_loss = [c.loss(val[i], R[i], args.a2c_value_coeff) for c in critics]
        stats["value_loss"] = all_critic_loss[0]
        all_critic_loss = sum(all_critic_loss)
        # critics[0].optimizer.zero_grad()
        # all_critic_loss[0].backward()
        # critics[0].optimizer.step()
       # stats["action_loss"] = 0

        
        
        #Actors loss: 
        all_actor_loss = []
        for i in range(n_agents):
            a_select = torch.tensor(batch[i].a_select).reshape(-1, 1).to(device)
            a_prob = torch.cat(batch[i].a_prob).to(device)
            all_actor_loss.append(actors[i].loss(a_prob, a_select, adv[i], args.a2c_entropy_coeff))
        stats["action_loss"] = all_actor_loss[0]
        all_actor_loss = sum(all_actor_loss)

        if args.a2c_share_actor:
            all_actor_loss /= n_agents
            actors[0].optimizer.zero_grad()
        else:
            for a in actors:
                a.optimizer.zero_grad()
       # total_loss = all_actor_loss + all_critic_loss #value coeff and entropy already taken account of

        #Update:
        #total_loss.backward()
        all_actor_loss.backward()
        if args.a2c_share_actor:
            actors[0].optimizer.step()
        else:
            for a in actors:
                a.optimizer.step()

        if args.a2c_share_critic:
            all_critic_loss /= n_agents
            critics[0].optimizer.zero_grad()
        else:
            for c in critics:
                c.optimizer.zero_grad()

        all_critic_loss.backward()
        if args.a2c_share_critic:
            critics[0].optimizer.step()
        else:
            for c in critics:
                c.optimizer.step()

        #Logging:
        stats["iterations"] = it
        stats["num_timesteps"] = len(batch[0].info)
        terminal_t_info = [inf for i,inf in enumerate(batch[0].info) if inf["terminate"]]
        stats["num_episodes"] = len(terminal_t_info)
        logger.log(stats, terminal_t_info, render_frames, checkpoint=False)


if __name__ == "__main__":
    pass

    

