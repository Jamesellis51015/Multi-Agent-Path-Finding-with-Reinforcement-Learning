import ray
import itertools
import torch
import config



def process_state_f(observation):
    (obs, rewards, dones, collisions, info) = observation
    x = obs[0][0]
    y = obs[0][1]
    x1 = torch.tensor(x)
    y1 = torch.tensor(y)
    #new_obs = {agent_id: \
    #(o[0], o[1]) for agent_id, o in obs.items()}
    new_obs = {agent_id: \
    ( torch.tensor(o[0]).reshape((1,-1)).to(config.device), torch.tensor(o[1]).reshape((1,-1)).to(config.device) ) for agent_id, o in obs.items()}

    return  new_obs

def s_a_adv_minibatch(policy, traj_list,gamma, lambda_, seperate_agents = False):# NB: TEST
    """traj_list; a list of (sarst, values, logging_info) tuples
        seperate_agents: Whether or not to append trajectories from
        different agents in the same environmnet. If homogeneous agents, 
        it can be appended"""
    states_p1 = {agent_id: [] for agent_id in traj_list[0][0].keys()}
    states_p2 = {agent_id: [] for agent_id in traj_list[0][0].keys()}
    actions = {agent_id: [] for agent_id in traj_list[0][0].keys()}
    advantages = {agent_id: [] for agent_id in traj_list[0][0].keys()}

    for trajectory in traj_list:
        (svarst, logging_info) = trajectory
        for agent_id, hldr in svarst.items():
            adv = None
            for i, (s, v, a, r, s_n, t) in enumerate(reversed(hldr)):
                if i == 0:
                    if t:
                        adv = r - v.item()
                    else:
                        a_prob,v_next = policy(*s_n)
                        adv = r + gamma * v_next.item() - v.item()
                else:
                    if t: 
                        adv = r - v.item()
                    else:
                        adv = r + gamma * lambda_ * adv - v.item()
                states_p1[agent_id].append(s[0])
                states_p2[agent_id].append(s[1])
                actions[agent_id].append(a)
                advantages[agent_id].append(adv)
        if seperate_agents != True:
            states_p1 = list(itertools.chain.from_iterable(states_p1.values()))
            states_p2 = list(itertools.chain.from_iterable(states_p2.values()))
            actions = list(itertools.chain.from_iterable(actions.values()))
            advantages = list(itertools.chain.from_iterable(advantages.values()))

            states_p1 = torch.cat(states_p1, dim=0)
            states_p2 = torch.cat(states_p2, dim=0)
            actions =  torch.cat(actions, dim = 0)
            advantages =  torch.tensor(advantages).reshape((-1,1))

        else:
            for agent_id, s in states_p1.items():
                states_p1[agent_id] = torch.cat(s, dim = 0)
            for agent_id, s in states_p2.items():
                states_p2[agent_id] = torch.cat(s, dim = 0)
            for agent_id, a in actions.items():
                actions[agent_id] = torch.cat(a, dim = 0)
            for agent_id, adv in advantages.items():
                advantages[agent_id] = torch.tensor(adv).reshape((-1,1))
        
        return (states_p1, states_p2), actions, advantages


        



#@ray.remote(num_cpus = 1, num_gpus=0.1)
class env_inst():
    def __init__(self, env, n_step, process_state_f):
        self.env = env
        self.n_step = n_step
        self.obs_next = None
        self.process_state = process_state_f

    def get_env(self):
        return self.env

    def rollout(self,policy):
        svarst = {agent_id:[] for agent_id in self.env.agents.keys()} #State, action, reward, ter
        values = {agent_id:[] for agent_id in self.env.agents.keys()} #State values list
        logging_info = []
        if self.obs_next == None:
            obs = self.env.reset()
        else:
            obs = self.obs_next
        
        for i in range(self.n_step):
            o = self.process_state(obs)
            a ={}
            for agent_id in self.env.agents.keys():
                a_prob, v = policy.forward(*o[agent_id])
                a[agent_id] = policy.get_action(a_prob)
                values[agent_id] = v

            

            self.obs_next = self.env.step(a)

            o_next = self.process_state(self.obs_next)

            for agent_id in self.env.agents.keys():
                svarst[agent_id].append((o[agent_id], v, a[agent_id], self.obs_next[1][agent_id], o_next[agent_id], self.obs_next[-1]["terminate"]))

            if svarst[0][-1][-1]:
                obs = self.env.reset()
                logging_info.append(self.obs_next[-1])
            else:
                obs = self.obs_next
        return (svarst, logging_info)
            







                










        trans = []
        #np.random.seed(1)
        for i in range(t):
            if self.global_T == 0:
                self.env.seed(sdr.get())
                self.this_state = torch.tensor([self.env.reset()], dtype=torch.float)
                
                

            _, a_prob = policy.forward(self.this_state)
            a = policy.get_action(a_prob).squeeze().item()

            #a = np.random.choice([0,1])

            s_next,r,ter,_ = self.env.step(a)
            s_next = torch.tensor([s_next], dtype=torch.float)
            self.r_log_hldr.append(r)
            trans.append(sar(self.this_state, a, r, s_next,a_prob.squeeze()[a].item(), ter))

            self.global_T += 1
            self.this_state = s_next

            if ter:
                self.global_T = 0
                self.r_log = self.r_log_hldr[:]
                self.r_log_hldr = []
                return trans
        return trans