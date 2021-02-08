import numpy as np
#from Agents.PPO.utils import make_parallel_env
import torch
import argparse
from tabulate import tabulate
from Env.make_env import make_env
from gym import spaces
import config
from PIL import Image
import time
import os
import ray

#ray.init()



#class DataGenerator():
    #def __init__(self):
    #    pass
        #self.args = args
        #self.data_path = data_path
 #       self.env = make_parallel_env(args, np.random.randint(5000), 2)
  #      self.n_envs = len(self.env)
#@ray.remote
def generate(env, episodes, n_agents):
    def make_start_postion_list(env_hldr):
        '''Assumes agent keys in evn.agents is the same as agent id's '''
        start_positions = []
        for i in range(len(env_hldr.agents.values())):
            start_positions.append(env_hldr.agents[i].pos)
        return start_positions

    def make_end_postion_list(env_hldr):
        '''Assumes agent keys in evn.agents is the same as agent id's '''
        end_positions = []
        for i in range(len(env_hldr.goals.values())):
            end_positions.append(env_hldr.goals[i].pos)
        return end_positions
    data = []
    observations = []
    actions = []
    ep_observations = []
    ep_actions = []
    for ep in range(episodes):
    #for ep in range(1):
        #continue_flag = False
        mstar_actions = None
        while mstar_actions is None:
            obs = env.reset()
            #env.render('human')
            start_pos = make_start_postion_list(env)
            end_pos = make_end_postion_list(env)
            #mstar_actions = env.graph.mstar_search3(start_pos, end_pos)

            inflation = 1.2
            mstar_actions =  env.graph.mstar_search4_OD(start_pos, end_pos, inflation)
            if mstar_actions is None:
                print("No M-star solution found for env generated")
        for a in mstar_actions:
           #assert continue_flag == False
            ep_observations.append(obs)
            #print("Observation: {}".format(obs[0][-1]))
            for i in range(n_agents):
                data.append((obs[i], a[i]))
            obs, r, dones, info = env.step(a)
            #env.render(mode = 'human')
            ep_actions.append(a)
            if info["terminate"]:
                if info["all_agents_on_goal"] == 1:
                    #add ep data
                    for ep_obs, ep_a in zip(ep_observations, ep_actions):
                      #  print("observations: {}".format(ep_obs))
                      #  print("\nActions: {}".format(ep_a))
                        for agent_handle, agent_obs in ep_obs.items():
                            pass
                           # data.append((agent_obs, ep_a[agent_handle]))
                            
                            ####################
                            # headers = ["Channels"]
                            # rows = [["Obstacle Channel"], ["Other Agent Channel"], ["Own Goals Channel"], \
                            #     ["Own Position Channel"], ["Other Goal Channel"]]
                            # rows[0].append(agent_obs[0])
                            # rows[1].append(agent_obs[1])
                            # rows[2].append(agent_obs[2])
                            # rows[3].append(agent_obs[4])
                            # rows[4].append(agent_obs[3])
                            # print(tabulate(rows, tablefmt = 'fancy_grid'))
                            # print("Action: {}".format(ep_a[agent_handle]))
                            #######################
                else:
                    print("M-star gave wrong solution. Episode data discarded.")
                ep_observations = []
                ep_a = []
               # continue_flag = True
        # #if continue_flag:
        # headers = ["Channels"]
        # rows = [["Obstacle Channel"], ["Other Agent Channel"], ["Own Goals Channel"], \
        #     ["Own Position Channel"], ["Other Goal Channel"]]
        # #for agnt in range(args.n_agents):
        # #headers.append("Agent {}".format(agnt))
        # rows[0].append(observations[ep*5][0])
        # rows[1].append(observations[ep*5][1])
        # rows[2].append(observations[ep*5][2])
        # rows[3].append(observations[ep*5][4])
        # rows[4].append(observations[ep*5][3])
        # print(tabulate(rows, tablefmt = 'fancy_grid'))
        # print("Action: {}".format(actions[ep*5]))
            #continue
    return data
    #return #{"observations": np.array(observations),
           # "actions": np.array(actions)}



def generate_data(name = "independent_navigation-v0", custom_args = None):
    
     #Environment:
    parser = argparse.ArgumentParser("Generate Data")
    parser.add_argument("--file_number", default = 0, type=int)
    parser.add_argument("--map_shape", default = 5, type=int)
    parser.add_argument("--n_agents", default = 4, type=int)
    parser.add_argument("--env_name", default = name, type= str)
    parser.add_argument("--use_default_rewards", default=True, type=bool)
    parser.add_argument("--view_d", default = 3, type=int)
    parser.add_argument("--obj_density", default = 0.2, type=float)
    parser.add_argument("--use_custom_rewards", default = False, action='store_true')
    parser.add_argument("--custom_env_ind", default= 1, type=int)
    parser.add_argument("--n_episodes", default= -1, type=int)
    parser.add_argument("--folder_name", default= "none", type=str)
    parser.add_argument("--data_name", default= "none", type=str)
    parser.add_argument("--base_path", default= "none", type=str)

    if custom_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(custom_args)

    args.map_shape = (args.map_shape, args.map_shape)

    if not args.n_episodes > 1:
        EPISODES = 5000
    else:
        EPISODES = args.n_episodes
    #WORKERS = 4
    if args.base_path == "none":
        import __main__
        base_path = os.path.dirname(__main__.__file__)
        base_path = os.path.join(base_path, "BC_Data")
    else:
        #base_path = args.base_path
        base_path = '/home/james/Desktop/Gridworld/BC_Data'

    data_folder = args.folder_name #'none'
    if args.data_name == "none":
        data_name = "none_" + str(args.file_number) + ".pt"
    else:
        data_name = args.data_name
    
    data_folder_dir = os.path.join(base_path, data_folder) #base_path + '/' + data_folder + '/'
    if not os.path.exists(data_folder_dir):
        os.makedirs(data_folder_dir)

    all_data = generate(make_env(args), EPISODES, args.n_agents)
    torch.save(all_data, os.path.join(data_folder_dir, data_name))






