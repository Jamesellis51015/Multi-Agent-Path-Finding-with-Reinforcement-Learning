import numpy as np
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
    ep_observations = []
    ep_actions = []
    for ep in range(episodes):
        mstar_actions = None
        while mstar_actions is None:
            obs = env.reset()
            start_pos = make_start_postion_list(env)
            end_pos = make_end_postion_list(env)
            inflation = 1.2
            mstar_actions =  env.graph.mstar_search4_OD(start_pos, end_pos, inflation)
            if mstar_actions is None:
                print("No M-star solution found for env generated")
        for a in mstar_actions:
            ep_observations.append(obs)
            for i in range(n_agents):
                data.append((obs[i], a[i]))
            obs, r, dones, info = env.step(a)
            ep_actions.append(a)
    return data


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
   
    if args.base_path == "none":
        import __main__
        base_path = os.path.dirname(__main__.__file__)
        base_path = os.path.join(base_path, "BC_Data")
    else:
  
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






