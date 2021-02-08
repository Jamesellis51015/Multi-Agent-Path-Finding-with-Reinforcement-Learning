import main
from sklearn.model_selection import ParameterGrid
#For manual testing:
import argparse
from tabulate import tabulate
from Env.make_env import make_env
from gym import spaces
import config
import numpy as np
from PIL import Image
import time
import signal
import torch
import os

def mstar_comparison_1_0_0():
    from Env.env import Independent_NavigationV8_0

    FILE_PATH = "/home/james/Desktop/Gridworld/EXPERIMENTS/odmstar_and_odrmstar_comparison2/"
    NUM_TRIALS = 100
    TIME_LIMIT = 5*60 + 1

    name ="test_mstar-v0"
    mode = "human"
    parser = argparse.ArgumentParser("Testing")

    #Environment:
    parser.add_argument("--map_shape", default = (32,32), type=object)
    parser.add_argument("--n_agents", default = 10, type=int)
    parser.add_argument("--view_d", default = 3, type=int)
    parser.add_argument("--env_name", default = name, type= str)
    parser.add_argument("--use_default_rewards", default=True, type=bool)
    parser.add_argument("--obj_density", default = 0.2, type=int)
    parser.add_argument("--use_custom_rewards", default = False, action='store_true')
    parser.add_argument("--custom_env_ind", default= 1, type=int)
    parser.add_argument("--ppo_heur_block", default=False, type=bool)

    args = parser.parse_args()

    env = Independent_NavigationV8_0(args)

    print(env.summary())

    obs = env.reset()

    def make_start_postion_list(env_hldr):
        '''Assumes agent keys in evn.agents is the same as agent id's '''
        start_positions = []
        for i in range(len(env_hldr.agents.values())):
            start_positions.append(env.agents[i].pos)
        return start_positions

    def make_end_postion_list(env_hldr):
        '''Assumes agent keys in evn.agents is the same as agent id's '''
        end_positions = []
        for i in range(len(env_hldr.goals.values())):
            end_positions.append(env.goals[i].pos)
        return end_positions

    def signal_handler(signum, frame):
        raise Exception("Time expired")

    signal.signal(signal.SIGALRM, signal_handler)

    for inf in [1.1]:
        for n in [40,50,60]:
            args.n_agents = n
            env = Independent_NavigationV8_0(args)
            print("Running inflation: {}    nagents: {}".format(inf, n))
            data = {
            "name": "32x32_density_0.2",
            "od_mstar_time": [],
            "od_mstar_length": [],
            "odr_mstar_time": [],
            "odr_mstar_length": []
            }
            #name_hldr = data["name"]
            data["name"] = data["name"] + "_nagents_" + str(n) + "_inflation_" + str(inf)
            for i in range(NUM_TRIALS): 
                obs = env.reset()          
                start_pos = make_start_postion_list(env)
                end_pos = make_end_postion_list(env)

                print("Start: {}   End {}".format(start_pos, end_pos))

                #Mstar_OD          
                all_actions = None
                time_taken = None
                signal.alarm(TIME_LIMIT)   
                try:
                    all_actions, time_taken =  env.graph.mstar_search4_OD(start_pos, end_pos, inflation=inf, return_time_taken=True)
                except:
                    print("ODMstar time ran out of time")

                if time_taken is not None:
                    data["od_mstar_time"].append(float(time_taken))
                else:
                    data["od_mstar_time"].append(None)
                
                if all_actions is not None:
                    data["od_mstar_length"].append(len(all_actions))
                else:
                    data["od_mstar_length"].append(None)

                # Mstar_ODr
                all_actions = None
                time_taken = None
                signal.alarm(TIME_LIMIT)   
                try:
                    all_actions, time_taken =  env.graph.mstar_search_ODrMstar(start_pos, end_pos, inflation=inf, return_time_taken=True)
                except:
                    print("ODrMstar time ran out of time")

                if time_taken is not None:
                    data["odr_mstar_time"].append(float(time_taken))
                else:
                    data["odr_mstar_time"].append(time_taken)
                
                if all_actions is not None:
                    data["odr_mstar_length"].append(len(all_actions))
                else:
                    data["odr_mstar_length"].append(None)
            
            torch.save(data, os.path.join(FILE_PATH, data["name"]+".pkl"))
    # t2 = time.time()
    # all_actions =  env.graph.mstar_search_ODrMstar(start_pos, end_pos)
    # print("Time for mstar_search4_ODrM* with inflation: {}".format(time.time() - t2))
    # print(all_actions)
    # print("Length all actions: {}".format(len(all_actions)))


    

    


def mstar10():
    from Env.env import Independent_NavigationV8_0

    FILE_PATH = "/home/james/Desktop/Gridworld/EXPERIMENTS/odmstarFinal/"
    NUM_TRIALS = 20
    TIME_LIMIT = 5*60

    name ="test_mstar-v0"
    mode = "human"
    parser = argparse.ArgumentParser("Testing")

    #Environment:
    parser.add_argument("--map_shape", default = (10,10), type=object)
    parser.add_argument("--n_agents", default = 10, type=int)
    parser.add_argument("--view_d", default = 3, type=int)
    parser.add_argument("--env_name", default = name, type= str)
    parser.add_argument("--use_default_rewards", default=True, type=bool)
    parser.add_argument("--obj_density", default = 0.2, type=int)
    parser.add_argument("--use_custom_rewards", default = False, action='store_true')
    parser.add_argument("--custom_env_ind", default= 1, type=int)
    parser.add_argument("--ppo_heur_block", default=False, type=bool)

    args = parser.parse_args()

    

    #print(env.summary())

    #obs = env.reset()
   # print("Time taken to reset env: {}".format(t01 - time.time()))


    def make_start_postion_list(env_hldr):
        '''Assumes agent keys in evn.agents is the same as agent id's '''
        start_positions = []
        for i in range(len(env_hldr.agents.values())):
            start_positions.append(env.agents[i].pos)
        return start_positions

    def make_end_postion_list(env_hldr):
        '''Assumes agent keys in evn.agents is the same as agent id's '''
        end_positions = []
        for i in range(len(env_hldr.goals.values())):
            end_positions.append(env.goals[i].pos)
        return end_positions

    def signal_handler(signum, frame):
        raise Exception("Time expired")

    signal.signal(signal.SIGALRM, signal_handler)

    for inf in [1.1,2.0,10.0]:
        for ob_dens in [0.0,0.1,0.2,0.3]:
            for n in [4,8,12,16,20,24,28]:
                args.n_agents = n
                args.obj_density = ob_dens
                name = "map_shape_" + str(args.map_shape[0]) + "_density_" + str(ob_dens) + "_nagents_" + str(n) + "_inf_" + str(inf)
                env = Independent_NavigationV8_0(args)
                print("Running inflation: {}    nagents: {} Density: {}".format(inf, n, ob_dens))
                # data = {
                # "name": name,
                # "od_mstar_time": [],
                # "od_mstar_length": [],
                # }
                results = {"agents_on_goal": [],
                "all_done": [],
                "episode_length": [],
                "execution_time": [],
                "obstacle_collisions": [],
                "agent_collisions": []}
                #name_hldr = data["name"]
                #data["name"] = data["name"] + "_nagents_" + str(n) + "_inflation_" + str(inf)
                for i in range(NUM_TRIALS): 
                    _ = env.reset()          
                    start_pos = make_start_postion_list(env)
                    end_pos = make_end_postion_list(env)

                   # print("Start: {}   End {}".format(start_pos, end_pos))

                    #Mstar_OD          
                    all_actions = None
                    time_taken = None
                    signal.alarm(TIME_LIMIT)   
                    try:
                        all_actions, time_taken =  env.graph.mstar_search4_OD(start_pos, end_pos, inflation=inf, return_time_taken=True)
                    except:
                        print("ODMstar time ran out of time")

                    if all_actions is not None:
                        results['agents_on_goal'].append(1)
                        results["all_done"].append(1)
                        results["episode_length"].append(len(all_actions))
                    else:
                        results['agents_on_goal'].append(0)
                        results["all_done"].append(0)
                        results["episode_length"].append(None)
                    
                    if time_taken is not None:
                        results["execution_time"].append(float(time_taken))
                    else:
                        results["execution_time"].append(None)
                
                torch.save(results, os.path.join(FILE_PATH, name+".pkl"))








def mstar32():
    from Env.env import Independent_NavigationV8_0

    FILE_PATH = "/home/james/Desktop/Gridworld/EXPERIMENTS/odmstarFinal/"
    NUM_TRIALS = 20
    TIME_LIMIT = 5*60

    name ="test_mstar-v0"
    mode = "human"
    parser = argparse.ArgumentParser("Testing")

    #Environment:
    parser.add_argument("--map_shape", default = (32,32), type=object)
    parser.add_argument("--n_agents", default = 10, type=int)
    parser.add_argument("--view_d", default = 3, type=int)
    parser.add_argument("--env_name", default = name, type= str)
    parser.add_argument("--use_default_rewards", default=True, type=bool)
    parser.add_argument("--obj_density", default = 0.2, type=int)
    parser.add_argument("--use_custom_rewards", default = False, action='store_true')
    parser.add_argument("--custom_env_ind", default= 1, type=int)
    parser.add_argument("--ppo_heur_block", default=False, type=bool)

    args = parser.parse_args()

    def make_start_postion_list(env_hldr):
        '''Assumes agent keys in evn.agents is the same as agent id's '''
        start_positions = []
        for i in range(len(env_hldr.agents.values())):
            start_positions.append(env.agents[i].pos)
        return start_positions

    def make_end_postion_list(env_hldr):
        '''Assumes agent keys in evn.agents is the same as agent id's '''
        end_positions = []
        for i in range(len(env_hldr.goals.values())):
            end_positions.append(env.goals[i].pos)
        return end_positions

    def signal_handler(signum, frame):
        raise Exception("Time expired")

    signal.signal(signal.SIGALRM, signal_handler)

    for inf in [1.1,2.0,10.0]:
        for ob_dens in [0.0,0.1,0.2,0.3]:
            for n in [20,40,80]:
                args.n_agents = n
                args.obj_density = ob_dens
                name = "map_shape_" + str(args.map_shape[0]) + "_density_" + str(ob_dens) + "_nagents_" + str(n) + "_inf_" + str(inf)
                env = Independent_NavigationV8_0(args)
                print("Running inflation: {}    nagents: {} Density: {}".format(inf, n, ob_dens))
                # data = {
                # "name": name,
                # "od_mstar_time": [],
                # "od_mstar_length": [],
                # }
                results = {"agents_on_goal": [],
                "all_done": [],
                "episode_length": [],
                "execution_time": [],
                "obstacle_collisions": [],
                "agent_collisions": []}
                #name_hldr = data["name"]
                #data["name"] = data["name"] + "_nagents_" + str(n) + "_inflation_" + str(inf)
                for i in range(NUM_TRIALS): 
                    _ = env.reset()          
                    start_pos = make_start_postion_list(env)
                    end_pos = make_end_postion_list(env)

                    all_actions = None
                    time_taken = None
                    signal.alarm(TIME_LIMIT)   
                    try:
                        all_actions, time_taken =  env.graph.mstar_search4_OD(start_pos, end_pos, inflation=inf, return_time_taken=True)
                    except:
                        print("ODMstar time ran out of time")

                    if all_actions is not None:
                        results['agents_on_goal'].append(1)
                        results["all_done"].append(1)
                        results["episode_length"].append(len(all_actions))
                    else:
                        results['agents_on_goal'].append(0)
                        results["all_done"].append(0)
                        results["episode_length"].append(None)
                    
                    if time_taken is not None:
                        results["execution_time"].append(float(time_taken))
                    else:
                        results["execution_time"].append(None)
                
                torch.save(results, os.path.join(FILE_PATH, name+".pkl"))


def mstar50():
    from Env.env import Independent_NavigationV8_0

    FILE_PATH = "/home/james/Desktop/Gridworld/EXPERIMENTS/odmstarFinal/"
    #FILE_PATH = "/home/jellis/workspace5/gridworld2/odmstarFinal/"
    NUM_TRIALS = 20
    TIME_LIMIT = 5*60

    name ="test_mstar-v0"
    mode = "human"
    parser = argparse.ArgumentParser("Testing")

    #Environment:
    parser.add_argument("--map_shape", default = (50,50), type=object)
    parser.add_argument("--n_agents", default = 10, type=int)
    parser.add_argument("--view_d", default = 3, type=int)
    parser.add_argument("--env_name", default = name, type= str)
    parser.add_argument("--use_default_rewards", default=True, type=bool)
    parser.add_argument("--obj_density", default = 0.2, type=int)
    parser.add_argument("--use_custom_rewards", default = False, action='store_true')
    parser.add_argument("--custom_env_ind", default= 1, type=int)
    parser.add_argument("--ppo_heur_block", default=False, type=bool)

    args = parser.parse_args()

    def make_start_postion_list(env_hldr):
        '''Assumes agent keys in evn.agents is the same as agent id's '''
        start_positions = []
        for i in range(len(env_hldr.agents.values())):
            start_positions.append(env.agents[i].pos)
        return start_positions

    def make_end_postion_list(env_hldr):
        '''Assumes agent keys in evn.agents is the same as agent id's '''
        end_positions = []
        for i in range(len(env_hldr.goals.values())):
            end_positions.append(env.goals[i].pos)
        return end_positions

    def signal_handler(signum, frame):
        raise Exception("Time expired")

    signal.signal(signal.SIGALRM, signal_handler)

    for inf in [1.1,2.0,10.0]:
        for ob_dens in [0.0,0.1,0.2,0.3]:
            for n in [120,80,20]:
                args.n_agents = n
                args.obj_density = ob_dens
                name = "map_shape_" + str(args.map_shape[0]) + "_density_" + str(ob_dens) + "_nagents_" + str(n) + "_inf_" + str(inf)
                env = Independent_NavigationV8_0(args)
                print("Running inflation: {}    nagents: {} Density: {}".format(inf, n, ob_dens))
                # data = {
                # "name": name,
                # "od_mstar_time": [],
                # "od_mstar_length": [],
                # }
                results = {"agents_on_goal": [],
                "all_done": [],
                "episode_length": [],
                "execution_time": [],
                "obstacle_collisions": [],
                "agent_collisions": []}
                #name_hldr = data["name"]
                #data["name"] = data["name"] + "_nagents_" + str(n) + "_inflation_" + str(inf)
                for i in range(NUM_TRIALS): 
                    _ = env.reset()          
                    start_pos = make_start_postion_list(env)
                    end_pos = make_end_postion_list(env)

                    all_actions = None
                    time_taken = None
                    signal.alarm(TIME_LIMIT)   
                    try:
                        all_actions, time_taken =  env.graph.mstar_search4_OD(start_pos, end_pos, inflation=inf, return_time_taken=True)
                    except:
                        print("ODMstar time ran out of time")

                    if all_actions is not None:
                        results['agents_on_goal'].append(1)
                        results["all_done"].append(1)
                        results["episode_length"].append(len(all_actions))
                    else:
                        results['agents_on_goal'].append(0)
                        results["all_done"].append(0)
                        results["episode_length"].append(None)
                    
                    if time_taken is not None:
                        results["execution_time"].append(float(time_taken))
                    else:
                        results["execution_time"].append(None)
                
                torch.save(results, os.path.join(FILE_PATH, name+".pkl"))