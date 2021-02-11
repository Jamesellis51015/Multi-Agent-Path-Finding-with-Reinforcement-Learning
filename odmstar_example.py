import argparse
from tabulate import tabulate
from Env.make_env import make_env
from gym import spaces
import config
import numpy as np
from PIL import Image
import time


if __name__ == "__main__":
    from Env.env import Independent_NavigationV8_0
    name ="test_mstar-v0"
    mode = "human"
    parser = argparse.ArgumentParser("Testing")

    #Environment:
    parser.add_argument("--map_shape", default = 32, type=int)
    parser.add_argument("--n_agents", default = 20, type=int)
    parser.add_argument("--view_d", default = 3, type=int)
    parser.add_argument("--env_name", default = name, type= str)
    parser.add_argument("--use_default_rewards", default=True, type=bool)
    parser.add_argument("--obj_density", default = 0.2, type=float)
    parser.add_argument("--use_custom_rewards", default = False, action='store_true')
    parser.add_argument("--ppo_heur_block", default= False, type=bool)
    parser.add_argument("--inflation", default = 1.1, type=float)
    parser.add_argument("--step_time", default = 0.25, type=float, help="Time taken between rendered env steps.")

    args = parser.parse_args()
    args.map_shape = (args.map_shape, args.map_shape)
    t0 = time.time()
    env = Independent_NavigationV8_0(args)
    t01 = time.time()

    obs = env.reset()
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

    start_pos = make_start_postion_list(env)
    end_pos = make_end_postion_list(env)
    t2 = time.time()
    print("Searching with inflation factor = {}".format(args.inflation))
    all_actions, time_taken =  env.graph.ODmstar(start_pos, end_pos, args.inflation, return_time_taken=True)
    print("Mstar time taken: {}  seconds".format(time_taken))
    #print(all_actions)

    #Execute and render search results:
    env.render(mode='human')
    for a in all_actions:
        env.step(a)
        time.sleep(args.step_time)
        env.render(mode='human')
    time.sleep(args.step_time)
        


