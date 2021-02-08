#For manual testing:
import argparse
from tabulate import tabulate
from Env.make_env import make_env
from gym import spaces
import config
import numpy as np
from PIL import Image
import time


def test_IndNav(name = 'cooperative_navigation-v0'):

        version = int(name[-1])

        if name == 'independent_navigation-v8_0':
            version = 3

        #mode = "rgb_array"
        mode = "human"
        parser = argparse.ArgumentParser("Testing")

        #Environment:
        parser.add_argument("--map_shape", default = (10,10), type=object)
        parser.add_argument("--n_agents", default = 4, type=int)
        parser.add_argument("--env_name", default = name, type= str)
        parser.add_argument("--use_default_rewards", default=True, type=bool)
        parser.add_argument("--obj_density", default = 0.3, type=int)
        parser.add_argument("--view_d", default = 3, type=int)
        parser.add_argument("--ppo_recurrent", default= False, action='store_true')
        parser.add_argument("--ppo_heur_block", default= False, action='store_true')
        parser.add_argument("--ppo_heur_valid_act", default= True, action='store_true')
        parser.add_argument("--ppo_heur_no_prev_state", default= False, action='store_true')

        parser.add_argument("--use_custom_rewards", default = False, action='store_true')
        parser.add_argument("--step_r", default = -10, type=float)
        parser.add_argument("--agent_collision_r", default = -10, type=float)
        parser.add_argument("--obstacle_collision_r", default = -10, type=float)
        parser.add_argument("--goal_reached_r", default = -10, type=float)
        parser.add_argument("--finish_episode_r", default = -10, type=float)
        parser.add_argument("--block_r", default = -10, type=float)

        args = parser.parse_args()
        env = make_env(args)
        #print(env.summary())
        #print("#################")
        #print("Observation space: {}".format(env.observation_space[0]))

        obs = env.reset()
        env.render(mode = mode)
    
        #print("Observation: {}".format(obs))
        # print("rewards: {}".format(rewards))
        # print("dones: {}".format(dones))
        # print("collisions: {}".format(collisions))
        # print("info: {}".format(info))

        print("Manual control: Actions: stay: 0 ; up: 1, right: 2, down: 3 ,left: 4 , step s \n Commands: agnt_handle action  e.g 0 2 s")

        cmd = ""
        while 'q' not in cmd:
            cmd = input(">> ")
            print(cmd)
            cmds = cmd.split(" ")
            action_dict = {}
            i = 0
            while i < len(cmds):
                if cmds[i] == 'q':
                    import sys
                    sys.exit()
                elif cmds[i] == 's':
                    (obs, rewards, dones, info) = env.step(action_dict)
                    print(obs[0])
                    if version == 0:
                        headers = ["Channels"]
                        rows = [["Obstacle Channel"], ["Other Agent Channel"], ["Own Goals Channel"], \
                            ["Own Position Channel"], ["Other Goal Channel"]]
                        for agnt in range(args.n_agents):
                            headers.append("Agent {}".format(agnt))
                            rows[0].append(obs[agnt][0])
                            rows[1].append(obs[agnt][1])
                            rows[2].append(obs[agnt][2])
                            rows[3].append(obs[agnt][4])
                            rows[4].append(obs[agnt][3])
                        print(tabulate(rows, headers = headers, tablefmt = 'fancy_grid'))
                        if args.ppo_heur_block:
                            print("Blocking is: {}".format(info["blocking"]))
                        if args.ppo_heur_valid_act:
                            print("Valid actions are: {}".format({k:env.graph.get_valid_actions(k) for k in env.agents.keys()}))
                    elif version == 1 or version == 2:
                        headers = ["Channels"]
                        rows = [["Obstacle Channel"], ["Other Agent Channel"], ["Own Goals Channel"], \
                             ["Other Goal Channel"], ["Vector"]]
                        for agnt in range(args.n_agents):
                            headers.append("Agent {}".format(agnt))
                            rows[0].append(obs[agnt][0][0])
                            rows[1].append(obs[agnt][0][1])
                            rows[2].append(obs[agnt][0][2])
                            rows[3].append(obs[agnt][0][3])
                            rows[4].append(obs[agnt][1])
                        print(tabulate(rows, headers = headers, tablefmt = 'fancy_grid'))
                        if args.ppo_heur_block:
                            print("Blocking is: {}".format(info["blocking"]))
                        if args.ppo_heur_valid_act:
                            print("Valid actions are: {}".format({k:env.graph.get_valid_actions(k) for k in env.agents.keys()}))
                    if version == 3:
                        headers = ["Channels"]
                        rows = [["Obstacle Channel"], ["Other Agent Channel"], ["Own Goals Channel"], \
                            ["Other Goal Channel"], ["Shortest Path Heur"]]
                        for agnt in range(args.n_agents):
                            headers.append("Agent {}".format(agnt))
                            rows[0].append(obs[agnt][0])
                            rows[1].append(obs[agnt][1])
                            rows[2].append(obs[agnt][2])
                            rows[3].append(obs[agnt][3])
                            rows[4].append(obs[agnt][4].round(3))
                        print(tabulate(rows, headers = headers, tablefmt = 'fancy_grid'))
                        if args.ppo_heur_block:
                            print("Blocking is: {}".format(info["blocking"]))
                        if args.ppo_heur_valid_act:
                            print("Valid actions are: {}".format({k:env.graph.get_valid_actions(k) for k in env.agents.keys()}))
                    else:
                        raise NotImplementedError
                    print("Rewards: ", rewards)
                    print("Dones: ", dones)
                    print("Collisions: ", info["step_collisions"])
                    print("info: {}".format(info))
                    
                else:
                    agent_id = int(cmds[i])
                    action = int(cmds[i + 1])
                    action_dict[agent_id] = action
                    i = i + 1
                i += 1
                r = env.render(mode = mode)
                print(r)

if __name__ == "__main__":

    name = 'independent_navigation-v0'

    test_IndNav(name)
  







