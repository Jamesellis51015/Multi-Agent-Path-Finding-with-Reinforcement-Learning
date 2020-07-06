#For manual testing:
import argparse
from tabulate import tabulate
from Env.make_env import make_env
from gym import spaces
import config
import numpy as np
from PIL import Image
import time

def test():
    from Env.env_entitiy_generator import random_obstacle_generator, csv_generator
    env_file = '/home/desktop123/Documents/Academics/Code/Multiagent_Gridworld/Env/custom/testcsv.csv'
    env_file = '/home/desktop123/Documents/Academics/Code/Multiagent_Gridworld/Env/custom/narrowCorridor1.csv'
    generator =  csv_generator(env_file)
    print(generator())


def test_mstar():
    from Env.env import Narrow_CorridorV0 

    name = 'narrow_corridor-v0'
    mode = "human"
    parser = argparse.ArgumentParser("Testing")

    #Environment:
    parser.add_argument("--map_shape", default = (7,7), type=object)
    parser.add_argument("--n_agents", default = 2, type=int)
    parser.add_argument("--env_name", default = name, type= str)
    parser.add_argument("--use_default_rewards", default=True, type=bool)
    parser.add_argument("--obj_density", default = 0.0, type=int)

    args = parser.parse_args()
    env = Narrow_CorridorV0(args, ind = 2)
    #from Env.env import Narrow_CorridorV0 
    #env = Narrow_CorridorV0(args, ind=i)
    print(env.summary())
    print("#################")
    print("Observation space: {}".format(env.observation_space))

   # obs = env.reset()#(env_ind=i)
    #obs = env.reset()#(env_ind=i)

    env.render(mode = mode)
    
    start_pos = tuple([a.pos for a in env.agents.values()])
    end_pos =[]
    for a in env.agents.values():
        for g in env.goals.values():
            if a.id == g.goal_id:
                end_pos.append(g.pos)
    end_pos = tuple(end_pos)

    #end_pos = tuple([a.pos for a in env.goals.values()])
    print("start_pos: {}    end_pos: {}".format(start_pos, end_pos))
    all_acts = env.heur.m_star_search(start_pos, end_pos)
    print(all_acts)

    all_steps = []
    for s in range(len(all_acts[0])):
        hldr = {}
        for k in all_acts.keys():
            hldr[k] = all_acts[k][s]
        all_steps.append(hldr)
    for h in all_steps:
        env.step(h)
        r = env.render(mode = mode)


    # print("Observation: {}".format(obs))
    # print("rewards: {}".format(rewards))
    # print("dones: {}".format(dones))
    # print("collisions: {}".format(collisions))
    # print("info: {}".format(info))

    print("Manual control: Actions: stay: 0 ; up: 1, right: 2, down: 3 ,left: 4 , step s \n Commands: agnt_handle action  e.g 0 2 s")

    cmd = ""
    while 'q' not in cmd:
        cmd = input(">> ")
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


def test_corridor(name = 'narrow_corridor-v0'):

    version = int(name[-1])
    i = 8
    mode = "human"
    parser = argparse.ArgumentParser("Testing")

    #Environment:
    parser.add_argument("--map_shape", default = (7,7), type=object)
    parser.add_argument("--n_agents", default = 2, type=int)
    parser.add_argument("--env_name", default = name, type= str)
    parser.add_argument("--use_default_rewards", default=True, type=bool)
    parser.add_argument("--obj_density", default = 0.3, type=int)

    args = parser.parse_args()
    env = make_env(args)
    #from Env.env import Narrow_CorridorV0 
    #env = Narrow_CorridorV0(args, ind=i)
    print(env.summary())
    print("#################")
    print("Observation space: {}".format(env.observation_space))

    obs = env.reset()#(env_ind=i)
    #obs = env.reset()#(env_ind=i)

    env.render(mode = mode)
    


    # print("Observation: {}".format(obs))
    # print("rewards: {}".format(rewards))
    # print("dones: {}".format(dones))
    # print("collisions: {}".format(collisions))
    # print("info: {}".format(info))

    print("Manual control: Actions: stay: 0 ; up: 1, right: 2, down: 3 ,left: 4 , step s \n Commands: agnt_handle action  e.g 0 2 s")

    cmd = ""
    while 'q' not in cmd:
        cmd = input(">> ")
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
                    #Obstacles;Other agents; own goal; other goals; own position
                    headers = ["Channels"]
                    rows = [["Obstacle Channel"], ["Other Agent Channel"], ["Own Goals Channel"], \
                        ["Other Goals Channel"], ["Own Position Channel"]]
                    for agnt in range(args.n_agents):
                        headers.append("Agent {}".format(agnt))
                        rows[0].append(obs[agnt][0])
                        rows[1].append(obs[agnt][1])
                        rows[2].append(obs[agnt][2])
                        rows[3].append(obs[agnt][3])
                        rows[4].append(obs[agnt][4])
                    print(tabulate(rows, headers = headers, tablefmt = 'fancy_grid'))
                # elif version == 1:
                #     headers = ["Channels"]
                #     rows = [["Obstacle Channel"], ["All Goals Channel"], \
                #         ["Own Position Channel"]]
                #     for agnt in range(args.n_agents):
                #         headers.append("Agent {}".format(agnt))
                #         rows[0].append(obs[agnt][0])
                #         rows[1].append(obs[agnt][1])
                #         rows[2].append(obs[agnt][2])
                #         #rows[3].append(obs[agnt][3])
                #     print(tabulate(rows, headers = headers, tablefmt = 'fancy_grid'))
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
            #print(r)

if __name__ == "__main__":
    #test()

    def test_IndNav(name = 'cooperative_navigation-v0'):

        version = int(name[-1])

        #mode = "rgb_array"
        mode = "human"
        parser = argparse.ArgumentParser("Testing")

        #Environment:
        parser.add_argument("--map_shape", default = (3,3), type=object)
        parser.add_argument("--n_agents", default = 2, type=int)
        parser.add_argument("--env_name", default = name, type= str)
        parser.add_argument("--use_default_rewards", default=True, type=bool)
        parser.add_argument("--obj_density", default = 0.0, type=int)
        parser.add_argument("--view_d", default = 2, type=int)
        parser.add_argument("--ppo_recurrent", default= False, action='store_true')
        parser.add_argument("--ppo_heur_block", default= False, action='store_true')
        parser.add_argument("--ppo_heur_valid_act", default= False, action='store_true')
        parser.add_argument("--ppo_heur_no_prev_state", default= False, action='store_true')

        args = parser.parse_args()
        env = make_env(args)
        print(env.summary())
        print("#################")
        print("Observation space: {}".format(env.observation_space[0]))

        obs = env.reset()
        obs = env.reset()

        env.render(mode = mode)

        start_pos = tuple([a.pos for a in env.agents.values()])
        end_pos = tuple([a.pos for a in env.goals.values()])
        print("start_pos: {}    end_pos: {}".format(start_pos, end_pos))
      #  all_acts = env.heur.m_star_search(start_pos, end_pos)
      #s  print(all_acts)

        #graph = GridToGraph(env.grid)
        #act, pos = graph.a_star_search(env.agents[0].pos, env.goals[0].pos)
        #print("a_star_results: {} \n {}".format(act, pos))
        #print("obs in way: {}".format(graph.get_blocking_obs(env.agents[0].pos, env.goals[0].pos)))
    

        # print("Observation: {}".format(obs))
        # print("rewards: {}".format(rewards))
        # print("dones: {}".format(dones))
        # print("collisions: {}".format(collisions))
        # print("info: {}".format(info))

        print("Manual control: Actions: stay: 0 ; up: 1, right: 2, down: 3 ,left: 4 , step s \n Commands: agnt_handle action  e.g 0 2 s")

        cmd = ""
        while 'q' not in cmd:
            cmd = input(">> ")
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
                    elif version == 1 or version == 2:
                        headers = ["Channels"]
                        rows = [["Obstacle Channel"], ["Other Agent Channel"], ["Own Goals Channel"], \
                             ["Other Goal Channel"], ["Vector"]]
                        for agnt in range(args.n_agents):
                            #obs, vec = obs
          
                            headers.append("Agent {}".format(agnt))
                            rows[0].append(obs[agnt][0][0])
                            rows[1].append(obs[agnt][0][1])
                            rows[2].append(obs[agnt][0][2])
                           # rows[3].append(obs[agnt][4])
                            rows[3].append(obs[agnt][0][3])
                            rows[4].append(obs[agnt][1])
                        print(tabulate(rows, headers = headers, tablefmt = 'fancy_grid'))
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
   
  
    def test_CN(name = 'cooperative_navigation-v0'):

        version = int(name[-1])

        #mode = "rgb_array"
        mode = "human"
        parser = argparse.ArgumentParser("Testing")

        #Environment:
        parser.add_argument("--map_shape", default = (5,5), type=object)
        parser.add_argument("--n_agents", default = 4, type=int)
        parser.add_argument("--env_name", default = name, type= str)
        parser.add_argument("--use_default_rewards", default=True, type=bool)
        parser.add_argument("--obj_density", default = 0.6, type=int)

        args = parser.parse_args()
        env = make_env(args)
        print(env.summary())
        print("#################")
        print("Observation space: {}".format(env.observation_space))

        obs = env.reset()
        obs = env.reset()

        env.render(mode = mode)

        #graph = GridToGraph(env.grid)
        #act, pos = graph.a_star_search(env.agents[0].pos, env.goals[0].pos)
        #print("a_star_results: {} \n {}".format(act, pos))
        #print("obs in way: {}".format(graph.get_blocking_obs(env.agents[0].pos, env.goals[0].pos)))
    

        # print("Observation: {}".format(obs))
        # print("rewards: {}".format(rewards))
        # print("dones: {}".format(dones))
        # print("collisions: {}".format(collisions))
        # print("info: {}".format(info))

        print("Manual control: Actions: stay: 0 ; up: 1, right: 2, down: 3 ,left: 4 , step s \n Commands: agnt_handle action  e.g 0 2 s")

        cmd = ""
        while 'q' not in cmd:
            cmd = input(">> ")
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
                        rows = [["Obstacle Channel"], ["Other Agent Channel"], ["All Goals Channel"], \
                            ["Own Position Channel"]]
                        for agnt in range(args.n_agents):
                            headers.append("Agent {}".format(agnt))
                            rows[0].append(obs[agnt][0])
                            rows[1].append(obs[agnt][1])
                            rows[2].append(obs[agnt][2])
                            rows[3].append(obs[agnt][3])
                        print(tabulate(rows, headers = headers, tablefmt = 'fancy_grid'))
                    elif version == 1:
                        headers = ["Channels"]
                        rows = [["Obstacle Channel"], ["All Goals Channel"], \
                            ["Own Position Channel"]]
                        for agnt in range(args.n_agents):
                            headers.append("Agent {}".format(agnt))
                            rows[0].append(obs[agnt][0])
                            rows[1].append(obs[agnt][1])
                            rows[2].append(obs[agnt][2])
                            #rows[3].append(obs[agnt][3])
                        print(tabulate(rows, headers = headers, tablefmt = 'fancy_grid'))
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

                #time.sleep(0.1)
  
    
  #  name = 'cooperative_navigation-v0'
    # name2 = 'cooperative_navigation-v1'
   # test_CN(name)
   # test_rendering()
    name = 'independent_navigation-v0'


    # name = 'independent_navigation-v2'
    test_IndNav(name)

   # test_corridor()

   # test_mstar()















 # def t():
    #   print("a")
    # t()
    # def test_rendering():
    #     parser = argparse.ArgumentParser("Testing")
    #       #Environment:
    #     parser.add_argument("--map_shape", default = (4,4), type=object)
    #     parser.add_argument("--n_agents", default = 2, type=int)
    #     parser.add_argument("--env_name", default = 'cooperative_navigation-v0', type= str)
    #     parser.add_argument("--use_default_rewards", default=True, type=bool)
    #     parser.add_argument("--use_custom_rewards", default=False, type=bool)
    #     parser.add_argument("--obj_density", default = 0.1, type=int)

    #     args = parser.parse_args()
    #     env = make_env(args)
    #     print(env.summary())
    #     print("#################")
    #     print("Observation space: {}".format(env.observation_space))

    #     obs = env.reset()
    #     obs = env.reset()

    #     a = env.render()

    #     img = Image.fromarray(a, 'RGB')

    #     img.save('t_render.png')
    #     img.show()

    #     print(type(a))

    # def calculate_return(episode_mask, value, reward, discount):
    #     """Calculates the montecarlo return for a particular agent"""
    #    # begin_flag = True
    #     size = len(reward)
    #     G = np.zeros(shape = (size,), dtype=float)
    #     for i, hldr in enumerate(reversed([a for a in zip(reward, value, episode_mask)])):
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
    #     return G 


    # def general_advantage_estimate(episode_mask, value, reward, discount, lambda_):
    #     size = len(reward)
    #     #Calculate deltas
    #     delta = np.zeros(shape = (size,), dtype=float)
    #     for i, hldr in enumerate(reversed([a for a in zip(reward, value, episode_mask)])):
    #         #NB: episode_mini_mask ignored ...episode does not finish until all agents finish
    #         if i == 0:
    #             if hldr[-1] == 0: #If the episode step is terminal
    #                 delta[size - i - 1] = hldr[0] #Reward in terminal state
    #             else:
    #                 delta[size - i - 1] = hldr[1] #The value [expected return] of being in that state (don't know the value of the next state)
    #         else:
    #             if hldr[-1]==1: #if the episode step is not terminal
    #                 delta[size - i -1] = hldr[0] + discount*value[size - i] - hldr[1]
    #             else:
    #                 delta[size - i -1] = hldr[0]
    #     #Calculate advantages:
    #     adv = np.zeros(shape = (size,), dtype=float)  
    #     for i, d in enumerate(reversed([k for k in zip(delta, episode_mask)])):
    #         if i ==0 or d[-1] == 0:
    #             adv[size - i -1] = d[0]
    #         else:
    #             adv[size - i -1] = d[0] + discount * lambda_ * adv[size - i]     
    #     return delta, adv#torch.tensor([adv]).double()

#     r = [0.01, 0.01, 0.01, 0.02, 3.0, 0.01, 0.01, 0.05,2.0]
#     value = [1 for i in range(len(r))]
#     ep_mask = [1 for i in r]
#     ep_mask[-1] = 0
#     ep_mask[4] = 0

#     d, a = general_advantage_estimate(ep_mask, value, r, discount = 1.0, lambda_ = 1.0)
#     print(ep_mask)
#     print(value)
#     print(r)
#     print(a)
#     print(d)

#     print("Returns")
#     g = calculate_return(ep_mask, value, r, discount = 1.0)
#     print(ep_mask)
#     print(value)
#     print(r)
#     print(g)
#    # print(d)

######################################