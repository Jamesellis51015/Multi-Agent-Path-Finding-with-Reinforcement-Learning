
"""Different environment variants and make_env funciton"""
import numpy as np
import argparse
import time
import copy
import os
from os import listdir
from os.path import isfile, join

from Env.grid_env import Grid_Env
from Env.env_entitiy_generator import random_obstacle_generator, csv_generator
from utils.pathfinding import Heuristics



class Narrow_CorridorV0(Grid_Env):
    class Rewards():
        step= -0.01
        object_collision = -0.02
        agent_collision = -0.4
        goal_reached= 0.5
        finish_episode = 1
    class Global_Cooperative_Rewards():
        """All agents share this reward """
        step= -0.01
        object_collision = -0.2
        agent_collision = -0.4
        goal_reached= 0.0
        finish_episode = 1
    def __init__(self, args, ind = None):
        import __main__
        curr_dir = os.path.dirname(__file__)
        env_folder = curr_dir + r"/custom/narrowCorridor/"
        all_files = listdir(env_folder)
        sort_f = lambda x: int(x.split('.')[0])
        all_files.sort(key=sort_f)
        if ind is None:
            env_ind = np.random.choice(np.arange(len(all_files)), 1).item()
            env_file = join(env_folder, all_files[env_ind])
        else:
            env_file = join(env_folder, all_files[ind])
        #env_file = '/home/desktop123/Documents/Academics/Code/Multiagent_Gridworld/Env/custom/narrowCorridor1.csv'
        self.name = "narrow_corridoorEnv"
        self.args = args
        generator =  csv_generator(env_file)
        #generator2 = random_obstacle_generator((7,7), 2, obj_density = args.obj_density)
        #print("from csv: {}".format(g))
        #print("from random obs gen: {}".format(generator2()))
        #time.sleep(1000)
        super(). __init__(generator, diagonal_actions = False, fully_obs = True, view_d = 2, agent_collisions = True)
        #Goals:
        assert len(self.goals) == len(self.agents)
        #goal_ids = np.random.choice(len(self.goals), len(self.goals), replace = False) #agents assigned to goals
        #for i,g in enumerate(self.goals):
        #    g.goal_id = goal_ids[i]
        self.goals = {g.goal_id : g for g in self.goals}

        #Set reward function, obsevation space etc
        self.rewards = self.Rewards()

        observation_space = self.init_observation_space()
        self.observation_space = [copy.deepcopy(observation_space) for _ in self.agents]

        self.heur = Heuristics(self.grid, self)
    def reset(self, env_ind = None):
        self.__init__(self.args, ind=env_ind)
        (obs, rewards, dones, info) = self.step({h:0 for h in self.agents.keys()})
        return obs

    def get_rewards(self, collisions):
        rewards = {}
        isdone = {}
        overall_dones = {}
        for handle, agent in self.agents.items():
            for g in self.goals.values():
                if g.pos == agent.pos and g.goal_id == agent.id:
                    rewards[handle] = self.rewards.goal_reached
                    isdone[handle] = True
                    break
                else:
                    rewards[handle] = self.rewards.step
                    isdone[handle] = False
        if sum(isdone.values()) == len(self.agents):
            for handle in self.agents.keys(): 
                rewards[handle] = self.rewards.finish_episode
                overall_dones[handle] = True
        else:
            for i, agnt in self.agents.items():
                overall_dones[i] = False

        if collisions['obs_col']:
            for key, val in collisions['obs_col'].items():
                if val: rewards[key] += self.rewards.object_collision
        if collisions['agent_col']:
            for key, val in collisions['agent_col'].items():
                if val: rewards[key] += self.rewards.agent_collision
        return overall_dones, rewards
    
    def get_global_cooperative_rewards(self, collisions):
        r = {i:0 for i, a in self.agents.items()}
        return r



# 
class Independent_NavigationV0(Grid_Env):
    class Rewards():
        step= -0.01
        object_collision = -0.015#-0.1
        agent_collision = -0.4
        goal_reached= 0.1#0.01
        finish_episode = 1

    class Global_Cooperative_Rewards():
        """All agents share this reward """
        step= -0.01
        object_collision = -0.2
        agent_collision = -0.4
        goal_reached= 0.0
        finish_episode = 1

    def __init__(self, args):
        #Make default env arguments if not exist in args
        #initialize base Grid_Env
        self.description = "Agents have to reach their own goal whilst \
            avoiding collisions with other agents"
        self.args =  args
        generator = random_obstacle_generator(args.map_shape, args.n_agents, obj_density = args.obj_density)
        super().__init__(generator, diagonal_actions = False, fully_obs = False, view_d = 2, agent_collisions = True)
        self.rewards = self.Rewards()
        #Goals:
        assert len(self.goals) == len(self.agents)
        # goal_ids = np.random.choice(len(self.goals), len(self.goals), replace = False) #agents assigned to goals
        # for i,g in enumerate(self.goals):
        #     g.goal_id = goal_ids[i]
        for key, val in self.agents.items():
            val.id = key

        self.goals = {i : g for i,g in enumerate(self.goals)}
        for key, val in self.goals.items():
            val.goal_id = key

        self.graph = Heuristics(self.grid, self)
        self.clear_paths()
        #Set reward function, obsevation space etc
        self.rewards = self.Rewards()
        
        # if args.use_custom_rewards == True:
        #     self.rewards.step = args.reward_step
        #     self.rewards.object_collision = args.reward_obj_collision
        #     self.rewards.agent_collision = args.reward_agent_collision
        #     self.rewards.goal_reached = args.reward_goal_reached
        #     self.rewards.finish_episode = args.reward_finish_episode

        #Observation settings:
        self.fully_obs = True
        self.inc_obstacles = True
        self.inc_other_agents = True
        self.inc_other_goals = True #If agent does not have specific goal, only include other agent's goals.
        self.inc_own_goals = True #This is only applicable to Independet Navigation tasks; In CN goals are everyones goals
        self.inc_direction_vector = False

        observation_space = self.init_observation_space()
        self.observation_space = [copy.deepcopy(observation_space) for _ in self.agents]

    def reset(self, env_ind = None):
        self.__init__(self.args)
        (obs, rewards, dones, info) = self.step({h:0 for h in self.agents.keys()})
        return obs

    def clear_paths(self):
        all_goal_ids = [g.goal_id for g in self.goals.values()]
        for a in self.agents.values():
            ind = all_goal_ids.index(a.id)
            g = self.goals[ind]
            blocks = self.graph.get_blocking_obs(a.pos, g.pos)
            while(len(blocks) != 0):
                self._remove_object(blocks[0], "obstacle")
                blocks = self.graph.get_blocking_obs(a.pos, g.pos)


            

    def get_rewards(self, collisions):
        rewards = {}
        isdone = {}
        overall_dones = {}
        for handle, agent in self.agents.items():
            for g in self.goals.values():
                if g.pos == agent.pos and g.goal_id == agent.id:
                    rewards[handle] = self.rewards.goal_reached
                    isdone[handle] = True
                    break
                else:
                    rewards[handle] = self.rewards.step
                    isdone[handle] = False
        if sum(isdone.values()) == len(self.agents):
            for handle in self.agents.keys(): 
                rewards[handle] = self.rewards.finish_episode
                overall_dones[handle] = True
        else:
            for i, agnt in self.agents.items():
                overall_dones[i] = False

        if collisions['obs_col']:
            for key, val in collisions['obs_col'].items():
                if val: rewards[key] += self.rewards.object_collision
        if collisions['agent_col']:
            for key, val in collisions['agent_col'].items():
                if val: rewards[key] += self.rewards.agent_collision
        return overall_dones, rewards
    
    def get_global_cooperative_rewards(self, collisions):
        r = {i:0 for i, a in self.agents.items()}
        return r
       


class Independent_NavigationV1(Grid_Env):
    class Rewards():
        step= -0.01
        object_collision = -0.015#-0.1
        agent_collision = -0.4
        goal_reached= 0.1#0.01
        finish_episode = 1
        blocking = -0.8

    class Global_Cooperative_Rewards():
        """All agents share this reward """
        step= -0.01
        object_collision = -0.2
        agent_collision = -0.4
        goal_reached= 0.0
        finish_episode = 1

    def __init__(self, args):
        #Make default env arguments if not exist in args
        #initialize base Grid_Env
        self.description = "Agents have to reach their own goal whilst \
            avoiding collisions with other agents"
        self.args =  args
        generator = random_obstacle_generator(args.map_shape, args.n_agents, obj_density = args.obj_density)
        super().__init__(generator, diagonal_actions = False, fully_obs = False, view_d = args.view_d, agent_collisions = True)
        
        self.heur_block = args.ppo_heur_block
        self.heur_valid_act = args.ppo_heur_valid_act
        #self.heur_no_prev_state = args.ppo_heur_no_prev_state
        
        self.rewards = self.Rewards()
        #Goals:
        assert len(self.goals) == len(self.agents)
        # goal_ids = np.random.choice(len(self.goals), len(self.goals), replace = False) #agents assigned to goals
        # for i,g in enumerate(self.goals):
        #     g.goal_id = goal_ids[i]
        for key, val in self.agents.items():
            val.id = key

        self.goals = {i : g for i,g in enumerate(self.goals)}
        for key, val in self.goals.items():
            val.goal_id = key

        self.heur = Heuristics(self.grid, self)
        self.clear_paths()
        #Set reward function, obsevation space etc
        self.rewards = self.Rewards()
        
        # if args.use_custom_rewards == True:
        #     self.rewards.step = args.reward_step
        #     self.rewards.object_collision = args.reward_obj_collision
        #     self.rewards.agent_collision = args.reward_agent_collision
        #     self.rewards.goal_reached = args.reward_goal_reached
        #     self.rewards.finish_episode = args.reward_finish_episode

        #Observation settings:
        self.fully_obs = False
        self.inc_obstacles = True
        self.inc_other_agents = True
        self.inc_other_goals = True #If agent does not have specific goal, only include other agent's goals.
        self.inc_own_goals = True #This is only applicable to Independet Navigation tasks; In CN goals are everyones goals
        self.inc_direction_vector = True

        observation_space = self.init_observation_space()
        self.observation_space = [copy.deepcopy(observation_space) for _ in self.agents]

        self.blocking_hldr = {i:None for i in self.agents.keys()}

    def reset(self, env_ind = None):
        self.__init__(self.args)
        (obs, rewards, dones, info) = self.step({h:0 for h in self.agents.keys()})
        return obs

    def clear_paths(self):
        all_goal_ids = [g.goal_id for g in self.goals.values()]
        for a in self.agents.values():
            ind = all_goal_ids.index(a.id)
            g = self.goals[ind]
            blocks = self.heur.get_blocking_obs(a.pos, g.pos)
            for b in blocks:
                self._remove_object(b, "obstacle")
            # while(len(blocks) != 0):
            #     self._remove_object(blocks[0], "obstacle")
            #     blocks = self.graph.get_blocking_obs(a.pos, g.pos)
            
    def step(self, action_dict):
        (obs, rewards, dones, info) = super().step(action_dict)
        if self.heur_block:
            info["blocking"] = copy.deepcopy(self.blocking_hldr)
        else:
            info["blocking"] = None

        if self.heur_valid_act:
            info["valid_act"] = {i:self.heur.get_valid_actions(i) for i in self.agents.keys()}
        else:
            info["valid_act"] = None
        return (obs, rewards, dones, info)

    def get_rewards(self, collisions):
        rewards = {}
        isdone = {}
        overall_dones = {}
        for handle, agent in self.agents.items():
            for g in self.goals.values():
                if g.pos == agent.pos and g.goal_id == agent.id:
                    rewards[handle] = self.rewards.goal_reached
                    isdone[handle] = True
                    break
                else:
                    rewards[handle] = self.rewards.step
                    isdone[handle] = False
        if sum(isdone.values()) == len(self.agents):
            for handle in self.agents.keys(): 
                rewards[handle] = self.rewards.finish_episode
                overall_dones[handle] = True
        else:
            for i, agnt in self.agents.items():
                overall_dones[i] = False

        if collisions['obs_col']:
            for key, val in collisions['obs_col'].items():
                if val: rewards[key] += self.rewards.object_collision
        if collisions['agent_col']:
            for key, val in collisions['agent_col'].items():
                if val: rewards[key] += self.rewards.agent_collision
        if self.heur_block:
            for key in self.agents.keys():
                block = self.heur.is_blocking(key)
                self.blocking_hldr[key] = block
                if block:
                    rewards[key] += self.rewards.blocking

        return overall_dones, rewards
    
    def get_global_cooperative_rewards(self, collisions):
        r = {i:0 for i, a in self.agents.items()}
        return r



class Independent_NavigationV2(Grid_Env):
    class Rewards():
        step= -0.01
        object_collision = -0.015#-0.1
        agent_collision = -0.4
        goal_reached= 0.1#0.01
        finish_episode = 1
        blocking = -0.8

    class Global_Cooperative_Rewards():
        """All agents share this reward """
        step= -0.01
        object_collision = -0.2
        agent_collision = -0.4
        goal_reached= 0.0
        finish_episode = 1

    def __init__(self, args):
        #Make default env arguments if not exist in args
        #initialize base Grid_Env
        self.description = "Agents have to reach their own goal whilst \
            avoiding collisions with other agents"
        self.args =  args
        generator = random_obstacle_generator(args.map_shape, args.n_agents, obj_density = args.obj_density)
        super().__init__(generator, diagonal_actions = False, fully_obs = False, view_d = args.view_d, agent_collisions = True)
        
        self.heur_block = args.ppo_heur_block
        self.heur_valid_act = args.ppo_heur_valid_act
        #self.heur_no_prev_state = args.ppo_heur_no_prev_state
        
        self.rewards = self.Rewards()
        #Goals:
        assert len(self.goals) == len(self.agents)
        # goal_ids = np.random.choice(len(self.goals), len(self.goals), replace = False) #agents assigned to goals
        # for i,g in enumerate(self.goals):
        #     g.goal_id = goal_ids[i]
        for key, val in self.agents.items():
            val.id = key

        self.goals = {i : g for i,g in enumerate(self.goals)}
        for key, val in self.goals.items():
            val.goal_id = key

        self.heur = Heuristics(self.grid, self)
        self.clear_paths()
        #Set reward function, obsevation space etc
        self.rewards = self.Rewards()
        
        # if args.use_custom_rewards == True:
        #     self.rewards.step = args.reward_step
        #     self.rewards.object_collision = args.reward_obj_collision
        #     self.rewards.agent_collision = args.reward_agent_collision
        #     self.rewards.goal_reached = args.reward_goal_reached
        #     self.rewards.finish_episode = args.reward_finish_episode

        #Observation settings:
        self.fully_obs = False
        self.inc_obstacles = True
        self.inc_other_agents = True
        self.inc_other_goals = True #If agent does not have specific goal, only include other agent's goals.
        self.inc_own_goals = False #This is only applicable to Independet Navigation tasks; In CN goals are everyones goals
        self.inc_direction_vector = False
        self.inc_path_grid = True

        self.path_grid = None

        observation_space = self.init_observation_space()
        self.observation_space = [copy.deepcopy(observation_space) for _ in self.agents]

        self.blocking_hldr = {i:None for i in self.agents.keys()}

        if self.inc_path_grid:
            self.init_path_grid()

        self.norm_path_len = 1#self.x_len + self.y_len

    def init_path_grid(self):
        #A dict of dict where each dict[id] = {position : Node, ...}
        self.path_grid = {}
        for agent_id, agnt in self.agents.items():
            agnt_goal_id = agnt.id
            goal = self.goals[agnt_goal_id]
            assert goal.goal_id == agnt_goal_id
            self.path_grid[agent_id] = self.heur.dijkstra_search(goal.pos, agnt.pos)

    def _get_path_grid(self, agent):
        # end_pos = agent.pos
        # goal_id = agent.id
        # goal = self.goals[goal_id]
        # assert goal.goal_id == goal_id
        # start_pos = goal.pos
        d_view = self.view_d
        view_dim =  2*d_view + 1
        view = np.empty((view_dim, view_dim), dtype = object)
        (x_pos, y_pos) = agent.pos
        for x_cursor in range(-d_view, d_view + 1):
            for y_cursor in range(-d_view, d_view + 1):
                x = x_pos + x_cursor
                y = y_pos + y_cursor
                
                if x >= self.x_len or x < 0 or y>= self.y_len or y < 0:
                    view[d_view + x_cursor,d_view + y_cursor] = -1.0
                else:
                    cell_obj = self.get(x,y)
                    cell_types = [c.type for c in cell_obj]
                    if 'obstacle' in cell_types:
                        view[d_view + x_cursor,d_view + y_cursor] = -1.0
                    else:
                        pos = tuple((x, y))
                        if pos in self.path_grid[agent.id]:
                            node = self.path_grid[agent.id][pos]
                            cost_to_go = node.move_cost / self.norm_path_len
                            view[d_view + x_cursor,d_view + y_cursor] = cost_to_go
                        else:
                            view[d_view + x_cursor,d_view + y_cursor] = -1.0
                            #print("agent {}  pos {}".format(agent.id, pos))
        hldr = view.flatten()
        return view.flatten().astype(np.float32)


    def reset(self, env_ind = None):
        self.__init__(self.args)
        (obs, rewards, dones, info) = self.step({h:0 for h in self.agents.keys()})
        return obs

    def clear_paths(self):
        all_goal_ids = [g.goal_id for g in self.goals.values()]
        for a in self.agents.values():
            ind = all_goal_ids.index(a.id)
            g = self.goals[ind]
            blocks = self.heur.get_blocking_obs(a.pos, g.pos)
            for b in blocks:
                self._remove_object(b, "obstacle")
            # while(len(blocks) != 0):
            #     self._remove_object(blocks[0], "obstacle")
            #     blocks = self.graph.get_blocking_obs(a.pos, g.pos)
            
    def step(self, action_dict):
        (obs, rewards, dones, info) = super().step(action_dict)
        if self.heur_block:
            info["blocking"] = copy.deepcopy(self.blocking_hldr)
        else:
            info["blocking"] = None

        if self.heur_valid_act:
            info["valid_act"] = {i:self.heur.get_valid_actions(i) for i in self.agents.keys()}
        else:
            info["valid_act"] = None
        return (obs, rewards, dones, info)

    def get_rewards(self, collisions):
        rewards = {}
        isdone = {}
        overall_dones = {}
        for handle, agent in self.agents.items():
            for g in self.goals.values():
                if g.pos == agent.pos and g.goal_id == agent.id:
                    rewards[handle] = self.rewards.goal_reached
                    isdone[handle] = True
                    break
                else:
                    rewards[handle] = self.rewards.step
                    isdone[handle] = False
        if sum(isdone.values()) == len(self.agents):
            for handle in self.agents.keys(): 
                rewards[handle] = self.rewards.finish_episode
                overall_dones[handle] = True
        else:
            for i, agnt in self.agents.items():
                overall_dones[i] = False

        if collisions['obs_col']:
            for key, val in collisions['obs_col'].items():
                if val: rewards[key] += self.rewards.object_collision
        if collisions['agent_col']:
            for key, val in collisions['agent_col'].items():
                if val: rewards[key] += self.rewards.agent_collision
        if self.heur_block:
            for key in self.agents.keys():
                block = self.heur.is_blocking(key)
                self.blocking_hldr[key] = block
                if block:
                    rewards[key] += self.rewards.blocking

        return overall_dones, rewards
    
    def get_global_cooperative_rewards(self, collisions):
        r = {i:0 for i, a in self.agents.items()}
        return r



class Cooperative_Navigation_V0(Grid_Env):
    """
    Agents have cover a number of landmarks (goals) equal 
    to the number of agents whilst avoiding collisions. It 
    does not matter which agent covers which landmark.
    
    v0: -No obstacles 
        -fully observable,
        -observations 'fixed' (not relative to agent)
        -agents are spawned randomly anywhere """
    class Rewards():
        step= -0.01
        object_collision = -0.015#-0.1
        agent_collision = -0.4
        goal_reached= 0.1#0.01
        finish_episode = 1

    class Global_Cooperative_Rewards():
        """All agents share this reward """
        step= -0.01
        object_collision = -0.2
        agent_collision = -0.4
        goal_reached= 0.0
        finish_episode = 1

    def __init__(self, args):
        #Make default env arguments if not exist in args
        #initialize base Grid_Env
        self.description = "Agents have cover a number of landmarks (goals) equal" + \
            " to the number of agents whilst avoiding collisions. It" + \
            "does not matter which agent covers which landmark. No obstacles"
        self.args =  args
        generator = random_obstacle_generator(args.map_shape, args.n_agents, obj_density = args.obj_density)
        super().__init__(generator, diagonal_actions = False, fully_obs = False, view_d = 2, agent_collisions = True)
        self.rewards = self.Rewards()
        #Goals:
        assert len(self.goals) == len(self.agents)
        # goal_ids = np.random.choice(len(self.goals), len(self.goals), replace = False) #agents assigned to goals
        # for i,g in enumerate(self.goals):
        #     g.goal_id = goal_ids[i]
        self.goals = {i : g for i,g in enumerate(self.goals)}

        self.graph = GridToGraph(self.grid)
        self.clear_paths()
        #Set reward function, obsevation space etc
        self.rewards = self.Rewards()
        
        # if args.use_custom_rewards == True:
        #     self.rewards.step = args.reward_step
        #     self.rewards.object_collision = args.reward_obj_collision
        #     self.rewards.agent_collision = args.reward_agent_collision
        #     self.rewards.goal_reached = args.reward_goal_reached
        #     self.rewards.finish_episode = args.reward_finish_episode

        #Observation settings:
        self.fully_obs = True
        self.inc_obstacles = True
        self.inc_other_agents = True
        self.inc_other_goals = True #If agent does not have specific goal, only include other agent's goals.
        self.inc_own_goals = False #This is only applicable to Independet Navigation tasks; In CN goals are everyones goals
        self.inc_direction_vector = False

        observation_space = self.init_observation_space()
        self.observation_space = [copy.deepcopy(observation_space) for _ in self.agents]

    def reset(self, custom_env = None):
        self.__init__(self.args)
        (obs, rewards, dones, info) = self.step({h:0 for h in self.agents.keys()})
        #return (obs, rewards, dones, collisions, info)
        return obs

    def clear_paths(self, any_goal = True):
        if any_goal:
            for a in self.agents.values():
                blocks = []
                for g in self.goals.values():
                    blocks.append(self.graph.get_blocking_obs(a.pos, g.pos))
                len_blocks = [len(b)==0 for b in blocks]
                #checks that there is path from agent to any goal
                if not any(len_blocks):
                    ind = len_blocks.index(False)
                    bl = blocks[ind]
                    g = list(self.goals.values())[ind]
                    assert len(bl) != 0
                    while len(bl) != 0:
                        self._remove_object(bl[0], "obstacle")
                        bl = self.graph.get_blocking_obs(a.pos, g.pos)
        else:
            raise "Goals not distinct in cooperative navigation"




                        


    def get_rewards(self, collisions):
        '''Agent get individual rewards for naviagting to a goal position as well a group reward for covering all positions '''
        rewards = {agent_id:self.rewards.step for agent_id in self.agents.keys()} #initialise each agnets reward to the step reward
        isdone = {} 

        goal_positions = [g.pos for g in self.goals.values()]
        for agent_id, agent in self.agents.items():
            if agent.pos in goal_positions:
                rewards[agent_id] += self.rewards.goal_reached
           # else:
           #     rewards[agent_id] = self.rewards.step

        agent_positions = [a.pos for a in self.agents.values()]

        if set(goal_positions) == set(agent_positions):
            for agent_id, agent in self.agents.items():
                rewards[agent_id] += self.rewards.finish_episode
                isdone[agent_id] = True
        else:
            for agent_id, agent in self.agents.items():
                isdone[agent_id] = False
    
        if collisions['obs_col']:
            for key, val in collisions['obs_col'].items():
                if val: rewards[key] += self.rewards.object_collision

        if self.agent_collisions:
            if collisions['agent_col']:
                for key, val in collisions['agent_col'].items():
                    if val: rewards[key] += self.rewards.agent_collision

        return isdone, rewards
    
    def get_global_cooperative_rewards(self, collisions):
        """Returns the same reward for each agent.
            The reward function is: 
            reward = ( step_r*n_agent + n_goals_reached * goal_reward + collision_obs_r * n_obs_collisions
            + collision_agent_r * n_agnet_collisions ) / n_agents + finish_reward"""

        step_r = len(self.agents) * self.global_cooperative_rewards.step
        goals_r = 0

        agent_positions = [a.pos for a in self.agents.values()]
        goal_positions = [g.pos for g in self.goals.values()]
        for agent_id, agent in self.agents.items():
            if agent.pos in goal_positions:
                goals_r += self.global_cooperative_rewards.goal_reached

        coll_r = sum(collisions['agent_col'].values()) * self.global_cooperative_rewards.agent_collision \
         + sum(collisions['obs_col'].values()) * self.global_cooperative_rewards.object_collision

        if set(goal_positions) == set(agent_positions):
            finish_r = self.global_cooperative_rewards.finish_episode
        else:
            finish_r = 0

        r = (step_r + + goals_r + coll_r)/ len(self.agents) + finish_r

        reward = {i: r for i in range(len(self.agents))}
        return reward
        

class Cooperative_Navigation_V1(Cooperative_Navigation_V0):
    '''A partially observable version of V0. Agents do not see other agent positions. '''
    def __init__(self, args):
        super().__init__(args)
        self.inc_other_agents = False
        observation_space = self.init_observation_space()
        self.observation_space = [copy.deepcopy(observation_space) for _ in self.agents]






# if __name__ == "__main__":

#     def test_CN():
#         parser = argparse.ArgumentParser("Experiment parameters")

#         #Environment:
#         parser.add_argument("--map_shape", default = (7,7), type=object)
#         parser.add_argument("--n_agents", default = 10, type=int)

#         args = parser.parse_args()
#         env = make_env("cooperative_navigation-v1", args)

#         print("Observation space: {}".format(env.observation_space))

#         (obs, rewards, dones, collisions, info) = env.reset()

#         env.render()
   

#         print("Observation: {}".format(obs))
#         print("rewards: {}".format(rewards))
#         print("dones: {}".format(dones))
#         print("collisions: {}".format(collisions))
#         print("info: {}".format(info))

#         print("Manual control: Actions: stay: 0 ; up: 1, right: 2, down: 3 ,left: 4 , step s \n Commands: agnt_handle action  e.g 0 2 s")

#         cmd = ""
#         while 'q' not in cmd:
#             cmd = input(">> ")
#             cmds = cmd.split(" ")

#             action_dict = {}

#             i = 0
#             while i < len(cmds):
#                 if cmds[i] == 'q':
#                     import sys


#                     sys.exit()
#                 elif cmds[i] == 's':
#                     (obs, rewards, dones, collisions, info) = env.step(action_dict)
#                     print("Observations: ", obs)
#                     print("Rewards: ", rewards)
#                     print("Dones: ", dones)
#                     print("Collisions: ", collisions)
#                     print("info: {}".format(info))
                    
#                 else:
#                     agent_id = int(cmds[i])
#                     action = int(cmds[i + 1])
#                     action_dict[agent_id] = action
#                     i = i + 1
#                 i += 1
#                 r = env.render("human")
#                 time.sleep(0.1)
#     test_CN()



#class Independent_Navigation(Grid_Env):
#     """Each agent navigates to its own individual goal """
#     class Rewards():
#         step= -0.01
#         object_collision = -0.2
#         agent_collision = -0.4
#         goal_reached= 0.5
#         finish_episode = 1
#     class Global_Cooperative_Rewards():
#         """All agents share this reward """
#         step= -0.01
#         object_collision = -0.2
#         agent_collision = -0.4
#         goal_reached= 0.0
#         finish_episode = 1
#     def __init__(self, args):
#         #Make default env arguments if not exist in args
#         #initialize base Grid_Env
#         self.args = args
#         generator = random_obstacle_generator(args.map_shape, args.n_agents, obj_density = args.obj_density)
#         super(). __init__(generator, diagonal_actions = False, fully_obs = False, view_d = 2, agent_collisions = True)
#         self.rewards = self.Rewards()
#         #Goals:
#         assert len(self.goals) == len(self.agents)
#         goal_ids = np.random.choice(len(self.goals), len(self.goals), replace = False) #agents assigned to goals
#         for i,g in enumerate(self.goals):
#             g.goal_id = goal_ids[i]
#         self.goals = {g.goal_id : g for g in self.goals}

#         #Set reward function, obsevation space etc
#         self.rewards = self.Rewards()

#         observation_space = self.init_observation_space()
#         self.observation_space = [copy.deepcopy(observation_space) for _ in self.agents]
        
#     def reset(self, custom_env = None):
#         self.__init__(self.args)
#         #super().__init__(self.generator, self.diagonal_actions, self.fully_obs, self.view_d, self.agent_collisions)
#         # self.info = {
#         #     'total_steps': 0,
#         #     'terminate': False,
#         #     'total_rewards': 0,  
#         #     'total_agent_collisions': 0,
#         #     'total_obstacle_collisions': 0,
#         #     'agents_done': 0
#         # }

#         # self.step_count = 0
#         # obj_map = self.generator()
#         # self.x_len, self.y_len = obj_map.shape
#         # super().__init__(self.x_len, self.y_len)
        
#         # #Agents:
#         # self.agents, self.goals = self._populate_grid(obj_map)
#         # self.agents = {i:agent for i,agent in enumerate(self.agents)}

#         # #Goals:
#         # assert len(self.goals) == len(self.agents)
#         # goal_ids = np.random.choice(len(self.goals), len(self.goals), replace = False) #agents assigned to goals
#         # for i,g in enumerate(self.goals):
#         #     g.goal_id = goal_ids[i]
#         # self.goals = {g.goal_id : g for g in self.goals}
#        # self.grid_renderer = None

#         (obs, rewards, dones, collisions, info) = self.step({h:0 for h in self.agents.keys()})
#         return (obs, rewards, dones, collisions, info)

#     def get_rewards(self, collisions):
#         #This is the incorrect implementation
#         raise NotImplementedError
#         rewards = {}
#         isdone = {}

#         for handle, agent in self.agents.items():
#             for g in self.goals.values():
#                 if g.pos == agent.pos and g.goal_id == handle:
#                     rewards[handle] = self.rewards.goal_reached
#                     isdone[handle] = True
#                     break
#                 else:
#                     rewards[handle] = self.rewards.step
#                     isdone[handle] = False
    
#         if sum(isdone.values()) == len(self.agents):
#             for handle in self.agents.keys(): 
#                 rewards[handle] = self.rewards.finish_episode
        
#         #Collision rewards:
#         # for handle, agent in self.agents.items():
#         #     if collisions['obs_col']:
#         #         if collisions['obs_col'][handle]: rewards[handle] += self.r.object_collision   
#         #     elif collisions['agent_col']:
#         #         if collisions['agent_col'][handle]: rewards[handle] += self.r.agent_collision

#         if collisions['obs_col']:
#             for key, val in collisions['obs_col'].items():
#                 if val: rewards[key] += self.rewards.object_collision

#         if collisions['agent_col']:
#             for key, val in collisions['agent_col'].items():
#                 if val: rewards[key] += self.rewards.agent_collision

#         return isdone, rewards
    