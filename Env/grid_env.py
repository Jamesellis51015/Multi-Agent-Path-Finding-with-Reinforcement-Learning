
# Some code/ideas from:

#@misc{gym_minigrid,
#  author = {Chevalier-Boisvert, Maxime and Willems, Lucas and Pal, Suman},
#  title = {Minimalistic Gridworld Environment for OpenAI Gym},
#  year = {2018},
#  publisher = {GitHub},
#  journal = {GitHub repository},
#  howpublished = {\url{https://github.com/maximecb/gym-minigrid}},
#}

# Map of object type to integers as well as everything relating 
# to rendering From: https://github.com/maximecb/gym-minigrid/blob/master/gym_minigrid/minigrid.py


import numpy as np
from numpy import genfromtxt
from gym import spaces
import math
import enum
import copy
from tabulate import tabulate

CELL_PIXELS = 32


OBJECT_TO_IDX = {
    'empty'     : '0',
    'obstacle'  : '1',
    'agent'     : '2',
    'goal'      : '3',
}

# Map of color names to RGB values
COLORS = {
    'red'   : np.array([255, 0, 0]),
    'green' : np.array([0, 255, 0]),
    'blue'  : np.array([0, 0, 255]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'grey'  : np.array([100, 100, 100]),
      0: np.array([255, 0, 0]),
      1: np.array([0, 0, 255]),
      2: np.array([255, 255, 0]),
      3: np.array([51, 255, 255])
}
# AGENT_COLOURS = {
#   0: np.array([255, 0, 0]),
#   1: np.array([0, 0, 255]),
#   2: np.array([255, 255, 0]),
#   3: np.array([51, 255, 255]),
# }

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

class ACTIONS(enum.Enum):
    stay = 0
    up = 1
    right = 2
    down = 3
    left = 4
    left_diag_up = 5
    right_diag_up = 6
    right_diag_down = 7
    left_diag_down = 8


class Empty():
    def __init__(self):
        self.type = 'empty'
        self.pos = (None, None)
        self.color = np.array([0, 0, 0]),
    def can_overlap(self):
        return True
    def render(self):
        print("render not implemented")
    def _set_color(self, r, custom = None):
        """Set the color of this object as the active drawing color"""
        if custom is None:
            c = COLORS[self.color]
        else: 
            c= custom
        r.setLineColor(c[0], c[1], c[2])
        r.setColor(c[0], c[1], c[2])

class Obstacle(Empty):
    def __init__(self, colour = 'grey'):
        super(Obstacle, self).__init__()
        self.type = 'obstacle'
        self.color = colour
    def can_overlap(self):
        return False
    def render(self, r):
        self._set_color(r)
        r.drawPolygon([
            (0          , CELL_PIXELS),
            (CELL_PIXELS, CELL_PIXELS),
            (CELL_PIXELS,           0),
            (0          ,           0)
        ])
        

class Goal(Empty):
    def __init__(self, x,y, colour = 'green'):
        super().__init__()
        self.type = 'goal'
        self.color = colour
        self.goal_id = None #The agents assigned to this goal
        self.pos = (x,y)
    def can_overlap(self):
        return True
    def render(self, r):
        custom = None
        if self.goal_id != None:
            if self.goal_id < 4:
                self.color = self.goal_id
            elif self.goal_id > 3:
                i = self.goal_id
                custom = np.array([255/(i-1), 255/(i-2), 255/(i-3)])
            else:
                raise Exception("ID not in range")
        self._set_color(r, custom=custom)
        r.drawPolygon([
            (0          , CELL_PIXELS),
            (CELL_PIXELS, CELL_PIXELS),
            (CELL_PIXELS,           0),
            (0          ,           0)
        ])

class Agent(Empty):
    def __init__(self, x_pos, y_pos, colour = 'blue'):
        super().__init__()
        self.type = 'agent'
        self.pos = (x_pos, y_pos)
        self.id = None
        self.color = colour
    def can_overlap(self):
        return True
    def render(self, r):
        custom = None
        if self.id != None:
            if self.id < 4:
                self.color = self.id
            elif self.id > 3:
                i = self.id
                custom = np.array([255/(i-1), 255/(i-2), 255/(i-3)])
            else:
                raise Exception("ID not in range")
        self._set_color(r, custom=custom)
        r.drawCircle(CELL_PIXELS * 0.5, CELL_PIXELS * 0.5, (CELL_PIXELS//2)*0.7)
       # print(self.id)
        # if self.id != None:
        #     i = str(self.id)
        #     self.color = 'red'
        #     self._set_color(r)
        #     r.drawText(CELL_PIXELS * 0.4, CELL_PIXELS * 0.7, i)
        #     self.color = 'blue'

class Grid():
    def __init__(self, x_len, y_len):
        self.x_len = x_len #number of rows
        self.y_len = y_len #number of columns
        #self.grid = [[None] for _ in range(self.x_len * self.y_len)] 
        self.grid = np.empty((x_len, y_len), dtype=object)

       # print("Env  np   {};    ".format(np.random.normal()))
        
    def get(self, x, y):
        return self.grid[x,y]#self.grid[y*self.y_len + x]
        
    def render_grid(self, r, tile_size):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        assert r.width == self.y_len * tile_size
        assert r.height == self.x_len * tile_size

        # Total grid size at native scale
        widthPx = self.y_len * CELL_PIXELS
        heightPx = self.x_len * CELL_PIXELS

        r.push()

        # Internally, we draw at the "large" full-grid resolution, but we
        # use the renderer to scale back to the desired size
        r.scale(tile_size / CELL_PIXELS, tile_size / CELL_PIXELS)

        # Draw the background of the in-world cells black
        r.fillRect(
            0,
            0,
            widthPx,
            heightPx,
            250, 250, 250
        )

        # Draw grid lines
        #r.setLineColor(100, 100, 100)
        r.setLineColor(0, 0, 0)
        for rowIdx in range(0, self.x_len):
            y = CELL_PIXELS * rowIdx
            r.drawLine(0, y, widthPx, y)
        for colIdx in range(0, self.y_len):
            x = CELL_PIXELS * colIdx
            r.drawLine(x, 0, x, heightPx)



        # Render the grid
        for j in range(0, self.x_len):
            for i in range(0, self.y_len):
                #print(self.get(i,j))
                cells = self.get(j, i)
                cell_types = [c.type for c in cells]
                if 'goal' in cell_types and 'agent' in cell_types:
                    r.push()
                    r.translate(i * CELL_PIXELS, j * CELL_PIXELS)
                    #(Goal(i,j)).render(r)
                    #(Agent(i,j)).render(r)
                    agnt_ind = cell_types.index('agent')
                    goal_ind = cell_types.index('goal')
                    (cells[goal_ind]).render(r)
                    (cells[agnt_ind]).render(r)
                    r.pop()
                else:
                    cell = cells[-1]
                    if cell.type == 'empty':
                        continue
                    r.push()
                    r.translate(i * CELL_PIXELS, j * CELL_PIXELS)
                    cell.render(r)
                    r.pop()

        r.pop()



class Grid_Env(Grid):
    # class Global_Cooperative_Rewards():
    #     """All agents share this reward """
    #     step= -0.01
    #     object_collision = -0.2
    #     agent_collision = -0.4
    #     goal_reached= 0.0
    #     finish_episode = 1
    def __init__(self, generator, diagonal_actions = False, fully_obs = False, view_d = 2, agent_collisions = True):
        #Get env, obstacles and agents from hand made csv file
        obj_map = generator()#genfromtxt(csv_file, delimiter=',')
        self.name = "None"
        self.agent_collisions = agent_collisions
        
        self.generator =  generator
        self.x_len, self.y_len = obj_map.shape
        if self.x_len != self.y_len:
            raise Exception("Currently only support square map shapes")
        super().__init__(self.x_len, self.y_len)
        self.max_step = int(math.floor(self.x_len * self.y_len*1.5))
        self.step_count = 0
        #Agents:
        self.agents, self.goals = self._populate_grid(obj_map)
        self.agents = {i:agent for i,agent in enumerate(self.agents)}
        for i, agnt in self.agents.items():
            agnt.id = i 
        self.n_agents = len(self.agents)
        #self.reward_function = 'navigation'
        #self.rewards = self.Rewards()
        self.rewards = None
        self.global_cooperative_rewards = self.Global_Cooperative_Rewards()

        #Goals:
        # assert len(self.goals) == len(self.agents)
        # goal_ids = np.random.choice(len(self.goals), len(self.goals), replace = False) #agents assigned to goals
        # for i,g in enumerate(self.goals):
        #     g.goal_id = goal_ids[i]
        # self.goals = {g.goal_id : g for g in self.goals} 
        #self.goals = None

        self.fully_obs = fully_obs
        self.view_d = view_d
        self.inc_obstacles = True
        self.inc_other_agents = True
        self.inc_other_goals = True
        self.inc_own_goals = True
        self.inc_direction_vector = False
        self.inc_path_grid = False #show 
        self.heur = None

        self.observation_space = None #Required to initialize manually
        
        self.grid_renderer = None
        if diagonal_actions:
            self.action_space = [spaces.Discrete(9) for _ in self.agents]
        else:
            self.action_space =[spaces.Discrete(5) for _ in self.agents]
        #Episode info
        self.info = {
            'total_steps': 0,
            'terminate': False,
            'total_rewards': 0,  #Total reward is averaged over the number of agents
            'total_agent_collisions': 0,
            'total_obstacle_collisions': 0,
            'agent_dones': 0,
            'all_agents_on_goal': 0,
            'total_avg_agent_r':0,
            'total_ep_global_r':0,
            'terminal_observation': None,
            'terminal_render': None
        }
    
    def init_observation_space(self):
        # Obstacles;Other agents; own goal; other goals; own position
        channels = 5
        if self.inc_obstacles == False:
            channels -= 1
        if self.inc_other_agents == False:
            channels -= 1
        if self.inc_other_goals == False:
            channels -= 1
        if self.inc_own_goals == False:
            channels -= 1

        if self.fully_obs:
            view_shape = (self.x_len, self.y_len)
        else:
            channels -= 1
            view_shape = (self.view_d*2 + 1, self.view_d*2 + 1)
        map_obs_shape = (channels, view_shape[1], view_shape[0])
        map_obs_space = spaces.Box(low=0, high=1, shape= map_obs_shape, dtype=int)

        assert not (self.inc_direction_vector == True and self.inc_path_grid==True)
            
        if self.inc_direction_vector:
            vector_obs_space = spaces.Box(low=0, high=1, shape= (2,), dtype=float)
        elif self.inc_path_grid:
            flat_input = view_shape[1] * view_shape[0]
            vector_obs_space = spaces.Box(low=0, high=1, shape= (flat_input,), dtype=float)

        if self.inc_direction_vector or self.inc_path_grid:
            return (map_obs_space, vector_obs_space)
        else:
            return map_obs_space

    def _populate_grid(self, obj_map):
        """Adds objects to grid list from numpy array corresponding to positions
        on grid.  
        Decoding conventions:
        'empty'     : '0',     
        'obstacle'  : '1',
        # agent id and goal id not specified:   |    Agent id and goals specified:
        'agent'     : '2',                      |    'a{n}'  where n is the id
        'goal'      : '3',                      |    'g{n}'
        """
        agents = []
        goals = []
        custom_g_ids = []
        custom_a_ids = []
        for x in range(self.x_len):
            for y in range(self.y_len):
                if obj_map[x,y] == OBJECT_TO_IDX['empty']:
                    self.grid[x,y] = [Empty()]
                elif obj_map[x,y] == OBJECT_TO_IDX['obstacle']:
                    #self.grid[y*self.y_len + x] = [Obstacle()]
                    self.grid[x,y] = [Obstacle()]
                elif obj_map[x,y] == OBJECT_TO_IDX['agent']:
                    obj = Agent(x,y)
                    agents.append(obj)
                    #self.grid[y*self.y_len + x] = [obj]
                    self.grid[x,y] = [obj]
                elif obj_map[x,y] == OBJECT_TO_IDX['goal']:
                    obj = Goal(x,y)
                   # self.grid[y*self.y_len + x] = [obj]
                    self.grid[x,y] = [obj]
                    goals.append(obj)
                elif obj_map[x,y][0] == 'a':
                    str_id = obj_map[x,y][1:]
                    int_id = int(str_id)
                    obj = Agent(x,y)
                    obj.id = int_id
                    agents.append(obj)
                    custom_a_ids.append(int_id)
                    self.grid[x,y] = [obj]
                elif obj_map[x,y][0] == 'g':
                    str_id = obj_map[x,y][1:]
                    int_id = int(str_id)
                    obj = Goal(x,y)
                    obj.goal_id = int_id
                    goals.append(obj)
                    custom_g_ids.append(int_id)
                    self.grid[x,y] = [obj]
                else:
                    print("No matching object ID")
        if len(custom_a_ids) != 0 and len(custom_g_ids)!=0:
            assert len(custom_a_ids)== len(custom_g_ids), "Number of custom goal ids not equal to agent ids"
            assert set(custom_a_ids) == set(custom_g_ids), "Custom goal id's and agent ids does not match up."
        return agents, goals
    

    def clear_paths(self, any_goal = True):
        raise NotImplementedError

    def _remove_object(self, position, type):
        assert type in ['empty', 'agent', 'obstacle', 'goal']
        all_obj_types = [ob.type for ob in self.get(position[0], position[1])]
        ind = all_obj_types.index(type)
        del self.grid[position[0], position[1]][ind]
        self.grid[position[0], position[1]].append(Empty())


    

    def render(self, mode='rgb_array', close=False, tile_size=CELL_PIXELS):
        """
        Render the entire grid 
        """

        if close:
            if self.grid_renderer:
                self.grid_renderer.close()
            return

        if self.grid_renderer is None or self.grid_renderer.window is None or (self.grid_renderer.width != self.y_len * tile_size):
            from .rendering import Renderer
            self.grid_renderer = Renderer(
                self.y_len * tile_size,
                self.x_len * tile_size,
                True if mode == 'human' else False
            )

        r = self.grid_renderer

        r.beginFrame()

        # Render the whole grid
        self.render_grid(r, tile_size)

        r.endFrame()
        
        if mode == 'rgb_array':
            return r.getArray()
        elif mode == 'pixmap':
            return r.getPixmap()
        return r
    
    def _next_position(self, agent_handle, action):
            assert agent_handle in range(len(self.agents))
            if type(action) == ACTIONS:
                action = action.value
           # print("act and space {}  {}".format(action, self.action_space[0]))
            assert action in range(self.action_space[0].n)

            (curr_x, curr_y) = self.agents[agent_handle].pos
            next_x, next_y = curr_x, curr_y
            
            if action == ACTIONS.stay.value:
                pass
            elif action == ACTIONS.right.value:
                #next_x += 1
                next_y += 1
            elif action == ACTIONS.down.value:
                next_x += 1
            elif action == ACTIONS.left.value:
                next_y -= 1
            elif action == ACTIONS.up.value:
                next_x -= 1 
            elif action == ACTIONS.left_diag_up.value:
                next_y -= 1
                next_x += 1
            elif action == ACTIONS.right_diag_up.value:
                next_y += 1
                next_x += 1
            elif action ==  ACTIONS.right_diag_down.value:
                next_y += 1
                next_x -= 1
            elif action == ACTIONS.left_diag_down.value:
                next_y -= 1
                next_x -= 1
            #If next position is outside of grid
            if next_x not in range(self.y_len) or next_y not in range(self.x_len):
                next_x, next_y = curr_x, curr_y
            return (next_x, next_y)
        
    def step(self, action_handles):
        """Input: A dictionary of handle-action key value pairs 
           Output: 
        """

        if self.info["terminate"]:
            return self.reset()

        self.step_count += 1
        agent_conflicts_global = {}
        obs_collisions_global = {}

        #Complete missing actions
        for handle in self.agents.keys():
            if handle not in action_handles.keys():
                action_handles[handle] = ACTIONS.stay.value
            

        def get_conflics(action_dict):
            """Returns new collision-free actions and agent handles for agents which are involved in collisions
            Has to be recursive since taking actions to avoid conflics for certain agents may involve causing 
            collisions for other agents """
            agnt_next_pos = {handle: self._next_position(handle,action) for handle, action in action_dict.items()}

            agnt_curr_pos = {i:agnt.pos for i,agnt in self.agents.items()}

            agent_conflicts_local = {}
            obs_collisions_local = {}
            new_handle_acion_dict = copy.deepcopy(action_dict)
            
            for handle in action_dict.keys():
                agent_next_positions_cpy = copy.deepcopy(agnt_next_pos)
                del agent_next_positions_cpy[handle]
                agnt_curr_pos_cpy = copy.deepcopy(agnt_curr_pos)
                del agnt_curr_pos_cpy[handle]

                agnt_conflict_flag = False
                #Taking into account other agent actions:
                if self.agent_collisions:
                    if agnt_next_pos[handle] in agent_next_positions_cpy.values():
                        agnt_conflict_flag = True
                        agent_conflicts_local[handle] = True
                        agent_conflicts_global[handle] = True
                        #conflict_handles.append(handle)
                        new_handle_acion_dict[handle] = ACTIONS.stay
                    # #Prevent agents from moving past each other:
                    # if agnt_next_pos[handle] in agnt_curr_pos_cpy.values():
                    #     npos = agnt_next_pos[handle]
                    #     this_agent = handle
                    #     other_agent = [k for (k,v) in agnt_curr_pos_cpy.items() if v == npos]
                    #     other_agent = other_agent[0]
                    #     hlsr = [action_dict[this_agent] == ACTIONS.right.value and action_dict[other_agent] == ACTIONS.left.value]
                    #     if action_dict[this_agent] == ACTIONS.left and action_dict[other_agent] == ACTIONS.right.value \
                    #         or action_dict[this_agent] == ACTIONS.right.value and action_dict[other_agent] == ACTIONS.left.value \
                    #         or action_dict[this_agent] == ACTIONS.up.value and action_dict[other_agent] == ACTIONS.down.value \
                    #         or action_dict[this_agent] == ACTIONS.down.value and action_dict[other_agent] == ACTIONS.up.value:
                    #         agnt_conflict_flag = True
                    #         agent_conflicts_local[handle] = True
                    #         agent_conflicts_global[handle] = True
                    #         #conflict_handles.append(handle)
                    #         new_handle_acion_dict[handle] = ACTIONS.stay


                #Taking into account obstacles and grid edges
                (x_p, y_p) = self.agents[handle].pos
                (x,y) = agnt_next_pos[handle]
                if x not in range(self.y_len) or y not in range(self.x_len):
                    x, y = x_p, y_p

                for cell_obj in self.grid[x,y]: #self.grid[y*self.y_len + x][-1]
                    if agnt_conflict_flag == False and cell_obj.type != 'empty' and not cell_obj is self.agents[handle]:
                        if cell_obj.can_overlap() == False:
                            obs_collisions_local[handle] = True
                            obs_collisions_global[handle] = True
                            #obs_collision_handles.append(handle)
                            new_handle_acion_dict[handle] = ACTIONS.stay

            if True in agent_conflicts_local.values() or True in obs_collisions_local.values():
                return get_conflics(new_handle_acion_dict)
            else:
                return new_handle_acion_dict

        
        
        #Get no conflict actions for all agents
        new_actions = get_conflics(action_handles)
        
        #Execute actions
        for handle, action in new_actions.items():
            (x_n, y_n) = self._next_position(handle, action)
            (x,y) = self.agents[handle].pos 
            
            index =None
            for i,obj in enumerate(self.grid[x,y]):
                if obj is self.agents[handle]: index = i
            if index == None: raise Exception

            del self.grid[x,y][index]

            if len(self.grid[x,y]) ==0:
                self.grid[x,y].append(Empty())


            # if len(self.grid[x,y]) > 1:
            #     del self.grid[x,y][-1]
            # else:
            #     self.grid[x,y][-1] = None

            # #self.grid[y_n*self.y_len + x_n][-1] = self.agents[handle]
            self.grid[x_n, y_n].append(self.agents[handle])
            self.agents[handle].pos = (x_n, y_n)

        collisions = {
            'agent_col': agent_conflicts_global,
            'obs_col': obs_collisions_global
        }

                
        dones,rewards = self.get_rewards(collisions)
        global_r = self.get_global_cooperative_rewards(collisions)
        global_r = global_r[0]
        obs = self._get_observations(self.view_d, self.fully_obs)
        
        #Updating info:
        if all([done for done in dones.values()]) or self.step_count > self.max_step:
            self.info["terminate"] = True
            self.info["terminal_observation"] = obs
            self.info["terminal_render"] = self.render()

        self.info["total_steps"] = self.step_count

        all_rewards = [r for r in rewards.values()]
        #self.info['total_rewards'] += sum(all_rewards)/len(all_rewards)
        #
        # for r in all_rewards:
        #     self.info["total_avg_agent_r"] += r
        self.info["total_avg_agent_r"] += sum(all_rewards) / len(all_rewards)
        #self.info["total_avg_agent_r"] /= len(all_rewards)
        self.info["total_ep_global_r"] += global_r 

        for collision in collisions["agent_col"].values():
            if collision: self.info["total_agent_collisions"] += 1
        for collision in collisions["obs_col"].values():
            if collision: self.info["total_obstacle_collisions"] += 1
        self.info["agent_dones"] = sum([d for d in dones.values()]) / len([d for d in dones.values()])
        
        if all([d for d in dones.values()]):
            self.info["all_agents_on_goal"] = 1
        else:
            self.info["all_agents_on_goal"] = 0


        #return (obs, rewards, dones, collisions, self.info)
        self.info["step_collisions"] = collisions
        return (obs, rewards, dones, copy.deepcopy(self.info))
    
    def reset(self, custom_env = None):
        raise NotImplementedError
  
    def get_rewards(self, collisions):
        raise NotImplementedError

    def get_global_cooperative_rewards(self, collisions):
        raise NotImplementedError

    def summary(self):
        '''Summary environment settings '''
        rows = []
        hldr = ["Name:",self.name, "None", "Size:", str((self.x_len,self.y_len)), "None", "Agents:", str(len(self.agents)), "None", "Max Steps:", str(self.max_step)]
        rows.append(hldr)
        
        if self.rewards == None:
            rows.append(["Rewards:", "None"])
        else:
            rows.append(["R_Step:", self.rewards.step, None, "R_Object_Collsion:", self.rewards.object_collision, None, \
            "R_Agent_Collision:", self.rewards.agent_collision, None, "R_Goal_Reached", self.rewards.goal_reached])

        rows.append(["Action space:", str(self.action_space), None, "Observation space", str(self.observation_space)])

        rows.append(["Observations:", None, "Obstacles:", self.inc_obstacles, None, "Other_Agents:", self.inc_other_agents, None, \
        "Other_Goals:", self.inc_other_goals, None, "Own_Goals", self.inc_own_goals, None, "Direction_Vector:", self.inc_direction_vector])

        return tabulate(rows)

    #Helper functions
    def _get_dir_vec(self,agent, goal):
        '''A direction normalized direction vector
            for large env where goals are outside of field of view '''
        #v = {}
        #for handle, agent in self.agents.items():
        (x1,y1) = agent.pos
        (x2, y2) = goal.pos #self.goals[handle].pos
        dx = x2 - x1
        dy = y2 - y1

        magnitude = math.sqrt(dx**2 + dy**2) / math.sqrt(self.x_len**2 + self.y_len**2)
        #if dx < 0.0000001: dx = 0.0000001
        angle = math.atan2(dy,dx)/(math.pi)
        return np.array([magnitude, angle])       

    def _get_path_grid(self, agent):
        raise NotImplementedError

    def _is_cell_type(self, view, cell = "obstacle", own_goal_only = True, agent_id = None):
        # view_shape = view.shape
        one_hot_view = np.zeros(view.shape)
        for x in range(view.shape[0]):
            for y in range(view.shape[1]):
                if all([obj.type == 'empty' for obj in view[x,y]]): continue
                goal = [obj for obj in view[x,y] if obj.type == 'goal']
                if cell == 'goal' and len(goal)!= 0:
                    goal = goal[-1]
                    if own_goal_only:
                        if goal.goal_id == agent_id: one_hot_view[x,y] = 1
                    else:
                        if goal.goal_id != agent_id: one_hot_view[x,y] = 1
                else:
                    if cell in [obj.type for obj in view[x,y]]: one_hot_view[x,y] = 1                    
        return one_hot_view

    def _get_view(self, x_pos, y_pos, d_view):
        view_dim =  2*d_view + 1
        view = np.empty((view_dim, view_dim), dtype = object)
        
        for x_cursor in range(-d_view, d_view + 1):
            for y_cursor in range(-d_view, d_view + 1):
                x = x_pos + x_cursor
                y = y_pos + y_cursor
                
                if x >= self.x_len or x < 0 or y>= self.y_len or y < 0:
                    view[d_view + x_cursor,d_view + y_cursor] = [Obstacle()]
                else:
                    view[d_view + x_cursor,d_view + y_cursor] = self.get(x,y)
        #view[d_view, d_view] = [Empty()] #Make own position None
        return view

    def _get_observations(self, d_view, fully_obs = False):
        """View is a square of size (view_d*2 + 1) x  (view_d*2 + 1)"""
        observations = {}
        
        for handle, agent in self.agents.items():
            if fully_obs:
                # top_obj_grid = np.empty(self.grid.shape, dtype = object)
                # for x in range(self.grid.shape[0]):
                #     for y in range(self.grid.shape[1]):
                #         top_obj_grid[x,y] = self.grid[x,y][-1]
                one_hot_obs = []
                if self.inc_obstacles: one_hot_obs.append(self._is_cell_type(self.grid, cell='obstacle'))
                if self.inc_other_agents:
                    hldr = self._is_cell_type(self.grid, cell='agent')
                    hldr[agent.pos[0], agent.pos[1]] = 0 #Exclude own position from other agent channel
                    one_hot_obs.append(hldr)
                if self.inc_own_goals: one_hot_obs.append(self._is_cell_type(self.grid, cell='goal', own_goal_only=True, agent_id=handle))
                if self.inc_other_goals: one_hot_obs.append(self._is_cell_type(self.grid, cell='goal', own_goal_only=False, agent_id=handle))
                
                #one_hot_obs = [is_cell_type(top_obj_grid, cell = obj_type) for obj_type in grid_obj_types.keys()]
                #one_hot_obs[1][agent.pos[0], agent.pos[1]] = 0 #Exclude own position from other agent channel
                
                one_hot_obs.append(np.zeros(self.grid.shape))
                one_hot_obs[-1][agent.pos[0], agent.pos[1]] = 1 #Include a channel for agents own position.
                #Observation channels: Obstacles;Other agents; own goal; other goals; own position
                if self.inc_direction_vector:
                    observations[handle] = (np.stack(one_hot_obs, 0), self._get_dir_vec(agent, self.goals[handle]))
                else:
                    observations[handle] = np.stack(one_hot_obs, 0)
            else:
                v = self._get_view(agent.pos[0], agent.pos[1], d_view)
                one_hot_observations = []
                #one_hot_observations = [is_cell_type(v, obj_type) for obj_type in grid_obj_types.keys()]
                if self.inc_obstacles: one_hot_observations.append(self._is_cell_type(v, cell='obstacle'))
                #remove self from observation
                hldr = self._is_cell_type(v, cell='agent')
                hldr[self.view_d, self.view_d] = 0.0
                if self.inc_other_agents: one_hot_observations.append(hldr)
                #if self.inc_other_agents: one_hot_observations.append(self._is_cell_type(v, cell='agent'))
                if self.inc_own_goals: one_hot_observations.append(self._is_cell_type(v, cell='goal', own_goal_only=True, agent_id=handle))
                if self.inc_other_goals: one_hot_observations.append(self._is_cell_type(v, cell='goal', own_goal_only=False, agent_id=handle))
                #observations[handle] = [np.stack(one_hot_observations, 0), get_dir_vec(agent, self.goals[handle])]
                if self.inc_direction_vector:
                    observations[handle] = (np.stack(one_hot_observations, 0), self._get_dir_vec(agent, self.goals[handle]))
                elif self.inc_path_grid:
                    observations[handle] = (np.stack(one_hot_observations, 0), self._get_path_grid(agent))
                else:
                    observations[handle] = np.stack(one_hot_observations, 0)
                #One hot obsevations contains one-hot representation of: obstacels, other agents, own goals, other agent goals                 
        return observations 






            
    
        
        
        
        













    
