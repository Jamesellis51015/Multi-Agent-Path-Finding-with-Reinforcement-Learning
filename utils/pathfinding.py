from queue import PriorityQueue
from queue import Queue
import copy
from sklearn.model_selection import ParameterGrid
import numpy as np
from tabulate import tabulate
from utils.mstar import Mstar
import time



class Heuristics():
    class Node():
        def __init__(self, pos, move_cost, prev_act, prev_pos):
            self.pos = pos
            self.move_cost = move_cost
            self.prev_act = prev_act
            self.prev_pos = prev_pos
        def __gt__(self, n2):
            return self.move_cost > n2.move_cost
        def __ge__(self, n2):
            return self.move_cost >= n2.move_cost
        def __lt__(self, n2):
            return self.move_cost < n2.move_cost
        def __le__(self, n2):
            return self.move_cost <= n2.move_cost
        def __eq__(self, n2):
            return self.move_cost == n2.move_cost
    
    # class vertex():
    #     def __init__(self, v, prev_v, cost):
    #         self.v = None
    #         self.move_cost = None
    #       #  self.g_cost = None
    #     #    self.nodes = None
    #         self.prev_vertex = None
    #         self.collision_set = None

    def __init__(self, grid, env):
        self.grid = grid
        self.env = env
        self.block_true_dist = env.view_d
        self.x_bound, self.y_bound = grid.shape
        self.directions = [(0,0), (0,1), (0,-1), (1,0), (-1,0)]

        #Treat these object types as obstacles
        self.obstacle_types = ["obstacle", "agent"]
        self.pos_act = {(0,1):2,
                        (1,0):3,
                        (0,-1):4,
                        (-1,0):1}
        self.cost = 1 #cost of transitioning to next node

        self.prev_pos_hldr = {i:None for i in self.env.agents.keys()}
        self.dijkstra_graphs = None

    def get_blocking_obs(self, agent_pos, goal_pos):
        '''If no path from agent to goal can be found,
            the locations of the objects in way of the shortest path
            from agent to goal is returned '''
        _, path = self.a_star_search(agent_pos, goal_pos, ignore=["agent"])
        obs_in_way = []
        if len(path) == 0:
            _, path = self.a_star_search(agent_pos, goal_pos, ignore=["agent", "obstacle"])
            path = path[1:-1]#remove start and end
            for p in path:
                obs_types = [ob.type for ob in self.grid[p[0], p[1]]]
                if "obstacle" in obs_types:
                    obs_in_way.append(p)
        return obs_in_way


    def init_joint_policy_graphs(self, start, end):
        '''Performs dijkstra search for each agent and stores the result
            in a dictionary. '''
        assert type(start) == list, Exception("start parameter has to be list")
        assert type(end) == list, Exception("end parameter has to be list")
        assert len(start) == len(end), Exception("start and end positions have to be of same length")

        self.dijkstra_graphs = {}
        for i,(p_start, p_end) in enumerate(zip(start, end)):
            assert type(p_start) == tuple and type(p_end) == tuple
            assert len(p_start) == 2 and len(p_end) == 2
            self.dijkstra_graphs[i] = self.dijkstra_search(p_end, p_start)

    def expand_position(self,agent_handle, position):
        '''Returns a list of possible next positions for an agent (ignoring other agents) '''
        assert not self.dijkstra_search is None
        assert type(position) == tuple and len(position) == 2
        assert agent_handle in self.dijkstra_graphs
        this_graph = self.dijkstra_graphs[agent_handle]
        next_postions = [self.add_tup(position, d, return_tuple=True) for d in self.directions]
        neighbours = [n_pos for n_pos in next_postions if n_pos in this_graph]
        return neighbours
    
    def get_next_joint_policy_position(self,agent_handle, position):
        '''Returns the shortest path next position for an agent'''
        assert not self.dijkstra_search is None
        assert type(position) == tuple and len(position) == 2
        assert agent_handle in self.dijkstra_graphs
        
        this_graph = self.dijkstra_graphs[agent_handle]

        assert position in this_graph

        next_postions = [self.add_tup(position, d, return_tuple=True) for d in self.directions]

        next_position_costs = {this_graph[n_pos].move_cost: n_pos for n_pos in next_postions if n_pos in this_graph}
        min_cost = min(next_position_costs.keys())
        min_cost_next_pos = next_position_costs[min_cost]
        return [min_cost_next_pos]

    def get_SIC(self, vertex):
        vertex = vertex.v
        assert type(vertex) == tuple
        assert len(vertex) == len(self.dijkstra_graphs)
        SIC = 0
        for i, pos in enumerate(vertex):
            SIC += self.dijkstra_graphs[i][pos].move_cost
        return SIC
            

    
    def mstar_search2(self, start, end):
        t_hldr1 = time.time()
        if self.dijkstra_graphs is None:
            print("Joint policy graphs not initialized. Initialzing now")
            self.init_joint_policy_graphs(start, end)
        else:
            print("Joint policy graphs already present...re-using graphs")

        t2 = time.time()
        print("Time taken for joint policy graphs: {}".format(t2 - t_hldr1))
        

        # this_graph = self.dijkstra_graphs[0]

        # table1 = []
        # table2 = []
        # table3 = []
        # for x in range(self.x_bound):
        #     row1 = []
        #     row2 = []
        #     row3 = []
        #     for y in range(self.y_bound):
        #         p = (x,y)
        #         row1.append(p)
        #         if p in this_graph:
        #             hldr = this_graph[p]
        #             row2.append(hldr.move_cost)
        #             row3.append(self.get_next_joint_policy_position(0, p))
        #         else:
        #             row2.append("x")
        #             row3.append("x")
        #     table1.append(row1)
        #     table2.append(row2)
        #     table3.append(row3)
        # print("positions: \n {} \n Cost:\n {}\n Optimal_next_pos:\n {}".format(tabulate(table1), tabulate(table2), tabulate(table3)))


        mstar = Mstar(start, end, self.expand_position, self.get_next_joint_policy_position, self.get_SIC)

        all_actions = mstar.search(start, end)
        print("Time taken for M star: {}".format(time.time() - t2))
        return all_actions




    def m_star_search(self, start_config, end_config, ignore = ["agent"]):

        def f_cost():
            return len(start_config)
        def h_cost(curr_pos):
            h_cost = 0
            for p_curr, p_end in zip(curr_pos, end_config):
                h_cost += self.abs_dist(p_curr, p_end)
            return h_cost


        class BackPtr():
            def __init__(self):
                self.hldr = {}
                self.pos_act = {(0,1):2,
                        (1,0):3,
                        (0,-1):4,
                        (-1,0):1,
                        (0,0): 0}
              #  self.prev_v = None
                self.last_v = None
            def add(self, vl, vk):
                self.hldr[vl] = vk
                self.last_v = vl

                # if curr != self.prev_v:
                #     self.hldr.append(curr)
                #     self.prev_v = curr

            def unpack(self):

                hldr = []
                hldr.append(self.last_v)
                this_v = self.last_v
                while this_v in self.hldr:
                    this_v = self.hldr[this_v]
                    hldr.append(this_v)
                
                # hldr = []
                # hldr.append(v)
                # if last_v in self.hldr:
                #     last_v = self.hldr[v]
                #     hldr.append(last_v)
                # while last_v in self.hldr: 
                #     last_v = self.hldr[v]
                #     hldr.append(last_v)
                #####################################
                #hldr = self.hldr
                hldr.reverse()
                actions = {i:[] for i in range(len(hldr[0]))}
                prev_v = hldr[0]
                for v in hldr[1:]:
                    for i, (p, p_v) in enumerate(zip(v, prev_v)):
                        p_diff = tuple(self.add_tup(p, self.mult_tup(p_v, -1)))
                        actions[i].append(self.pos_act[p_diff])
                    prev_v = v
                return actions

            def add_tup(self, a,b):
                assert len(a) == len(b)
                ans = []
                for ia,ib in zip(a,b):
                    ans.append(ia+ib)
                return ans

            def mult_tup(self, a, m):
                ans = []
                for ai in a:
                    ans.append(ai*m)
                return ans


        class BackTrack():
            def __init__(self):
                self.hldr = {}
                # self.pos_act = {(0,1):2,
                #         (1,0):3,
                #         (0,-1):4,
                #         (-1,0):1}
            def add(self, curr, past):
               # self.hldr[curr] = past
                if curr in self.hldr:
                    self.hldr[curr].append(past)
                else:
                    self.hldr[curr] = [past]
            def get(self, curr):
                if curr in self.hldr:
                    return self.hldr[curr]
                else:
                    return []
            


        class Collisions():
            def __init__(self):
                self.coll = {}
            def add(self, v):
                if not v in self.coll:
                    self.coll[v] = set()
                #Determines if collision and add index if coll
                col2 = self.is_colliding(v)
                self.union(v, col2)
            def get(self, v):
                if v in self.coll:
                    return self.coll[v]
                else:
                    return set()
            def union(self,v, other_set):
                if not v in self.coll:
                    self.coll[v] = set()
                self.coll[v] = self.coll[v].union(other_set)
            
            def is_colliding(self, v):
                raise Exception("This function has a bug, use function in mstar.py")
                for i, vi in enumerate(v):
                    for i2, vi2 in enumerate(v):
                        if i != i2:
                            if vi == vi2:
                                hldr = set()
                                hldr.add(i)
                                hldr.add(i2)

                                return hldr
                                #self.coll[v].add(i)
                                #self.coll[v].add(i2)
                return set()


        class Cost():
            def __init__(self):
                self.hldr = {}
            def add(self, v, c):
                self.hldr[v] = c
            def get(self, v):
                if v in self.hldr:
                    return self.hldr[v]
                else:
                    return 10000000
                
        def backprop(v, c_l, c, open, cost, back_set):
            if not c_l.issubset(c.get(v)):
                #c[v] = c[v].union(c_l) 
                c.union(v, c_l)
                if len(open.queue) != 0:
                    (_, all_open) = list(zip(*open.queue))
                else:
                    all_open = []
                #h = list(zip(*open.queue))
                #(_, all_open) = list(zip(*open.queue))
                if not v in all_open:
                    v_cost = cost.get(v) + f_cost()
                    priority = v_cost + h_cost(v)
                    open.put((priority, v))
                vm_back = back_set.get(v)
              #  if not vm_back is None:
                for vm in vm_back:
                    backprop(vm, c.get(v), c, open, cost, back_set)
        
        obstacle_types = copy.deepcopy(self.obstacle_types)
        for ig in ignore:
            if ig in self.obstacle_types:
                ind = obstacle_types.index(ig)
                del obstacle_types[ind]
        assert len(start_config) == len(end_config)

        graph = {i:self.dijkstra_search(end, start) for i ,(start, end) in enumerate(zip(start_config, end_config))}
        coll_set = Collisions()
        back_set = BackTrack()
        back_ptr = BackPtr()
        start_v = tuple(start_config)
        cost = Cost()
       #closed = {}
        open = PriorityQueue()
        cost.add(start_v, 0)
        open.put((0, start_v))
        while not open.empty():
            (_, curr_v) = open.get()
            if curr_v == end_config:
                #closed[curr_v] = curr_v
               # back_ptr.add(curr_v)
                return back_ptr.unpack()
            hldr2 = self._m_expand(curr_v, graph, coll = coll_set.get(curr_v))
            for v_n in hldr2:
                back_set.add(v_n, curr_v)
                coll_set.add(v_n)
                backprop(curr_v, coll_set.get(v_n), coll_set,open, cost, back_set)
                if len(coll_set.is_colliding(v_n)) == 0 and ( cost.get(curr_v) + f_cost() ) < cost.get(v_n):
                    v_n_cost = cost.get(curr_v) + f_cost()
                    cost.add(v_n, v_n_cost)
                    back_ptr.add(v_n, curr_v)
                    priority = v_n_cost + h_cost(v_n)
                    open.put((priority, v_n))
        return None




    def _m_expand(self, pos, graph, coll = None):
        '''Returns a list of tuples which is the expanded vertices '''
        pos_act = {(0,1):2,
                        (1,0):3,
                        (0,-1):4,
                        (-1,0):1,
                        (0,0): 0}
        ####
        n_pos = {}
        directions = [(0,0), (0,1), (0,-1), (1,0), (-1,0)]
        for i, p in enumerate(pos):
            this_graph = graph[i]
            n_pos_hldr = []
            for d in directions:
                new_pos = tuple(self.add_tup(p, d))

                p_diff = tuple(self.add_tup(new_pos, self.mult_tup(p, -1)))
                assert p_diff in pos_act.keys()

                if new_pos in this_graph:
                    n_pos_hldr.append(new_pos)
            if i in coll:
                #Get all posible next pos
                n_pos[i] = n_pos_hldr
            else:
                #get shortest path next pos
                node_costs = [this_graph[p2].move_cost for p2 in n_pos_hldr]
                min_cost_ind = np.argmin(node_costs)
                n_node = n_pos_hldr[min_cost_ind]
                n_pos[i] = [n_node]
        
        #Create all possible vertex combinations:
        combinations = ParameterGrid(n_pos)
        all_v = [tuple([c[i] for i in n_pos.keys()]) for c in combinations]
        
        #Temp test code
        
       
        hldr = all_v
        actions = {i:[] for i in range(len(hldr[0]))}
        prev_v = hldr[0]
        for v in hldr[1:]:
            for i, (p, p_v) in enumerate(zip(pos, v)):
                p_diff = tuple(self.add_tup(p, self.mult_tup(p_v, -1)))
                actions[i].append(pos_act[p_diff])
            prev_v = v
        return all_v




            
    
   # def _vertex_cost(self, v_current, v_end):




    def dijkstra_search(self, start_pos, end_pos, ignore = ["agent"]):
        '''
        NB: The cost of starts at 0 at the start position and continues thougough 
        all postions. In order to use the cost to search for the cheapest path,
        start, place end postion at start position.
        Searches the entire search space and returns a dictionary of the closed set '''
        obstacle_types = copy.deepcopy(self.obstacle_types)
        for ig in ignore:
            if ig in self.obstacle_types:
                ind = obstacle_types.index(ig)
                del obstacle_types[ind]
        open = Queue()
        closed = {}
        start_node = self.Node(start_pos, 0, None, None)
        path_found_flag = False
        open.put(start_node)
        while not open.empty():
            #(_, curr_node) = open.get()
            curr_node = open.get()
            if curr_node.pos == end_pos:
                closed[curr_node.pos] = curr_node
                path_found_flag = True
            next_nodes_pos = self._get_neigbours(curr_node.pos, obstacle_types)
            move_cost = curr_node.move_cost + 1
            for n in next_nodes_pos:
                (p,a) = n 
                p = tuple(p)
                if not p in closed.keys(): #or (closed[p]).move_cost > move_cost:
                    #priority = move_cost + heuristic(p, end_pos)
                    #open.put((priority, self.Node(p, move_cost, a, curr_node.pos)))
                    open.put(self.Node(p, move_cost, a, curr_node.pos))
            closed[curr_node.pos] = curr_node
        return closed

        


    def a_star_search(self,start_pos, goal_pos, ignore = ["agent"], pos_obstacle = None):
        '''Single agent, single goal, ignore object of type agent.
            Automatically ignore goals which are not this agent's goal '''
            
        obstacle_types = copy.deepcopy(self.obstacle_types)
        for ig in ignore:
            if ig in self.obstacle_types:
                ind = obstacle_types.index(ig)
                del obstacle_types[ind]
        
        heuristic = self.abs_dist
        closed = {}
        start_node = self.Node(start_pos, 0, None, None)
        open = PriorityQueue()
        open.put((0, start_node))
        while not open.empty():
            (_, curr_node) = open.get()
            if curr_node.pos in goal_pos:
                closed[curr_node.pos] = curr_node
                break
            next_nodes_pos = self._get_neigbours(curr_node.pos, obstacle_types, pos_obstacle)
            move_cost = curr_node.move_cost + 1
            for n in next_nodes_pos:
                (p,a) = n 
                p = tuple(p)
                if not p in closed.keys() or (closed[p]).move_cost > move_cost:
                    priority = move_cost + heuristic(p, goal_pos)
                    open.put((priority, self.Node(p, move_cost, a, curr_node.pos)))
            closed[curr_node.pos] = curr_node

        action_path = []
        node_path = []
        if goal_pos in closed.keys():
            n = closed[goal_pos]
            node_path.append(n.pos)
            while not n.prev_act is None:
                action_path.append(n.prev_act)
                n = closed[n.prev_pos]
                node_path.append(n.pos)
            action_path.reverse()
        return action_path, node_path
            


    def _get_neigbours(self, position, obstacle_types, pos_obstacle = None):
        '''pos_obstacle =  a postion to treat as obstacle '''
        assert self.assert_in_bounds(position)

        n_pos = []
        for n, a in self.pos_act.items():
            pos = self.add_tup(position, n)
            if self.assert_in_bounds(pos):
                obj = self.grid[pos[0], pos[1]]
                obj_types = [o.type for o in obj]

                if not any([obs in obj_types for obs in obstacle_types]) and pos_obstacle != (pos[0], pos[1]):
                    n_pos.append((pos, a))
        return n_pos


    def assert_in_bounds(self, pos):
        (x, y) = pos
        if (x>= 0 and x<self.x_bound and y>=0 and y<self.y_bound):
            return True
        else:
            return False
    
    def abs_dist(self, curr_pos, goal_pos):
        (dx, dy) = self.add_tup(curr_pos, self.mult_tup(goal_pos, -1))
        path_len = abs(dx) + abs(dy)
        return path_len
    def abs_dist_comp(self,curr_pos, goal_pos):
        (dx, dy) = self.add_tup(curr_pos, self.mult_tup(goal_pos, -1))
        return (abs(dx),abs(dy))



    def add_tup(self, a,b, return_tuple = False):
        assert len(a) == len(b)
        ans = []
        for ia,ib in zip(a,b):
            ans.append(ia+ib)
        if return_tuple:
            ans = tuple(ans)
        return ans
    
    def mult_tup(self, a, m):
        ans = []
        for ai in a:
            ans.append(ai*m)
        return ans

    def get_valid_actions(self, agent_id):
        pos = self.env.agents[agent_id].pos
        valid_neighbours = self._get_neigbours(pos, ["obstacle"])
        states, act = zip(*valid_neighbours)
        act = list(act)
        states = [tuple(s) for s in states]
        if self.prev_pos_hldr[agent_id] in states:
            ind = states.index(self.prev_pos_hldr[agent_id])
            del act[ind]
        self.prev_pos_hldr[agent_id] = pos
        act.append(0)
        return act


    def is_blocking(self, agent_id):
        '''Is a particular agent blocking any other agent '''
        pos = self.env.agents[agent_id].pos
        for key, val in self.env.agents.items():
            if key == agent_id:
                continue
            goal_pos = self.env.goals[agent_id].pos
            (dx, dy) = self.abs_dist_comp(pos, val.pos)
            if dx <= self.block_true_dist and dy <= self.block_true_dist:
                act_path1, node_path1 = self.a_star_search(val.pos, goal_pos)
                act_path2, node_path2 = self.a_star_search(val.pos, goal_pos, pos_obstacle=pos)
                # if len(act_path2) != len(act_path1):
                #     print("here")
                if abs(len(act_path2) - len(act_path1)) >= self.block_true_dist or len(act_path2) == 0:
                    return True
        return False




