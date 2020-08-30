from queue import PriorityQueue
from queue import Queue
import copy
from sklearn.model_selection import ParameterGrid
import numpy as np
from tabulate import tabulate
from utils.mstar import Mstar
from utils.mstar2 import Mstar_OD
import time
import heapq


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
                        (-1,0):1,
                        (0,0):0}
        self.cost = 1 #cost of transitioning to next node

        self.prev_pos_hldr = {i:None for i in self.env.agents.keys()}
        self.dijkstra_graphs = None

    def get_blocking_obs(self, agent_pos, goal_pos):
        '''If no path from agent to goal can be found,
            the locations of the objects in way of the shortest path
            from agent to goal is returned '''
        _, path = self.a_star_search2(agent_pos, goal_pos, ignore=["agent"])
        obs_in_way = []
        if len(path) == 0:
            _, path = self.a_star_search2(agent_pos, goal_pos, ignore=["agent", "obstacle"])
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
           # self.dijkstra_graphs[i] = self.dijkstra_search(p_end, p_start)
            self.dijkstra_graphs[i] = self.BFS(p_end, p_start)

    def expand_position(self,agent_handle, position):
        '''Returns a list of possible next positions for an agent (ignoring other agents) '''
        assert not self.dijkstra_search is None
        assert type(position) == tuple and len(position) == 2
        assert agent_handle in self.dijkstra_graphs
        this_graph = self.dijkstra_graphs[agent_handle]
        next_postions = [self.add_tup(position, d, return_tuple=True) for d in self.directions]
        neighbours = [n_pos for n_pos in next_postions if n_pos in this_graph]
        return neighbours
    
    def get_next_joint_policy_position(self,agent_handle, position, goal_pos=None):
        '''Returns the shortest path next position for an agent'''
        assert not self.dijkstra_search is None
        assert type(position) == tuple and len(position) == 2
        assert agent_handle in self.dijkstra_graphs
        
        this_graph = self.dijkstra_graphs[agent_handle]

        assert position in this_graph

        next_postions = [self.add_tup(position, d, return_tuple=True) for d in self.directions]

        

        next_position_costs = {this_graph[n_pos].move_cost: n_pos for n_pos in next_postions if n_pos in this_graph}
        # next_position_costs = dict()
        # next_position_pos = dict()
        # for i,n_pos in enumerate(next_postions):
        #     if n_pos in this_graph:
        #         next_position_costs[i] = this_graph[n_pos].move_cost
        #         next_position_pos[i] = n_pos

        # next_position_costs = {this_graph[n_pos].move_cost: n_pos for n_pos in next_postions if n_pos in this_graph}
        # next_position_costs2 = dict()
        # for pos in next_postions:
        #     if pos in this_graph:
        #         next_position_costs2[this_graph[pos].move_cost] = pos

        # for i, k in enumerate(next_position_costs.keys()):
        #     if k in list(next_position_costs.keys())[i:]:
        #         pass
        # for p in next_postions:
        #     if p in this_graph:
        #         pass
        min_cost = min(next_position_costs.keys())
        min_cost_next_pos = next_position_costs[min_cost]
        # if position == goal_pos:
        #     assert min_cost_next_pos == goal_pos
        return [min_cost_next_pos]

    def get_SIC(self, vertex):
        vertex = vertex.v
        assert type(vertex) == tuple
        assert len(vertex) == len(self.dijkstra_graphs)
        SIC = 0
        for i, pos in enumerate(vertex):
            SIC += self.dijkstra_graphs[i][pos].move_cost
        return SIC
    
    def get_shorterst_path_cost(self, agent_id, pos_tup):
        assert type(pos_tup) == tuple
        return self.dijkstra_graphs[agent_id][pos_tup].move_cost

    
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

        all_actions = mstar.search_OD(start, end)
        print("Time taken for M star: {}".format(time.time() - t2))
        return all_actions

    def mstar_search3(self, start, end):
        t_hldr1 = time.time()
        if self.dijkstra_graphs is None:
            print("Joint policy graphs not initialized. Initialzing now")
            self.init_joint_policy_graphs(start, end)
        else:
            print("Joint policy graphs already present...re-using graphs")

        t2 = time.time()
        print("Time taken for joint policy graphs: {}".format(t2 - t_hldr1))

        mstar = Mstar(start, end, self.expand_position, self.get_next_joint_policy_position, self.get_SIC)

        all_actions = mstar.search(start, end)
        print("Time taken for M star: {}".format(time.time() - t2))
        return all_actions

    def mstar_search4_OD(self, start, end):
        t_hldr1 = time.time()
        if self.dijkstra_graphs is None:
            print("Joint policy graphs not initialized. Initialzing now")
            self.init_joint_policy_graphs(start, end)
        else:
            print("Joint policy graphs already present...re-using graphs")

        t2 = time.time()
        print("Time taken for joint policy graphs: {}".format(t2 - t_hldr1))

        mstar = Mstar_OD(start, end, self.expand_position, self.get_next_joint_policy_position, self.get_shorterst_path_cost)

        all_actions = mstar.search(OD = True)
        print("Time taken for M star4 OD: {}".format(time.time() - t2))
        return all_actions
    
    def mstar_search4_Not_OD(self, start, end):
        t_hldr1 = time.time()
        if self.dijkstra_graphs is None:
            print("Joint policy graphs not initialized. Initialzing now")
            self.init_joint_policy_graphs(start, end)
        else:
            print("Joint policy graphs already present...re-using graphs")

        t2 = time.time()
        print("Time taken for joint policy graphs: {}".format(t2 - t_hldr1))

        mstar = Mstar_OD(start, end, self.expand_position, self.get_next_joint_policy_position, self.get_shorterst_path_cost)

        all_actions = mstar.search(OD = False)
        print("Time taken for M star4 Not OD: {}".format(time.time() - t2))
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




    # def dijkstra_search(self, start_pos, end_pos, ignore = ["agent"]):
    #     '''
    #     NB: The cost of starts at 0 at the start position and continues thougough 
    #     all postions. In order to use the cost to search for the cheapest path,
    #     start, place end postion at start position.
    #     Searches the entire search space and returns a dictionary of the closed set '''
    #     obstacle_types = copy.deepcopy(self.obstacle_types)
    #     for ig in ignore:
    #         if ig in self.obstacle_types:
    #             ind = obstacle_types.index(ig)
    #             del obstacle_types[ind]
    #     #open = Queue()
    #     open = []
    #     closed = {}
    #     start_node = self.Node(start_pos, 0, None, None)
    #     path_found_flag = False
    #     #open.put(start_node)
    #     open.append(start_node)
    #     #while not open.empty():
    #     while len(open) != 0:
    #         #(_, curr_node) = open.get()
    #         #curr_node = open.get()
    #         curr_node = open[-1]
    #         del open[-1]
    #         if curr_node.pos == end_pos:
    #             closed[curr_node.pos] = curr_node
    #             path_found_flag = True
    #         next_nodes_pos = self._get_neigbours(curr_node.pos, obstacle_types)
    #         move_cost = curr_node.move_cost + 1
    #         for n in next_nodes_pos:
    #             (p,a) = n 
    #             p = tuple(p)
    #             if not p in closed.keys(): #or (closed[p]).move_cost > move_cost:
    #                 #priority = move_cost + heuristic(p, end_pos)
    #                 #open.put((priority, self.Node(p, move_cost, a, curr_node.pos)))

    #                 #open.put(self.Node(p, move_cost, a, curr_node.pos))
    #                 open.append(self.Node(p, move_cost, a, curr_node.pos))
    #         closed[curr_node.pos] = curr_node
    #     return closed
    

    # def dijkstra_search(self, start_pos, end_pos, ignore = ["agent"]):
    #     '''
    #     NB: The cost of starts at 0 at the start position and continues thougough 
    #     all postions. In order to use the cost to search for the cheapest path,
    #     start, place end postion at start position.
    #     Searches the entire search space and returns a dictionary of the closed set '''

    #     class myQueue():
    #         def __init__(self):
    #             self.q = Queue
    #             self.lookup = dict()
    #         def push(self, item):
    #             if not item.pos in self.lookup:
    #                 self.q.put(item)
    #         def pop(self):
    #             item = self.q.get()
    #             del self.lookup[item.pos]


            
    #     obstacle_types = copy.deepcopy(self.obstacle_types)
    #     for ig in ignore:
    #         if ig in self.obstacle_types:
    #             ind = obstacle_types.index(ig)
    #             del obstacle_types[ind]
    #     open = Queue()
    #     #open = []
    #     closed = {}
    #     start_node = self.Node(start_pos, 0, None, None)
    #     path_found_flag = False
    #     open.put(start_node)
    #     #open.append(start_node)
    #     while not open.empty():
    #     #while len(open) != 0:
    #         #(_, curr_node) = open.get()
    #         curr_node = open.get()
    #         #curr_node = open[-1]
    #         #del open[-1]
    #         if curr_node.pos == end_pos:
    #             closed[curr_node.pos] = curr_node
    #             path_found_flag = True
    #         next_nodes_pos = self._get_neigbours(curr_node.pos, obstacle_types)
    #         move_cost = curr_node.move_cost + 1
    #         for n in next_nodes_pos:
    #             (p,a) = n 
    #             p = tuple(p)
    #             if not p in closed.keys(): #or (closed[p]).move_cost > move_cost:
    #                 #priority = move_cost + heuristic(p, end_pos)
    #                 #open.put((priority, self.Node(p, move_cost, a, curr_node.pos)))

    #                 open.put(self.Node(p, move_cost, a, curr_node.pos))
    #                 #open.append(self.Node(p, move_cost, a, curr_node.pos))
    #         closed[curr_node.pos] = curr_node
    #     return closed


    def dijkstra_search(self, start_pos, end_pos, ignore = ["agent"]):
        '''
        NB: The cost of starts at 0 at the start position and continues thougough 
        all postions. In order to use the cost to search for the cheapest path,
        start, place end postion at start position.
        Searches the entire search space and returns a dictionary of the closed set '''

        raise Exception("Use Breadth First Search instead")

        class myQueue():
            def __init__(self):
                self.q = Queue()
                self.lookup = dict()
            def put(self, item):
                if not item.pos in self.lookup:
                    self.q.put(item)
                    self.lookup[item.pos] = item
            def get(self):
                item = self.q.get()
                del self.lookup[item.pos]
                return item
            def empty(self):
                return self.q.empty()
        
        # class AllNodes():
        #     def __init__(self):
        #         self.lookup = dict()
        #     def get(self, pos, move_cost, a, curr_node.pos):
        #         if pos in self.lookup:
        #             return self.lookup[pos]
        #         else:
        #             self.lookup[pos]

                

        obstacle_types = copy.deepcopy(self.obstacle_types)
        for ig in ignore:
            if ig in self.obstacle_types:
                ind = obstacle_types.index(ig)
                del obstacle_types[ind]
       # open = Queue()
        open = myQueue()
        #open = []
        closed = {}
        start_node = self.Node(start_pos, 0, None, None)
        path_found_flag = False
        open.put(start_node)
        #open.append(start_node)
        while not open.empty():
        #while len(open) != 0:
            #(_, curr_node) = open.get()
            curr_node = open.get()
            #curr_node = open[-1]
            #del open[-1]
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
                    #open.append(self.Node(p, move_cost, a, curr_node.pos))
            closed[curr_node.pos] = curr_node
        return closed

    def BFS(self, start_pos, end_pos, ignore = ["agent"]):
        assert type(start_pos) == tuple
        assert type(end_pos) == tuple

        obstacle_types = copy.deepcopy(self.obstacle_types)
        for ig in ignore:
            if ig in self.obstacle_types:
                ind = obstacle_types.index(ig)
                del obstacle_types[ind]

        visited = dict()
        q = Queue()
        start_node = self.Node(start_pos, 0, None, None)
        q.put(start_node)

        while not q.empty():
            n = q.get()
            #assert not n.pos in visited
            if not n.pos in visited:
                visited[n.pos] = n
                next_nodes_pos = self._get_neigbours(n.pos, obstacle_types)
                for next_n in next_nodes_pos:
                    hldr = tuple(next_n[0])
                    if not hldr in visited:
                        q.put(self.Node(tuple(next_n[0]), n.move_cost + 1, next_n[1], n.pos))
        return visited



        
    def a_star_search2(self,start_pos, goal_pos, ignore = ["agent"], pos_obstacle = None):
        '''Single agent, single goal, ignore object of type agent.
            Automatically ignore goals which are not this agent's goal '''

        class Node2():
            def __init__(self, pos, prev_act, prev_pos):
                self.pos = pos
                self.g = None
                self.f = None #f = g + h
                self.prev_act = prev_act
                self.prev_pos = prev_pos
            # For for python heapq: 
            def __gt__(self, n2):
                return self.g > n2.g
            def __ge__(self, n2):
                return self.g >= n2.g
            def __lt__(self, n2):
                return self.g < n2.g
            def __le__(self, n2):
                return self.g <= n2.g
            def __eq__(self, n2):
                return self.g == n2.g

        class myPriorityQ2():
            ''' Specifically for use with Node class objects'''
            def __init__(self):
                self.q = []
                self.lookup = dict()
            def push(self, item):
                heapq.heappush(self.q, item)
                node = item[-1]
                self.lookup[node.pos] = node
                # if node.pos in self.lookup:
                #     self.lookup[node.pos].append(node)
                # else:
                #     self.lookup[node.pos] = [node]
            def pop(self):
                (_, n) = heapq.heappop(self.q)
                while not n.pos in self.lookup:
                    assert len(self.q) != 0
                    n = heapq.heappop(self.q)
                # list_of_nodes = self.lookup[n.pos]
                # if len(list_of_nodes) == 1:
                #     del self.lookup[n.pos]
                # else:
                #     found_ind = None
                #     for i, item in enumerate(list_of_nodes):
                #         if item.f == n.f:
                #             found_ind = i
                #             break
                #     assert not found_ind is None
                #     del list_of_nodes[found_ind] 
                #     self.lookup[n.pos] = list_of_nodes
                return n
 
            def __len__(self,):
                return len(self.q)
            def empty(self):
                if len(self.q) == 0:
                    return True
                else:
                    return False
            def __contains__(self, other_node):
                if other_node.pos in self.lookup:
                    return True
                else:
                    return False
            def contains_less_than(self, node):
                ''' If priority Q contains same node
                    with f value less than input node.
                    If False returned, node should be 
                    added to open list'''
                if node.pos in self.lookup:
                    # lst_at_pos = self.lookup[node.pos]
                    # found_true_flag = False
                    # for n_stored in lst_at_pos:
                    #     if n_stored.f < node.f:
                    #         found_true_flag = True
                    # return found_true_flag

                    if self.lookup[node.pos].f <= node.f:
                        return True
                    else:
                        return False
                else:
                    return False

        class myPriorityQ():
            ''' Specifically for use with Node class objects'''
            def __init__(self):
                self.q = []
                self.lookup = dict()
            def push(self, item):
                heapq.heappush(self.q, item)
                node = item[-1]
                if node.pos in self.lookup:
                    self.lookup[node.pos].append(node)
                else:
                    self.lookup[node.pos] = [node]
            def pop(self):
                (_, n) = heapq.heappop(self.q)
                list_of_nodes = self.lookup[n.pos]
                if len(list_of_nodes) == 1:
                    del self.lookup[n.pos]
                else:
                    found_ind = None
                    for i, item in enumerate(list_of_nodes):
                        if item.f == n.f:
                            found_ind = i
                            break
                    assert not found_ind is None
                    del list_of_nodes[found_ind] 
                    self.lookup[n.pos] = list_of_nodes
                return n
 
            def __len__(self,):
                return len(self.q)
            def empty(self):
                if len(self.q) == 0:
                    return True
                else:
                    return False
            def __contains__(self, other_node):
                if other_node.pos in self.lookup:
                    return True
                else:
                    return False
            def contains_less_than(self, node):
                ''' If priority Q contains same node
                    with f value less than input node.
                    If False returned, node should be 
                    added to open list'''
                if node.pos in self.lookup:
                    lst_at_pos = self.lookup[node.pos]
                    found_true_flag = False
                    for n_stored in lst_at_pos:
                        if n_stored.f < node.f:
                            found_true_flag = True
                    return found_true_flag

                    # if self.lookup[node.pos].f < node.f:
                    #     return True
                    # else:
                    #     return False
                else:
                    return False
    
            
        obstacle_types = copy.deepcopy(self.obstacle_types)
        for ig in ignore:
            if ig in self.obstacle_types:
                ind = obstacle_types.index(ig)
                del obstacle_types[ind]
        
        #######################
        assert type(start_pos) == tuple
        assert type(goal_pos) == tuple
        def get_cost(v_current, v_next):
            return 1

        heuristic_f = self.abs_dist
        #open = myPriorityQ()
        open = myPriorityQ2()
        closed = dict()
        vs = Node2(start_pos, None, None)
        vs.g = 0
        vs.f = vs.g + heuristic_f(vs.pos, goal_pos)
        open.push((vs.f, vs))
        while not open.empty():
            vk = open.pop()
            if vk.pos == goal_pos:
                closed[vk.pos] = vk
                break #Solution found
            for vn_pos_act in self._get_neigbours(vk.pos, obstacle_types, pos_obstacle):
                (pos, act) = vn_pos_act
                pos = tuple(pos)
                vn = Node2(pos, act, vk.pos)
                vn.g = vk.g + get_cost(vk, vn)
                vn.f = vn.g + heuristic_f(vn.pos, goal_pos)
                #If vertex in open orclosed and vetex in 
                # open or closed has f vlaue less
                # than vn.f, then vn not added to open
                if open.contains_less_than(vn):
                    continue
                elif vn.pos in closed:
                    if closed[vn.pos].f <= vn.f:
                        continue
                open.push((vn.f, vn))
                # Parent already added implicity by adding action
            closed[vk.pos] = vk
        
        #######################
        # heuristic = self.abs_dist
        # closed = {}
        # start_node = self.Node(start_pos, 0, None, None)
        # open = PriorityQueue()
        # open.put((0, start_node))
        # while not open.empty():
        #     (_, curr_node) = open.get()
        #     if curr_node.pos in goal_pos:
        #         closed[curr_node.pos] = curr_node
        #         break
        #     next_nodes_pos = self._get_neigbours(curr_node.pos, obstacle_types, pos_obstacle)
        #     move_cost = curr_node.move_cost + 1
        #     for n in next_nodes_pos:
        #         (p,a) = n 
        #         p = tuple(p)
        #         if not p in closed.keys() or (closed[p]).move_cost > move_cost:
        #        # if not p in closed.keys() or (closed[p]).move_cost < move_cost:
        #             priority = move_cost + heuristic(p, goal_pos)
        #             open.put((priority, self.Node(p, move_cost, a, curr_node.pos)))
        #     closed[curr_node.pos] = curr_node

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


    def a_star_search3(self,start_pos, goal_pos, ignore = ["agent"], pos_obstacle = None):
        '''Single agent, single goal, ignore object of type agent.
            Automatically ignore goals which are not this agent's goal '''

        class Node2():
            def __init__(self, pos, prev_act, prev_pos):
                self.pos = pos
                self.g = None
                self.f = None #f = g + h
                self.prev_act = prev_act
                self.prev_pos = prev_pos
            # For for python heapq: 
            def __gt__(self, n2):
                return self.g > n2.g
            def __ge__(self, n2):
                return self.g >= n2.g
            def __lt__(self, n2):
                return self.g < n2.g
            def __le__(self, n2):
                return self.g <= n2.g
            def __eq__(self, n2):
                return self.g == n2.g

        class myPriorityQ2():
            ''' Specifically for use with Node class objects'''
            def __init__(self):
                self.q = []
                self.lookup = dict()
            def push(self, item):
                heapq.heappush(self.q, item)
                node = item[-1]
                self.lookup[node.pos] = node
                # if node.pos in self.lookup:
                #     self.lookup[node.pos].append(node)
                # else:
                #     self.lookup[node.pos] = [node]
            def pop(self):
                (_, n) = heapq.heappop(self.q)
                while not n.pos in self.lookup:
                    assert len(self.q) != 0
                    n = heapq.heappop(self.q)
                
                del self.lookup[n.pos]
                # list_of_nodes = self.lookup[n.pos]
                # if len(list_of_nodes) == 1:
                #     del self.lookup[n.pos]
                # else:
                #     found_ind = None
                #     for i, item in enumerate(list_of_nodes):
                #         if item.f == n.f:
                #             found_ind = i
                #             break
                #     assert not found_ind is None
                #     del list_of_nodes[found_ind] 
                #     self.lookup[n.pos] = list_of_nodes
                return n
 
            def __len__(self,):
                return len(self.q)
            def empty(self):
                if len(self.q) == 0:
                    return True
                else:
                    return False
            def __contains__(self, other_node):
                if other_node.pos in self.lookup:
                    return True
                else:
                    return False
            def contains_less_than(self, node):
                ''' If priority Q contains same node
                    with f value less than input node.
                    If False returned, node should be 
                    added to open list'''
                if node.pos in self.lookup:
                    # lst_at_pos = self.lookup[node.pos]
                    # found_true_flag = False
                    # for n_stored in lst_at_pos:
                    #     if n_stored.f < node.f:
                    #         found_true_flag = True
                    # return found_true_flag

                    if self.lookup[node.pos].f <= node.f:
                        return True
                    else:
                        return False
                else:
                    return False

        class myPriorityQ():
            ''' Specifically for use with Node class objects'''
            def __init__(self):
                self.q = []
                self.lookup = dict()
            def push(self, item):
                heapq.heappush(self.q, item)
                node = item[-1]
                if node.pos in self.lookup:
                    self.lookup[node.pos].append(node)
                else:
                    self.lookup[node.pos] = [node]
            def pop(self):
                (_, n) = heapq.heappop(self.q)
                list_of_nodes = self.lookup[n.pos]
                if len(list_of_nodes) == 1:
                    del self.lookup[n.pos]
                else:
                    found_ind = None
                    for i, item in enumerate(list_of_nodes):
                        if item.f == n.f:
                            found_ind = i
                            break
                    assert not found_ind is None
                    del list_of_nodes[found_ind] 
                    self.lookup[n.pos] = list_of_nodes
                return n
 
            def __len__(self,):
                return len(self.q)
            def empty(self):
                if len(self.q) == 0:
                    return True
                else:
                    return False
            def __contains__(self, other_node):
                if other_node.pos in self.lookup:
                    return True
                else:
                    return False
            def contains_less_than(self, node):
                ''' If priority Q contains same node
                    with f value less than input node.
                    If False returned, node should be 
                    added to open list'''
                if node.pos in self.lookup:
                    lst_at_pos = self.lookup[node.pos]
                    found_true_flag = False
                    for n_stored in lst_at_pos:
                        if n_stored.f < node.f:
                            found_true_flag = True
                    return found_true_flag

                    # if self.lookup[node.pos].f < node.f:
                    #     return True
                    # else:
                    #     return False
                else:
                    return False
    
            
        obstacle_types = copy.deepcopy(self.obstacle_types)
        for ig in ignore:
            if ig in self.obstacle_types:
                ind = obstacle_types.index(ig)
                del obstacle_types[ind]
        
        #######################
        assert type(start_pos) == tuple
        assert type(goal_pos) == tuple
        def get_cost(v_current, v_next):
            return 1

        heuristic_f = self.abs_dist
        #open = myPriorityQ()
        open = myPriorityQ2()
        closed = dict()
        vs = Node2(start_pos, None, None)
        vs.g = 0
        vs.f = vs.g + heuristic_f(vs.pos, goal_pos)
        open.push((vs.f, vs))
        while not open.empty():
            vk = open.pop()
            if vk.pos == goal_pos:
                closed[vk.pos] = vk
                break #Solution found
            for vn_pos_act in self._get_neigbours(vk.pos, obstacle_types, pos_obstacle):
                (pos, act) = vn_pos_act
                pos = tuple(pos)
                vn = Node2(pos, act, vk.pos)
                vn.g = vk.g + get_cost(vk, vn)
                vn.f = vn.g + heuristic_f(vn.pos, goal_pos)
                #If vertex in open orclosed and vetex in 
                # open or closed has f vlaue less
                # than vn.f, then vn not added to open
                if open.contains_less_than(vn):
                    continue
                elif vn.pos in closed:
                    if closed[vn.pos].f <= vn.f:
                        continue
                open.push((vn.f, vn))
                # Parent already added implicity by adding action
            closed[vk.pos] = vk
        
        #######################
        # heuristic = self.abs_dist
        # closed = {}
        # start_node = self.Node(start_pos, 0, None, None)
        # open = PriorityQueue()
        # open.put((0, start_node))
        # while not open.empty():
        #     (_, curr_node) = open.get()
        #     if curr_node.pos in goal_pos:
        #         closed[curr_node.pos] = curr_node
        #         break
        #     next_nodes_pos = self._get_neigbours(curr_node.pos, obstacle_types, pos_obstacle)
        #     move_cost = curr_node.move_cost + 1
        #     for n in next_nodes_pos:
        #         (p,a) = n 
        #         p = tuple(p)
        #         if not p in closed.keys() or (closed[p]).move_cost > move_cost:
        #        # if not p in closed.keys() or (closed[p]).move_cost < move_cost:
        #             priority = move_cost + heuristic(p, goal_pos)
        #             open.put((priority, self.Node(p, move_cost, a, curr_node.pos)))
        #     closed[curr_node.pos] = curr_node

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




    def a_star_search(self,start_pos, goal_pos, ignore = ["agent"], pos_obstacle = None):
        '''Single agent, single goal, ignore object of type agent.
            Automatically ignore goals which are not this agent's goal '''
        
        raise Exception("Use a_star_search2 instead")
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
               # if not p in closed.keys() or (closed[p]).move_cost < move_cost:
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

###########################################################

    def _is_colliding(self, v):
        hldr = set()
        for i, vi in enumerate(v):
            for i2, vi2 in enumerate(v):
                if i != i2:
                    if vi == vi2:
                        hldr.add(i)
                        hldr.add(i2)
        return hldr 

    def _get_neighbours_joint(self, joint_position, obstacle_types, pos_obstacle):
        '''Takes vertex object as input and returns a 
        list of expanded v according to M* alg'''
        assert type(joint_position) == tuple
        indiv_pos = dict()
        v_len = len(joint_position)
        for i, pos in enumerate(joint_position):
            indiv_neighbours = self._get_neigbours(pos, obstacle_types, pos_obstacle)
            #indiv_neighbours_pos = [tuple(hldr[0]) for hldr in indiv_neighbours]
            #indiv_neighbours_act = [tuple(hldr[1]) for hldr in indiv_neighbours]
            indiv_pos[i] = indiv_neighbours
        
        combinations = ParameterGrid(indiv_pos)
    
        all_combinations = []
        for c in combinations:
            this_joint_position = tuple([tuple(c[i][0]) for i in range(v_len)])
            this_joint_action = {i:c[i][1] for i in range(v_len)}

            if len(self._is_colliding(this_joint_position)) == 0:
                all_combinations.append([this_joint_position, this_joint_action])
        return all_combinations
            
        
    def _get_neighbours_joint_OD(self, inter_vertex, obstacle_types, pos_obstacle):
        '''Takes an intermediate vertex of the form ( (1,2, ...) , ( (x1,y1), (...),..)
        and returns a list of the next intermediate nodes. '''
        assert type(inter_vertex) == tuple
       # indiv_pos = dict()
        num_inter_levels = len(inter_vertex[1])
        current_inter_level = inter_vertex[0][-1]

        next_inter_level = current_inter_level + 1
        if next_inter_level == num_inter_levels:
            next_inter_level = 0

        joint_position = list(inter_vertex[1])

        pos_to_expand = joint_position[current_inter_level]
        neighbours = self._get_neigbours(pos_to_expand, obstacle_types, pos_obstacle)
        pos_already_assigned = joint_position[:current_inter_level]
        neighbours = [tuple(n[0]) for n in neighbours if not tuple(n[0]) in pos_already_assigned]
        #assert len(neighbours) != 0

        inter_v = []
        for n in neighbours:
            joint_p_cyp = copy.deepcopy(joint_position)
            joint_p_cyp[current_inter_level] = tuple(n)
            inter_v.append(tuple( ((next_inter_level,), tuple(joint_p_cyp))))

        return inter_v



    def a_star_search5(self, start_pos, goal_pos, ignore = ["agent"], pos_obstacle = None):
        '''Single agent, single goal, ignore object of type agent.
            Automatically ignore goals which are not this agent's goal '''


        def abs_dist_SIC(start, end):
            assert type(start) == tuple
            assert type(end) == tuple
            total= 0
            for i, (p1,p2) in enumerate(zip(start, end)):
                total += self.abs_dist(p1, p2)
            return total

        class Node2():
            def __init__(self, pos, prev_act, prev_pos):
                self.pos = pos
                self.v = None
                self.g = 1e6 #None
                self.f = None #f = g + h
                self.prev_act = prev_act
                self.prev_pos = prev_pos
                self.back_ptr = None

            def add_parent(self, v):
                self.back_ptr = v

            # For python heapq: 
            def __gt__(self, n2):
                return self.g > n2.g
            def __ge__(self, n2):
                return self.g >= n2.g
            def __lt__(self, n2):
                return self.g < n2.g
            def __le__(self, n2):
                return self.g <= n2.g
            def __eq__(self, n2):
                return self.g == n2.g

        class AllVertex():
            '''Keeps track of all nodes created
                such that nodes are created only once '''
            def __init__(self, use_intermediate_nodes = False):
                self.all_v = dict()
                self.intermediate = use_intermediate_nodes
            def get(self, v_pos):
                if v_pos in self.all_v:
                    v = self.all_v[v_pos]
                else:
                    if self.intermediate:
                        (inter_level, pos) = v_pos
                        v = Node2(pos, None, None)
                        v.v = v_pos
                    else:
                        v = Node2(v_pos, None, None)
                    self.all_v[v_pos] = v
                return v
            def update(self, v, prev_act, prev_pos):
                assert v.v in self.all_v
                v.prev_act = prev_act
                v.prev_pos = prev_pos

            def add_parent(self, v_current, v_parent):
                v_current.add_parent(v_parent)

        class simplePriorityQ():
            def __init__(self):
                self.q = []
            def push(self, item):
                heapq.heappush(self.q, item)
            def pop(self):
                (_, n) = heapq.heappop(self.q)
                return n
            def empty(self):
                if len(self.q) == 0:
                    return True
                else:
                    return False
        
        class ReplacePriorityQ():
            '''Replaces same nodes in q by only poping the lowest f-valued node '''
            def __init__(self):
                self.q = []
                self.lookup = dict()
            def push(self, item):
                heapq.heappush(self.q, item)
                node = item[-1]
                self.lookup[node.v] = node
            def pop(self):
                (_, n) = heapq.heappop(self.q)
                while not n.v in self.lookup:
                    assert len(self.q) != 0
                    n = heapq.heappop(self.q)
                del self.lookup[n.v]
                return n
            def empty(self):
                if len(self.q) == 0:
                    return True
                else:
                    return False
    
            
        obstacle_types = copy.deepcopy(self.obstacle_types)
        for ig in ignore:
            if ig in self.obstacle_types:
                ind = obstacle_types.index(ig)
                del obstacle_types[ind]
        
        #######################
        assert len(start_pos) == len(goal_pos)
        assert type(start_pos) == tuple
        assert type(goal_pos) == tuple
        #Assert input is tuple of tuples:
        for hldr1, hldr2 in zip(start_pos, goal_pos):
            assert type(hldr1) == tuple
            assert type(hldr2) == tuple
        N_AGENTS = len(start_pos)

        def get_cost(v_current, v_next, intermediate = False):
            if intermediate:
                inter_level = v_next.v[0][-1]
                return 1 #1*(inter_level+1)
            else:
                return N_AGENTS

        heuristic_f = abs_dist_SIC
        #open = simplePriorityQ() 
        open = ReplacePriorityQ()
        all_v = AllVertex(use_intermediate_nodes=True)
        vs = all_v.get(((0,),start_pos))
        vs.g = 0
        vs.f = vs.g + heuristic_f(vs.pos, goal_pos)
        open.push((vs.f, vs))
        solution_v = None
        while not open.empty():
            vk = open.pop()
            if vk.pos == goal_pos and vk.v[0][-1] == 0:
                solution_v = vk
                break #Solution found
            #for vn_pos_act in self._get_neigbours(vk.pos, obstacle_types, pos_obstacle):
            for vn_pos_act in self._get_neighbours_joint_OD(vk.v, obstacle_types, pos_obstacle):
                v = vn_pos_act
                #pos = tuple(pos)
                vn = all_v.get(v)
                if vk.g + get_cost(vk, vn) < vn.g:
                    #all_v.update(vn, act, vk.v)
                    all_v.add_parent(vn, vk)
                    vn.g = vk.g + get_cost(vk, vn, intermediate=True)
                    vn.f = vn.g + heuristic_f(vn.pos, goal_pos)
                    open.push((vn.f, vn))
        # action_path = []
        # node_path = []
        # closed = all_v.all_v
        # if goal_pos in closed.keys():
        #     n = closed[goal_pos]
        #     node_path.append(n.pos)
        #     while not n.prev_act is None:
        #         action_path.append(n.prev_act)
        #         n = closed[n.prev_pos]
        #         node_path.append(n.pos)
        #     action_path.reverse()
        if solution_v is None:
            return None
        else:
            return self._back_track(solution_v) #action_path, node_path

    def _back_track(self, goal_v):
        '''Returns a dictionary of actions for the optimal path '''
        self.pos_act = {(0,1):2,
                        (1,0):3,
                        (0,-1):4,
                        (-1,0):1,
                        (0,0): 0}
        
        #get vertices:
        all_v = []
        all_v.append(goal_v.pos)
        next_v = goal_v.back_ptr
        while not next_v is None:
            if next_v.v[0][-1] == 0:
                all_v.append(next_v.pos)
            next_v = next_v.back_ptr

        #Get actions from vertices:
        all_actions = []
        prev_v = all_v[-1]
        for v in reversed(all_v[:-1]):
            actions = {}
            for i, (previous_position, next_postion) in enumerate(zip(prev_v, v)):
                position_diff = self._add_tup(next_postion, self._mult_tup(previous_position, -1))
                actions[i] = self.pos_act[position_diff]
            prev_v = v
            all_actions.append(actions)
        return all_actions


    def _add_tup(self, a,b):
        assert len(a) == len(b)
        ans = []
        for ia,ib in zip(a,b):
            ans.append(ia+ib)
        return tuple(ans)

    def _mult_tup(self, a, m):
        ans = []
        for ai in a:
            ans.append(ai*m)
        return tuple(ans)














    def a_star_search4(self,start_pos, goal_pos, ignore = ["agent"], pos_obstacle = None):
        '''Single agent, single goal, ignore object of type agent.
            Automatically ignore goals which are not this agent's goal '''

        class Node2():
            def __init__(self, pos, prev_act, prev_pos):
                self.pos = pos
                self.g = 1e6 #None
                self.f = None #f = g + h
                self.prev_act = prev_act
                self.prev_pos = prev_pos
            # For python heapq: 
            def __gt__(self, n2):
                return self.g > n2.g
            def __ge__(self, n2):
                return self.g >= n2.g
            def __lt__(self, n2):
                return self.g < n2.g
            def __le__(self, n2):
                return self.g <= n2.g
            def __eq__(self, n2):
                return self.g == n2.g

        class AllVertex():
            '''Keeps track of all nodes created
                such that nodes are created only once '''
            def __init__(self):
                self.all_v = dict()
            def get(self, pos):
                if pos in self.all_v:
                    v = self.all_v[pos]
                else:
                    v = Node2(pos, None, None)
                    self.all_v[pos] = v
                return v
            def update(self, v, prev_act, prev_pos):
                assert v.pos in self.all_v
                v.prev_act = prev_act
                v.prev_pos = prev_pos

        class simplePriorityQ():
            def __init__(self):
                self.q = []
            def push(self, item):
                heapq.heappush(self.q, item)
            def pop(self):
                (_, n) = heapq.heappop(self.q)
                return n
            def empty(self):
                if len(self.q) == 0:
                    return True
                else:
                    return False
        
        class ReplacePriorityQ():
            '''Replaces same nodes in q by only poping the lowest f-valued node '''
            def __init__(self):
                self.q = []
                self.lookup = dict()
            def push(self, item):
                heapq.heappush(self.q, item)
                node = item[-1]
                self.lookup[node.pos] = node
            def pop(self):
                (_, n) = heapq.heappop(self.q)
                while not n.pos in self.lookup:
                    assert len(self.q) != 0
                    n = heapq.heappop(self.q)
                del self.lookup[n.pos]
                return n
            def empty(self):
                if len(self.q) == 0:
                    return True
                else:
                    return False
    
            
        obstacle_types = copy.deepcopy(self.obstacle_types)
        for ig in ignore:
            if ig in self.obstacle_types:
                ind = obstacle_types.index(ig)
                del obstacle_types[ind]
        
        #######################
        assert type(start_pos) == tuple
        assert type(goal_pos) == tuple
        def get_cost(v_current, v_next):
            return 1

        heuristic_f = self.abs_dist
        open = simplePriorityQ() 
        all_v = AllVertex()
        vs = all_v.get(start_pos)
        vs.g = 0
        vs.f = vs.g + heuristic_f(vs.pos, goal_pos)
        open.push((vs.f, vs))
        while not open.empty():
            vk = open.pop()
            if vk.pos == goal_pos:
                break #Solution found
            for vn_pos_act in self._get_neigbours(vk.pos, obstacle_types, pos_obstacle):
                (pos, act) = vn_pos_act
                pos = tuple(pos)
                vn = all_v.get(pos)
                if vk.g + get_cost(vk, vn) < vn.g:
                    all_v.update(vn, act, vk.pos)
                    vn.g = vk.g + get_cost(vk, vn)
                    vn.f = vn.g + heuristic_f(vn.pos, goal_pos)
                    open.push((vn.f, vn))
        action_path = []
        node_path = []
        closed = all_v.all_v
        if goal_pos in closed.keys():
            n = closed[goal_pos]
            node_path.append(n.pos)
            while not n.prev_act is None:
                action_path.append(n.prev_act)
                n = closed[n.prev_pos]
                node_path.append(n.pos)
            action_path.reverse()
        return action_path, node_path








