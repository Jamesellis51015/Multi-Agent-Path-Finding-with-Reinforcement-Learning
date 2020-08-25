from queue import PriorityQueue
from sklearn.model_selection import ParameterGrid
import copy

class NewPriorityQueue(PriorityQueue):
    def __init__(self):
        super().__init__()
        self.lookup_table = set()
    def put(self, item):
        if not item[-1].v in self.lookup_table:
            super().put(item)
            self.lookup_table.add(item[-1].v)
    def get(self):
        result = super().get()
        self.lookup_table.remove(result[-1].v)
        return result
    def __contains__(self, key):
        return key.v in self.lookup_table

class NewPriorityQueue2(PriorityQueue):
    def __init__(self):
        super().__init__()
        self.lookup_table = {}
    def add_lookup(self, item):
        if item[-1].v in self.lookup_table:
            self.lookup_table[item[-1].v] += 1
        else:
            self.lookup_table[item[-1].v] = 1
    def remove_lookup(self, item):
        if item[-1].v in self.lookup_table:
            if self.lookup_table[item[-1].v] < 2:
                del self.lookup_table[item[-1].v]
            else:
                self.lookup_table[item[-1].v] -= 1
    def put(self, item):
       # if not item[-1].v in self.lookup_table:
        super().put(item)
        self.add_lookup(item)
    def get(self):
        result = super().get()
        self.remove_lookup(result)
        return result
    def __contains__(self, key):
        return key.v in self.lookup_table


class Mstar():
    class Vertex():
        def __init__(self, v, intermediate_level = None, inherit_col_set = None, inherit_is_col = None):
            MAXCOST = 1e6
            assert type(v) == tuple, "Input is not tuple"
            self.v = v
            self.cost_g = MAXCOST
            self._collision_set = set()
            self._back_set = {}
            self.back_ptr = None

            # Intermediary level: the level in the search tree of agent expansion (for intermediary vertices)
            # if intermediary_level is empty, then its a standard node
            #self.intermediate_level = {}
            #self.intermediate_level = self.add_intermediate(intermediate_level)
            self.intermediate_level = set()
            self.init_intermediate_level(intermediate_level)

            self.is_colliding = self._is_colliding(v)
            self.joint_collision_set(self.is_colliding)
            # #Prevents computing collision set for intermediate nodes:
            # if inherit_is_col is None:
            #     self.is_colliding = self._is_colliding(v)
            # else:
            #     self.is_colliding = inherit_is_col
            # if inherit_col_set is None:
            #     self.joint_collision_set(self.is_colliding)
            # else:
            #     self._collision_set = inherit_col_set
            
        def init_intermediate_level(self, intermediate):
            if not intermediate is None:
                self.intermediate_level = intermediate
            else:
                self.intermediate_level = set()
        #def add_intermediate(self, intermediate):
        #   return self.intermediate_level.add(intermediate)
        
        def add_back_set(self, new_v):
            assert isinstance(new_v, type(self)), "Input is not a Vertex class"

            # if new_v.v in self._back_set:
            #     print("WARNING: overwriting vertex which already exist in _back_set")
            self._back_set[new_v.v] = new_v
        
        def get_back_set(self):
            return self._back_set.values()
        def set_back_ptr(self, ptr):
            self.back_ptr = ptr
        
        def add_collision(self, agent_handle):
            self._collision_set.add(agent_handle)
        def joint_collision_set(self, other_set):
            self._collision_set = self._collision_set.union(other_set)
        def is_col_subset(self, other_set):
            return other_set.issubset(self._collision_set)
        def get_collision_set(self):
            return self._collision_set
        def set_cost(self, new_cost):
            self.cost_g = new_cost
        @property
        def get_cost(self):
            return self.cost_g

        def _is_colliding(self, v):
            hldr = set()
            for i, vi in enumerate(v):
                for i2, vi2 in enumerate(v):
                    if i != i2:
                        if vi == vi2:
                            # hldr.add(i)
                            # hldr.add(i2)
                            #Do not count intermediate vertices
                            if len(self.intermediate_level) == 0:
                                hldr.add(i)
                                hldr.add(i2)
                            elif i2 in self.intermediate_level:
                                hldr.add(i)
                                hldr.add(i2)
            return hldr 
        #For priority que:
        def __eq__(self, other_v):
            return self.get_cost == other_v.get_cost
        def __gt__(self, other_v):
            return self.get_cost > other_v.get_cost
        def __ge__(self, other_v):
            return self.get_cost >= other_v.get_cost
        def __lt__(self, other_v):
            return self.get_cost < other_v.get_cost
        def __le__(self, other_v):
            return self.get_cost <= other_v.get_cost

    def __init__(self, start, end, expand_position, get_next_joint_policy_position, get_SIC):
        '''
        This class implements subdimensional expansion with a star as the search algorithm.
        It assumes the following functions which are external to the class:
            -- expand_position: returns the neighbouring vertices of a single position
            -- get_next_joint_policy_position: Returns the next vertex of a particular agents 
                                        joint policy action
                                        where the joint policy is the shortest path action
                                        where there is no other agents.
            -- get_SIC: returns the sum of individual cost (individual 
                        optimal path cost from vertex vk to vf)
         '''
        assert type(start) == list, "start parameter has to be list"
        assert type(end) == list, "end parameter has to be list"
        assert len(start) == len(end), "start and end positions have to be of same length"

        self.v_len = len(start)
        self.f_e_kl = self.v_len * 1 #cost of traversing an edge
        self.expand_position = expand_position
        self.get_next_joint_policy_position = get_next_joint_policy_position
        self.get_SIC = get_SIC

        self.all_v = {}

       # self.start = start

    def fekl(self, vertex):
        if len(vertex.intermediate_level) == 0:
            return self.v_len * 1
        else:
            return len(vertex.intermediate_level) * 1

    # def test(self):
    #     for i in range(5):
    #         print({i:self.get_next_joint_policy_position(i, self.start[i]) for i in range(self.v_len)})

    def get_vertex(self, v, intermediate = None, inherit_col_set=None, inherit_is_col=None):
    #def get_vertex(self, v, intermediate = None):
        '''If the vertex is not fount in all_v 
            lookup table, a new vertex is initiated '''
        assert type(v) == tuple
        if intermediate is None:
            intermediate = set()
        id = tuple((v, tuple(intermediate)))
        if not v in self.all_v:
            new_v = self.Vertex(v, intermediate)
            self.all_v[id] = new_v
        return self.all_v[id]


    def search(self, start, end):
        '''Performs M* search. Assumes M* has been initialized '''
        assert type(start) == list, "start parameter has to be list"
        assert type(end) == list, "end parameter has to be list"
        assert len(start) == len(end), "start and end positions have to be of same length"

        
        #v_f = self.Vertex(self._combine_tuples(end))
        v_f = self.get_vertex(self._combine_tuples(end))
        open = NewPriorityQueue2()
        #v_s = self.Vertex(self._combine_tuples(start))
        v_s = self.get_vertex(self._combine_tuples(start))
        v_s.set_cost(0)

        open.put((0, v_s))

        while not open.empty():
            (_, v_k) = open.get()
            if v_k.v == v_f.v:
                return self._back_track(v_k)
            
            for v_l in self._get_neighbours(v_k):
                v_l.add_back_set(v_k)
                # C_l already updated at initiation of vertex
                c_l = v_l.get_collision_set()
                self._backprop(v_k, c_l, open)
                if len(v_l.is_colliding) == 0 and (v_k.get_cost + self.f_e_kl) < v_l.get_cost:
                    v_l.set_cost(v_k.get_cost + self.f_e_kl)
                    v_l.set_back_ptr(v_k)
                    priority = v_l.get_cost + self.get_SIC(v_l)
                    open.put((priority, v_l))
        return None

    def search_OD(self, start, end):
        '''Performs M* search. Assumes M* has been initialized '''
        assert type(start) == list, "start parameter has to be list"
        assert type(end) == list, "end parameter has to be list"
        assert len(start) == len(end), "start and end positions have to be of same length"

        
        #v_f = self.Vertex(self._combine_tuples(end))
        v_f = self.get_vertex(self._combine_tuples(end))
        open = NewPriorityQueue2()
        #v_s = self.Vertex(self._combine_tuples(start))
        v_s = self.get_vertex(self._combine_tuples(start))
        v_s.set_cost(0)

        open.put((0, v_s))

        while not open.empty():
            (_, v_k) = open.get()
            if len(v_k.intermediate_level) == 0:
                prev_v_k = v_k
            #Intermediary vertices not valid solutions
            if v_k.v == v_f.v:
                return self._back_track(v_k)
            
            for v_l in self._get_neighbours_OD(v_k):
                
                #For intermediary nodes
                if len(v_l.intermediate_level) != 0:
                    v_l.set_cost(v_k.get_cost + self.fekl(v_l))
                    v_l.set_back_ptr(v_k)
                    priority = v_l.get_cost + self.get_SIC(v_l)
                    open.put((priority, v_l))
                    continue
                
               # assert len(v_l.intermediate_level) == 0
               # v_l.add_back_set(v_k)
                v_l.add_back_set(prev_v_k)
                # C_l already updated at initiation of vertex
                c_l = v_l.get_collision_set()
                self._backprop(v_k, c_l, open)
                if len(v_l.is_colliding) == 0 and (v_k.get_cost + self.fekl(v_l)) < v_l.get_cost:
                    v_l.set_cost(v_k.get_cost + self.fekl(v_l))
                    v_l.set_back_ptr(v_k)
                    priority = v_l.get_cost + self.get_SIC(v_l)
                    open.put((priority, v_l))
        return None

            

    def _backprop(self, v_k, c_l, open):
        if not c_l.issubset(v_k.get_collision_set()):
            v_k.joint_collision_set(c_l)
            if not v_k in open:
                priority = v_k.get_cost + self.get_SIC(v_k)
                open.put((priority, v_k))
            for v_m in v_k.get_back_set():
                self._backprop(v_m, v_k.get_collision_set(), open)


    def _get_neighbours(self, vertex):
        '''Takes vertex object as input and returns a 
        list of expanded v according to M* alg'''
        collisions = vertex.get_collision_set()
        v = vertex.v
        v_len = len(v)
        all_v_pos = {}
        for i, pos in enumerate(v):
            if i in collisions:
                n_pos = self.expand_position(i, pos)
            else:
                n_pos = self.get_next_joint_policy_position(i, pos)
            all_v_pos[i] = n_pos
        combinations = ParameterGrid(all_v_pos)
        all_v = [self.get_vertex(tuple([c[i] for i in range(v_len)])) for c in combinations]
        #all_v = [self.Vertex(tuple([c[i] for i in range(v_len)])) for c in combinations]
        return all_v


    def _get_neighbours_OD(self, vertex):
        '''Takes vertex object as input and returns a 
        list of expanded v according to M* alg'''
        collisions = vertex.get_collision_set()
        v = vertex.v
        v_len = len(v)

        this_level = len(vertex.intermediate_level)
        if this_level == len(v)-1:
            next_inter_level = set()
        else:
            next_inter_level = copy.deepcopy(vertex.intermediate_level)
            next_inter_level.add(this_level+1)

        # #For intermediate vertices:
        if len(next_inter_level) != 0:
            inherit_col_set = copy.deepcopy(collisions)
            #inherit_is_col = copy.deepcopy(vertex.is_colliding)
        else:
            inherit_col_set = None
           # inherit_is_col = None



        v = list(v)

        #this_level = len(vertex.intermediate_level)
        this_pos = v[this_level]
       # n_pos = self.expand_position(this_level, this_pos)
        if this_level in collisions:
            n_pos = self.expand_position(this_level, this_pos)
            
            #remove position options in conflict with previous inter agent action choices
            for i,p in enumerate(v[:this_level]):
                if p in n_pos:
                    n_pos.remove(p)
        else:
            n_pos = self.get_next_joint_policy_position(this_level, this_pos)

        all_v_tup = []
        for p in n_pos:
            v[this_level] = p
            all_v_tup.append(tuple(v))
        
        assert len(all_v_tup) != 0, "Something went wrong with OD expansion"

        all_v = [self.get_vertex(v_hldr, copy.deepcopy(next_inter_level), inherit_col_set=inherit_col_set) for v_hldr in all_v_tup]

        # all_v_pos = {}
        # for i, pos in enumerate(v):
        #     if i in collisions:
        #         n_pos = self.expand_position(i, pos)
        #     else:
        #         n_pos = self.get_next_joint_policy_position(i, pos)
        #     all_v_pos[i] = n_pos
        # combinations = ParameterGrid(all_v_pos)
        # all_v = [self.get_vertex(tuple([c[i] for i in range(v_len)])) for c in combinations]
        # #all_v = [self.Vertex(tuple([c[i] for i in range(v_len)])) for c in combinations]
        return all_v


    def _get_neighbours_OD2(self, vertex):
        '''Takes vertex object as input and returns a 
        list of expanded v according to M* alg'''
        collisions = vertex.get_collision_set()
        v = vertex.v
        v_len = len(v)

        # this_level = len(vertex.intermediate_level)
        # if this_level == len(v)-1:
        #     next_inter_level = set()
        # else:
        #     next_inter_level = copy.deepcopy(vertex.intermediate_level)
        #     next_inter_level.add(this_level+1)

        # # #For intermediate vertices:
        # if len(next_inter_level) != 0:
        #     inherit_col_set = copy.deepcopy(collisions)
        #     #inherit_is_col = copy.deepcopy(vertex.is_colliding)
        # else:
        #     inherit_col_set = None
           # inherit_is_col = None



        v = list(v)

        #this_level = len(vertex.intermediate_level)
        this_pos = v[this_level]
       # n_pos = self.expand_position(this_level, this_pos)
        if this_level in collisions:
            n_pos = self.expand_position(this_level, this_pos)
            
            #remove position options in conflict with previous inter agent action choices
            for i,p in enumerate(v[:this_level]):
                if p in n_pos:
                    n_pos.remove(p)
        else:
            n_pos = self.get_next_joint_policy_position(this_level, this_pos)

        all_v_tup = []
        for p in n_pos:
            v[this_level] = p
            all_v_tup.append(tuple(v))
        
        assert len(all_v_tup) != 0, "Something went wrong with OD expansion"

        all_v = [self.get_vertex(v_hldr, copy.deepcopy(next_inter_level), inherit_col_set=inherit_col_set) for v_hldr in all_v_tup]

        # all_v_pos = {}
        # for i, pos in enumerate(v):
        #     if i in collisions:
        #         n_pos = self.expand_position(i, pos)
        #     else:
        #         n_pos = self.get_next_joint_policy_position(i, pos)
        #     all_v_pos[i] = n_pos
        # combinations = ParameterGrid(all_v_pos)
        # all_v = [self.get_vertex(tuple([c[i] for i in range(v_len)])) for c in combinations]
        # #all_v = [self.Vertex(tuple([c[i] for i in range(v_len)])) for c in combinations]
        return all_v


    def _back_track(self, goal_v):
        '''Returns a dictionary of actions for the optimal path '''
        self.pos_act = {(0,1):2,
                        (1,0):3,
                        (0,-1):4,
                        (-1,0):1,
                        (0,0): 0}
        
        #get vertices:
        all_v = []
        all_v.append(goal_v.v)
        next_v = goal_v.back_ptr
        while not next_v is None:
            if len(next_v.intermediate_level) == 0:
                all_v.append(next_v.v)
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

        

    def _combine_tuples(self, positions):
        '''Takes a list of tuples and returns a single tuple of tuples'''
        return tuple(positions)

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




    
    #def a_star(self, start, end, expand_position):
    #    raise NotImplementedError