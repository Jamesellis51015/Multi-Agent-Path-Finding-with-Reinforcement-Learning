from queue import PriorityQueue
from sklearn.model_selection import ParameterGrid


class NewPriorityQueue(PriorityQueue):
    def __init__(self):
        super().__init__()
        self.lookup_table = set()
    def put(self, item):
        super().put(item)
        self.lookup_table.add(item[-1].v)
    def get(self):
        result = super().get()
        self.lookup_table.remove(result[-1].v)
        return result
    def __contains__(self, key):
        return key.v in self.lookup_table

class Mstar():
    class Vertex():
        def __init__(self, v):
            MAXCOST = 1e6
            assert type(v) == tuple, "Input is not tuple"
            self.v = v
            self.cost_g = MAXCOST
            self._collision_set = set()
            self._back_set = {}
            self.back_ptr = None
            self.is_colliding = self._is_colliding(v)
            self.joint_collision_set(self.is_colliding)
        
        def add_back_set(self, new_v):
            assert isinstance(new_v, self), "Input is not a Vertex class"

            if new_v.v in self._back_set:
                print("WARNING: overwriting vertex which already exist in _back_set")
            self._back_set[new_v.v] = new_v
        
        def get_back_set(self):
            return self._back_set.values()
        def set_back_ptr(self, ptr):
            self.back_ptr = ptr
        
        def add_collision(self, agent_handle):
            self._collision_set.add(agent_handle)
        def joint_collision_set(self, other_set):
            self._collision_set.union(other_set)
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

        self.start = start


    # def test(self):
    #     for i in range(5):
    #         print({i:self.get_next_joint_policy_position(i, self.start[i]) for i in range(self.v_len)})

    def _search(self, start, end):
        '''Performs M* search. Assumes M* has been initialized '''
        assert type(start) == list, "start parameter has to be list"
        assert type(end) == list, "end parameter has to be list"
        assert len(start) == len(end), "start and end positions have to be of same length"

        all_v = {}
        v_f = self.Vertex(self._combine_tuples(end))
        open = NewPriorityQueue()
        v_s = self.Vertex(self._combine_tuples(start))
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
        #NB need to create vertex object as well
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
        all_v = [tuple([c[i] for i in range(v_len)]) for c in combinations]
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
            all_v.append(next_v.v)
            next_v = next_v.back_ptr
        #####
        

    def _combine_tuples(self, positions):
        '''Takes a list of tuples and returns a single tuple of tuples'''
        return tuple(positions)

    def _add_tup(self, a,b):
        assert len(a) == len(b)
        ans = []
        for ia,ib in zip(a,b):
            ans.append(ia+ib)
        return ans

    def _mult_tup(self, a, m):
        ans = []
        for ai in a:
            ans.append(ai*m)
        return ans




    
    #def a_star(self, start, end, expand_position):
    #    raise NotImplementedError