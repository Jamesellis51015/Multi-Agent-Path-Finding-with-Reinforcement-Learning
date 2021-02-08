from queue import PriorityQueue
from sklearn.model_selection import ParameterGrid
import copy
import heapq

class Vertex():
    def __init__(self, v_id):
        assert type(v_id) == tuple
        assert len(v_id) == 2

        self.v_id = v_id
        self.collision_set = set()
        self.g = 1e6
        self.f = None
        self.back_set = dict()
        self.back_ptr = None
        
    @property
    def is_standard(self):
        return self.v_id[0] == self.v_id[1]
    
    def add_collision(self, other_collision_set):
        self.collision_set = self.collision_set.union(other_collision_set)

    def is_col_subset(self, other_set):
        return other_set.issubset(self.collision_set)

    def add_back_set(self, new_v):
        assert isinstance(new_v, type(self)), "Input is not a Vertex class"
        self.back_set[new_v.v_id] = new_v
    
    def get_back_set(self):
        return self.back_set.values()
    
    #For priority que:
    def __eq__(self, other_v):
        return self.g == other_v.g
    def __gt__(self, other_v):
        return self.g > other_v.g
    def __ge__(self, other_v):
        return self.g >= other_v.g
    def __lt__(self, other_v):
        return self.g < other_v.g
    def __le__(self, other_v):
        return self.g <= other_v.g


class SimplePriorityQ():
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

class PriorityQueue2(SimplePriorityQ):
    '''PQ which implements __contains__ member'''
    def __init__(self):
        super().__init__()
        self.lookup_table = {}
    def add_lookup(self, item):
        if item[-1].v_id in self.lookup_table:
            self.lookup_table[item[-1].v_id] += 1
        else:
            self.lookup_table[item[-1].v_id] = 1
    def remove_lookup(self, v):
        if v.v_id in self.lookup_table:
            if self.lookup_table[v.v_id] < 2:
                del self.lookup_table[v.v_id]
            else:
                self.lookup_table[v.v_id] -= 1
    def push(self, item):
        super().push(item)
        self.add_lookup(item)
    def pop(self):
        result = super().pop()
        self.remove_lookup(result)
        return result
    # def get(self):
    #     result = super().pop()
    #     self.remove_lookup(result)
    #     return result
    def __contains__(self, key):
        return key.v_id in self.lookup_table

    def __len__(self):
        return len(self.q)

class AllVertex():
    '''Keeps track of all nodes created
        such that nodes are created only once '''
    def __init__(self):
        self.all_v = dict()
        #self.intermediate = use_intermediate_nodes
    def get(self, v_id):
        if v_id in self.all_v:
            return self.all_v[v_id]
        else:
            self.all_v[v_id] = Vertex(v_id)
            return self.all_v[v_id]
    


class Mstar_OD():
    def __init__(self, start, end, expand_position, get_next_joint_policy_position, get_shortest_path_cost, inflation = 1.0):
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
       # assert type(start) == list, "start parameter has to be list"
       # assert type(end) == list, "end parameter has to be list"
        assert len(start) == len(end), "start and end positions have to be of same length"

        start = tuple(start)
        end = tuple(end)
        self.start_pos = start
        self.end_pos = end
        self.v_len = len(start)
       # self.f_e_kl = self.v_len * 1 #cost of traversing an edge
        self.expand_position = expand_position
        self.get_next_joint_policy_position = get_next_joint_policy_position
        self.heuristic_shortest_path_cost = get_shortest_path_cost
        self.all_v = AllVertex()
        self.inflation = inflation
    
    def search(self, OD = True, memory_limit = None):
        open = PriorityQueue2() #SimplePriorityQ()
        start_v = (self.start_pos, self.start_pos)
        end_v = (self.end_pos, self.end_pos)
        vs = self.all_v.get(start_v)
        vs.g = 0
        vs.f = vs.g + self.heuristic_SIC(vs.v_id)
        open.push((vs.f, vs))
        if OD:
            expand_function = self.expand_OD
        else:
            expand_function = self.expand_joint_actions

        while not open.empty():
            if memory_limit is not None:
                if len(open) > memory_limit:
                    del open
                    return None
                    
            vk = open.pop()
            if vk.v_id == end_v:
                return self._back_track(vk)
            for vl_id in expand_function(vk):
                # Intermediate nodes not part of backprop
                #For standard v only
                vl = self.all_v.get(vl_id)
                v_pos = vl.v_id[-1]
                col = self._is_pos_colliding(v_pos)
                if vl.is_standard:
                    vl.add_back_set(vk)
                    vl.add_collision(col)
                    self._backprop(vk, vl.collision_set, open)
                if (len(col) == 0 or vl.is_standard==False) and vk.g + self.get_move_cost(vk,vl) < vl.g:
                    vl.g = vk.g + self.get_move_cost(vk,vl)
                    vl.f = vl.g + self.heuristic_SIC(vl.v_id)
                    vl.back_ptr = vk
                    open.push((vl.f, vl))
        return None

    def _backprop(self, v_k, c_l, open):
        #NB check that not intermediate node
        if v_k.is_standard:
            if not c_l.issubset(v_k.collision_set):
                v_k.add_collision(c_l)
                if not v_k in open:
                    priority = v_k.g + self.heuristic_SIC(v_k.v_id)
                    open.push((priority, v_k))
                for v_m in v_k.get_back_set():
                    self._backprop(v_m, v_k.collision_set, open)


    def heuristic_SIC(self, v_id):
        #Need to check which positions has been assigned and which not for intermediate nodes
        (inter_tup, vertex_pos_tup) = v_id
        total_cost = 0
        for i, pos in enumerate(inter_tup):
            if pos == "_":
                total_cost += self.heuristic_shortest_path_cost(i, vertex_pos_tup[i])
            else:
                total_cost += self.heuristic_shortest_path_cost(i, pos)
        return total_cost * self.inflation


    
    def _is_pos_colliding(self, v_pos):
        '''Returns set of coll agents '''
        hldr = set()
        for i, vi in enumerate(v_pos):
            for i2, vi2 in enumerate(v_pos):
                if i != i2:
                    if vi == vi2:
                        hldr.add(i)
                        hldr.add(i2)
        return hldr 

    def get_move_cost(self, vk, vn):
        '''Cost of moving from vertex vk to vn '''
        #It is possible for vk and vn to both be standard nodes. Need to account for this in cost
        # Due to subdimensional expansion, expanded node neighbours are not always 1 appart. 
        # eg. expanding a standard node where x agents follow individually optimal policies
        end = list(self.end_pos)
        if vk.is_standard:
            if vn.is_standard:
                cost = self.v_len
                #count number of transitions from goal to goal pos
                num_agents_stay_on_goal = 0
                for gp, pk,pn in zip(end, vk.v_id[0], vn.v_id[0]):
                    if pk == gp and pn == gp:
                        num_agents_stay_on_goal += 1
                cost -= num_agents_stay_on_goal
                assert cost >= 0
            else:
                #vk should be root node of vn
                assert vk.v_id[1] == vn.v_id[1]
                cnt_vn = 0
                for g, pn,pk in zip(end, vn.v_id[0], vk.v_id[0]):
                    if not pn == '_':
                        if pn == g and pk == g: #if agent stayed on goal
                            cnt_vn += 0
                        else:
                            cnt_vn += 1
                cost = cnt_vn
        else:
            if vn.is_standard:
                num_pos_canged = 0
                cost = 0
                for gp, pk, pn, pk_root in zip(end, vk.v_id[0], vn.v_id[0], vk.v_id[1]):
                    if pk == '_':
                        assert not pn == "_"
                        num_pos_canged += 1
                        if pn == gp and pk_root == gp:
                            cost += 0
                        else:
                            cost += 1
                assert num_pos_canged == 1
            else:
                num_pos_canged = 0
                cost = 0
                for gp, pk, pn, pk_root in zip(end, vk.v_id[0], vn.v_id[0], vk.v_id[1]):
                    if pk == '_' and not pn == "_":
                        num_pos_canged += 1
                        if pn == gp and pk_root == gp:
                            cost += 0
                        else:
                            cost += 1
                assert num_pos_canged == 1
        
        assert cost >= 0 #vn should always be of higher count
        return cost
                        

    def expand_OD(self, v):
        (inter_tup, vertex_pos_tup) = v.v_id
        collision_set = set()
        next_inter_tup = [] #list(inter_tup)
        # If standard node create next intermediate node base
        # else convert current inter_tup to list
        if not "_" in inter_tup: #if stadard node
            assert(v.is_standard)
            collision_set = v.collision_set
            for i,p in enumerate(inter_tup):
                if i in collision_set:
                    next_inter_tup.append("_")
                else:
                    n_pos = self.get_next_joint_policy_position(i, p, self.end_pos[i])
                    next_inter_tup.append(n_pos[-1])
        else:
            next_inter_tup = list(inter_tup)
        
        #Deterimine intermediate node level
        this_inter_level = None
        for i,p in enumerate(next_inter_tup):
            if p == '_':
                this_inter_level = i
                break
        
        all_next_inter_tup = []
        if not this_inter_level is None:
            #if not a standard vertex
            pos = vertex_pos_tup[this_inter_level]
            positions_taken = [p for p in next_inter_tup if p != '_']
            n_pos = self.expand_position(i, pos)
            valid_n_pos = [p for p in n_pos if not p in positions_taken]

            if len(valid_n_pos) == 0:
                return []
                #If no valid positions, produce coliding vertex
                #valid_n_pos = [vertex_pos_tup[this_inter_level]]

            for p in valid_n_pos:
                next_inter_tup[this_inter_level] = p 
                all_next_inter_tup.append(tuple(next_inter_tup))
        else:
            all_next_inter_tup.append(tuple(next_inter_tup))
            assert not "_" in next_inter_tup #should be standard node

        #Make v_id's:
        v_ids = []
        for inter_v in all_next_inter_tup:
            if not "_" in inter_v:
                v_ids.append((tuple(inter_v), tuple(inter_v)))
            else:
                v_ids.append((tuple(inter_v), vertex_pos_tup))
        
        return v_ids
    

    def expand_joint_actions(self, v):
        (inter_tup, vertex_pos_tup) = v.v_id
        assert inter_tup == vertex_pos_tup

        num_agents = len(vertex_pos_tup)
        all_positions = dict()
        collisions = v.collision_set

        for i,p in enumerate(vertex_pos_tup):
            if i in collisions:

                all_positions[i] = self.expand_position(i, p)
            else:
                n_pos = self.get_next_joint_policy_position(i, p)
                #assert type(n_pos) == list
                all_positions[i] = n_pos#self.get_next_joint_policy_position(i, pos)
            
        joint_positions = ParameterGrid(all_positions)

        next_v_id = []
        for j_pos in joint_positions:
            v_id = tuple([j_pos[i] for i in range(num_agents)])
            v_id = (v_id, v_id)
            next_v_id.append(v_id)

        return next_v_id
            
            


    def _back_track(self, goal_v):
        '''Returns a dictionary of actions for the optimal path '''
        self.pos_act = {(0,1):2,
                        (1,0):3,
                        (0,-1):4,
                        (-1,0):1,
                        (0,0): 0}
        
        #get vertices:
        all_v = []
        all_v.append(goal_v.v_id[-1])
        next_v = goal_v.back_ptr
        while not next_v is None:
            if next_v.is_standard:
                all_v.append(next_v.v_id[-1])
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



