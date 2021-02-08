from queue import PriorityQueue
from sklearn.model_selection import ParameterGrid
import copy
import heapq
import time

class Vertex():
    def __init__(self, v_id):
        assert type(v_id) == tuple
        assert len(v_id) == 2

        self.v_id = v_id
        self.collision_set = frozenset()
        self.g = 1e6
        self.f = None
        self.back_set = dict()
        self.back_ptr = None
        self.forward_pointer = None
        
    @property
    def is_standard(self):
        return self.v_id[0] == self.v_id[1]
    
    def add_collision(self, other_collision_set):
       # print("other col set: {}".format(other_collision_set))
        other_col_set2 = []
        # make work with single set or sets of sets:
        for temp in other_collision_set:
            if type(temp) == frozenset:
                other_col_set2.append(temp)
            else:
                other_col_set2.append(other_collision_set)
                break
        #assert len(other_collision_set) > 1
        assert self._check_valid_collision_set() == True
        for other_collision_set in other_col_set2:
            to_merge = []
            keep_same = []
            for i in other_collision_set:
                for existing_set in self.collision_set:
                        if i in existing_set:
                            if not existing_set in to_merge:
                                    to_merge.append(existing_set)
                        else:
                            keep_same.append(existing_set)
            to_merge.append(other_collision_set)
            #remove sets in keep_same which is also in to_marge
            keep_same = [k for k in keep_same if not k in to_merge]
            if len(to_merge) == 0:
                keep_same.append(other_collision_set)
            else:
                big_set = []
                for item in to_merge:
                        for i2 in item:
                            big_set.append(i2)
                big_set = frozenset(big_set)
            new_col_set = []
            new_col_set.append(big_set)
            for k in keep_same:
                new_col_set.append(k)
            new_col_set = frozenset(new_col_set)
            self.collision_set = new_col_set
       # print("Added col set {}".format(self.collision_set))
        # if len(other_collision_set) == 0:
        #     return
        # #assert len(other_collision_set) > 1
        # assert self._check_valid_collision_set() == True
        # to_merge = []
        # keep_same = []
        # for i in other_collision_set:
        #         for existing_set in self.collision_set:
        #             if i in existing_set:
        #                     if not existing_set in to_merge:
        #                         to_merge.append(existing_set)
        #             else:
        #                     keep_same.append(existing_set)
        # to_merge.append(other_collision_set)
        # #remove sets in keep_same which is also in to_marge
        # keep_same = [k for k in keep_same if not k in to_merge]
        # if len(to_merge) == 0:
        #         keep_same.append(other_collision_set)
        # else:
        #         big_set = []
        #         for item in to_merge:
        #             for i2 in item:
        #                     big_set.append(i2)
        #         big_set = frozenset(big_set)
        # new_col_set = []
        # new_col_set.append(big_set)
        # for k in keep_same:
        #         new_col_set.append(k)
        # new_col_set = frozenset(new_col_set)
        # self.collision_set = new_col_set
        # print("new col set: {}".format(new_col_set))

    def _check_valid_collision_set(self):
        temp = dict()
        flag = True
        for item in self.collision_set:
            if type(item) == frozenset:
                for it in item:
                    if not it in temp:
                        temp[it] = it
                    else:
                        flag = False
            else:
                raise Exception("invalid object type in collision set")
        return flag

    # def is_col_subset(self, other_set):
    #     flag = False
    #     for sub in self.collision_set:
    #         if other_set.issubset(sub):
    #             flag = True
    #     if len(other_set) == 0:
    #               flag = True
    #     return flag
    def is_col_subset(self, other_set):
        #make work with set of sets and single set:
        other_set2 = []
        is_sets_of_sets = [True if type(i) == frozenset else False for i in other_set]
        if True in is_sets_of_sets:
                assert all(is_sets_of_sets)
        if False in is_sets_of_sets:
                is_sets_of_sets_cpy = [True if i==False else True for i in is_sets_of_sets]
                assert all(is_sets_of_sets_cpy)
        
        if all(is_sets_of_sets) and len(is_sets_of_sets)!=0:
                for s in other_set:
                    other_set2.append(s)
        else:
                other_set2.append(other_set)

        col_set_flat = []
        is_sets_of_sets = [True if type(i) == frozenset else False for i in self.collision_set]
        if True in is_sets_of_sets:
                assert all(is_sets_of_sets)
        if False in is_sets_of_sets:
                is_sets_of_sets_cpy = [True if i==False else True for i in is_sets_of_sets]
                assert all(is_sets_of_sets_cpy)

        if all(is_sets_of_sets) and len(is_sets_of_sets)!=0:
                for s in self.collision_set:
                    col_set_flat.append(s)
        else:
                col_set_flat.append(self.collision_set)

        #Check if otherset is subset of collision set
        other_set_dict = {other: False for other in other_set2}
        for k in other_set_dict.keys():
                for sub in col_set_flat:
                    if k.issubset(sub):
                            other_set_dict[k] = True
        flag = all(list(other_set_dict.values()))
        return flag


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
    


class Mstar_ODr():
    def __init__(self, end, expand_position, get_next_joint_policy_position, get_shortest_path_cost, sub_graphs = None, inflation = None):
        '''
        This class implements subdimensional expansion with a star as the search algorithm.
        It assumes the following functions which are external to the class:
            -- expand_position: returns the neighbouring vertices of a single position
            -- get_next_joint_policy_position: Returns the next vertex of a particular agents 
                                        joint policy action
                                        where the joint policy is the shortest path action
                                        where there is no other agents.
            -- het_shortest_path_cost = the shortest path cost of single agent.
            -- get_SIC: returns the sum of individual cost (individual 
                        optimal path cost from vertex vk to vf)
         '''
       # assert type(start) == list, "start parameter has to be list"
       # assert type(end) == list, "end parameter has to be list"
        #assert len(start) == len(end), "start and end positions have to be of same length"

       # start = tuple(start)
        if type(end) == list:
            end = tuple(end)
        
        if type(end) == tuple:
            end = {i:v for i,v in enumerate(end)}

        assert type(end) == dict
        self.v_len = len(end)
        self.end_dict = end
        self.agent_ids = frozenset(end.keys())
        self.expand_position = expand_position
        self.get_next_joint_policy_position = get_next_joint_policy_position
        self.heuristic_shortest_path_cost = get_shortest_path_cost
        self.all_v = AllVertex()

        if inflation is None:
            self.inflation = 1.0
        else:
            self.inflation = inflation


        if sub_graphs is None:
            self.sub_graphs = dict()
        else:
            self.sub_graphs = sub_graphs
        
        self._init_own_sub_graph()

        
       # print("init")

        
    
    def _init_own_sub_graph(self):
        hldr = list(self.end_dict.keys())
        hldr.sort()
        own_id = tuple(hldr)
        #this method should only be called once
        if not own_id in self.sub_graphs:
            self.sub_graphs[own_id] = self.retrieve_next_optimal_pos
            print("Creating sub graph: {}".format(own_id))
        else:
            print("_init_own_sub_graph called but id already in sub_graph")


    def query_sub_graph_optimal_policy(self, this_graph_sub_id, sub_start_v):
        '''this_graph_sub_id the collision set in this instance of rM*. 
            sub_start_v the full position tuple of the vertex which has the collision '''
            #map this_graph_sub_id to global ids
            # create sub_start_dic and sub_end_dict
            # get next sub_graph position
            # map next_sub_graph position back to dict with keys of this_sub_graph_id
        if type(this_graph_sub_id) == int:
            this_graph_sub_id = frozenset([this_graph_sub_id])
        assert type(this_graph_sub_id) == frozenset
        def sort_iterable(variable):
            a = list(variable)
            a.sort()
            return a
        this_graph_sub_id = frozenset(sort_iterable(this_graph_sub_id))
        #assume v positions always arranged in ascending order of global id keys
        true_ids = sort_iterable(self.end_dict.keys())
        sub_start_dict = dict()
        for id in this_graph_sub_id:
            sub_start_dict[true_ids[id]] = sub_start_v[id]
        
        sub_end_dict = dict()
        for id in this_graph_sub_id:
            sub_end_dict[true_ids[id]] = self.end_dict[true_ids[id]]




        #Graphs of higher dimension than self should not be queried.
        
       # assert type(graph_id) == frozenset or type(graph_id) == int
        #if type(graph_id) == set:
        graph_id = tuple(sort_iterable(sub_start_dict.keys()))
        assert graph_id == tuple(sort_iterable(sub_end_dict.keys()))
        
        # NB: need function to rerieve shortest path cost for single agent.

        if not graph_id in self.sub_graphs:
            #Init sub graph
            if len(graph_id) > 1:
                print("End dict is {}".format(sub_end_dict))
                temp = type(self)(sub_end_dict, self.expand_position, \
                    self.get_next_joint_policy_position, self.heuristic_shortest_path_cost, self.sub_graphs, inflation=self.inflation)
                if not graph_id in self.sub_graphs:
                    print("Id is: {}  Sub end dict is: {}".format(graph_id, sub_end_dict))
                next_sub_v = self.sub_graphs[graph_id](sub_start_dict, sub_end_dict)
            elif len(graph_id) == 1:
                agent_id = graph_id[0]
                pos = sub_start_dict[agent_id]
                next_sub_v = {agent_id: self.get_next_joint_policy_position(agent_id, pos)[-1]}
            else:
                raise Exception("Graph id has to be len >= 1")
        else:
            assert len(graph_id) > 1
            next_sub_v = self.sub_graphs[graph_id](sub_start_dict, sub_end_dict)
        
        #map glabal keys back to relative keys:
        next_sub_v_relative_id = {}
        this_graph_sub_id_keys =[k for k in this_graph_sub_id] #list(this_graph_sub_id.keys())
        this_graph_sub_id_keys.sort()
        for i,val in enumerate(next_sub_v.values()):
            next_sub_v_relative_id[this_graph_sub_id_keys[i]] = val
        return next_sub_v_relative_id


    def retrieve_next_optimal_pos(self, start_dict, end_dict):
        ''' For this methods own vetices and graph. '''
        # assert self.end == end #end should be same

        #check if start in all_v dict. If not do search.
        # if start in all_v dict: check if has forward pointer; if not do search.
        #return dict of {agent_id: vertex_tuple}
        #NB also check that solution was found if search is done
        def sort_iterable(variable):
            a = list(variable)
            a.sort()
            return a
        
        assert self.end_dict == end_dict
        assert type(start_dict) == dict
        assert set(start_dict.keys()) == set(end_dict.keys())
        #print("In retrieve_next_optimal_pos")
        start_tup = []
        for k in sort_iterable(start_dict.keys()):
            start_tup.append(start_dict[k])
        start_tup = tuple(start_tup)
        #print("start_dict: {}     start_tup: {}".format(start_dict, start_tup))

        end_tup = []
        for k in sort_iterable(end_dict.keys()):
            end_tup.append(end_dict[k])
        end_tup = tuple(end_tup)
        #print("end_dict: {}     end_tup: {}".format(end_dict, end_tup))

        start_v = (start_tup, start_tup)
        end_v = (end_tup, end_tup)
        actions = self.search(start_tup, end_tup)
        v = self.all_v.all_v[start_v]
        next_v_tup = None
        if v.v_id == end_v:
            next_v_tup = start_v
        elif v.forward_pointer is None:
            assert actions is None
            next_v_tup = None
            #print("forward_pointer is None")
        else:
            
            #Makes sure next_v is standard vertex
            next_v = v.forward_pointer
            #print("After search, getting v.fwdptr. Next)v is: {} is_standard: {}".format(next_v.v_id, next_v.is_standard))
            cntr = 0
            while (not next_v is None) and next_v.is_standard == False:
                next_v = next_v.forward_pointer
                #print("next_d: {}   is_standard {}".format(next_v.v_id, next_v.is_standard))
                cntr += 1
                if cntr > 50000:
                    raise Exception("Infinite while loop")
            #time.sleep(10)
            #print("while loop done")
            if next_v is None:
                next_v_tup = None
            else:
                next_v_tup = next_v.v_id
        #print("prev_v is {} || next v is  {}".format(v.v_id, next_v_tup))
        if not next_v_tup is None:
            #assert next_v.is_standard == True
            assert next_v_tup[0] == next_v_tup[1]
            next_v_dict = {}
            inter_v, root_v = next_v_tup
            for k, inter_pos, root_pos in zip(sort_iterable(end_dict.keys()), inter_v, root_v):
                next_v_dict[k] = inter_pos #(inter_pos, root_pos)
        else:
            next_v_dict = None
        #print("next_v_dict is {}".format(next_v_dict))
       # time.sleep(2)
        return next_v_dict

    
    def search(self,start_pos, end_pos, OD = True):
        #print("Begin search in graph: {}".format(tuple(list(self.end_dict.keys()))))
        # assert self.end == end_pos
        open = PriorityQueue2() #SimplePriorityQ()
        start_v = (start_pos, start_pos)
        end_v = (end_pos, end_pos)
        #self.all_v = AllVertex()
        if len(self.all_v.all_v) > 1:
            for k,v in self.all_v.all_v.items():
                v.collision_set = frozenset()
                v.g = 1e6
                v.f = None
                v.back_set = dict()
                v.back_ptr = None
        
        vs = self.all_v.get(start_v)
        #####################
        # 
        # if vs.v_id == end_v or not vs.forward_pointer is None:
        #     print("Solution found")
        #     #self._set_forward_pointers(vs)
        #     if vs.v_id == end_v:
        #         #print("vk.v_id == end_v")
        #         #print("vk.v_id : {}  end_v: {}".format(vk.v_id, end_v))
        #         return self._back_track(vs)
        #     else:
        #         #print("fwd ptr not none")
        #         cntr1 = 0
        #         while not vs.forward_pointer is None:
        #             vs = vs.forward_pointer
        #             cntr1 +=1
        #             assert cntr1 < 50000
        #         print("goal node found")
        #         assert vs.v_id == end_v
        #         return self._back_track(vs)



        #####################
        vs.g = 0
        vs.f = vs.g + self.heuristic_SIC(vs.v_id)
        open.push((vs.f, vs))
        if OD:
            expand_function = self.expand_rOD
        else:
            print("Not implemented")
            #raise NotImplementedError
            #expand_function = self.expand_joint_actions

        while not open.empty():
            vk = open.pop()
            test = vk.v_id
            #print("vk from open: {}".format(test))
            if vk.v_id == end_v or vk.forward_pointer is not None:
                #print("Solution found")
                #self._set_forward_pointers(vk)
                if vk.v_id == end_v:
                    self._set_forward_pointers(vk)
                    #print("vk.v_id == end_v")
                    #print("vk.v_id : {}  end_v: {}".format(vk.v_id, end_v))
                    return self._back_track(vk)
                else:
                    self._set_forward_pointers(vk)
                    #print("fwd ptr not none: {}".format(vk.forward_pointer.v_id))
                    cntr1 = 0
                    while vk.forward_pointer is not None:
                        vk = vk.forward_pointer
                        cntr1 +=1
                        assert cntr1 < 50000
                    #print("goal node found")
                    assert vk.v_id == end_v
                    return self._back_track(vk)
                
            for vl_id in expand_function(vk, self.end_dict):
                # Intermediate nodes not part of backprop
                #For standard v only
                vl = self.all_v.get(vl_id)
                v_pos = vl.v_id[-1]
                col = self._is_pos_colliding(v_pos)
                if vl.is_standard:
                    vl.add_back_set(vk)
                    #print("vl_col before: {}  col to be added: {}".format(vl.collision_set, col))
                    vl.add_collision(col)
                    #print("vl_col after: {}  col was added: {}".format(vl.collision_set, col))
                    self._backprop(vk, vl.collision_set, open)
                if (len(col) == 0 or vl.is_standard==False) and vk.g + self.get_move_cost(vk,vl, end_pos) < vl.g:
                    vl.g = vk.g + self.get_move_cost(vk,vl, end_pos)
                    vl.f = vl.g + self.heuristic_SIC(vl.v_id)
                    vl.back_ptr = vk
                    open.push((vl.f, vl))
                    #print("V added to open: {}".format(vl.v_id))
        print("returning no solution")
        print("This graph is {}".format(tuple(list(self.end_dict.keys()))))
        for k,v in self.all_v.all_v.items():
            f_ptr = None
            if not v.forward_pointer is None:
                f_ptr = v.forward_pointer.v_id
            print("V: {}   Forward_ptr: {} ".format(k, f_ptr))
        return None

    # def _col_is_none(self, other_set):
    #     flag = True
    #     other_set2 = []
    #     is_sets_of_sets = [True if type(i) == frozenset else False for i in other_set]
    #     if True in is_sets_of_sets:
    #             assert all(is_sets_of_sets)
    #     if False in is_sets_of_sets:
    #             is_sets_of_sets_cpy = [True if i==False else True for i in is_sets_of_sets]
    #             assert all(is_sets_of_sets_cpy)
        
    #     if all(is_sets_of_sets) and len(is_sets_of_sets)!=0:
    #             for s in other_set:
    #                 other_set2.append(s)
    #     else:
    #             other_set2.append(other_set)
    #     for s in other_set2:
    #             if len(s) != 0:
    #                 flag = False
    #     return flag

    def _backprop(self, v_k, c_l, open):
        #NB check that not intermediate node
        #print("in backprop, col set {}".format(c_l))
        if v_k.is_standard:
            #if not c_l.issubset(v_k.collision_set):
            if not v_k.is_col_subset(c_l):
                #print("is subset false:  {}  vk_col set: {}".format(v_k.is_col_subset(c_l), v_k.collision_set))
               # print("In backprop vk is {} with col set {} Col being added: {}".format(v_k.v_id, v_k.collision_set, c_l))
                v_k.add_collision(c_l)
                if not v_k in open:
                    priority = v_k.g + self.heuristic_SIC(v_k.v_id)
                    open.push((priority, v_k))
                for v_m in v_k.get_back_set():
                    self._backprop(v_m, v_k.collision_set, open)
            #else:
                #print("is subset true")
        #else:
            #print("not standard v")


    def heuristic_SIC(self, v_id):
        #Need to check which positions has been assigned and which not for intermediate nodes
        (inter_tup, vertex_pos_tup) = v_id
        total_cost = 0
        true_id = list(self.end_dict.keys())
        for i, pos in enumerate(inter_tup):
            i_true = true_id[i]
            if pos == "_":
                total_cost += self.heuristic_shortest_path_cost(i_true, vertex_pos_tup[i])
            else:
                total_cost += self.heuristic_shortest_path_cost(i_true, pos)
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
        hldr = frozenset([i for i in hldr])
        return hldr 

    def get_move_cost(self, vk, vn, end_pos):
        '''Cost of moving from vertex vk to vn '''
        #It is possible for vk and vn to both be standard nodes. Need to account for this in cost
        # Due to subdimensional expansion, expanded node neighbours are not always 1 appart. 
        # eg. expanding a standard node where x agents follow individually optimal policies
        
        assert len(end_pos) == self.v_len
        end = list(end_pos)
        # Four possible conditions for vk and vn being either standard or
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
                        

    def expand_rOD(self, v, end_dict):
        #print("Expanding v: {} with end_dict: {}".format(v.v_id, end_dict))
        assert len(end_dict) == self.v_len
        (inter_tup, vertex_pos_tup) = v.v_id
        #collision_set = set()
        next_inter_tup = [] #list(inter_tup)
        # If standard node create next intermediate node base
        # else convert current inter_tup to list
        if not "_" in inter_tup: #if stadard node
            assert(v.is_standard)
            collision_set = v.collision_set
            next_tup = {i:None for i in range(self.v_len)}
            this_all_ids = frozenset([i for i in range(self.v_len)])
            for c in collision_set:
                if len(c) == len(self.agent_ids):
                    assert len(collision_set) == 1
                    assert c == this_all_ids
                    for i in this_all_ids:
                        next_tup[i] = "_"
                else:
                    n_p = self.query_sub_graph_optimal_policy(c, v.v_id[0])
                    hldr = set([i for i in c])
                    hldr2 = set(n_p.keys())
                    assert hldr == hldr2
                    #print("Sub graph optimal pol keys: {}".format(hldr))
                    for k,val in n_p.items():
                        assert next_tup[k] is None
                        next_tup[k] = val
            all_col_ids = []
            for c in collision_set:
                for c2 in c:
                    all_col_ids.append(c2)
            all_col_ids = frozenset(all_col_ids)
            diff = this_all_ids.difference(all_col_ids)

            #Get next shortest path position for non-colliding agents
            for d in diff:
                assert not "_" in next_inter_tup
                n_pos = self.query_sub_graph_optimal_policy(d, v.v_id[0])
                for k,val in n_pos.items():
                    assert next_tup[k] is None
                    next_tup[k] = val
            next_inter_tup = list(next_tup.values())
            assert not None in next_inter_tup

            ####
            # for i,p in enumerate(inter_tup):
            #     if i in collision_set:
            #         next_inter_tup.append("_")
            #     else:
            #         n_pos = self.get_next_joint_policy_position(i, p, end_pos[i])
            #         next_inter_tup.append(n_pos[-1])
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
            #print("before expand postion...")
            n_pos = self.expand_position(i, pos)
            #print("after expand_positions n_pos: {}".format(n_pos))
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
       # print("Expanded v: {}".format(v_ids))
        return v_ids
    

    def expand_joint_actions(self, v):
        raise Exception("This function should not be called")
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
            
            
    def _set_forward_pointers(self, goal_v):
        #print("begin _set_forward_pointers")
        this_v = goal_v
        while not this_v.back_ptr is None:
            back_v = this_v.back_ptr
            back_v.forward_pointer = this_v
            this_v = back_v
        #print("end_set_forward_pointers")


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
               # print("In _backtrack while: {}".format(next_v.v_id[-1]))
            next_v = next_v.back_ptr
       # print("exit while")
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
       # print("exit backtrack")
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



