

class Mstar():

    class Vertex():
        def __init__(self, v):
            assert type(v) == tuple, Exception("Input is not tuple")
            self.v = v
            self.cost_g = None
            self._collision_set = set()
            self._back_set = {}
        
        def add_back_set(self, new_v):
            assert isinstance(new_v, self), Exception("Input is not a Vertex class")

            if new_v.v in self._back_set:
                print("WARNING: overwriting vertex which already exist in _back_set")
            self._back_set[new_v.v] = new_v
        
        def add_collision(self, agent_handle):
            self._collision_set.add(agent_handle)
        def get_collision_set(self):
            return self._collision_set

    def __init__(self, start, end, expand_position, get_next_joint_policy_position):
        '''
        This class implements subdimensional expansion with a star as the search algorithm.
        It assumes the following functions which are external to the class:
            -- expand_position: returns the neighbouring vertices of a single position
            -- get_next_joint_policy_position: Returns the next vertex of a particular agents 
                                        joint policy action
                                        where the joint policy is the shortest path action
                                        where there is no other agents.
         '''
        assert type(start) == list, Exception("start parameter has to be list")
        assert type(end) == list, Exception("end parameter has to be list")
        assert len(start) == len(end), Exception("start and end positions have to be of same length")

        self.v_len = len(start)
        self.expand_position = expand_position
        self.get_next_joint_policy_position = get_next_joint_policy_position

        self.start = start


    # def test(self):
    #     for i in range(5):
    #         print({i:self.get_next_joint_policy_position(i, self.start[i]) for i in range(self.v_len)})

    def search(self, start, end):
        assert type(start) == list, Exception("start parameter has to be list")
        assert type(end) == list, Exception("end parameter has to be list")
        assert len(start) == len(end), Exception("start and end positions have to be of same length")


    def _combine_tuples(self, positions):
        '''Takes a list of tuples and returns a single tuple of tuples'''
        return tuple(positions)




    
    #def a_star(self, start, end, expand_position):
    #    raise NotImplementedError