import argparse 

def ppo_cn():
    map_size = 5
    class currBase():
        def __init__(self, n_agents, obj_density):
            self.env_name = "cooperative_navigation-v0"
            self.map_shape = (map_size, map_size)
            self.n_agents = n_agents
            self.obj_density = obj_density
            self.name = "n_agents" + str(n_agents) + "_obj_density" + str(obj_density)
    
    curr = []
    for ob_dens in [0.0, 0.1, 0.2]:
        for a in range(6):
            hldr = currBase(a+1, ob_dens)
            curr.append(hldr)
    curr = {i:c for i, c in enumerate(curr)}
    return curr


    