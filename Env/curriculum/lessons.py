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
            self.use_custom_rewards = False
     
    curr = []
    for ob_dens in [0.0, 0.1, 0.2]:
        for a in range(6):
            hldr = currBase(a+1, ob_dens)
            curr.append(hldr)
    curr = {i:c for i, c in enumerate(curr)}
    return curr


def ppo_cl_4():
    map_size = 5
    class currBase():
        def __init__(self, n_agents, obj_density):
            self.env_name = "independent_navigation-v0"
            self.map_shape = (map_size, map_size)
            self.n_agents = n_agents
            self.obj_density = obj_density
            self.name = "n_agents" + str(n_agents) + "_obj_density" + str(obj_density)
            self.use_custom_rewards = False
            self.custom_env_ind = -1

    class currBase2():
        def __init__(self, ind = 0):
            self.env_name = "ind_navigation_custom-v0"
            self.map_shape = (map_size, map_size)
            self.n_agents = 4
            self.obj_density = 0.0
            self.name = "Env_" + self.env_name
            self.use_custom_rewards = False
            self.custom_env_ind = ind
     
    curr = []
    for ob_dens in [0.0, 0.1, 0.2]:
        for a in range(2,9):
            hldr = currBase(a, ob_dens)
            curr.append(hldr)
    for i in range(7):
        hldr = currBase2(i)
        curr.append(hldr)
        
    curr = {i:c for i, c in enumerate(curr)}
    return curr


def ppo_cl_inc_size():
    # map_size = 5
    class currBase():
        def __init__(self, n_agents, obj_density, map_size):
            self.env_name = "independent_navigation-v8_0"
            self.map_shape = (map_size, map_size)
            self.n_agents = n_agents
            self.view_d = 3
            self.obj_density = obj_density
            self.name = "mapsize_" + str(map_size) + "nagents" + str(n_agents) + "_objdensity" + str(obj_density)
            self.use_custom_rewards = False
            self.custom_env_ind = -1

    # class currBase2():
    #     def __init__(self, ind = 0):
    #         self.env_name = "ind_navigation_custom-v0"
    #         self.map_shape = (map_size, map_size)
    #         self.n_agents = 4
    #         self.obj_density = 0.0
    #         self.name = "Env_" + self.env_name
    #         self.use_custom_rewards = False
    #         self.custom_env_ind = ind
     
    curr = []
    c_env_size = [5,7,10]
    c_obs_dens = [0.0,0.1,0.2]
    c_agents = [2,4]

    for agents in c_agents:
        for dens in c_obs_dens:
            for size in c_env_size:
                hldr = currBase(agents, dens, size)
                curr.append(hldr)


    # for ob_dens in [0.0, 0.1, 0.2]:
    #     for a in range(2,9):
    #         hldr = currBase(a, ob_dens)
    #         curr.append(hldr)
    # for i in range(7):
    #     hldr = currBase2(i)
    #     curr.append(hldr)
        
    curr = {i:c for i, c in enumerate(curr)}
    return curr


def ppo_cl_inc_size_dirvec():
    # map_size = 5
    class currBase():
        def __init__(self, n_agents, obj_density, map_size):
            self.env_name = "independent_navigation-v8_1"
            self.map_shape = (map_size, map_size)
            self.n_agents = n_agents
            self.view_d = 3
            self.obj_density = obj_density
            self.name = "mapsize_" + str(map_size) + "nagents" + str(n_agents) + "_objdensity" + str(obj_density)
            self.use_custom_rewards = False
            self.custom_env_ind = -1

    # class currBase2():
    #     def __init__(self, ind = 0):
    #         self.env_name = "ind_navigation_custom-v0"
    #         self.map_shape = (map_size, map_size)
    #         self.n_agents = 4
    #         self.obj_density = 0.0
    #         self.name = "Env_" + self.env_name
    #         self.use_custom_rewards = False
    #         self.custom_env_ind = ind
     
    curr = []
    c_env_size = [5,7,10]
    c_obs_dens = [0.0,0.1,0.2]
    c_agents = [2,4]

    for agents in c_agents:
        for dens in c_obs_dens:
            for size in c_env_size:
                hldr = currBase(agents, dens, size)
                curr.append(hldr)


    # for ob_dens in [0.0, 0.1, 0.2]:
    #     for a in range(2,9):
    #         hldr = currBase(a, ob_dens)
    #         curr.append(hldr)
    # for i in range(7):
    #     hldr = currBase2(i)
    #     curr.append(hldr)
        
    curr = {i:c for i, c in enumerate(curr)}
    return curr
    