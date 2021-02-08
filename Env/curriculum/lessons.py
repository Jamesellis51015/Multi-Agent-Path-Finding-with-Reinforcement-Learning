import argparse 
import numpy as np

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
    c_env_size = [10,15,20]
    c_obs_dens = [0.0,0.1]
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
    

def ppo_primal():
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
            self.ppo_heur_block = False
    
    all_curr = []
    all_curr.append(currBase(5, 0.2, 10))
    all_curr.append(currBase(10, 0.2, 20))
    all_curr.append(currBase(15, 0.2, 32))

    curr = {i:c for i, c in enumerate(all_curr)}
    return curr

def ppo_all_same(args):
    class currBase():
        def __init__(self, n_agents, obj_density, map_size):
            self.env_name = "independent_navigation-v8_1"
            self.map_shape = (map_size, map_size)
            self.n_agents = n_agents
            self.view_d = 4
            self.obj_density = obj_density
            self.name = "mapsize_" + str(map_size) + "nagents" + str(n_agents)# + "_objdensity" + str(obj_density)
            self.use_custom_rewards = False
            self.custom_env_ind = -1
            self.ppo_heur_block = True
        
        def sample_agents_obstacle_density(self):
            obs_density = np.random.triangular(0.0, 0.33, 0.5)
            self.obj_density = obs_density 



    
    all_curr = []
    all_curr.append(currBase(8, 0.2, 10))
    # for arg in vars(args):
    #     if not hasattr(all_curr[-1], arg):
    #         setattr()
    # for k, v in args_dict.items():
    #     if not hasattr(env_args, k):
    #         print("Adding new attribute: {}".format(k))
    #     setattr(env_args, k, v)
    all_curr.append(currBase(8, 0.2, 20))
    all_curr.append(currBase(8, 0.2, 35))

    curr = {i:c for i, c in enumerate(all_curr)}
    return curr

def ppo_all_same_no_sampling(args):
    class currBase():
        def __init__(self, n_agents, obj_density, map_size):
            self.env_name = "independent_navigation-v8_1"
            self.map_shape = (map_size, map_size)
            self.n_agents = n_agents
            self.view_d = 4
            self.obj_density = obj_density
            self.name = "mapsize_" + str(map_size) + "nagents" + str(n_agents)# + "_objdensity" + str(obj_density)
            self.use_custom_rewards = False
            self.custom_env_ind = -1
            self.ppo_heur_block = True
        
        def sample_agents_obstacle_density(self):
            #obs_density = np.random.triangular(0.0, 0.33, 0.5)
            obs_density = np.random.uniform(0.0, 0.5)
            self.obj_density = obs_density 



    
    all_curr = []
    all_curr.append(currBase(8, 0.2, 10))
    # for arg in vars(args):
    #     if not hasattr(all_curr[-1], arg):
    #         setattr()
    # for k, v in args_dict.items():
    #     if not hasattr(env_args, k):
    #         print("Adding new attribute: {}".format(k))
    #     setattr(env_args, k, v)
    all_curr.append(currBase(8, 0.2, 20))
    all_curr.append(currBase(8, 0.2, 35))

    curr = {i:c for i, c in enumerate(all_curr)}
    return curr



def ppo_all_same_no_sampling2(args):
    class currBase():
        def __init__(self, n_agents, obj_density, map_size):
            self.env_name = "independent_navigation-v8_1"
            self.map_shape = (map_size, map_size)
            self.n_agents = n_agents
            self.view_d = 4
            self.obj_density = obj_density
            self.name = "mapsize_" + str(map_size) + "nagents" + str(n_agents)  # + "_objdensity" + str(obj_density)
            self.use_custom_rewards = False
            self.custom_env_ind = -1
            self.ppo_heur_block = False
        
        def sample_agents_obstacle_density(self):
            #obs_density = np.random.triangular(0.0, 0.33, 0.5)
            obs_density = np.random.uniform(0.0, 0.5)
            self.obj_density = obs_density 



    
    all_curr = []
    all_curr.append(currBase(8, 0.2, 35))
    # for arg in vars(args):
    #     if not hasattr(all_curr[-1], arg):
    #         setattr()
    # for k, v in args_dict.items():
    #     if not hasattr(env_args, k):
    #         print("Adding new attribute: {}".format(k))
    #     setattr(env_args, k, v)
    all_curr.append(currBase(8, 0.2, 35))
    all_curr.append(currBase(8, 0.2, 35))

    curr = {i:c for i, c in enumerate(all_curr)}
    return curr

def ppo_final():
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

    
    all_curr = []
    obs_densities = [0.0,0.1, 0.2, 0.3]
    
    
    nagents = [2,4,6]
    for n in nagents:
        for ob in obs_densities:
            hldr = currBase(n, ob, 7)
            all_curr.append(hldr)

    nagents = [6,8,10]
    for n in nagents:
        for ob in obs_densities:
            hldr = currBase(n, ob, 10)
            all_curr.append(hldr)


    nagents = [10,14,18]
    for n in nagents:
        for ob in obs_densities:
            hldr = currBase(n, ob, 12)
            all_curr.append(hldr)

    nagents = [14,18,22]
    for n in nagents:
        for ob in obs_densities:
            hldr = currBase(n, ob, 14)
            all_curr.append(hldr)
    
    nagents = [18, 22, 26]
    for n in nagents:
        for ob in obs_densities:
            hldr = currBase(n, ob, 16)
            all_curr.append(hldr)

    nagents = [20, 24, 28]
    for n in nagents:
        for ob in obs_densities:
            hldr = currBase(n, ob, 18)
            all_curr.append(hldr)
    
    nagents = [24, 28, 30]
    for n in nagents:
        for ob in obs_densities:
            hldr = currBase(n, ob, 20)
            all_curr.append(hldr)

    nagents = [26, 30]
    for n in nagents:
        for ob in obs_densities:
            hldr = currBase(n, ob, 22)
            all_curr.append(hldr)
    
    nagents = [30]
    for n in nagents:
        for ob in obs_densities:
            hldr = currBase(n, ob, 24)
            all_curr.append(hldr)

    nagents = [30]
    for n in nagents:
        for ob in obs_densities:
            hldr = currBase(n, ob, 26)
            all_curr.append(hldr)

    
    nagents = [30]
    for n in nagents:
        for ob in obs_densities:
            hldr = currBase(n, ob, 28)
            all_curr.append(hldr)

    nagents = [30]
    for n in nagents:
        for ob in obs_densities:
            hldr = currBase(n, ob, 30)
            all_curr.append(hldr)

    
    nagents = [30]
    for n in nagents:
        for ob in obs_densities:
            hldr = currBase(n, ob, 32)
            all_curr.append(hldr)

        
    curr = {i:c for i, c in enumerate(all_curr)}
    return curr


def ppo_final2(): #NB Independent Navigation v8_0 or Independent Navigation V8_1
    # 1) Learn to follow shortest path to goal
    # 2) Learn to coordinate with other agents
    # 3) Learn to navigate with obstacles and other agents.
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

    
    all_curr = []
    obs_densities = [0.0,0.1, 0.2, 0.3]
    
    # 1) Learn to follow shortest path
    nagents = [1]
    obs_densities = [0.0]
    env_sizes = [5,7,10,12,14,16,18,20,22,24,26,28,30,32]
    for n in nagents:
        for ob in obs_densities:
            for en in env_sizes:
                hldr = currBase(n, ob, en)
                all_curr.append(hldr)

    #Learn to coordinate with other agents:
    nagents = [2,4,6,8,10]
    obs_densities = [0.0]
    env_sizes = [10]  
    for ob in obs_densities:
        for en in env_sizes:
            for n in nagents:
                hldr = currBase(n, ob, en)
                all_curr.append(hldr)
    #

    nagents = [6,8,10,12,14,16,18]
    obs_densities = [0.0]
    env_sizes = [12]
    for ob in obs_densities:
        for en in env_sizes:
            for n in nagents:
                hldr = currBase(n, ob, en)
                all_curr.append(hldr)

    #
    nagents = [10,12,14,16,18,20,22,24]
    obs_densities = [0.0]
    env_sizes = [14]
    for ob in obs_densities:
        for en in env_sizes:
            for n in nagents:
                hldr = currBase(n, ob, en)
                all_curr.append(hldr)

    #
    nagents = [10,12,14,16,18,20,22,24]
    obs_densities = [0.0]
    env_sizes = [16]
    for ob in obs_densities:
        for en in env_sizes:
            for n in nagents:
                hldr = currBase(n, ob, en)
                all_curr.append(hldr)
    
    #
    nagents = [10,12,14,16,18,20,22,24,26]
    obs_densities = [0.0]
    env_sizes = [18]
    for ob in obs_densities:
        for en in env_sizes:
            for n in nagents:
                hldr = currBase(n, ob, en)
                all_curr.append(hldr)

    #
    nagents = [12,14,16,18,20,22,24,26,28]
    obs_densities = [0.0]
    env_sizes = [20]
    for ob in obs_densities:
        for en in env_sizes:
            for n in nagents:
                hldr = currBase(n, ob, en)
                all_curr.append(hldr)
    #
    nagents = [12,14,16,18,20,22,24,26,28,30]
    obs_densities = [0.0]
    env_sizes = [22]
    for ob in obs_densities:
        for en in env_sizes:
            for n in nagents:
                hldr = currBase(n, ob, en)
                all_curr.append(hldr)

    #
    nagents = [12,14,16,18,20,22,24,26,28,30,32]
    obs_densities = [0.0]
    env_sizes = [24]
    for ob in obs_densities:
        for en in env_sizes:
            for n in nagents:
                hldr = currBase(n, ob, en)
                all_curr.append(hldr)

    #
    nagents = [12,14,16,18,20,22,24,26,28,30,32,34]
    obs_densities = [0.0]
    env_sizes = [26]
    for ob in obs_densities:
        for en in env_sizes:
            for n in nagents:
                hldr = currBase(n, ob, en)
                all_curr.append(hldr)

    #
    nagents = [12,14,16,18,20,22,24,26,28,30,32,34,36]
    obs_densities = [0.0]
    env_sizes = [28]
    for ob in obs_densities:
        for en in env_sizes:
            for n in nagents:
                hldr = currBase(n, ob, en)
                all_curr.append(hldr)

    #
    nagents = [14,16,18,20,22,24,26,28,30,32,34,36,38]
    obs_densities = [0.0]
    env_sizes = [30]
    for ob in obs_densities:
        for en in env_sizes:
            for n in nagents:
                hldr = currBase(n, ob, en)
                all_curr.append(hldr)

    #
    nagents = [16,18,20,22,24,26,28,30,32,34,36,38,40]
    obs_densities = [0.0]
    env_sizes = [32]
    for ob in obs_densities:
        for en in env_sizes:
            for n in nagents:
                hldr = currBase(n, ob, en)
                all_curr.append(hldr)

    #Learn to coordinate with other agents AND Obstacles:
    for ob_hldr2 in [0.1, 0.2, 0.3]:
        nagents = [2,4,6,8,10]
        obs_densities = [ob_hldr2]
        env_sizes = [10]  
        for ob in obs_densities:
            for en in env_sizes:
                for n in nagents:
                    hldr = currBase(n, ob, en)
                    all_curr.append(hldr)
        #

        nagents = [6,8,10,12,14,16,18]
        obs_densities = [ob_hldr2]
        env_sizes = [12]
        for ob in obs_densities:
            for en in env_sizes:
                for n in nagents:
                    hldr = currBase(n, ob, en)
                    all_curr.append(hldr)

        #
        nagents = [10,12,14,16,18,20,22,24]
        obs_densities = [ob_hldr2]
        env_sizes = [14]
        for ob in obs_densities:
            for en in env_sizes:
                for n in nagents:
                    hldr = currBase(n, ob, en)
                    all_curr.append(hldr)

        #
        nagents = [10,12,14,16,18,20,22,24]
        obs_densities = [ob_hldr2]
        env_sizes = [16]
        for ob in obs_densities:
            for en in env_sizes:
                for n in nagents:
                    hldr = currBase(n, ob, en)
                    all_curr.append(hldr)
        
        #
        nagents = [10,12,14,16,18,20,22,24,26]
        obs_densities = [ob_hldr2]
        env_sizes = [18]
        for ob in obs_densities:
            for en in env_sizes:
                for n in nagents:
                    hldr = currBase(n, ob, en)
                    all_curr.append(hldr)

        #
        nagents = [12,14,16,18,20,22,24,26,28]
        obs_densities = [ob_hldr2]
        env_sizes = [20]
        for ob in obs_densities:
            for en in env_sizes:
                for n in nagents:
                    hldr = currBase(n, ob, en)
                    all_curr.append(hldr)
        #
        nagents = [12,14,16,18,20,22,24,26,28,30]
        obs_densities = [ob_hldr2]
        env_sizes = [22]
        for ob in obs_densities:
            for en in env_sizes:
                for n in nagents:
                    hldr = currBase(n, ob, en)
                    all_curr.append(hldr)

        #
        nagents = [12,14,16,18,20,22,24,26,28,30,32]
        obs_densities = [ob_hldr2]
        env_sizes = [24]
        for ob in obs_densities:
            for en in env_sizes:
                for n in nagents:
                    hldr = currBase(n, ob, en)
                    all_curr.append(hldr)

        #
        nagents = [12,14,16,18,20,22,24,26,28,30,32,34]
        obs_densities = [ob_hldr2]
        env_sizes = [26]
        for ob in obs_densities:
            for en in env_sizes:
                for n in nagents:
                    hldr = currBase(n, ob, en)
                    all_curr.append(hldr)

        #
        nagents = [12,14,16,18,20,22,24,26,28,30,32,34,36]
        obs_densities = [ob_hldr2]
        env_sizes = [28]
        for ob in obs_densities:
            for en in env_sizes:
                for n in nagents:
                    hldr = currBase(n, ob, en)
                    all_curr.append(hldr)

        #
        nagents = [14,16,18,20,22,24,26,28,30,32,34,36,38]
        obs_densities = [ob_hldr2]
        env_sizes = [30]
        for ob in obs_densities:
            for en in env_sizes:
                for n in nagents:
                    hldr = currBase(n, ob, en)
                    all_curr.append(hldr)

        #
        nagents = [16,18,20,22,24,26,28,30,32,34,36,38,40]
        obs_densities = [ob_hldr2]
        env_sizes = [32]
        for ob in obs_densities:
            for en in env_sizes:
                for n in nagents:
                    hldr = currBase(n, ob, en)
                    all_curr.append(hldr)
        
    curr = {i:c for i, c in enumerate(all_curr)}
    return curr


    


