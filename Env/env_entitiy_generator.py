from numpy import genfromtxt
import numpy as np

OBJECT_TO_IDX = {
    'empty'     : '0',
    'obstacle'  : '1',
    'agent'     : '2',
    'goal'      : '3',
}


def csv_generator(csv_file):
    def generator():
        return genfromtxt(csv_file, delimiter=',', dtype=str)

    return generator

def custom_env_batches_generator(train_env, test_env = None, validation_env = None):
    """Arguments are folder directories to different environment sets """
    def get_envs_from_folder(directory):
        if directory == None:
            return 0
        else:
            #get files and check that csv
            #convert each to np and return list
            return 0
    train_env = get_envs_from_folder(train_env)
    test_env =  get_envs_from_folder(test_env)
    validation_env = get_envs_from_folder(validation_env)

    def generator(env_set = 'train'):
        """Possible env_sets: train, validation, test """
        if env_set == 'train':
            return train_env
        elif env_set ==  'test':
            return test_env
        elif env_set == 'val':
            return validation_env

    return generator


def random_obstacle_generator(map_size, n_agents, obj_density = 0, obj_distribution = 'uniform'):

    def generator():
       # print(" generator     np   {};  ".format(np.random.normal()))
        #obj_map = np.zeros(map_size, dtype= str)
        obj_map = np.full(map_size, '0', dtype=str)
        if obj_distribution == 'uniform':
            obj_ind =  np.arange(map_size[0]*map_size[1]).reshape(map_size)
        else:
            def get_distribution(disribution):
                pass
            pass
        #Obstacles
        n_samples = np.floor(map_size[0]*map_size[1] * obj_density)
        #Check the nr of obs does not exceed nr of agents and goals
        if n_samples > obj_map.size-2*n_agents: n_samples -= 2*n_agents
        if n_samples == 0: 
            selected_ind = np.array([])
        else:
            selected_ind = np.random.choice(obj_ind.reshape((-1)), int(n_samples), replace= False)


        for ind in selected_ind:
            obj_map[ind%map_size[0], ind//map_size[0]] = OBJECT_TO_IDX["obstacle"]

        #Agents
        if selected_ind.size == 0:
            avail_ind = obj_ind.reshape((-1,))
        else:
            avail_ind = np.array([x for x in obj_ind.reshape((-1,)).tolist() if x not in selected_ind.tolist()])
        agent_ind = np.random.choice(avail_ind, n_agents, replace=False)
        for ind in agent_ind:
            obj_map[ind%map_size[0], ind//map_size[0]] = OBJECT_TO_IDX["agent"]

        #Goals
        avail_ind = [x for x in avail_ind.reshape((-1,)).tolist() if x not in agent_ind.tolist()]
        goal_ind = np.random.choice(avail_ind, n_agents, replace=False)
        for ind in goal_ind:
            obj_map[ind%map_size[0], ind//map_size[0]] = OBJECT_TO_IDX["goal"]

        return obj_map
    return generator









        

