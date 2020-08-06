import numpy as np
import math
from Env.curriculum.lessons import ppo_cn, ppo_cl_4

class CurriculumManager():
    def __init__(self, args, logger):
        #if args.curriculum == 'ppo_cn':
          #  from Env.curriculum.lessons import ppo_cn
        self.curr = ppo_cl_4() #ppo_cn()
        self.curr_env_id = 0 #the sampled/currently active env
        self.highest_priority_id = 0 #The curriculum with the highest probability of being selected
        self.logger = logger
        self.n_updates = args.ppo_curr_n_updates
        self.args = args
        self.threshold_var_key = "agent_dones"
        self.threshold_value = 0.9 #threshold needed to be reached to continue to next curr
        self.done_flag = False

        self.H_PROB = 0.6 #probability of the highest priority curr

    def init_env_args(self):
        return self.curr[0]

    def init_logger(self, model, bench_func):
        self.logger = self.logger(self.args, model, self.curr, bench_func, self.threshold_var_key, self)
        return self.logger

    def sample_env(self):
        hldr = self.logger.get_threshold_var_mean(self.highest_priority_id)
        if self.logger.get_threshold_var_mean(self.highest_priority_id) > self.threshold_value:
            self.highest_priority_id += 1
            if self.highest_priority_id > (len(self.curr) - 1):
                self.highest_priority_id = len(self.curr) - 1
                self.done_flag = True

        # num = np.random.uniform(0,1)
        # ind_sample = math.floor(num)
        all_env_ind = [i for i in range(len(self.curr))]
        uniform_prob = (1- self.H_PROB) / (len(all_env_ind) - 1)
        curr_select_prob = [uniform_prob for i in range(len(all_env_ind))]
        curr_select_prob[self.highest_priority_id] = self.H_PROB
        ind_sample = np.random.choice(all_env_ind, size = 1, p = curr_select_prob)
        self.curr_env_id = ind_sample[0]
        return self.curr[ind_sample[0]]

    @property
    def is_done(self):
        return self.done_flag