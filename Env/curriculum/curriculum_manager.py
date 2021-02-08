import numpy as np
import math
from Env.curriculum.lessons import ppo_cn, ppo_cl_4, ppo_cl_inc_size, ppo_cl_inc_size_dirvec, ppo_final, ppo_final2, ppo_primal, ppo_all_same, ppo_all_same_no_sampling, ppo_all_same_no_sampling2

class CurriculumManager():
    def __init__(self, args, logger):
        #if args.curriculum == 'ppo_cn':
          #  from Env.curriculum.lessons import ppo_cn
        #self.curr = ppo_final() #ppo_cl_inc_size() #ppo_cl_4() #ppo_cn()
        self.curr =ppo_all_same(args)  # ppo_all_same_no_sampling(args) #ppo_primal() 
        self.curr_env_id = 0 #the sampled/currently active env
        self.highest_priority_id = 2 #0 #The curriculum with the highest probability of being selected
        self.logger = logger
        self.n_updates = args.ppo_curr_n_updates
        self.args = args
        self.threshold_var_key = "agent_dones"
        self.threshold_value = 1.1 #0.9 #threshold needed to be reached to continue to next curr
        self.done_flag = False

        # Distribution shape 1:
        # self.L_PROB = 0.05
        # self.M_PROB = 0.1
        # self.H_PROB = 0.3 #probability of the highest priority curr

        # Distribution shape 2:
        # self.LL_PROB = 0.05
        # self.L_PROB = 0.15
        # self.M_PROB = 0.25
        # self.H_PROB = 0.5 #probability of the highest priority curr

        # # Distribution shape 3 for primal:
        self.LL_PROB = 0.0
        self.L_PROB = 0.5
        self.M_PROB = 0.25
        self.H_PROB = 0.25 #probability of the highest priority curr

        # self.L_PROB = 1 - 2*0.333
        # self.M_PROB = 0.333
        # self.H_PROB = 0.333 #probability of the highest priority curr

        #For immitation learning:
        # self.L_PROB = 0.3
        # self.M_PROB = 0.3
        # self.H_PROB = 0.4 #probability of the highest priority curr


    def init_env_args(self):
        return self.curr[0]

    def init_logger(self, model, bench_func):
        self.logger = self.logger(self.args, model, self.curr, bench_func, self.threshold_var_key, self)
        return self.logger

    # def sample_env(self):
    #     #hldr = self.logger.get_threshold_var_mean(self.highest_priority_id)
    #     if self.logger.get_threshold_var_mean(self.highest_priority_id) > self.threshold_value:
    #         self.highest_priority_id += 1
    #         if self.highest_priority_id > (len(self.curr) - 1):
    #             self.highest_priority_id = len(self.curr) - 1
    #             self.done_flag = True

    #     # num = np.random.uniform(0,1)
    #     # ind_sample = math.floor(num)
    #     all_env_ind = [i for i in range(len(self.curr))]
    #     total_non_uniform_prob = 0
        
    #     if self.highest_priority_id-1 in all_env_ind:
    #         total_non_uniform_prob += self.M_PROB
    #     if self.highest_priority_id-2 in all_env_ind:
    #         total_non_uniform_prob += self.L_PROB
    #     if self.highest_priority_id+1 in all_env_ind:
    #         total_non_uniform_prob += self.M_PROB
    #     if self.highest_priority_id+2 in all_env_ind:
    #         total_non_uniform_prob += self.L_PROB
        
    #     total_non_uniform_prob += self.H_PROB
        
    #     uniform_prob = (1- total_non_uniform_prob) / (len(all_env_ind) - 1)
    #     curr_select_prob = [uniform_prob for i in range(len(all_env_ind))]
    #     curr_select_prob[self.highest_priority_id] = self.H_PROB

    #     if self.highest_priority_id-1 in all_env_ind:
    #         curr_select_prob[self.highest_priority_id-1] += self.M_PROB
    #     if self.highest_priority_id-2 in all_env_ind:
    #         curr_select_prob[self.highest_priority_id-2] += self.L_PROB
    #     if self.highest_priority_id+1 in all_env_ind:
    #         curr_select_prob[self.highest_priority_id+1] += self.M_PROB
    #     if self.highest_priority_id+2 in all_env_ind:
    #         curr_select_prob[self.highest_priority_id+2] += self.L_PROB

    #     ind_sample = np.random.choice(all_env_ind, size = 1, p = curr_select_prob)
    #     self.curr_env_id = ind_sample[0]
    #     print("Highest Priority Curriculum ID: {}".format(self.highest_priority_id))
    #     return self.curr[ind_sample[0]]

    def sample_env(self):
        #hldr = self.logger.get_threshold_var_mean(self.highest_priority_id)
        if self.logger.get_threshold_var_mean(self.highest_priority_id) > self.threshold_value:
            self.highest_priority_id += 1
            if self.highest_priority_id > (len(self.curr) - 1):
                self.highest_priority_id = len(self.curr) - 1
                self.done_flag = True
        
        if self.logger.global_info["global_iterations"] > self.args.ppo_iterations:
            self.done_flag = True

        # num = np.random.uniform(0,1)
        # ind_sample = math.floor(num)
        all_env_ind = [i for i in range(len(self.curr))]
        total_non_uniform_prob = 0
        
        if self.highest_priority_id-1 in all_env_ind:
            total_non_uniform_prob += self.M_PROB
        if self.highest_priority_id-2 in all_env_ind:
            total_non_uniform_prob += self.L_PROB
        if self.highest_priority_id-3 in all_env_ind:
            total_non_uniform_prob += self.LL_PROB

        
        total_non_uniform_prob += self.H_PROB
        
        uniform_prob = (1- total_non_uniform_prob) / (len(all_env_ind) - 1)
        curr_select_prob = [uniform_prob for i in range(len(all_env_ind))]
        curr_select_prob[self.highest_priority_id] = self.H_PROB

        if self.highest_priority_id-1 in all_env_ind:
            curr_select_prob[self.highest_priority_id-1] += self.M_PROB
        if self.highest_priority_id-2 in all_env_ind:
            curr_select_prob[self.highest_priority_id-2] += self.L_PROB
        if self.highest_priority_id-3 in all_env_ind:
            curr_select_prob[self.highest_priority_id-3] += self.LL_PROB
        

        ind_sample = np.random.choice(all_env_ind, size = 1, p = curr_select_prob)
        self.curr_env_id = ind_sample[0]
        print("Highest Priority Curriculum ID: {}".format(self.highest_priority_id))
        self.curr[ind_sample[0]].sample_agents_obstacle_density()
        return self.curr[ind_sample[0]]

    @property
    def is_done(self):
        return self.done_flag