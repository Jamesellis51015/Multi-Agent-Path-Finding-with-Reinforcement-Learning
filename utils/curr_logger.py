import os
import itertools
from tensorboardX import SummaryWriter
import torch
import moviepy.editor as mpy
import config
import time
import numpy as np

class CurriculumLogger():
    def __init__(self, args, model, curriculum, benchmark_func, threshold_key, curr_manager):
        #make work dir
        #make plot dir or central plot dir
        #Mkae dir for eac curr in work/exp dir
        #Make track variables
        #inint log keys
        #nint summary logger for each curr and for global
        #inint global iteration and iteration for each curr
        self.log_keys = ['total_steps', 'total_rewards', \
         'total_agent_collisions', 'total_obstacle_collisions',\
         'total_avg_agent_r', 'total_ep_global_r', 'agent_dones','all_agents_on_goal']

        self.args = args
        self.model = model
        self.benchmark_func = benchmark_func
        self.curr = curriculum
        self.curr_manager = curr_manager

        track = ["iterations", "episode_count", "time_step_count"]
        self.tracking_info = {id: {t:0 for t in track} for id in curriculum.keys()}
        self.global_info = {"global_iterations": 0}


        #Over the number of iterations
        self.NUM_THRESHOLD_VAR = 20
        self.threshold_key = threshold_key
        self.thresh_var_hldr = [np.zeros(self.NUM_THRESHOLD_VAR) for _ in self.curr]
        self.thresh_var_cntr = [0 for _ in self.curr]

        #Naming of directories:
        if self.args.working_directory == 'none':
            import __main__
            self.args.working_directory = os.path.dirname(__main__.__file__) + r"/" + self.args.name
        else:
            self.args.working_directory += r"/" + self.args.name

        if args.ppo_continue_from_checkpoint == False:
            cntr = 0
            work_dir_cpy = self.args.working_directory + r"_N" + str(cntr)
            
            while os.path.exists(work_dir_cpy):
                cntr += 1
                work_dir_cpy = work_dir_cpy.replace("_N" + str(cntr-1), "") + r"_N" + str(cntr)
                
                if cntr == 500: raise Exception("Cntr == 500")

            self.args.working_directory = work_dir_cpy

            if self.args.alternative_plot_dir == 'none':
                self.global_plot_dir = self.args.working_directory + r"/global_tensorboard_plot"
            else:
                self.global_plot_dir = self.args.alternative_plot_dir + r"/" + self.args.name
            cntr2 = 0
            while os.path.exists(self.global_plot_dir):
                cntr2 += 1
                self.global_plot_dir  = self.global_plot_dir.replace("_N" + str(cntr-1), "") + r"_N" + str(cntr)
                
                if cntr2 == 500: raise Exception("Cntr == 500")

            self.checkpoint_dir = self.args.working_directory + r"/checkpoint"
           # self.render_dir = self.args.working_directory + r"/render"
           # self.benchmark_dir = self.args.working_directory + r"/benchmark"
            
            os.makedirs(self.args.working_directory)
            os.makedirs(self.global_plot_dir)
            os.makedirs(self.checkpoint_dir)
          #  os.makedirs(self.render_dir)

            #Make sub dir for each environment:
            self.sub_working_dir = []
            for key, val in self.curr.items():
                hldr = self.args.working_directory + "/" + str(key) + val.name
                self.sub_working_dir.append(hldr)
                os.makedirs(hldr)
            
            self.sub_plot_dir = []
            for key, val in self.curr.items():
                hldr = self.sub_working_dir[key] + "/" + "tensorboard_plot_0"
                self.sub_plot_dir.append(hldr)
                os.makedirs(hldr)

            self.sub_render_dir = []
            for key, val in self.curr.items():
                hldr = self.sub_working_dir[key] + "/" + "render"
                self.sub_render_dir.append(hldr)
                os.makedirs(hldr)
            self.sub_benchmark_dir = []

            for key, val in self.curr.items():
                hldr = self.sub_working_dir[key] + "/" + "benchmark"
                self.sub_benchmark_dir.append(hldr)
                os.makedirs(hldr)
        else:
            #Find correct directories:
            #Find the work dir with the highest {n} suffing in N_{n}
            work_dir_split = self.args.working_directory.split('/')

            work_dir_name = work_dir_split[-1]
            parent_work_dir = ""
            for w in work_dir_split[:-1]:
                parent_work_dir += w + "/"
            
            all_ = os.walk(parent_work_dir)
            all_dirs = None
            for a in all_:
                all_dirs = a[1]
                break

            #all_dirs = [a[1] for a in all_]
            the_dir = [a for a in all_dirs if work_dir_name in a]

            if len(the_dir) >= 1:
                ind_ = [int(a.split("N")[-1]) for a in the_dir]
                max_ind = max(ind_)
                max_ind = ind_.index(max_ind)
                the_dir = the_dir[max_ind]
            elif len(the_dir) == 0:
                raise Exception("Working directory not found")

            #print(parent_work_dir)
            #print(the_dir)
            work_dir = parent_work_dir + the_dir
            self.args.working_directory = work_dir
            self.saved_info = self.load_info(work_dir + "/checkpoint")
            prev_sub_plot_dirs = self.saved_info['sub_plot_dir']
            this_sub_plot_dirs = []
            for d in prev_sub_plot_dirs:
                ind = d.split('_')[-1]
                n_ind = str(int(ind)+ 1)
                hldr = d.replace("_"+ind, "_" + n_ind)
                this_sub_plot_dirs.append(hldr)
                os.makedirs(hldr)
            self.sub_plot_dir = this_sub_plot_dirs
            self.sub_render_dir = self.saved_info["sub_render_dir"]
            self.checkpoint_dir = self.saved_info["checkpoint_dir"]
            self.sub_benchmark_dir = self.saved_info["sub_benchmark_dir"]
            self.global_plot_dir = self.saved_info["global_plot_dir"]

            self.tracking_info = self.saved_info["tracking_info"]
            self.global_info = self.saved_info["global_info"]
            self.model.load(self.saved_info["model"])

            #Over the number of iterations
            self.NUM_THRESHOLD_VAR = self.saved_info["num_threshold_var"]
            self.threshold_key = self.saved_info["threshold_key"]
            self.thresh_var_hldr = self.saved_info["thresh_var_hldr"]
            self.thresh_var_cntr = self.saved_info["thresh_var_cntr"]
            self.curr_manager.highest_priority_id = self.saved_info["highest_priority_id"]

        self.global_plot_writer = SummaryWriter(self.global_plot_dir)
        self.sub_plot_writer =[SummaryWriter(d) for d in self.sub_plot_dir]

        self.previous_checkpoint = None

        self.temp_render_info = {"flag": False,
                                "episode": 0,
                                "frames": [], 
                                "prev_iter": None}

        self.bc_counter = 0
        
    
    def add_threshold_var(self, curr_id, var):
        cntr = self.thresh_var_cntr[curr_id]
        self.thresh_var_hldr[curr_id][cntr] = var
        self.thresh_var_cntr[curr_id] += 1
        if self.thresh_var_cntr[curr_id] >= self.NUM_THRESHOLD_VAR:
            self.thresh_var_cntr[curr_id] = 0
        
    def get_threshold_var_mean(self, curr_id):
        val = self.thresh_var_hldr[curr_id]
        return np.mean(val)

    def init_sub_sub_benchmark_dir(self,parent_bench_dir, episode):
        bench_dir = parent_bench_dir + "/benchmark_" + str(episode)
        os.makedirs(bench_dir)
        return SummaryWriter(bench_dir)

        
    def benchmark(self, curr_id, own_iteration = None):
        #run benchmark func and log results
        if own_iteration is None:
            this_iteration = self.tracking_info[curr_id]["iterations"]
        else:
            this_iteration = own_iteration
        env_args = self.curr[curr_id]
        frames, terminal_infos = self.benchmark_func(env_args, self.args, \
            self.model, self.args.benchmark_num_episodes, self.args.benchmark_render_length, self.model.current_device)
        render_path = self.sub_benchmark_dir[curr_id] + "/benchmark_" + \
            str(this_iteration)+ '.mp4'
        clip = mpy.ImageSequenceClip(frames, fps = 5)
        clip.write_videofile(render_path)
        
        writer = self.init_sub_sub_benchmark_dir(self.sub_benchmark_dir[curr_id]\
            , this_iteration)
        for i, inf in enumerate(terminal_infos):
            for key in self.log_keys:
                writer.add_scalar(key, inf[key], i)
        writer.close()


    def log_bc(self, info_dic):
        global_writer = self.global_plot_writer
        for k,v in info_dic.items():
            global_writer.add_scalar(k, v, self.bc_counter)     
        self.bc_counter += 1

    def log(self,curr_id, non_ter_infos, extra_stats):
        '''Logs infos at end of every iteration '''
        #ignore extra stats

        ter_info = [inf for inf in non_ter_infos if inf["terminate"]]
        if len(ter_info) == 0:
            raise Exception("No episodes terminated")
        num_episodes = len(ter_info)
        self.tracking_info[curr_id]["episode_count"] += num_episodes
        self.tracking_info[curr_id]["time_step_count"] += len(non_ter_infos)
        

        sub_writer = self.sub_plot_writer[curr_id]
        global_writer = self.global_plot_writer
        iteration = self.tracking_info[curr_id]["iterations"] 
        #Average iteration stats and log:
        for key in self.log_keys:
            avrg = sum([inf[key] for inf in ter_info]) / num_episodes
            if key == self.threshold_key:
                self.add_threshold_var(curr_id, avrg)
            sub_writer.add_scalar(key, avrg, iteration)
            global_writer.add_scalar(key, avrg, self.global_info["global_iterations"])
       
        #log extra stats:
        for k, v in extra_stats.items():
            if v is not None:
                global_writer.add_scalar(k, v, self.global_info["global_iterations"])

        #log tracking info
        for key, value in self.tracking_info[curr_id].items():
            sub_writer.add_scalar(key, value, iteration)
            global_writer.add_scalar(key, value, self.global_info["global_iterations"])

        self.tracking_info[curr_id]["iterations"] += 1
        self.global_info["global_iterations"] += 1
        #Benchmarking:
        if self.tracking_info[curr_id]["iterations"] % self.args.benchmark_frequency == 0:
            self.benchmark(curr_id)
        #Checkpointing
        global_iter = self.global_info["global_iterations"]
        if global_iter % self.args.checkpoint_frequency == 0 and global_iter!=0:
            self.checkpoint(global_iter)

    def checkpoint(self, global_iter):
        #if global_iter % self.args.checkpoint_frequency and global_iter!=0:
        path = self.checkpoint_dir + "/checkpoint_" + str(global_iter)
        self.save_info(path)
        if not self.previous_checkpoint is None and self.args.replace_checkpoints:
            os.remove(self.previous_checkpoint)
        self.previous_checkpoint = path

    

    def load_info(self, checkpoint_path):
        #Get highest saved checkpoint
        all_ = os.walk(checkpoint_path)
        for a in all_:
            all_files = a[2]
            break
       # all_files = [a[2] for a in all_]
        all_check_numbers = [a.split("_")[-1] for a in all_files]
        max_ = max(all_check_numbers)
        ind = all_check_numbers.index(max_)
        the_file = all_files[ind]
        return torch.load(checkpoint_path + "/" + the_file)

    def record_render(self, env_id, env, info):
        if self.temp_render_info["flag"] == False:
            if self.tracking_info[env_id]["iterations"] % self.args.render_rate == 0 \
                and self.tracking_info[env_id]["iterations"] != self.temp_render_info["prev_iter"]:
                self.temp_render_info["flag"] = True
                self.temp_render_info["prev_iter"] = self.tracking_info[env_id]["iterations"]
        if self.temp_render_info["flag"]:
            if info["terminate"]:
                self.temp_render_info["frames"].append(info["terminal_render"])
                self.temp_render_info["episode"] += 1
            else:
                self.temp_render_info["frames"].append(env.render(indices = [0])[0])
            if self.temp_render_info["episode"] > self.args.render_length:
                path = self.sub_render_dir[env_id] + "/render_" \
                    + str(self.tracking_info[env_id]["iterations"]) + ".mp4"
                clip = mpy.ImageSequenceClip(self.temp_render_info["frames"], fps = 5)
                clip.write_videofile(path)
                self.temp_render_info["flag"] = False
                self.temp_render_info["episode"] = 0
                self.temp_render_info["frames"] = []

    def release_render(self, env_id):
        if len(self.temp_render_info["frames"]) > 0:
            path = self.sub_render_dir[env_id] + "/render_" \
                        + str(self.tracking_info[env_id]["iterations"]) + ".mp4"
            clip = mpy.ImageSequenceClip(self.temp_render_info["frames"], fps = 5)
            clip.write_videofile(path)
            self.temp_render_info["flag"] = False
            self.temp_render_info["episode"] = 0
            self.temp_render_info["frames"] = []

    



    def save_info(self, path):
        model_info = self.model.get_model_dict()
        save_dict = {"model":model_info,
        "sub_render_dir":self.sub_render_dir,
        "checkpoint_dir": self.checkpoint_dir,
        "sub_benchmark_dir": self.sub_benchmark_dir,
        'sub_plot_dir': self.sub_plot_dir,
        "tracking_info": self.tracking_info,
        "global_info": self.global_info,
        "num_threshold_var": self.NUM_THRESHOLD_VAR,
        "threshold_key": self.threshold_key ,
        "thresh_var_hldr": self.thresh_var_hldr,
        "thresh_var_cntr": self.thresh_var_cntr,
        "highest_priority_id": self.curr_manager.highest_priority_id,
        "global_plot_dir": self.global_plot_dir}

        torch.save(save_dict, path)


