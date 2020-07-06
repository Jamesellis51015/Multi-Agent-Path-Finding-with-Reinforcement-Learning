import os
import itertools
from tensorboardX import SummaryWriter
import torch
import moviepy.editor as mpy
import config
import time

class Maac_Logger():
    def __init__(self, args):
        self.args = args
        #Naming of directories:
        if self.args.working_directory == 'none':
            import __main__
            self.args.working_directory = os.path.dirname(__main__.__file__) + r"/" + self.args.name
            print("os.path.dirname(__main__.__file__) is {}".format(os.path.dirname(__main__.__file__)))
            print("Working dir not specified, making working dir in {}".format(self.args.working_directory))
        else:
            self.args.working_directory += r"/" + self.args.name
          #  print("Working dir is {}".format(self.args.working_directory))

        cntr = 0
        work_dir_cpy = self.args.working_directory + r"_" + str(cntr)
        
        while os.path.exists(work_dir_cpy):
            cntr += 1
            work_dir_cpy = work_dir_cpy.replace("_" + str(cntr-1), "") + r"_" + str(cntr)
            
            if cntr == 500: raise Exception("Cntr == 500")

        self.args.working_directory = work_dir_cpy

        if self.args.alternative_plot_dir == 'none':
            self.plot_dir = self.args.working_directory + r"/tensorboard_plot"
        else:
            #cntr2 = 0
          #  while os.path.exists(self.args.alternative_plot_dir):
          #      cntr2 += 1
          #      self.args.alternative_plot_dir = self.args.alternative_plot_dir.replace("_" + str(cntr-1), "") + r"_" + str(cntr) 
            self.plot_dir = self.args.alternative_plot_dir + r"/" + self.args.name
        cntr2 = 0
        while os.path.exists(self.plot_dir):
            cntr2 += 1
            self.plot_dir  = self.plot_dir.replace("_N" + str(cntr-1), "") + r"_N" + str(cntr)
            
            if cntr2 == 500: raise Exception("Cntr == 500")
       # print("Working dir is {}".format(self.args.working_directory))
        self.checkpoint_dir = self.args.working_directory + r"/checkpoint"
        self.render_dir = self.args.working_directory + r"/render"
        self.benchmark_dir = self.args.working_directory + r"/benchmark"
        
        os.makedirs(self.args.working_directory)
        os.makedirs(self.plot_dir)
        os.makedirs(self.checkpoint_dir)
        os.makedirs(self.render_dir)

        self.writer = SummaryWriter(self.plot_dir)
      #  self.benchmark_writer_dir = None
        self.benchmark_writer = None #SummaryWriter(self.benchmark_dir)
        self.benchmark_episode = None
    
    def log_ep_info(self,all_infos, render_frames, episode, parallel_env = True):
        #print("All infos is \n {}".format(all_infos))
        info = []
        if parallel_env:
            for i in all_infos:
                for i2 in i:
                    if i2["terminate"]:
                        info.append(i2)
        else:
            for i in all_infos:
                if i["terminate"]:
                    info.append(i)
        if len(info)==0: print("Logging info is empty")

        self.log_keys = ['total_steps', 'total_rewards', \
         'total_agent_collisions', 'total_obstacle_collisions',\
         'total_avg_agent_r', 'total_ep_global_r', 'agent_dones']
        for i,inf in enumerate(info):
         # print("Logging ep: {}".format(episode+i))
          for key in self.log_keys:
              #print("Logging:: key: {}  info[key]: {}  ep: {}".format(key, inf[key], episode + i))
              self.writer.add_scalar(key, inf[key], episode + i)
          self.writer.add_scalar("test_ep", episode + i, episode + i)
        
        if len(render_frames) != 0:
            self.save_render(render_frames, episode)

    def init_benchmark_logging(self, episode):
        bench_dir = self.benchmark_dir + "/benchmark_" + str(episode)
        os.makedirs(bench_dir)
        self.benchmark_writer = SummaryWriter(bench_dir)

    def benchmark_info(self,all_infos, render_frames, episode, parallel_env = True):
        '''The infos should be off the entire benchmark nr of episode, not per episode '''
        if episode != self.benchmark_episode:
            self.benchmark_episode = episode
            self.init_benchmark_logging(episode)

        if self.benchmark_writer is None:
            raise Exception("Benchmark logging has not been initiated")
        info = []
        if parallel_env:
            for i in all_infos:
                for i2 in i:
                    if i2["terminate"]:
                        info.append(i2)
        else:
            for i in all_infos:
                if i["terminate"]:
                    info.append(i)
        #prefix = "bench_"+ str(episode)
        self.log_keys = ['total_steps', 'total_rewards', \
         'total_agent_collisions', 'total_obstacle_collisions',\
         'total_avg_agent_r', 'total_ep_global_r', 'agent_dones']
        self.log_names = self.log_keys
        # for i,key in enumerate(self.log_keys): 
        #     self.log_names.append(prefix + key)

        for i,inf in enumerate(info):
          for key,name in zip(self.log_keys, self.log_names):
              self.benchmark_writer.add_scalar(name, inf[key], i)
        
        if len(render_frames) != 0:
            end = str(episode)
            clip = mpy.ImageSequenceClip(render_frames, fps = 5)
            clip.write_videofile(self.benchmark_dir + "/benchmark_render_" + end + '.mp4')
                
    def save_render(self, frames, ep):
        if self.args.render_rate < 1000:
            end = str(ep)
        else:
            end = str(ep // 1000) + "k"
        
        clip = mpy.ImageSequenceClip(frames, fps = 5)
        #clip.write_gif(self.render_dir + "/render_" + str(episode) + '.gif')
        clip.write_videofile(self.render_dir + "/render_" + end + '.mp4')




class Logger():
    def __init__(self, args, env_summary, policy_summary, policy):
        self.batch_average_stats = {} 
        #self.episode_stats = {}
        self.args = args
        self.iterations = 0
        self.episodes = 0
        self.total_time_steps = 0
        self.render_count = 0
        self.policy = policy
        self.checkpoint_counter = 0
        self.previous_checkpoint_dir = None
        self.start_time = time.time()
        self.log_keys = None

        #Naming of directories:
        if self.args.working_directory == 'none':
            import __main__
            self.args.working_directory = os.path.dirname(__main__.__file__) + r"/" + self.args.name
        else:
            self.args.working_directory += r"/" + self.args.name

        cntr = 0
        work_dir_cpy = self.args.working_directory + r"_N" + str(cntr)
        
        while os.path.exists(work_dir_cpy):
            cntr += 1
            work_dir_cpy = work_dir_cpy.replace("_N" + str(cntr-1), "") + r"_N" + str(cntr)
            
            if cntr == 500: raise Exception("Cntr == 500")

        self.args.working_directory = work_dir_cpy

        if self.args.alternative_plot_dir == 'none':
            self.plot_dir = self.args.working_directory + r"/tensorboard_plot"
        else:
            #cntr2 = 0
          #  while os.path.exists(self.args.alternative_plot_dir):
          #      cntr2 += 1
          #      self.args.alternative_plot_dir = self.args.alternative_plot_dir.replace("_" + str(cntr-1), "") + r"_" + str(cntr) 
            self.plot_dir = self.args.alternative_plot_dir + r"/" + self.args.name
        cntr2 = 0
        while os.path.exists(self.plot_dir):
            cntr2 += 1
            self.plot_dir  = self.plot_dir.replace("_N" + str(cntr-1), "") + r"_N" + str(cntr)
            
            if cntr2 == 500: raise Exception("Cntr == 500")

        self.checkpoint_dir = self.args.working_directory + r"/checkpoint"
        self.render_dir = self.args.working_directory + r"/render"
        self.benchmark_dir = self.args.working_directory + r"/benchmark"
        
        os.makedirs(self.args.working_directory)
        os.makedirs(self.plot_dir)
        os.makedirs(self.checkpoint_dir)
        os.makedirs(self.render_dir)

        experiment_description = ""
        experiment_description += "Environment Summary: \n" + env_summary + "\n\n"
        experiment_description += "Arguments Summary: \n" + str(args) + "\n\n"
        experiment_description += "Policy Summary: \n" + policy_summary

        self.write_txt(self.args.working_directory, experiment_description, "Experiment_Description.txt")

        self.writer = SummaryWriter(self.plot_dir)
        self.benchmark_writer = None 
        self.benchmark_episode = None
        

    def init_benchmark_logging(self, episode):
        bench_dir = self.benchmark_dir + "/benchmark_" + str(episode)
        os.makedirs(bench_dir)
        self.benchmark_writer = SummaryWriter(bench_dir)

    def benchmark_info(self, all_infos, render_frames, episode, parallel_env = True):
        '''The infos should be off the entire benchmark nr of episode, not per episode '''
        if episode != self.benchmark_episode:
            self.benchmark_episode = episode
            self.init_benchmark_logging(episode)

        if self.benchmark_writer is None:
            raise Exception("Benchmark logging has not been initiated")
        info = []
        if parallel_env:
            for i in all_infos:
                for i2 in i:
                    if i2["terminate"]:
                        info.append(i2)
        else:
            for i in all_infos:
                if i["terminate"]:
                    info.append(i)
        #prefix = "bench_"+ str(episode)
        self.log_keys = ['total_steps', 'total_rewards', \
         'total_agent_collisions', 'total_obstacle_collisions',\
         'total_avg_agent_r', 'total_ep_global_r', 'agent_dones']
        self.log_names = self.log_keys
        # for i,key in enumerate(self.log_keys): 
        #     self.log_names.append(prefix + key)

        for i,inf in enumerate(info):
          for key,name in zip(self.log_keys, self.log_names):
              self.benchmark_writer.add_scalar(name, inf[key], i)
        
        if len(render_frames) != 0:
            end = str(episode)
            clip = mpy.ImageSequenceClip(render_frames, fps = 5)
            clip.write_videofile(self.benchmark_dir + "/benchmark_render_" + end + '.mp4')

    def log(self,stats, terminal_t_info, render_frames, checkpoint = True):
        '''stats: > Episodes and timesteps PER BATCH.
                    --Each batch has the actions taken per agent 
                  > Iteration number 
            terminal_t_into: the info dictionaries at the end of each episode
                  > Total obstacle and agent collisions
                  > Total reward ...
            render_frames: First nr of episode frames recorded
            
            Assume log is called every batch'''
        def set_dict_keys(dict, keys):
            for key in keys:
                dict[key] = 0

        

        if len(self.batch_average_stats.keys()) == 0 and self.iterations == 0: 
            #self.log_keys = terminal_t_info[0].keys()
            self.log_keys = ['total_steps', 'terminate', 'total_rewards', 'total_agent_collisions', 'total_obstacle_collisions', 'total_avg_agent_r', 'total_ep_global_r', 'agent_dones']

            set_dict_keys(self.batch_average_stats,self.log_keys)
        if len(terminal_t_info) == 0:
            terminal_t_info.append({k:0 for k in self.log_keys})
        batch_size = len(terminal_t_info)
        self.iterations = stats["iterations"]
        self.episodes += stats["num_episodes"]
        self.total_time_steps += stats["num_timesteps"]

        loss = { "value_loss_per_update": stats["value_loss"]}
        loss["action_loss_per_update"] = stats["action_loss"]
        self.plot_tensorboard(loss, only=["value_loss_per_update", "action_loss_per_update"])        

        for key in self.log_keys: self.batch_average_stats[key] += sum([inf[key] for inf in terminal_t_info])
        # self.batch_average_stats["total_ep_global_r"] += sum([inf["total_ep_global_r"] for inf in terminal_t_info]) 
        # self.batch_average_stats["total_avg_agent_r"] += sum([inf["total_avg_agent_r"] for inf in terminal_t_info]) 
        # self.batch_average_stats["total_agent_collisions"] += sum([inf["total_agent_collisions"] for inf in terminal_t_info]) 
        # self.batch_average_stats["total_obstacle_collisions"] += sum([inf["total_obstacle_collisions"] for inf in terminal_t_info])


        if (self.iterations + 1) % self.args.mean_stats == 0:
            for key in self.log_keys: self.batch_average_stats[key] /= self.args.mean_stats * batch_size
            # self.batch_average_stats["total_ep_global_r"] /= self.args.mean_stats
            # self.batch_average_stats["total_avg_agent_r"] /= self.args.mean_stats
            # self.batch_average_stats["total_agent_collisions"] /= self.args.mean_stats
            # self.batch_average_stats["total_obstacle_collisions"] /= self.args.mean_stats

            self.plot_tensorboard(self.batch_average_stats)
            set_dict_keys(self.batch_average_stats, self.log_keys)
        
        if len(render_frames) != 0:
            self.save_render(render_frames)
            self.render_count += 1
        
        if checkpoint: self.checkpoint_policy()

        if (self.iterations + 1) % self.args.print_frequency == 0:
            print("Iteration Nr: {}".format(self.iterations))
            print("Average time per iteration: {}".format((time.time() - self.start_time) / self.iterations))


    def should_render(self):
        if self.iterations > ((self.render_count+1) * self.args.render_rate):
            return self.args.render_length
        else:
            return 0
        

    def plot_tensorboard(self, stats, only = None):
        #Plot compared to iterations:
        if only == None:
            for key in self.log_keys:
                self.writer.add_scalar(key , stats[key], self.iterations)
        else:
            for key in only:
                self.writer.add_scalar(key, stats[key], self.iterations)



    def store_data(self):
        pass

    def write_txt(self, directory,string, file_name):
        f = open(directory + r"/" + file_name, "a")
        f.write(string)
        f.close
    def save_render(self, frames):
        if self.args.render_rate < 1000:
            end = str(self.iterations)
        else:
            end = str(self.iterations // 1000) + "k"
        
        clip = mpy.ImageSequenceClip(frames, fps = 5)
        #clip.write_gif(self.render_dir + "/render_" + str(episode) + '.gif')
        clip.write_videofile(self.render_dir + "/render_" + end + '.mp4')

    def checkpoint_policy(self):
        if self.iterations > self.checkpoint_counter * self.args.checkpoint_frequency:
            self.checkpoint_counter += 1
            

            checkpoint_path = self.checkpoint_dir + "/checkpoint_" + str(self.checkpoint_counter)

            if self.previous_checkpoint_dir != None and self.args.replace_checkpoints:
                os.remove(self.previous_checkpoint_dir)

            self.policy.save(checkpoint_path)
            self.previous_checkpoint_dir = checkpoint_path

    def summary(self):
        s =""
        s += "Iterations: " + str(self.iterations) + "\n"
        s += "Episodes: " + str(self.episodes) + "\n"
        s += "Time steps: " + str(self.total_time_steps)
        return s
    
    def done(self):
        self.write_txt(self.args.working_directory, self.summary(), "Experiment_Description.txt")



        


        
        

class Logger2():
    def __init__(self, args):
        self.name = args.name
        self.description = args.note
        self.working_dir = args.working_directory
        self.previous_checkpoint = None

        self.args = args

        #make working directory
        ans = "n"
        while ans == "n":
            if not os.path.exists(self.working_dir):
                os.makedirs(self.working_dir)
            
            if len(os.listdir(self.working_dir)) != 0:
                ans = input("Working directory not empty; overwrite it? (y/n) ")

                if ans == "n":
                    self.working_dir = input("Enter working directory: ")
                else:
                    import shutil
                    shutil.rmtree(self.working_dir)
                    os.makedirs(self.working_dir)
                   
            else:
                ans = "y"

        #Make logging folders:
        self.checkpoint_dir = self.working_dir + "/checkpoints"
        os.makedirs(self.checkpoint_dir)
        self.render_dir = self.working_dir + "/render"
        os.makedirs(self.render_dir)
        self.plot_dir = self.working_dir + "/plot_data"
       # os.makedirs(self.plot_dir)

        self.writer = SummaryWriter(self.plot_dir)

        self.mean_episode_reward = 0 
        self.total_time_steps = 0
        self.mean_timesteps_per_episode = 0
        self.mean_agent_collisions = 0
        self.mean_obstacle_collisions = 0
        self.mean_agents_done = 0

        self.episode_cntr = 0
        self.total_timestep_cntr = 0
        self.mean_const = args.mean_const

        self.network_updates = 0

    def dict_to_str(self, dictionary):  
        for key, value in dictionary.items():
            string += key.remove("--") + "\t |  " + value + " \n"
        return string

    def make_log_file(self, additional_info):
       
        file_name =  "/Experiment_summary.txt"

        content = "EXPERIMENT PARAMETERS: \n"
        content += self.dict_to_str(vars(self.args))
        content += " \n \n"
        content += "OTHER INFORMATION: \n"
        
        content += "Episodes: \t {} \n".format(self.episode_cntr)
        content += "Total timesteps: \t {} \n".format(self.total_time_steps)
        content += "\n"
        content += self.dict_to_str(additional_info)

        f = open(self.working_dir + file_name, "w")
        f.write(content)
        f.close

    def add_training_info(self, loss):
        self.writer.add_scalar("Loss / update", loss, self.network_updates)
        self.network_updates += 1


    def add_ep_info(self, info):
        """info: List of dictionaries containing information about an episode"""
        if len(info) == 0: return
        
        inf = [hldr for hldr in info if hldr["terminate"]] #Only take into account episodes which reached end of episode

        for i in inf:
            self.episode_cntr +=1
            self.total_time_steps += i["total_steps"]
            self.mean_timesteps_per_episode += i["total_steps"]
            self.mean_episode_reward += i["total_rewards"]
            self.mean_agents_done += i["agents_done"]
            self.mean_obstacle_collisions += i["total_obstacle_collisions"]
            self.mean_agent_collisions += i["total_agent_collisions"]

            if self.episode_cntr % self.mean_const == 0:
                self.writer.add_scalar("Mean_Ep_R /" + str(self.mean_const) + \
                "Episodes",self.mean_episode_reward / self.mean_const, self.episode_cntr // self.mean_const)

                #Mean dones are average fraction of agents who reached their goal:
                self.writer.add_scalar("Mean_Dones /" + str(self.mean_const) + \
                "Episodes",self.mean_agents_done / self.mean_const, self.episode_cntr // self.mean_const)

                #Mean cumulative timesteps till all agents reach their goals
                self.writer.add_scalar("Mean_Ep_Timesteps /" + str(self.mean_const) + \
                "Episodes",self.mean_timesteps_per_episode / self.mean_const, self.episode_cntr // self.mean_const)

                self.writer.add_scalar("Mean_Ep_Agent_Collisions /" + str(self.mean_const) + \
                "Episodes",self.mean_agent_collisions / self.mean_const, self.episode_cntr // self.mean_const)

                self.writer.add_scalar("Mean_Ep_Obstacle_Collisions /" + str(self.mean_const) + \
                "Episodes",self.mean_obstacle_collisions / self.mean_const, self.episode_cntr // self.mean_const)

                self.mean_timesteps_per_episode = 0
                self.mean_episode_reward = 0
                self.mean_agents_done = 0
                self.mean_obstacle_collisions = 0
                self.mean_agent_collisions = 0

    def record(self, env, policy, episode):
        frames = []
        (s, rewards, dones, collisions, info) = env.reset()
        frames.append(env.render("rgb_array"))
        for ep in range(self.args.render_length):
            for t in itertools.count():
                a = {}
                for agent_id in env.agents.keys():
                    o = (torch.from_numpy(s[agent_id][0]).reshape((1,-1)).to(config.device), torch.from_numpy(s[agent_id][1]).reshape((1,-1)).to(config.device))
                    a_prob , v = policy.forward(o[0], o[1])
                    a[agent_id] = policy.get_action(a_prob).item()

                (s, rewards, dones, collisions, info) = env.step(a)
                frames.append(env.render("rgb_array"))

                if info["terminate"]:
                    break
        clip = mpy.ImageSequenceClip(frames, fps = 10)
        #clip.write_gif(self.render_dir + "/render_" + str(episode) + '.gif')
        clip.write_videofile(self.render_dir + "/render_" + str(int(episode//1000)) + '.mp4')

    def checkpoint(self, policy_network, i):
        checkpoint_path = self.checkpoint_dir + "/checkpoint_" + str(i)

        if self.previous_checkpoint != None and self.args.replace_checkpoints:
            os.remove(self.previous_checkpoint)

        policy_network.save(checkpoint_path)
        self.previous_checkpoint = checkpoint_path

        


                






    
    


