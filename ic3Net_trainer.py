"""Handles sampling from the environment. """
import numpy as np
#import ray
from collections import namedtuple
import copy
import torch
from Env.make_env import make_env


Transition = namedtuple('Transition', ('state', 'action_taken', 'action_prob', 'value', 'episode_mask', 'episode_mini_mask', 'next_state',
                                       'reward', 'misc'))


class Trainer():
    def __init__(self, args, policy, env):
        self.policy = policy
        self.env = env
        self.args = args
        self.batch_size = args.batch_size
        self.batch = []

    def get_episode_normal_pg(self, render):
        info = {}
        batch = []
        render_frames = []
        (obs, reward, dones, collisions, info) = self.env.reset()
        if render > 0: render_frames.append(self.env.render('rgb_array'))
        terminate = False
        for t in range(self.env.max_step):

            actions, action_prob, value= self.policy.take_action(obs)
            (obs_next, reward, dones, collisions, info) = self.env.step(actions)
            if render > 0: render_frames.append(self.env.render('rgb_array'))

            if info["terminate"]:
                terminate = True
                episode_mask = 0 
            else:
                episode_mask = 1 
            
            episode_mini_mask = dones #TODO:Implement episode mini_mask in policy update
                
            batch.append(Transition(obs, actions, action_prob, value, episode_mask, episode_mini_mask, obs_next, reward, copy.deepcopy(info)))
            obs = obs_next

            if terminate:
                return batch, render_frames
        return batch, render_frames

    def benchmark_ic3(self, policy,logger, num_episodes, render_length, curr_episode):
        all_infos = []
        render_frames = []
        for ep in range(num_episodes):
            info = {}
            info['alive_mask'] = np.ones(self.args.n_agents) 
            obs = self.env.reset()
            if ep < render_length:
                render_frames.append(self.env.render('rgb_array'))
            terminate = False
            prev_hid = torch.zeros(1, self.args.n_agents, self.args.hid_size)

            for t in range(self.env.max_step):
                if t == 0 and self.args.hard_attn:
                    info['comm_action'] = np.zeros(self.args.n_agents, dtype=int)

                # recurrence over time
                if self.args.recurrent:
                    prev_hid = self.policy.init_hidden(batch_size=1) #Assume batch size always one since this doing one step at a time
                    x = [obs, prev_hid]
                    action_dict, action, action_prob, value, prev_hid = self.policy.take_action(x, info, greedy=True)

                    if (t + 1) % self.args.detach_gap == 0:
                        prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())
                else:
                    x = obs
                    action_dict, action, action_prob, value = self.policy.take_action(x, info)

                (obs_next, reward, dones, info2) = self.env.step(action_dict)

                all_infos.append(info2)

                if ep < render_length:
                    render_frames.append(self.env.render('rgb_array'))
                for key in info2.keys():
                    info[key] = info2[key]

                # store comm_action in info for next step
                if self.args.hard_attn:
                    info['comm_action'] = action[-1] if not self.args.comm_action_one else np.ones(self.args.n_agents, dtype=int)

                info['alive_mask'] = np.ones(self.args.n_agents)

                if info["terminate"]:
                    terminate = True
                    episode_mask = np.zeros(self.args.n_agents) 
                else:
                    episode_mask = np.ones(self.args.n_agents) 
                
               # episode_mini_mask =np.array([1 if (d==False) else 0 for d in dones.values()]) 
                    
                #batch.append(Transition(obs, action, action_prob, value, episode_mask, episode_mini_mask, obs_next, reward, copy.deepcopy(info)))
                #Note(debug):Expecting obs: dict of obs | action: list of actions of all agents | action_prob: tensor [batch,n,n-actions] | value: [batch*n, 1]
                obs = obs_next

                if terminate:
                    break
        logger.benchmark_info(all_infos, render_frames, curr_episode, parallel_env = False)
           

    def get_episode_ic3net(self, render):
        info = {}
        info['alive_mask'] = np.ones(self.args.n_agents) #For now agents are always alive
        batch = []
        render_frames = []
        #(obs, reward, dones, collisions, info) = self.env.reset()
        obs = self.env.reset()
        if render > 0: render_frames.append(self.env.render('rgb_array'))
        terminate = False

        prev_hid = torch.zeros(1, self.args.n_agents, self.args.hid_size)
        

        for t in range(self.env.max_step):
            if t == 0 and self.args.hard_attn:
                info['comm_action'] = np.zeros(self.args.n_agents, dtype=int)

            # recurrence
            if self.args.recurrent:
             #   prev_hid = self.policy.init_hidden(batch_size=1) #Assume batch size always one since this doing one step at a time
                x = [obs, prev_hid]
                action_dict, action, action_prob, value, prev_hid = self.policy.take_action(x, info)

                if (t + 1) % self.args.detach_gap == 0:
                    prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())
            else:
                x = obs
                action_dict, action, action_prob, value = self.policy.take_action(x, info)

            #action = select_action(self.args, action_out)
            #action, actual = translate_action(self.args, self.env, action)
            (obs_next, reward, dones, info2) = self.env.step(action_dict) #ignore the comm action 
            
            for key in info2.keys():
                info[key] = info2[key]

            # store comm_action in info for next step
            if self.args.hard_attn:
                info['comm_action'] = action[-1] if not self.args.comm_action_one else np.ones(self.args.n_agents, dtype=int)

            # if 'alive_mask' in info:
            #     misc['alive_mask'] = info['alive_mask'].reshape(reward.shape)
            # else:
            #     misc['alive_mask'] = np.ones_like(reward)
            
            ##################################
            info['alive_mask'] = np.ones(self.args.n_agents)
            if render > 0: render_frames.append(self.env.render('rgb_array'))

            if info["terminate"]:
                terminate = True
                episode_mask = np.zeros(self.args.n_agents) 
            else:
                episode_mask = np.ones(self.args.n_agents) 
            
            episode_mini_mask =np.array([1 if (d==False) else 0 for d in dones.values()]) #TODO:Implement episode mini_mask in policy update
                
            batch.append(Transition(obs, action, action_prob, value, episode_mask, episode_mini_mask, obs_next, reward, copy.deepcopy(info)))
            #Note(debug):Expecting obs: dict of obs | action: list of actions of all agents | action_prob: tensor [batch,n,n-actions] | value: [batch*n, 1]
            obs = obs_next

            if terminate:
                return batch, render_frames
        return batch, render_frames

    def get_episode(self, render):
        normal_policies = ['PG_Shared', 'AC_Shared', 'PG_Separate', 'AC_Separate']

        if self.args.policy in normal_policies:
            return self.get_episode_normal_pg(render)
        elif self.args.policy == 'IC3':
            return self.get_episode_ic3net(render)
        else:
            raise Exception ("No get_episode training loop defined for {} policy".format(self.args.policy))
        
    def sample_batch(self, render = 0):
        self.batch = []
        self.render_frames = []
        stats = dict()
        stats["num_episodes"] = 0
        class should_rend():
            def __init__(self, rend):
                self.render = rend
            def rend(self):
                hldr = self.render
                if self.render > 0:self.render -= 1
                return hldr
        render_class = should_rend(render)

        if self.args.n_workers == 1:
            #Execute sequentially:
            while len(self.batch) < self.batch_size:
                bat, rend = self.get_episode(render)
                self.batch.extend(bat)
                self.render_frames.extend(rend)
                render = render_class.rend()
                stats["num_episodes"] += 1
            stats["num_timesteps"] = len(self.batch)
            self.batch = Transition(*zip(*self.batch))
        else:
            #Execute in parallel:
            workers = [env_inst.remote(self.args) for _ in range(self.args.n_workers)]
            while len(self.batch) < self.batch_size:
                ep_results_ids = [worker.rollout.remote(self.policy, render_class.rend()) for worker in workers]
                ep_results = ray.get(ep_results_ids)
                render = render_class.render
                (ep_traj_lst, rend_lst) = zip(*ep_results)
                stats["num_episodes"] += len(ep_traj_lst)
                print("Workers: {}".format(len(ep_traj_lst)))
                for traj in ep_traj_lst:
                    for trans in traj:
                        self.batch.append(trans)
                    if len(self.batch) >= self.batch_size:
                        break
                for lst in rend_lst:
                    for frame in lst:
                        self.render_frames.append(frame)
            stats["num_timesteps"] = len(self.batch)
            self.batch = Transition(*zip(*self.batch))

        return self.batch, stats, self.render_frames

    def train_maac(self, logger):
        info = {}
        batch = []
        render_frames = []
        (obs, reward, dones, collisions, info) = self.env.reset()
       # if render > 0: render_frames.append(self.env.render('rgb_array'))
        for ep in range(self.args.maac_n_episode):
            terminate = False
            for t in range(self.env.max_step):

                actions, action_prob, value= self.policy.take_action(obs)
                (obs_next, reward, dones, collisions, info) = self.env.step(actions)
                #if render > 0: render_frames.append(self.env.render('rgb_array'))
                if info["terminate"]:
                    terminate = True
                    episode_mask = 0 
                else:
                    episode_mask = 1 
                
                episode_mini_mask = dones #TODO:Implement episode mini_mask in policy update
                    
                batch.append(Transition(obs, actions, action_prob, value, episode_mask, episode_mini_mask, obs_next, reward, copy.deepcopy(info)))
                obs = obs_next

                if terminate:
                    break

            return batch, render_frames



    


        

