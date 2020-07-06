import argparse
import torch
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
#from torch.autograd import Variable
from tensorboardX import SummaryWriter
#from utils.make_env import make_env
from Env.make_env import make_env
from Agents.maac_utils.buffer import ReplayBuffer
from Agents.maac_utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from Agents.maac_utils.attention_sac import AttentionSAC
import time

from utils.wrappers import flat_np_lst, lst_to_dict, flat_np_lst_env_stack, wrap_actions


def make_parallel_env(args, n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            env = make_env(args)
            #Make env obs space flat:
            # obs_shape = env.observation_space[0].shape
            # flat_dim = obs_shape[0]*obs_shape[1]*obs_shape[2]
            # env.observation_space = [Box(low=0, high=1, shape= (flat_dim,), dtype=int) \
            #                         for _ in env.agents]
            #obs_space = env.observation_space
            #env.observation_space = [obs_space for _ in env.agents]
           # env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        #return DummyVecEnv([get_env_fn(0), get_env_fn(2)])
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def run(config, logger0):
   # model_dir = Path('./models') / config.env_id / config.model_name
   # if not model_dir.exists():
   #     run_num = 1
    # else:
    #     exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
    #                      model_dir.iterdir() if
    #                      str(folder.name).startswith('run')]
    #     if len(exst_run_nums) == 0:
    #         run_num = 1
    #     else:
    #         run_num = max(exst_run_nums) + 1
    # curr_run = 'run%i' % run_num
    # run_dir = model_dir / curr_run
    # log_dir = run_dir / 'logs'
    # os.makedirs(log_dir)
    config.maac_n_rollout_threads = 1
    start_time = time.time()
    logger = None #logger0.writer
    run_num = 1
    torch.manual_seed(run_num)
    np.random.seed(run_num)
    env = make_parallel_env(config, config.maac_n_rollout_threads, run_num)

    hldr = make_env(config)
    episode_length = hldr.max_step
    del hldr

   # quit()

    model = AttentionSAC.init_from_env(env, 
                                       tau=config.maac_tau,
                                       pi_lr=config.maac_pi_lr,
                                       q_lr=config.maac_q_lr,
                                       gamma=config.maac_gamma,
                                       pol_hidden_dim=config.maac_pol_hidden_dim,
                                       critic_hidden_dim=config.maac_critic_hidden_dim,
                                       attend_heads=config.maac_attend_heads,
                                       reward_scale=config.maac_reward_scale,
                                       share_actor = config.maac_share_actor,
                                       base_policy_type = config.maac_base_policy_type)
    print("model.nagents: {}".format(model.nagents))
    replay_buffer = ReplayBuffer(config.maac_buffer_length, model.nagents,
                                 [obsp.shape for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])

    t = 0
    render_frames = []
    render_counter = 0
    for ep_i in range(0, config.maac_n_episodes, config.maac_n_rollout_threads):
        ETA = ((((time.time() - start_time) /3600.0) / float(ep_i + 1)) * float(config.maac_n_episodes) ) - ((time.time() - start_time)/3600.0)
        print("Episodes %i-%i of %i ETA %f" % (ep_i + 1,
                                        ep_i + 1 + config.maac_n_rollout_threads,
                                        config.maac_n_episodes, ETA))
        #obs = np.stack([flat_np_lst(i) for i in env.reset()])
        obs = flat_np_lst_env_stack(env.reset(), flat=False) #*
        model.prep_rollouts(device='cpu')

        all_infos = []

        for et_i in range(episode_length):
            # rearrange observations to be per agent, and convert to torch Variable
            # hldr1 = obs[:, 1]
            # hldr = np.vstack(obs[:, 1])
            # torch_obs = [torch.tensor(np.vstack(obs[:, i]),
            #                       requires_grad=False)
            #              for i in range(model.nagents)]
            torch_obs = [torch.tensor(obs[:, i],
                                  requires_grad=False)
                         for i in range(model.nagents)]
            # get actions as torch Variables
            torch_agent_actions = model.step(torch_obs, explore=True) 
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.maac_n_rollout_threads)]
            #print("actions in maac: {}".format(actions))
            actions_dict = wrap_actions(actions)#[lst_to_dict(a) for a in actions] #*
            next_obs, rewards, dones, infos = env.step(actions_dict)

           # rewards = [{i:0.01 for i in range(model.nagents)}]

            if (ep_i+ 1)%config.render_rate <= (config.render_length * config.maac_n_rollout_threads):
                render_frames.append(env.render(indices = [0])[0])
                if et_i == 0: render_counter += 1
                
            all_infos.append(infos)

            #print("Obs before: {}".format(next_obs))
            next_obs = flat_np_lst_env_stack(next_obs, flat = False) #*
           # print("Rewards before: {}".format(rewards))
            rewards = flat_np_lst_env_stack(rewards, flat=False) #*
            if np.isnan(rewards).any():
                hldr = 1
            dones = flat_np_lst_env_stack(dones, flat = False) #*
            #print("Dones looks like: {}".format(dones))

            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            t += config.maac_n_rollout_threads
            if (len(replay_buffer) >= config.maac_batch_size and
                (t % config.maac_steps_per_update) < config.maac_n_rollout_threads):
                if config.maac_use_gpu:
                    model.prep_training(device='gpu')
                else:
                    model.prep_training(device='cpu')
                for u_i in range(config.maac_num_updates):
                    sample = replay_buffer.sample(config.maac_batch_size,
                                                  to_gpu=config.maac_use_gpu)
                    model.update_critic(sample, logger=logger)
                    model.update_policies(sample, logger=logger)
                    model.update_all_targets()
                model.prep_rollouts(device='cpu')

            if infos[0]["terminate"]:
                obs = flat_np_lst_env_stack(env.reset(), flat=False)
                break

        
        if render_counter == config.render_length:
          render_counter = 0
          logger0.log_ep_info(all_infos, render_frames, ep_i)
          render_frames = []
        else:
          logger0.log_ep_info(all_infos, [], ep_i)


        # ep_rews = replay_buffer.get_average_rewards(
        #     episode_length * config.maac_n_rollout_threads)
        # for a_i, a_ep_rew in enumerate(ep_rews):
        #     logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)

        if ep_i % config.checkpoint_frequency < config.maac_n_rollout_threads:
            model.prep_rollouts(device='cpu')
            #os.makedirs(run_dir / 'incremental', exist_ok=True)
            model.save(logger0.checkpoint_dir + '/checkpoint_' + str(ep_i //config.checkpoint_frequency) + '.pt')
            model.save(logger0.checkpoint_dir + '/model.pt')

        if ep_i % config.benchmark_frequency < config.maac_n_rollout_threads and ep_i != 0:
            benchmark(config, logger0, model, config.benchmark_num_episodes, config.benchmark_render_length, ep_i)

    model.save(logger0.checkpoint_dir + '/model.pt')
    benchmark(config, logger0, model, config.benchmark_num_episodes, config.benchmark_render_length, ep_i)
    env.close()

def benchmark(config, logger, policy, num_episodes, render_length, curr_episode):
    env = make_parallel_env(config, 1, seed=num_episodes//100)
    
    hldr = make_env(config)
    max_steps = hldr.max_step

    render_frames = []
    all_infos = []

    for ep_i in range(num_episodes):
        obs = flat_np_lst_env_stack(env.reset(), flat=False) #*
        if ep_i < render_length:
            render_frames.append(env.render(indices = [0])[0])

        policy.prep_rollouts(device='cpu')
        
        for et_i in range(max_steps):
            torch_obs = [torch.tensor(obs[:, i],
                                requires_grad=False)
                        for i in range(policy.nagents)]
            torch_agent_actions = policy.step(torch_obs, explore=False) 
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(1)]
            actions_dict = wrap_actions(actions)#[lst_to_dict(a) for a in actions] #*
            next_obs, rewards, dones, infos = env.step(actions_dict)
            if ep_i < render_length:
                render_frames.append(env.render(indices = [0])[0])

            all_infos.append(infos)
            next_obs = flat_np_lst_env_stack(next_obs, flat = False) #*)
            rewards = flat_np_lst_env_stack(rewards, flat=False) #*
            dones = flat_np_lst_env_stack(dones, flat = False) #*
            obs = next_obs
            if infos[0]["terminate"]:
                obs = flat_np_lst_env_stack(env.reset(), flat=False)
                break

    logger.benchmark_info(all_infos, render_frames, curr_episode, parallel_env = True)
                

    
