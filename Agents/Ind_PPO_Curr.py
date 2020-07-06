import torch
import numpy as np
from Agents.PPO.ppo import PPO
from Agents.PPO.utils import make_parallel_env
from Agents.PPO.buffer import PPO_Buffer
import itertools
from utils.logger import Logger

from utils.curr_logger import CurriculumLogger
from Env.curriculum.curriculum_manager import CurriculumManager


#heplper functions:
def get_n_obs(next_obs, info):
    nobs = []
    for ob, inf in zip(next_obs, info):
        if inf["terminate"]:
            nobs.append(inf["terminal_observation"])
        else:
            nobs.append(ob)
    return nobs



# def run(args):
#     if args.ppo_use_gpu:
#         device = 'gpu'
#     else:
#         device = 'cpu'
    
#     buff = PPO_Buffer(args.n_agents, args.ppo_workers, args.ppo_rollout_length)
#     env = make_parallel_env(args, args.seed, buff.nworkers)
#     ppo = PPO(env.action_space[0].n, env.observation_space[0].shape,
#             args.ppo_base_policy_type, 
#             env.n_agents[0], args.ppo_share_actor, 
#             args.ppo_share_value, args.ppo_k_epochs, 
#             args.ppo_minibatch_size, 
#             args.ppo_lr_a, args.ppo_lr_v, 
#             args.ppo_hidden_dim, args.ppo_eps_clip, args.ppo_entropy_coeff)
#     buff.init_model(ppo)
#     logger = Logger(args, "No summary", "no policy summary", None)

#     obs = env.reset()
#     stats = {}
#     for it in range(args.ppo_iterations):
#         print("Iteration: {}".format(it))
#         render_frames = []
#         render_cntr = 0
#         while buff.is_full == False:
#             a_probs, a_select, value = zip(*[ppo.forward(ob) for ob in obs])
#             a_env_dict = [{key:val.item() for key,val in hldr.items()} for hldr in a_select]
#             next_obs, r, dones, info = env.step(a_env_dict)
#             if it % args.render_rate == 0:
#                 if render_cntr < args.render_length:
#                     render_frames.append(env.render(indices= [0])[0])
#                     if info[0]["terminate"]:
#                         render_cntr += 1

#             next_obs_ = get_n_obs(next_obs, info)
#             buff.add(obs, r, value, next_obs_, a_probs, a_select, info, dones)
#             obs = next_obs

#         if buff.is_full:
#             observations, a_prob, a_select, adv, v, infos = buff.sample(args.ppo_discount, args.ppo_gae_lambda)
#             stats["action_loss"], stats["value_loss"] \
#             = ppo.update(observations, a_prob, a_select, adv, v, agent_id = 0, dev=device)
        
#         #Logging:
#         stats["iterations"] = it
#         stats["num_timesteps"] = len(infos)
#         terminal_t_info = [inf for inf in infos if inf["terminate"]]
#         stats["num_episodes"] = len(terminal_t_info)
#         logger.log(stats, terminal_t_info, render_frames, checkpoint=False)
#         render_frames = []
#         render_cntr = 0

def benchmark_func(env_args, model, num_episodes, render_len):
    #Assume parallel env
    env = make_parallel_env(env_args, np.random.randint(0, 10000), 1)
    render_frames = []
    obs = env.reset()
    terminal_info = []
    render_frames.append(env.render(indices = [0])[0])
    for ep in range(num_episodes):
        for t in itertools.count():
            a_probs, a_select, value = zip(*[model.forward(ob, greedy =True) for ob in obs])
            a_env_dict = [{key:val.item() for key,val in hldr.items()} for hldr in a_select]
            next_obs, r, dones, info = env.step(a_env_dict)
            if ep < render_len:
                if info[0]["terminate"]:
                    render_frames.append(info[0]["terminal_render"])
                else:
                    render_frames.append(env.render(indices = [0])[0])
            obs = next_obs
            if info[0]["terminate"]:
                terminal_info.append(info[0])
                break
    return render_frames, terminal_info


    


def run(args): #Curriculum tain:
    if args.ppo_use_gpu:
        device = 'gpu'
    else:
        device = 'cpu'

    curr_manager = CurriculumManager(args, CurriculumLogger)
    #Get env args for first env
    env_args = curr_manager.init_env_args()
    #Determine number of workers for buffer and init env
    buff = PPO_Buffer(env_args.n_agents, args.ppo_workers, args.ppo_rollout_length)
    seed = np.random.randint(0, 10000)
    env = make_parallel_env(env_args, seed, buff.nworkers)
    #Init ppo model
    ppo = PPO(env.action_space[0].n, env.observation_space[0],
            args.ppo_base_policy_type, 
            env.n_agents[0], args.ppo_share_actor, 
            args.ppo_share_value, args.ppo_k_epochs, 
            args.ppo_minibatch_size, 
            args.ppo_lr_a, args.ppo_lr_v, 
            args.ppo_hidden_dim, args.ppo_eps_clip, args.ppo_entropy_coeff)
    #Add model to buffer
    buff.init_model(ppo)

    logger = curr_manager.init_logger(ppo, benchmark_func)
    global_iterations = 0
    env.close()
    while not curr_manager.is_done:
        env_args = curr_manager.sample_env()
        buff.__init__(env_args.n_agents, args.ppo_workers, args.ppo_rollout_length)#recalculates nworkers
        ppo.extend_agent_indexes(env_args.n_agents)
        buff.init_model(ppo)
        seed = np.random.randint(0, 10000)
        env = make_parallel_env(env_args, seed, buff.nworkers)

        obs = env.reset()
        extra_stats = {}
        env_id = curr_manager.curr_env_id
        for up_i in range(curr_manager.n_updates):
            print("Iteration: {}".format(global_iterations))
            while buff.is_full == False:
                a_probs, a_select, value = zip(*[ppo.forward(ob) for ob in obs])
                a_env_dict = [{key:val.item() for key,val in hldr.items()} for hldr in a_select]
                next_obs, r, dones, info = env.step(a_env_dict)
                logger.record_render(env_id, env, info[0])
                next_obs_ = get_n_obs(next_obs, info)
                buff.add(obs, r, value, next_obs_, a_probs, a_select, info, dones)
                obs = next_obs

            if buff.is_full:
                observations, a_prob, a_select, adv, v, infos = buff.sample(args.ppo_discount, args.ppo_gae_lambda)
                extra_stats["action_loss"], extra_stats["value_loss"] \
                = ppo.update(observations, a_prob, a_select, adv, v, agent_id = 0, dev=device)
                global_iterations += 1
                logger.log(env_id, infos, extra_stats)
                if up_i == curr_manager.n_updates-1:
                    logger.release_render(env_id)
        env.close()
    print("Done")


