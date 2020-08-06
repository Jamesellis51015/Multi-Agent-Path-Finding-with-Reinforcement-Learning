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

def benchmark_func(env_args,main_args, model, num_episodes, render_len, device):
    #Assume parallel env
    env = make_parallel_env(env_args, np.random.randint(0, 10000), 1)
    render_frames = []
   # obs = env.reset()
    

    terminal_info = []
    render_frames.append(env.render(indices = [0])[0])
    for ep in range(num_episodes):
        obs = env.reset()
        if main_args.ppo_recurrent:
            hx_cx_actr = [{i:model.init_hx_cx(device) for i in ob.keys()} for ob in obs]
            hx_cx_cr = [{i:model.init_hx_cx(device) for i in ob.keys()} for ob in obs]
        else:
            hx_cx_actr = [{i:(None,None) for i in ob.keys()} for ob in obs]
            hx_cx_cr = [{i:(None,None) for i in ob.keys()} for ob in obs]
        info2 = [{"valid_act":{i:None for i in ob.keys()}} for ob in obs]
        for t in itertools.count():
            #a_probs, a_select, value = zip(*[model.forward(ob, greedy =True) for ob in obs])
            a_probs, a_select, value, hx_cx_actr_n, hx_cx_cr_n, blocking = zip(*[model.forward(ob,ha,hc, dev=device, valid_act_heur = inf2["valid_act"] ) \
                                    for ob,ha,hc, inf2 in zip(obs, hx_cx_actr, hx_cx_cr, info2)])
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


    


def run(args): #Curriculum train:
    if args.ppo_use_gpu:
        device = 'gpu'
    else:
        device = 'cpu'

    curr_manager = CurriculumManager(args, CurriculumLogger)
    #Get env args for first env
    env_args = curr_manager.init_env_args()
    #Determine number of workers for buffer and init env
    buff = PPO_Buffer(env_args.n_agents, args.ppo_workers, args.ppo_rollout_length, args.ppo_recurrent)
    seed = np.random.randint(0, 10000)
    env = make_parallel_env(env_args, seed, buff.nworkers)
    #Init ppo model
    ppo = PPO(env.action_space[0].n, env.observation_space[0],
            args.ppo_base_policy_type, 
            env.n_agents[0], args.ppo_share_actor, 
            args.ppo_share_value, args.ppo_k_epochs, 
            args.ppo_minibatch_size, 
            args.ppo_lr_a, args.ppo_lr_v, 
            args.ppo_hidden_dim, args.ppo_eps_clip, args.ppo_entropy_coeff,
            args.ppo_recurrent,args.ppo_heur_block)
    #Add model to buffer
    buff.init_model(ppo)

    logger = curr_manager.init_logger(ppo, benchmark_func)
    global_iterations = 0
    env.close()
    while not curr_manager.is_done:
        env_args = curr_manager.sample_env()
        buff.__init__(env_args.n_agents, args.ppo_workers, args.ppo_rollout_length, args.ppo_recurrent)#recalculates nworkers
        ppo.extend_agent_indexes(env_args.n_agents)
        buff.init_model(ppo)
        seed = np.random.randint(0, 10000)
        env = make_parallel_env(env_args, seed, buff.nworkers)

        obs = env.reset()
        if args.ppo_recurrent:
            hx_cx_actr = [{i:ppo.init_hx_cx(device) for i in ob.keys()} for ob in obs]
            hx_cx_cr = [{i:ppo.init_hx_cx(device) for i in ob.keys()} for ob in obs]
        else:
            hx_cx_actr = [{i:(None,None) for i in ob.keys()} for ob in obs]
            hx_cx_cr = [{i:(None,None) for i in ob.keys()} for ob in obs]

        extra_stats = {}
        env_id = curr_manager.curr_env_id
        for up_i in range(curr_manager.n_updates):
            print("Iteration: {}".format(global_iterations))
            info2 = [{"valid_act":{i:None for i in ob.keys()}} for ob in obs]
            while buff.is_full == False:
                #a_probs, a_select, value = zip(*[ppo.forward(ob) for ob in obs])
                a_probs, a_select, value, hx_cx_actr_n, hx_cx_cr_n, blocking = zip(*[ppo.forward(ob,ha,hc, dev=device, valid_act_heur = inf2["valid_act"] ) \
                                    for ob,ha,hc, inf2 in zip(obs, hx_cx_actr, hx_cx_cr, info2)])

                a_env_dict = [{key:val.item() for key,val in hldr.items()} for hldr in a_select]
                next_obs, r, dones, info = env.step(a_env_dict)
                logger.record_render(env_id, env, info[0])
                next_obs_ = get_n_obs(next_obs, info)
                buff.add(obs, r, value, next_obs_, a_probs, a_select, info, dones, hx_cx_actr, hx_cx_cr, hx_cx_actr_n, hx_cx_cr_n, blocking)
                #Reset hidden and cell states when epidode done
                for i, inf in enumerate(info):
                    if inf["terminate"] and args.ppo_recurrent:
                        hx_cx_actr, hx_cx_cr = list(hx_cx_actr), list(hx_cx_cr)
                        hx_cx_actr[i] = {i2:ppo.init_hx_cx(device) for i2 in hx_cx_actr[i].keys()}
                        hx_cx_cr[i] = {i2:ppo.init_hx_cx(device) for i2 in hx_cx_cr[i].keys()}
                obs = next_obs

            if buff.is_full:
                observations, a_prob, a_select, adv, v, infos, h_actr, h_cr, blk_labels, blk_pred = buff.sample(args.ppo_discount,\
                 args.ppo_gae_lambda,args.ppo_gae_lambda, blocking=args.ppo_heur_block)
                extra_stats["action_loss"], extra_stats["value_loss"] \
                = ppo.update(observations, a_prob, a_select, adv, v, 0, h_actr, h_cr, blk_labels, blk_pred, dev=device)
                if args.ppo_recurrent:
                    for a, c in zip(hx_cx_actr, hx_cx_cr):
                        for a2, c2 in zip(a.values(), c.values()):
                            a2[0].detach_()
                            a2[1].detach_()
                            c2[0].detach_()
                            c2[1].detach_()
                global_iterations += 1
                logger.log(env_id, infos, extra_stats)
                if up_i == curr_manager.n_updates-1:
                    logger.release_render(env_id)
        env.close()
    print("Done")


