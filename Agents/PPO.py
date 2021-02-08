import torch
from Agents.PPO.ppo import PPO
from Agents.PPO.utils import make_parallel_env
from Agents.PPO.buffer import PPO_Buffer
import itertools
from utils.logger import Logger

import numpy as np


#heplper functions:
def get_n_obs(next_obs, info):
    nobs = []
    for ob, inf in zip(next_obs, info):
        if inf["terminate"]:
            nobs.append(inf["terminal_observation"])
        else:
            nobs.append(ob)
    return nobs



def run(args):
    if args.ppo_use_gpu:
        device = 'gpu'
    else:
        device = 'cpu'
    buff = PPO_Buffer(args.n_agents, args.ppo_workers, args.ppo_rollout_length, args.ppo_recurrent)
    env = make_parallel_env(args, args.seed, buff.nworkers)
    ppo = PPO(env.action_space[0].n, env.observation_space[0],
            args.ppo_base_policy_type, 
            env.n_agents[0], args.ppo_share_actor, 
            args.ppo_share_value, args.ppo_k_epochs, 
            args.ppo_minibatch_size, 
            args.ppo_lr_a, args.ppo_lr_v, 
            args.ppo_hidden_dim, args.ppo_eps_clip, args.ppo_entropy_coeff, args.ppo_recurrent,
            args.ppo_heur_block)
    buff.init_model(ppo)
    logger = Logger(args, "No summary", "no policy summary", ppo)

    obs = env.reset()
    if args.ppo_recurrent:
        hx_cx_actr = [{i:ppo.init_hx_cx(device) for i in ob.keys()} for ob in obs]
        hx_cx_cr = [{i:ppo.init_hx_cx(device) for i in ob.keys()} for ob in obs]
    else:
        hx_cx_actr = [{i:(None,None) for i in ob.keys()} for ob in obs]
        hx_cx_cr = [{i:(None,None) for i in ob.keys()} for ob in obs]

    stats = {}
    for it in range(args.ppo_iterations):
        print("Iteration: {}".format(it))
        render_frames = []
        render_cntr = 0
        info2 = [{"valid_act":{i:None for i in ob.keys()}} for ob in obs]
        while buff.is_full == False:
            ppo.prep_device(device)
            a_probs, a_select, value, hx_cx_actr_n, hx_cx_cr_n, blocking = zip(*[ppo.forward(ob,ha,hc, dev=device, valid_act_heur = inf2["valid_act"] ) \
                                    for ob,ha,hc, inf2 in zip(obs, hx_cx_actr, hx_cx_cr, info2)])
            a_env_dict = [{key:val.item() for key,val in hldr.items()} for hldr in a_select]
            next_obs, r, dones, info = env.step(a_env_dict)
            if it % args.render_rate == 0:
                if render_cntr < args.render_length:
                    render_frames.append(env.render(indices= [0])[0])
                    if info[0]["terminate"]:
                        render_cntr += 1
            next_obs_ = get_n_obs(next_obs, info)
            buff.add(obs, r, value, next_obs_, a_probs, a_select, info, dones, hx_cx_actr, hx_cx_cr, hx_cx_actr_n, hx_cx_cr_n, blocking)
            hx_cx_actr, hx_cx_cr = hx_cx_actr_n, hx_cx_cr_n 
            #Reset hidden and cell states when epidode done
            for i, inf in enumerate(info):
                if inf["terminate"] and args.ppo_recurrent:
                    hx_cx_actr, hx_cx_cr = list(hx_cx_actr), list(hx_cx_cr)
                    hx_cx_actr[i] = {i2:ppo.init_hx_cx(device) for i2 in hx_cx_actr[i].keys()}
                    hx_cx_cr[i] = {i2:ppo.init_hx_cx(device) for i2 in hx_cx_cr[i].keys()}

            obs = next_obs

        if buff.is_full:
            observations, a_prob, a_select, adv, v, infos, h_actr, h_cr, blk_labels, blk_pred = buff.sample(args.ppo_discount, \
                args.ppo_gae_lambda, blocking=args.ppo_heur_block)
            stats["action_loss"], stats["value_loss"] \
            = ppo.update(observations, a_prob, a_select, adv, v, 0, h_actr, h_cr, blk_labels, blk_pred, dev=device)

            if args.ppo_recurrent:
                for a, c in zip(hx_cx_actr, hx_cx_cr):
                    for a2, c2 in zip(a.values(), c.values()):
                        a2[0].detach_()
                        a2[1].detach_()
                        c2[0].detach_()
                        c2[1].detach_()
        if (it+1) % args.benchmark_frequency == 0:
            rend_f, ter_i = benchmark_func(args, ppo, args.benchmark_num_episodes, args.benchmark_render_length, device)
            logger.benchmark_info(ter_i, rend_f, it)

        
        #Logging:
        stats["iterations"] = it
        stats["num_timesteps"] = len(infos)
        terminal_t_info = [inf for inf in infos if inf["terminate"]]
        stats["num_episodes"] = len(terminal_t_info)
        logger.log(stats, terminal_t_info, render_frames, checkpoint=True)
        render_frames = []
        render_cntr = 0
    
    rend_f, ter_i = benchmark_func(args, ppo, args.benchmark_num_episodes, args.benchmark_render_length, device)
    logger.benchmark_info(ter_i, rend_f, it)

    





def benchmark_func(args, model, num_episodes, render_len, device, greedy=False):
    #Assume parallel env
    env = make_parallel_env(args, np.random.randint(0, 10000), 1)
    render_frames = []
    obs = env.reset()
    #terminal_info = []
    all_info = []
    render_frames.append(env.render(indices = [0])[0])
    

    for ep in range(num_episodes):
        if args.ppo_recurrent:
            hx_cx_actr = [{i:model.init_hx_cx(device) for i in ob.keys()} for ob in obs]
            hx_cx_cr = [{i:model.init_hx_cx(device) for i in ob.keys()} for ob in obs]
        else:
            hx_cx_actr = [{i:(None,None) for i in ob.keys()} for ob in obs]
            hx_cx_cr = [{i:(None,None) for i in ob.keys()} for ob in obs]
        info2 = [{"valid_act":{i:None for i in ob.keys()}} for ob in obs]
        for t in itertools.count():
            model.prep_device(device)

           # a_probs, a_select, value, _, _, _ = zip(*[model.forward(ob, ) for ob in obs])

            a_probs, a_select, value, _, _, _ = zip(*[model.forward(ob,ha,hc, dev=device, valid_act_heur = inf2["valid_act"], greedy=greedy) \
                                    for ob,ha,hc, inf2 in zip(obs, hx_cx_actr, hx_cx_cr, info2)])


            a_env_dict = [{key:val.item() for key,val in hldr.items()} for hldr in a_select]
            next_obs, r, dones, info = env.step(a_env_dict)
            if ep < render_len:
                if info[0]["terminate"]:
                    render_frames.append(info[0]["terminal_render"])
                else:
                    render_frames.append(env.render(indices = [0])[0])
            obs = next_obs

            all_info.append(info)
            if info[0]["terminate"]:
             #   terminal_info.append(info[0])
                break
    return render_frames, all_info



        



        
    
