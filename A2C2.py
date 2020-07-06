import sys
import torch
import argparse
import numpy as np
from gym.spaces import Box, Discrete
import itertools
import time 

from Env.make_env import make_env
from Agents.maac_utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from collections import namedtuple
from Agents.a2c2_utils.policy import make_policies
from utils.logger import Maac_Logger
Transition = namedtuple('Transition', ('state', 'action_taken', 'action_prob', 'r_message', 's_message', 'comm_map', 'value', 'episode_mask', 'next_state',
                                       'reward', 'info'))


#From: https://github.com/shariqiqbal2810/MAAC/blob/master/main.py
def make_parallel_env(args, n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            env = make_env(args)
            #Make env obs space flat:
            obs_shape = env.observation_space[0].shape
            flat_dim = obs_shape[0]*obs_shape[1]*obs_shape[2]
            env.observation_space = [Box(low=0, high=1, shape= (flat_dim,), dtype=int) \
                                    for _ in env.agents]
           # env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def get_comm_map(n_agnets):
    return [[1 if i1 != i2 else 0 for i1 in range(n_agnets)] for i2 in range(n_agnets)]

def map_sent_to_received_messages(sent, comm_map, mode = "concat"):
    assert mode=="concat" or mode == "sum" or mode == "mean" , "Comm Mode not recognized, either sum, mean or concat"
    received = []
    
    if mode == "concat":
        for i in range(len(sent)):
            agent_r_messages = sent[0:i] + sent[i+1::]
            hldr = torch.cat(agent_r_messages, dim=-1)
            received.append(hldr)
    else:
        sent = torch.cat(sent)
        for c in comm_map:
            mask = torch.tensor(c).unsqueeze(-1).expand_as(sent)
            rec = sent * mask
            if mode == "sum":
                rec = rec.sum(0)
            elif mode == "average":
                rec = rec.mean(0)
            else:
                Exception("In map_sent_to_received_messages, neithen sum nor average is specified.")
            received.append(rec)
    return received

def map_received_m_loss_to_sent_m_loss(m_r_gradients, comm_map, message_size, comm_mode):
    #Assumes comm map is the same for all time steps
    number_map = []
    for c in comm_map:
        n =[]
        for i, e in enumerate(c):
            if e==1:
                n.append(i)
        number_map.append(n)

    t = m_r_gradients[0].size()[0]
    sent_m_loss = [torch.zeros((t,message_size)) for _ in range(len(m_r_gradients))]
    mean_counter = [0 for _ in range(len(m_r_gradients))]
    if comm_mode == "concat":
        for i,agent_grad in enumerate(m_r_gradients):
            for i2, num in enumerate(number_map[i]):
                sent_m_loss[num] += agent_grad[:,i2*message_size:(i2+1)*message_size]
                mean_counter[num] += 1
    for l,cntr in zip(sent_m_loss, mean_counter):
        l /= cntr
    return sent_m_loss

                    




def get_advantages(rewards, values, episode_mask, discount, GAE = False, lmbda_ = False):
    #Assume entire episodes are given (will not need to estimate v(next))
    if GAE: assert lmbda_ != False , "Lambda needs to be specified when using GAE"

    if GAE:
        raise NotImplementedError
    else:
        #One step update
        n_agents = len(rewards)
        adv = [[] for _ in  range(n_agents)]
        for i in range(n_agents):
            G = None
            for r,v,ep in zip(reversed(rewards[i]), reversed(values[i]), reversed(episode_mask)):
                if ep == 0:
                    #val = v.item().detach()
                    G = r
                    est = G # - val
                    #v_next = val
                    adv[i].append(G)
                else:
                    if G == None: G =  v
                    G = r + discount*G
                    # val = v.item().detach()
                    # est = r + discount*v_next - val
                    # v_next = val
                    adv[i].append(G)
        for a in adv:
            a.reverse()
        return adv

def save(path,args, ac, cr, me):
    save_dict = {}
    save_dict["init_info"] = args
    ac_param = []
    cr_pram = []
    me_param = []
    share_flags = [args.share_actor, args.share_critic, args.share_comm_network]
    labels = ["actors", "critics", "cnets"]
    for flag, net, buf, labl in zip(share_flags, [ac,cr,me], [ac_param, cr_pram, me_param], labels):
        if flag:
            buf.append(net[0].state_dict())
        else:
            for n in net:
                buf.append(n.state_dict())
        save_dict[labl] = buf
    torch.save(save_dict, path)

def benchmark(env,args, episode,logger, num_runs, render_length, actors, critics, comm_nets):
    f_o = lambda x: torch.from_numpy(x).unsqueeze(0).float()
    render_flag =False


    if num_runs%args.n_rollout_threads !=0:
        num_runs += args.n_rollout_threads - num_runs%args.n_rollout_threads 
    all_transitions = []
    render_frames = []
    for ep in range(0, num_runs, args.n_rollout_threads):
        transitions = []
        obs = env.reset()
        
        if ep%args.render_rate < render_length: #Only for one thread
            render_frames.append(env.render())
            render_flag = True
        else:
            render_flag = False
        s_message = [c_net.init_message().float() for c_net in comm_nets]
        comm_map = get_comm_map(args.n_agents)
        r_message = map_sent_to_received_messages(s_message, comm_map)
        for i in range(args.n_rollout_threads):
            for t in itertools.count():
                actions, action_probs = zip(*[actor.take_action(f_o(obs[i]), r_message[i], greedy=True) for i,actor in enumerate(actors)])
                s_message = [c_net.forward(f_o(obs[i]), r_message[i]) for i, c_net in enumerate(comm_nets)]
                actions_dict = {i:a for i,a in enumerate(actions)}
                #action_probs = {i:a for i,a in enumerate(action_probs)}
                next_obs, rewards, dones, info = env.step(actions_dict)
                values = [crit.forward(f_o(obs[i])) for i, crit in enumerate(critics)]
                if ep%args.render_rate < render_length: #Only for one thread
                    render_frames.append(env.render())
                if ep%args.render_rate + 1 < render_length:
                    render_flag = True
                else:
                    render_flag = False
         
                #comm_map = get_comm_map() #Comm_map stays the same for now
                if info["terminate"]:
                    transitions.append(Transition(obs, actions, action_probs,r_message, s_message, comm_map, values,\
                         0, next_obs, rewards, info))
                    break
                else:
                    transitions.append(Transition(obs, actions, action_probs, r_message, s_message, comm_map, values,\
                         1, next_obs, rewards, info))
                    
                r_message = map_sent_to_received_messages(s_message, comm_map)
                obs = next_obs
        all_transitions.append(transitions)
    all_transitions = list(itertools.chain.from_iterable(all_transitions))
    all_t = Transition(*zip(*all_transitions))
    logger.benchmark_info(all_t.info,render_frames, episode, parallel_env = False)
        # tr = Transition(*zip(*transitions))
        # if render_flag:          
        #     logger.log_ep_info(tr.info, [], ep, parallel_env=False)
        # else:
        #     logger.log_ep_info(tr.info, render_frames, ep, parallel_env=False)
        #     render_frames = []



def run(args):
   # print("args in run {}".format(args))
    torch.manual_seed(0)
    np.random.seed(0)
    assert args.n_rollout_threads == 1 , "Only implemented for one rollout thread, but n_rollout_threads not 1"
    logger = Maac_Logger(args)
    env = make_env(args) #make_parallel_env(args, 1, 0)
    num_agents = len(env.action_space)
    #print("num agents {}".format(num_agents))
    policies = make_policies(args.network_decoder_type)
    (Actor , Critic, CommNet) = policies

    to_agents = num_agents - 1 #In this instance
    #print("ags.share_actor {} share critic: {}  messages net: {}".format(args.share_actor, args.share_critic, args.share_comm_network))
    if args.share_actor:
        print("sharing actor")
        actr = Actor(env.observation_space[0], env.action_space[0], args.comm_channels*to_agents, lr = args.lr_a, \
            h_dim = args.h_dim_size)
        actors = [actr for _ in range(num_agents)]
    else:
        actr_args = [env.observation_space[0], env.action_space[0], args.comm_channels*to_agents]
        actr_kwargs = {"lr": args.lr_a, "h_dim": args.h_dim_size}
        actors = [Actor(*actr_args, **actr_kwargs) for _ in range(num_agents)]

    if args.share_critic:
        print("Sharing critic")
        crit = Critic(env.observation_space[0],lr = args.lr_v, h_dim = args.h_dim_size)
        critics = [crit for _ in range(num_agents)]
    else:
        crit_args = [env.observation_space[0]]
        crit_kwargs = {"lr":args.lr_v, "h_dim":args.h_dim_size}
        critics = [Critic(*crit_args, **crit_kwargs) for _ in range(num_agents)]
    
    if args.share_comm_network:
        print("Sharing message network")
        comm_net = CommNet(env.observation_space[0], comm_in=args.comm_channels*to_agents,\
            comm_out=args.comm_channels, lr = args.lr_c, h_dim = args.h_dim_size)
        comm_nets = [comm_net for _ in range(num_agents)]
    else:
        comm_args = [env.observation_space[0]]
        comm_kwargs = {"comm_in": args.comm_channels*to_agents, "comm_out": args.comm_channels,\
            "lr": args.lr_c, "h_dim":args.h_dim_size}
        comm_nets = [CommNet(*comm_args, **comm_kwargs) for _ in range(num_agents)]
    
    #Serial for now:
    #obs = env.reset()
    # obs_buff = []
    # rew_buff = []
    # dones_buff = []
    # info_buff = []
    f_o = lambda x: torch.from_numpy(x).unsqueeze(0).float()
    # test = np.random.rand(2,3)
    # print(test.shape)
    # print(f_o(test).size())
    start_t = time.time()
    render_frames = []
    for ep in range(0, args.n_episodes, args.n_rollout_threads):
        if ep%50 == 0:
            print("Episode: {}  Time: {}".format(ep, (time.time()-start_t)/3600))
        transitions = []
        obs = env.reset()
        
        if (ep+1)%args.render_rate < args.render_length: #Only for one thread
            render_frames.append(env.render())
            render_flag = True
        else:
            render_flag = False
        s_message = [c_net.init_message().float() for c_net in comm_nets]
        comm_map = get_comm_map(num_agents)
        r_message = map_sent_to_received_messages(s_message, comm_map)
        for i in range(args.n_rollout_threads):
            for t in itertools.count():
                actions, action_probs = zip(*[actor.take_action(f_o(obs[i]), r_message[i]) for i,actor in enumerate(actors)])
                if args.comm_zero:
                    s_message = [c_net.init_message().float() for c_net in comm_nets]
                else:
                    s_message = [c_net.forward(f_o(obs[i]), r_message[i]) for i, c_net in enumerate(comm_nets)]
                actions_dict = {i:a for i,a in enumerate(actions)}
                #action_probs = {i:a for i,a in enumerate(action_probs)}
                next_obs, rewards, dones, info = env.step(actions_dict)
                values = [crit.forward(f_o(obs[i])) for i, crit in enumerate(critics)]
                if (ep+1)%args.render_rate < args.render_length: #Only for one thread
                    render_frames.append(env.render())
                if (ep+1)%args.render_rate + 1 < args.render_length:
                    render_flag = True
                else:
                    render_flag = False
                #print("Info in loop is {}".format(info))
                #comm_map = get_comm_map() #Comm_map stays the same for now
                if info["terminate"]:
                    transitions.append(Transition(obs, actions, action_probs,r_message, s_message, comm_map, values,\
                         0, next_obs, rewards, info))
                    break
                else:
                    transitions.append(Transition(obs, actions, action_probs, r_message, s_message, comm_map, values,\
                         1, next_obs, rewards, info))
                    
                r_message = map_sent_to_received_messages(s_message, comm_map)
                obs = next_obs

        tr = Transition(*zip(*transitions))
        #print("transitions info is {}".format(tr.info))
        if render_flag:          
            logger.log_ep_info(tr.info, [], ep, parallel_env=False)
        else:
            logger.log_ep_info(tr.info, render_frames, ep, parallel_env=False)
            render_frames = []
        #Change to per agent:
        actions = [[act[i] for act in tr.action_taken] for i in range(num_agents)]
        action_probs = [[act[i] for act in tr.action_prob] for i in range(num_agents)]
        rewards = [[r[i] for r in tr.reward] for i in range(num_agents)]

        r_message = [[hldr[i] for hldr in tr.r_message] for i in range(num_agents)]
        s_message = [[hldr[i] for hldr in tr.s_message] for i in range(num_agents)]
        values = [[hldr[i] for hldr in tr.value] for i in range(num_agents)]
        states = [[hldr[i] for hldr in tr.state] for i in range(num_agents)]
        

        #print("Transitions after: {}".format(transitions))

        #Update: 
        advantages = get_advantages(rewards, values, tr.episode_mask,\
             args.discount, args.GAE, args.lambda_) 
        
         #Message nets
        #############
        all_actor_loss2 = []
        all_m_r_grad2 = []
       # print("a_prob before: {}".format(action_probs[0]))
        for actr, o, a,val, r_m, adv in zip(actors, states,actions,values, r_message, advantages):
            try:
              obs = torch.cat([f_o(o2) for o2 in o])
            except:
              print("o: {} \n f_o(o): {}".format(o, f_o(o)))
              print("a: {} \n val: {}".format(a,val))
              raise
            # if o==[] or f_o(o[0])==[]:
            #   print("o: {} \n f_o(o): {}".format(o, f_o(o)))
            #   print("a: {} \n val: {}".format(a,val))
            #   time.sleep(1000)
            # else:
            #   obs = torch.cat([f_o(o2) for o2 in o])
            r_m = torch.cat(r_m)
            _, a_prob2 = actr.take_action(obs, r_m)
            #print("a_prob after {}".format(a_prob2))
            entropy = torch.distributions.Categorical(probs = a_prob2).entropy().unsqueeze(-1) * torch.tensor(args.entropy_coeff)
            a2 = torch.cat(a)
            a_select2 = torch.gather(a_prob2, -1, a2)
            val =torch.cat(val)
            adv = torch.tensor(adv).unsqueeze(-1) #torch.cat(adv)#
            actor_loss2 = -torch.log(a_select2) * (adv - val.detach())
            actor_loss2 -= entropy
            actor_m_loss = actor_loss2.sum()
            #r_mess_tens = r_m[1::]
            if args.comm_zero ==False:
                m_r_gradients = torch.autograd.grad(actor_m_loss, r_m, retain_graph=True)[0]
                all_m_r_grad2.append(m_r_gradients[1::])
            all_actor_loss2.append(actor_loss2.sum())



        # #############
        if len(states[0]) > 1 and args.comm_zero == False:
          message_net_losses = map_received_m_loss_to_sent_m_loss(all_m_r_grad2, comm_map, message_size=args.comm_channels, comm_mode = args.comm_mode)
          #print("s_m_before {}".format(s_message[0]))
          all_c_losses = []
          for c_net, o, r_m, s_m, grads in zip(comm_nets, states, r_message, s_message, message_net_losses):
              try:
                obs = torch.cat([f_o(o2) for o2 in o][:-1:])
              except:
                print("o[0]: {} \n\n f_o(o[0]): {}".format(o[0], f_o(o[0])))
                print("len(o): {} \n len(r_m)): {}".format(len(o),len(r_m)))
                hldr = [f_o(o2) for o2 in o]
                print("[f_o(o2) for o2 in o] is {}".format(hldr))
                print("len(states): {} len(states[1]: {}".format(len(states), len(states[1])))
                raise
              r_m2 = torch.cat(r_m[:-1:])
              s_m2 = torch.cat(s_m[:-1:])
              s_out = c_net(obs, r_m2)
              #print("s_out after {}".format(s_out))
              m_loss = ((s_m2 - grads - s_out)**2).mean()
              all_c_losses.append(m_loss)

          total_m_losses = sum(all_c_losses)
          if args.share_comm_network:
              total_m_losses /= num_agents
              comm_nets[0].optimizer.zero_grad()
              total_m_losses.backward(retain_graph = True)
              comm_nets[0].optimizer.step()
          else:
              for c_net in comm_nets:
                  c_net.optimizer.zero_grad()
              total_m_losses.backward(retain_graph = True)
              for c_net in comm_nets:
                  c_net.optimizer.step()

        #Critics
        all_crit_loss = []
        for i,(crit, adv) in enumerate(zip(critics, advantages)):
            adv = torch.tensor(adv).unsqueeze(-1)
            crit_loss = ((adv - torch.cat(values[i]) )**2 ).mean()
            all_crit_loss.append(crit_loss)

        total_crit_loss = sum(all_crit_loss)
        if args.share_critic:
            total_crit_loss /= num_agents
            critics[0].optimizer.zero_grad()
           # total_crit_loss.backward()
            #critics[0].optimizer.step()
        else:
            for crit in critics:
                crit.optimizer.zero_grad()
            #total_crit_loss.backward()
            #for crit in critics:
            #    crit.optimizer.step()
        
        #Actors:
        all_actor_loss = []
       # all_m_r_grad = []
        for i, (adv, actr,action, act_prob, val, r_mess) in enumerate(zip(advantages, actors, actions, action_probs, values, r_message)):
            act_prob = torch.cat(act_prob)
            val = torch.cat(val)
            entropy = torch.distributions.Categorical(probs = act_prob).entropy().unsqueeze(-1) * torch.tensor(args.entropy_coeff)
            a = torch.cat(action)#torch.tensor(action).unsqueeze(-1)
            selected_action = torch.gather(act_prob, -1, a)
            adv = torch.tensor(adv).unsqueeze(-1) #torch.cat(adv)#
            actor_loss = -torch.log(selected_action) * (adv - val.detach())
            actor_loss -= entropy
            actor_m_loss = actor_loss.sum()
           # r_mess_tens = torch.cat(r_mess)
          #  m_r_gradients = torch.autograd.grad(actor_m_loss, r_mess_tens, retain_graph=True)
          #  all_m_r_grad.append(m_r_gradients)
            all_actor_loss.append(actor_loss.sum())
        
        total_actor_loss = sum(all_actor_loss)
        if args.share_actor:
            total_actor_loss /= num_agents
            actors[0].optimizer.zero_grad()
            #total_actor_loss.backward()
            #actors[0].optimizer.step()
        else:
            for actor in actors:
                actor.optimizer.zero_grad()
            #total_actor_loss.backward()
            #for actor in actors:
            #    actor.optimizer.step()
        total_loss = 0.5*total_crit_loss + total_actor_loss
        total_loss.backward()
        if args.share_actor:
            actors[0].optimizer.step()
        else:
            for actr in actors:
                actr.optimizer.step()
        if args.share_critic:
            critics[0].optimizer.step()
        else:
            for crit in critics:
                crit.optimizer.step()
        
        if (ep + 1) % args.checkpoint_frequency == 0:
            path = logger.checkpoint_dir +"/checkpoint_" + str(ep)+ ".pt"
            save(path, args, actors, critics, comm_nets)
        
    
    path = logger.checkpoint_dir +"/checkpoint_" + str(ep)+ ".pt"
    save(path, args, actors, critics, comm_nets)

    benchmark(env,args, args.n_episodes,logger, args.bench_num_runs, args.bench_render_length, actors, critics, comm_nets)

       


        




    

def main(args):
    experiment_group_name = "Test3_A2C2"
    work_dir = experiment_group_name
    plot_dir = experiment_group_name + "_Central3"

    parser = argparse.ArgumentParser("A2C2_parameters")

    #Environment
    parser.add_argument("--map_shape", default = 5, type=int)
    parser.add_argument("--n_agents", default = 4, type=int)
    parser.add_argument("--obj_density", default = 0.0, type=int)
    parser.add_argument("--view_size", default=2, type = int)
    parser.add_argument("--use_custom_rewards", default=False, action = 'store_true')
    parser.add_argument("--reward_step", default = -0.01, type = float)
    parser.add_argument("--reward_obj_collision", default = 0.0, type = float)
    parser.add_argument("--reward_agent_collision", default = -0.4, type = float)
    parser.add_argument("--reward_goal_reached", default = 0.5, type = float)
    parser.add_argument("--reward_finish_episode", default = 1.0, type = float)
    parser.add_argument("--env_name", default = "cooperative_navigation-v0", type = str)

    parser.add_argument("--n_episodes", default = 100000, type=int) 
    parser.add_argument("--n_rollout_threads", default = 1, type=int)
    parser.add_argument("--share_actor", default = False,action = 'store_true')
    parser.add_argument("--share_critic", default = False,action = 'store_true')
    parser.add_argument("--share_comm_network", default = False,action = 'store_true')
    parser.add_argument("--entropy_coeff", default = 0.01, type = float)
    parser.add_argument("--lr_a", default = 0.001, type = float)#actor
    parser.add_argument("--lr_v", default = 0.001, type = float)#critic
    parser.add_argument("--lr_c", default = 0.001, type = float)#comm
    parser.add_argument("--h_dim_size", default = 120, type = int)
    parser.add_argument("--GAE", default = False, type = bool, \
        help="Use advantage estimate; else use use one_step return")
    parser.add_argument("--discount", default = 0.99, type = float)
    parser.add_argument("--lambda_", default = 1.0, type = float)
    parser.add_argument("--network_decoder_type", default = "mlp", type = str)
    parser.add_argument("--comm_channels", default = 20, type = int)
    parser.add_argument("--comm_zero", default = False,action = 'store_true')
    parser.add_argument("--comm_mode", default="concat", type= str, help="Concat, sum or mean comm vectors.")
    
    parser.add_argument("--checkpoint_frequency", default = int(10e4), type=int)
    parser.add_argument("--render_rate", default= int(5e2 - 10), type=int)
    parser.add_argument("--render_length", default = 20, type= int, help="Number of episodes of rendering to save")
    parser.add_argument("--alternative_plot_dir", default=plot_dir, help = "Creates a single folder to store tensorboard plots for comparison")
    #parser.add_argument("--working_directory", default='home/desktop123/Desktop/a2c2_results', type = str)
    parser.add_argument("--working_directory", default=work_dir, type = str)
    parser.add_argument("--name", default='Test', type = str)

    parser.add_argument("--bench_num_runs", default = 500, type = int)
    parser.add_argument("--bench_render_length", default = 200, type = int)
    #print("Args in a2c2 main: \n {}".format(args))
    args =  parser.parse_args(args)
    #print("Args after parsing: \n {}".format(args))
    if args.map_shape % 1 != 0:
        print("Env map size even... adding one to map size")
        args.map_shape += 1
    args.map_shape = (args.map_shape, args.map_shape)
    
    run(args)
    # env = make_env(args)
    # print("Summary: {}".format(env.summary()))
    # obs = env.reset()
    # a = [{i:0 for i in range(3)}]
    # obs2 = env.step(a)
    # print(obs2)

if __name__ == "__main__":
    main(sys.argv[1:])