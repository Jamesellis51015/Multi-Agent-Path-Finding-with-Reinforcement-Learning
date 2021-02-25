import torch
import torch.nn.functional as F
import numpy as np
from Agents.PPO.ppo import PPO
from Agents.PPO.PRIMAL_ppo import PPO_PRIMAL
from Agents.PPO.utils import make_parallel_env
from Agents.PPO.buffer import PPO_Buffer
import itertools
from utils.logger import Logger
import copy

from utils.curr_logger import CurriculumLogger
from Env.curriculum.curriculum_manager import CurriculumManager

#from Experiments.final_experiments import run_ppo_benchmark
#CrossEntropyLoss = nn.CrossEntropyLoss()

#heplper functions:
def get_n_obs(next_obs, info):
    nobs = []
    for ob, inf in zip(next_obs, info):
        if inf["terminate"]:
            nobs.append(inf["terminal_observation"])
        else:
            nobs.append(ob)
    return nobs



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
            a_probs, a_select, value, hx_cx_actr_n, hx_cx_cr_n, blocking = zip(*[model.forward(ob,ha,hc, dev=device, valid_act_heur = inf2["valid_act"]) \
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


def bc_training_iteration2(args, env_args, ppo, device, minibatch_size, n_sub_updates, logger = None, env_id =None, bench_freq = None, iteration=None):
    def make_start_postion_list(env_hldr):
        '''Assumes agent keys in evn.agents is the same as agent id's '''
        start_positions = []
        for i in range(len(env_hldr.agents.values())):
            start_positions.append(env_hldr.agents[i].pos)
        return start_positions

    def make_end_postion_list(env_hldr):
        '''Assumes agent keys in evn.agents is the same as agent id's '''
        end_positions = []
        for i in range(len(env_hldr.goals.values())):
            end_positions.append(env_hldr.goals[i].pos)
        return end_positions
    #make single env:
    env = make_parallel_env(env_args, np.random.randint(0, 1e6), 1)
    #for i in range(n_sub_updates):
    obs = env.reset()

    info2 = [{"valid_act":{i:None for i in ob.keys()}} for ob in obs]
    if args.ppo_recurrent:
        hx_cx_actr = [{i:ppo.init_hx_cx(device) for i in ob.keys()} for ob in obs]
        hx_cx_cr = [{i:ppo.init_hx_cx(device) for i in ob.keys()} for ob in obs]
    else:
        hx_cx_actr = [{i:(None,None) for i in ob.keys()} for ob in obs]
        hx_cx_cr = [{i:(None,None) for i in ob.keys()} for ob in obs]

    # Get start and end_positions
    # run mstar
    # 
    
    num_samples = minibatch_size*n_sub_updates
    #for i in range(n_sub_updates):
    inflation = 1.2
    buff_a_probs = []
    buff_expert_a = []
    buff_blocking_pred = []
    buff_is_blocking = []
    buff_valid_act = []
    info = None
    #Collect samples:
    update_cntr = 0
    #while len(buff_a_probs) < num_samples:
    while update_cntr < n_sub_updates:
        if not info is None:
            assert info[0]["terminate"] == True
        env_hldr = env.return_env()
        #env_hldr[0].render(mode='human')
        start_pos = make_start_postion_list(env_hldr[0])
        end_pos = make_end_postion_list(env_hldr[0])

        for i in range(5):
            all_actions =  env_hldr[0].graph.mstar_search4_OD(start_pos, end_pos, inflation = inflation, memory_limit=4*1e6)
            if all_actions is None:
                inflation += 1.5
                if i==4:
                    print("Mstar ran out of memory. No solution found. Exiting behaviour cloning")
                    return
            else:
                break

        for i, mstar_action in enumerate(all_actions):
            env_hldr2 = env.return_env()
            if args.ppo_heur_valid_act:
                #val_act_hldr = copy.deepcopy(env.return_valid_act())
                #info2 = [{"valid_act":hldr} for hldr in val_act_hldr]
                buff_valid_act.append({k:env_hldr2[0].graph.get_valid_actions(k) for k in env_hldr2[0].agents.keys()})

            a_probs, a_select, value, hx_cx_actr_n, hx_cx_cr_n, blocking = zip(*[ppo.forward(ob,ha,hc, dev=device, valid_act_heur = inf2["valid_act"] ) \
                                                for ob,ha,hc, inf2 in zip(obs, hx_cx_actr, hx_cx_cr, info2)])
            
            #env_hldr2[0].render(mode='human')
            buff_a_probs.append(a_probs[0])
            buff_expert_a.append(mstar_action)
            if args.ppo_heur_block:
                buff_blocking_pred.append(blocking[0])
                #env_hldr2 = env.return_env()
                buff_is_blocking.append(copy.deepcopy(env_hldr2[0].blocking_hldr))

            next_obs, r, dones, info = env.step([mstar_action])

            next_obs_ = get_n_obs(next_obs, info)

            hx_cx_actr = [{k:v for k,v in hldr.items()} for hldr in hx_cx_actr_n]
            hx_cx_cr = [{k:v for k,v in hldr.items()} for hldr in hx_cx_cr_n]
            for i, inf in enumerate(info):
                if inf["terminate"] and args.ppo_recurrent:
                    hx_cx_actr, hx_cx_cr = list(hx_cx_actr), list(hx_cx_cr)
                    hx_cx_actr[i] = {i2:ppo.init_hx_cx(device) for i2 in hx_cx_actr[i].keys()}
                    hx_cx_cr[i] = {i2:ppo.init_hx_cx(device) for i2 in hx_cx_cr[i].keys()}
            obs = next_obs

            # if len(buff_a_probs) == num_samples:
            #     break
            if len(buff_a_probs) == minibatch_size:
                # Update:
                update_cntr += 1
                keys = buff_a_probs[0].keys()
                buff_a_probs_flat = []
                for ap in buff_a_probs:
                    for k in keys:
                        buff_a_probs_flat.append(ap[k])

                buff_expert_a_flat = []
                for hldr in buff_expert_a:
                    for k in keys:
                        buff_expert_a_flat.append(hldr[k])

                if args.ppo_heur_block:
                    buff_blocking_pred_flat = []
                    for hldr in buff_blocking_pred:
                        for k in keys:
                            buff_blocking_pred_flat.append(hldr[k])

                    buff_is_blocking_flat = []
                    for hldr in buff_is_blocking:
                        for k in keys:
                            buff_is_blocking_flat.append(hldr[k])
                if args.ppo_heur_valid_act:
                    buff_valid_act_flat = []
                    for time_step in buff_valid_act:
                        for k in keys:
                            hldr = time_step[k]
                            multi_hot = torch.zeros(size=(1, 5), dtype=torch.float32) 
                            for i in hldr:
                                multi_hot[0][i] = 1
                            buff_valid_act_flat.append(multi_hot)
                    

                #Discard excess samples:
                #all_data = []
                a_probs = torch.cat(buff_a_probs_flat[:num_samples])
                expert_a = torch.from_numpy(np.array(buff_expert_a_flat[:num_samples])).reshape(-1,1)
                expert_a = ppo.tens_to_dev(device, expert_a)
                #all_data.append(a_probs)
                #all_data.append(expert_a)

                if args.ppo_heur_valid_act:
                    buff_valid_act_flat = torch.cat(buff_valid_act_flat[:num_samples])
                    buff_valid_act_flat = ppo.tens_to_dev(device, buff_valid_act_flat)
                    #print("Buff valid act: {} \n {}".format(buff_valid_act, buff_valid_act_flat))

                if args.ppo_heur_block:
                    blocking_pred = torch.cat(buff_blocking_pred_flat[:num_samples])
                    #all_data.append(blocking_pred)
                    #make tensor of ones and zeros
                    block_bool = buff_is_blocking_flat[:num_samples]
                    is_blocking = torch.zeros(len(block_bool))
                    is_blocking[block_bool] = 1
                    #is_blocking = torch.cat(is_blocking)
                    is_blocking = is_blocking.reshape(-1, 1)
                    is_blocking = ppo.tens_to_dev(device, is_blocking)
                    #all_data.append(is_blocking)



                #train_dataset = torch.utils.data.TensorDataset(*all_data)
                #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = minibatch_size, shuffle=True)

                def loss_f_actions(pred, label):
                    action_label_prob = torch.gather(F.softmax(pred),-1, label.long())
                    log_actions = -torch.log(action_label_prob)
                    #Should sum before taking mean...
                    #This is equivalent to scaling the correct loss by 0.2.
                    loss = log_actions.mean()
                    return loss

                def loss_f_blocking(pred, label):
                    #pred_action_prob = torch.gather(pred,-1, label.long())
                    #categorical loss:
                    #test_categorical_loss = CrossEntropyLoss(pred, label.detach())
                    pred = torch.clamp(pred, 1e-15, 1.0)
                    loss = -(label*torch.log(pred) + (1 - label)*torch.log(1-pred))
                    #loss = -(label*torch.log(pred_action_prob) + (1-label)*torch.log(1-pred_action_prob))
                    #log_actions = -torch.log(action_label_prob)
                    #loss = log_actions.mean()
                    return loss.mean()

                def loss_f_valid(all_act_prob, valid_act_multi_hot):
                    sigmoid_act = F.sigmoid(all_act_prob)
                    
                    #hldr1 = 1-sigmoid_act
                    #hldr2 = 1-valid_act_multi_hot
                    #hldr3 = torch.log(sigmoid_act)

                    valid_act_loss = -(torch.log(sigmoid_act) * valid_act_multi_hot + torch.log(1-sigmoid_act)*(1-valid_act_multi_hot))
                    return valid_act_loss.mean()

                #indexes = np.arange(0,num_samples)
                #np.random.shuffle(indexes)
                #mb_start = 0
                #for i in range(minibatch_size, num_samples, minibatch_size):
                #ind = indexes[mb_start:i]
                #mb_start = i

                a_pred = a_probs #[ind] #data[0]
                a = expert_a #[ind] #data[1]
                action_loss = loss_f_actions(a_pred, a)
                loss2 = action_loss
                block_loss, vld_loss = None, None

                if args.ppo_heur_block:
                    block_loss = 0.5*loss_f_blocking(blocking_pred, is_blocking)
                    loss2 += block_loss

                if args.ppo_heur_valid_act:
                    vld_loss = 0.5*loss_f_valid(a_pred, buff_valid_act_flat)
                    loss2 += vld_loss
                
                
                ppo.actors[0].optimizer.zero_grad()
                loss2.backward(retain_graph = True)
                torch.nn.utils.clip_grad_norm_(ppo.actors[0].parameters(), 1000)
                ppo.actors[0].optimizer.step()
                buff_a_probs = []
                buff_expert_a = []
                buff_blocking_pred = []
                buff_is_blocking = []
                buff_valid_act = []
                if logger is not None:
                    hldr = dict()
                    hldr["bc_action_loss"] = action_loss.item()
                    if block_loss is not None:
                        hldr["bc_block_loss"] = block_loss.item()
                    if vld_loss is not None:
                        hldr["bc_valid_loss"] = vld_loss.item()
                    # if block_loss is not None and vld_loss is not None:
                    #     hldr = {"bc_action_loss": action_loss.item(),
                    #             "bc_block_loss": block_loss.item(),
                    #             "bc_valid_loss": vld_loss.item()}
                    logger.log_bc(hldr)
    # if bench_freq is not None and logger is not None and iteration is not None:
    #     if (iteration+1) % bench_freq  == 0:
    #         if env_id is not None:
    #             temp_id = env_id
    #         else:
    #             temp_id = 0
    #         hldr = iteration+5
    #         logger.benchmark(temp_id, hldr)
        #benchmark_func(env_args, args, ppo, 10,10, device) #(env_args,main_args, model, num_episodes, render_len, device)




def bc_training_iteration(args, env_args, ppo, device, minibatch_size, n_sub_updates):
    def make_start_postion_list(env_hldr):
        '''Assumes agent keys in evn.agents is the same as agent id's '''
        start_positions = []
        for i in range(len(env_hldr.agents.values())):
            start_positions.append(env_hldr.agents[i].pos)
        return start_positions

    def make_end_postion_list(env_hldr):
        '''Assumes agent keys in evn.agents is the same as agent id's '''
        end_positions = []
        for i in range(len(env_hldr.goals.values())):
            end_positions.append(env_hldr.goals[i].pos)
        return end_positions
    #make single env:
    env = make_parallel_env(env_args, np.random.randint(0, 1e6), 1)
    for i in range(n_sub_updates):
        obs = env.reset()

        info2 = [{"valid_act":{i:None for i in ob.keys()}} for ob in obs]
        if args.ppo_recurrent:
            hx_cx_actr = [{i:ppo.init_hx_cx(device) for i in ob.keys()} for ob in obs]
            hx_cx_cr = [{i:ppo.init_hx_cx(device) for i in ob.keys()} for ob in obs]
        else:
            hx_cx_actr = [{i:(None,None) for i in ob.keys()} for ob in obs]
            hx_cx_cr = [{i:(None,None) for i in ob.keys()} for ob in obs]

        # Get start and end_positions
        # run mstar
        # 
        buff_a_probs = []
        buff_expert_a = []
        buff_blocking_pred = []
        buff_is_blocking = []
        num_samples = minibatch_size
        info = None
        while len(buff_a_probs) < num_samples:
            if not info is None:
                assert info[0]["terminate"] == True
            env_hldr = env.return_env()
            #env_hldr[0].render(mode='human')
            start_pos = make_start_postion_list(env_hldr[0])
            end_pos = make_end_postion_list(env_hldr[0])

            all_actions =  env_hldr[0].graph.mstar_search4_OD(start_pos, end_pos, inflation = 1.2)
            
            for i, mstar_action in enumerate(all_actions):
                a_probs, a_select, value, hx_cx_actr_n, hx_cx_cr_n, blocking = zip(*[ppo.forward(ob,ha,hc, dev=device, valid_act_heur = inf2["valid_act"] ) \
                                                    for ob,ha,hc, inf2 in zip(obs, hx_cx_actr, hx_cx_cr, info2)])
                env_hldr2 = env.return_env()
                #env_hldr2[0].render(mode='human')
                buff_a_probs.append(a_probs[0])
                buff_expert_a.append(mstar_action)
                if args.ppo_heur_block:
                    buff_blocking_pred.append(blocking[0])
                    #env_hldr2 = env.return_env()
                    buff_is_blocking.append(copy.deepcopy(env_hldr2[0].blocking_hldr))


                #a_env_dict = [{key:val.item() for key,val in hldr.items()} for hldr in a_select]
                #next_obs, r, dones, info = env.step(a_env_dict)
                next_obs, r, dones, info = env.step([mstar_action])

                next_obs_ = get_n_obs(next_obs, info)

                

            # buff.add(obs, r, value, next_obs_, a_probs, a_select, info, dones, hx_cx_actr, hx_cx_cr, hx_cx_actr_n, hx_cx_cr_n, blocking)
                #Reset hidden and cell states when epidode done
                for i, inf in enumerate(info):
                    if inf["terminate"] and args.ppo_recurrent:
                        hx_cx_actr, hx_cx_cr = list(hx_cx_actr), list(hx_cx_cr)
                        hx_cx_actr[i] = {i2:ppo.init_hx_cx(device) for i2 in hx_cx_actr[i].keys()}
                        hx_cx_cr[i] = {i2:ppo.init_hx_cx(device) for i2 in hx_cx_cr[i].keys()}
                #if info[0]["terminate"]:
                #    assert i == len(all_actions)
                obs = next_obs

                if len(buff_a_probs) == num_samples:
                    break
        
        #train PPO policy with expert data:
       #print("test imitation learning ")

        keys = buff_a_probs[0].keys()
        buff_a_probs_flat = []
        for ap in buff_a_probs:
            for k in keys:
                buff_a_probs_flat.append(ap[k])

        buff_expert_a_flat = []
        for hldr in buff_expert_a:
            for k in keys:
                buff_expert_a_flat.append(hldr[k])

        if args.ppo_heur_block:
            buff_blocking_pred_flat = []
            for hldr in buff_blocking_pred:
                for k in keys:
                    buff_blocking_pred_flat.append(hldr[k])

            buff_is_blocking_flat = []
            for hldr in buff_is_blocking:
                for k in keys:
                    buff_is_blocking_flat.append(hldr[k])

        #Discard excess samples:
        all_data = []
        a_probs = torch.cat(buff_a_probs_flat[:num_samples])
        expert_a = torch.from_numpy(np.array(buff_expert_a_flat[:num_samples])).reshape(-1,1)
        expert_a = ppo.tens_to_dev(device, expert_a)
        all_data.append(a_probs)
        all_data.append(expert_a)

        if args.ppo_heur_block:
            blocking_pred = torch.cat(buff_blocking_pred_flat[:num_samples])
            all_data.append(blocking_pred)
            is_blocking = torch.cat(buff_is_blocking_flat[:num_samples])
            all_data.append(is_blocking)



        #train_dataset = torch.utils.data.TensorDataset(*all_data)
        #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = minibatch_size, shuffle=True)

        def loss_f_actions(pred, label):
            action_label_prob = torch.gather(pred,-1, label.long())
            log_actions = -torch.log(action_label_prob)
            loss = log_actions.mean()
            return loss

        def loss_f_blocking(pred, label):
            action_label_prob = torch.gather(pred,-1, label.long())
            log_actions = -torch.log(action_label_prob)
            loss = log_actions.mean()
            return loss
        a_pred = a_probs #data[0]
        a = expert_a #data[1]
        if args.ppo_heur_block:
            loss2 = loss_f_actions(a_pred, a) + loss_f_blocking(blocking_pred, is_blocking)
        loss2 = loss_f_actions(a_pred, a)
        ppo.actors[0].optimizer.zero_grad()
        loss2.backward()
        ppo.actors[0].optimizer.step()
    del env
        # for ind in range(minibatch_size, minibatch_size+num_samples, minibatch_size):
        #     a_pred = a_probs[ind-minibatch_size:ind] #data[0]
        #     a = expert_a[ind-minibatch_size:ind] #data[1]
        #     loss2 = loss_f_actions(a_pred, a)
        #     ppo.actors[0].optimizer.zero_grad()
        #     loss2.backward()
        #     ppo.actors[0].optimizer.step()
        # loss2 = None
        # for data in train_loader:
        #     if len(data) == 2:
        #         #No blocking present
        #         a_pred = data[0]
        #         a = data[1]
        #         loss2 = loss_f_actions(a_pred, a)
        #     elif len(data) == 4:
        #         #blocking present
        #         a_pred = data[0]
        #         a = data[1]
        #         block_pred = data[2]
        #         is_block = data[3]
        #         loss2 = loss_f_actions(a_pred, a) + loss_f_blocking(block_pred, is_block)
        #     else:
        #         raise Exception("Incorrect data length")
            
        #     ppo.actors[0].optimizer.zero_grad()
        #     loss2.backward()
        #     ppo.actors[0].optimizer.step()




def run(args): #Curriculum train:
    if args.ppo_use_gpu:
        device = 'gpu'
    else:
        device = 'cpu'

    assert args.ppo_bc_iteration_prob >= 0.0 and args.ppo_bc_iteration_prob <= 1.0

    PRIMAL = True
    if PRIMAL:
        print("Using PRIMAL network.")

    curr_manager = CurriculumManager(args, CurriculumLogger)
    #Get env args for first env
    env_args = curr_manager.init_env_args()
    #Determine number of workers for buffer and init env
    buff = PPO_Buffer(env_args.n_agents, args.ppo_workers, args.ppo_rollout_length, args.ppo_recurrent)
    seed = np.random.randint(0, 100000)
    env = make_parallel_env(env_args, seed, buff.nworkers)
    #Init ppo model
    if PRIMAL:
        ppo = PPO_PRIMAL(env.action_space[0].n, env.observation_space[0],
            args.ppo_base_policy_type, 
            env.n_agents[0], args.ppo_share_actor, 
            args.ppo_share_value, args.ppo_k_epochs, 
            args.ppo_minibatch_size, 
            args.ppo_lr_a, args.ppo_lr_v, 
            args.ppo_hidden_dim, args.ppo_eps_clip, args.ppo_entropy_coeff,
            args.ppo_recurrent,args.ppo_heur_block)
    else:
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
    up_bc = 0
    env.close()
    #primal_lr_start = 2e-5
    #primal_lr_const = 1e-3
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
       
        up_i = 0
        #for up_i in range(curr_manager.n_updates):
        new_lr=None
        while up_i < curr_manager.n_updates:
            #if PRIMAL:
            #    new_lr = primal_lr_start / np.sqrt(1 + primal_lr_const*(global_iterations+up_bc))
            #    ppo.set_lr(new_lr)
            bc_iteration = np.random.choice([True, False], p=[args.ppo_bc_iteration_prob, 1-args.ppo_bc_iteration_prob])
            if bc_iteration:
                print("bc iteration: {}, lr= {}".format(up_bc, new_lr))
                n_sub_updates = 4 #((args.ppo_workers * args.ppo_rollout_length) // args.ppo_minibatch_size)
                #if up_bc > 300:
                #    n_sub_updates = 1
                bc_training_iteration2(args, env_args, ppo, device, 128, n_sub_updates=n_sub_updates, logger = logger, env_id = env_id, bench_freq=20, iteration=up_bc)
                up_bc += 1
            else:
                print("Iteration: {}".format(global_iterations))
                info2 = [{"valid_act":{i:None for i in ob.keys()}} for ob in obs]
                while buff.is_full == False:
                    if args.ppo_heur_valid_act:
                        val_act_hldr = copy.deepcopy(env.return_valid_act())
                        info2 = [{"valid_act":hldr} for hldr in val_act_hldr]
                    else:
                        val_act_hldr = [{i:None for i in ob.keys()} for ob in obs]

                    a_probs, a_select, value, hx_cx_actr_n, hx_cx_cr_n, blocking = zip(*[ppo.forward(ob,ha,hc, dev=device, valid_act_heur = inf2["valid_act"] ) \
                                        for ob,ha,hc, inf2 in zip(obs, hx_cx_actr, hx_cx_cr, info2)])

                    a_env_dict = [{key:val.item() for key,val in hldr.items()} for hldr in a_select]
                    next_obs, r, dones, info = env.step(a_env_dict)
                    logger.record_render(env_id, env, info[0])
                    next_obs_ = get_n_obs(next_obs, info)

                    buff.add(obs, r, value, next_obs_, a_probs, a_select, info, dones, hx_cx_actr, hx_cx_cr, hx_cx_actr_n, hx_cx_cr_n, blocking, val_act_hldr)
                    
                    hx_cx_actr = [{k:v for k,v in hldr.items()} for hldr in hx_cx_actr_n]
                    hx_cx_cr = [{k:v for k,v in hldr.items()} for hldr in hx_cx_cr_n]
                    #Reset hidden and cell states when epidode done
                    for i, inf in enumerate(info):
                        if inf["terminate"] and args.ppo_recurrent:
                            hx_cx_actr, hx_cx_cr = list(hx_cx_actr), list(hx_cx_cr)
                            hx_cx_actr[i] = {i2:ppo.init_hx_cx(device) for i2 in hx_cx_actr[i].keys()}
                            hx_cx_cr[i] = {i2:ppo.init_hx_cx(device) for i2 in hx_cx_cr[i].keys()}
                    obs = next_obs
                    

                if buff.is_full:
                    observations, a_prob, a_select, adv, v, infos, h_actr, h_cr, blk_labels, blk_pred, val_act = buff.sample(args.ppo_discount,\
                    args.ppo_gae_lambda,args.ppo_gae_lambda, blocking=args.ppo_heur_block, use_valid_act = args.ppo_heur_valid_act)
                    if PRIMAL:
                        extra_stats["action_loss"], extra_stats["value_loss"],extra_stats["blocking_loss"],extra_stats["validact_loss"] \
                        = ppo.update(observations, a_prob, a_select, adv, v, 0, h_actr, h_cr, blk_labels, blk_pred,val_act, dev=device)
                    else:
                        extra_stats["action_loss"], extra_stats["value_loss"] \
                        = ppo.update(observations, a_prob, a_select, adv, v, 0, h_actr, h_cr, blk_labels, blk_pred,val_act, dev=device)
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
                up_i += 1
                # if up_i == 300:
                #     run_ppo_benchmark(ppo, up_i, number_trials=5)
                # else:
                #     if up_i > 300 and up_i % 100 == 0:
                #         run_ppo_benchmark(ppo, up_i, number_trials=5)
        env.close()
    print("Done")





# def bc_training_iteration2(args, env_args, ppo, device, minibatch_size, n_sub_updates, logger = None):
#     def make_start_postion_list(env_hldr):
#         '''Assumes agent keys in evn.agents is the same as agent id's '''
#         start_positions = []
#         for i in range(len(env_hldr.agents.values())):
#             start_positions.append(env_hldr.agents[i].pos)
#         return start_positions

#     def make_end_postion_list(env_hldr):
#         '''Assumes agent keys in evn.agents is the same as agent id's '''
#         end_positions = []
#         for i in range(len(env_hldr.goals.values())):
#             end_positions.append(env_hldr.goals[i].pos)
#         return end_positions
#     #make single env:
#     env = make_parallel_env(env_args, np.random.randint(0, 1e6), 1)
#     #for i in range(n_sub_updates):
#     obs = env.reset()

#     info2 = [{"valid_act":{i:None for i in ob.keys()}} for ob in obs]
#     if args.ppo_recurrent:
#         hx_cx_actr = [{i:ppo.init_hx_cx(device) for i in ob.keys()} for ob in obs]
#         hx_cx_cr = [{i:ppo.init_hx_cx(device) for i in ob.keys()} for ob in obs]
#     else:
#         hx_cx_actr = [{i:(None,None) for i in ob.keys()} for ob in obs]
#         hx_cx_cr = [{i:(None,None) for i in ob.keys()} for ob in obs]

#     # Get start and end_positions
#     # run mstar
#     # 
    
#     num_samples = minibatch_size
#     for i in range(n_sub_updates):
#         inflation = 1.2
#         buff_a_probs = []
#         buff_expert_a = []
#         buff_blocking_pred = []
#         buff_is_blocking = []
#         buff_valid_act = []
#         info = None
#         while len(buff_a_probs) < num_samples-1:
#             if not info is None:
#                 assert info[0]["terminate"] == True
#             env_hldr = env.return_env()
#             #env_hldr[0].render(mode='human')
#             start_pos = make_start_postion_list(env_hldr[0])
#             end_pos = make_end_postion_list(env_hldr[0])

#             for i in range(5):
#                 all_actions =  env_hldr[0].graph.mstar_search4_OD(start_pos, end_pos, inflation = inflation, memory_limit=3.5*1e6)
#                 if all_actions is None:
#                     inflation += 1.5
#                     if i==4:
#                         print("Mstar ran out of memory. No solution found. Exiting behaviour cloning")
#                         return
#                 else:
#                     break
                
            
#             for i, mstar_action in enumerate(all_actions):
#                 env_hldr2 = env.return_env()
#                 if args.ppo_heur_valid_act:
#                     #val_act_hldr = copy.deepcopy(env.return_valid_act())
#                     #info2 = [{"valid_act":hldr} for hldr in val_act_hldr]
#                     buff_valid_act.append({k:env_hldr2[0].graph.get_valid_actions(k) for k in env_hldr2[0].agents.keys()})

#                 a_probs, a_select, value, hx_cx_actr_n, hx_cx_cr_n, blocking = zip(*[ppo.forward(ob,ha,hc, dev=device, valid_act_heur = inf2["valid_act"] ) \
#                                                     for ob,ha,hc, inf2 in zip(obs, hx_cx_actr, hx_cx_cr, info2)])
                
#                 #env_hldr2[0].render(mode='human')
#                 buff_a_probs.append(a_probs[0])
#                 buff_expert_a.append(mstar_action)
#                 if args.ppo_heur_block:
#                     buff_blocking_pred.append(blocking[0])
#                     #env_hldr2 = env.return_env()
#                     buff_is_blocking.append(copy.deepcopy(env_hldr2[0].blocking_hldr))

#                 next_obs, r, dones, info = env.step([mstar_action])

#                 next_obs_ = get_n_obs(next_obs, info)

#                 hx_cx_actr = [{k:v for k,v in hldr.items()} for hldr in hx_cx_actr_n]
#                 hx_cx_cr = [{k:v for k,v in hldr.items()} for hldr in hx_cx_cr_n]
#                 for i, inf in enumerate(info):
#                     if inf["terminate"] and args.ppo_recurrent:
#                         hx_cx_actr, hx_cx_cr = list(hx_cx_actr), list(hx_cx_cr)
#                         hx_cx_actr[i] = {i2:ppo.init_hx_cx(device) for i2 in hx_cx_actr[i].keys()}
#                         hx_cx_cr[i] = {i2:ppo.init_hx_cx(device) for i2 in hx_cx_cr[i].keys()}
#                 obs = next_obs

#                 if len(buff_a_probs) == num_samples:
#                     keys = buff_a_probs[0].keys()
#                     buff_a_probs_flat = []
#                     for ap in buff_a_probs:
#                         for k in keys:
#                             buff_a_probs_flat.append(ap[k])

#                     buff_expert_a_flat = []
#                     for hldr in buff_expert_a:
#                         for k in keys:
#                             buff_expert_a_flat.append(hldr[k])

#                     if args.ppo_heur_block:
#                         buff_blocking_pred_flat = []
#                         for hldr in buff_blocking_pred:
#                             for k in keys:
#                                 buff_blocking_pred_flat.append(hldr[k])

#                         buff_is_blocking_flat = []
#                         for hldr in buff_is_blocking:
#                             for k in keys:
#                                 buff_is_blocking_flat.append(hldr[k])
#                     if args.ppo_heur_valid_act:
#                         buff_valid_act_flat = []
#                         for time_step in buff_valid_act:
#                             for k in keys:
#                                 hldr = time_step[k]
#                                 multi_hot = torch.zeros(size=(1, 5), dtype=torch.float32) 
#                                 for i in hldr:
#                                     multi_hot[0][i] = 1
#                                 buff_valid_act_flat.append(multi_hot)
                        

#                     #Discard excess samples:
#                     #all_data = []
#                     a_probs = torch.cat(buff_a_probs_flat[:num_samples])
#                     expert_a = torch.from_numpy(np.array(buff_expert_a_flat[:num_samples])).reshape(-1,1)
#                     expert_a = ppo.tens_to_dev(device, expert_a)
#                     #all_data.append(a_probs)
#                     #all_data.append(expert_a)

#                     if args.ppo_heur_valid_act:
#                         buff_valid_act_flat = torch.cat(buff_valid_act_flat[:num_samples])
#                         buff_valid_act_flat = ppo.tens_to_dev(device, buff_valid_act_flat)

#                     if args.ppo_heur_block:
#                         blocking_pred = torch.cat(buff_blocking_pred_flat[:num_samples])
#                         #all_data.append(blocking_pred)
#                         #make tensor of ones and zeros
#                         block_bool = buff_is_blocking_flat[:num_samples]
#                         is_blocking = torch.zeros(len(block_bool))
#                         is_blocking[block_bool] = 1
#                         #is_blocking = torch.cat(is_blocking)
#                         is_blocking = is_blocking.reshape(-1, 1)
#                         is_blocking = ppo.tens_to_dev(device, is_blocking)
#                         #all_data.append(is_blocking)



#                     #train_dataset = torch.utils.data.TensorDataset(*all_data)
#                     #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = minibatch_size, shuffle=True)

#                     def loss_f_actions(pred, label):

#                         action_label_prob = torch.gather(F.softmax(pred),-1, label.long())
#                         log_actions = -torch.log(action_label_prob)
#                         loss = log_actions.mean()
#                         return loss

#                     def loss_f_blocking(pred, label):
#                         #pred_action_prob = torch.gather(pred,-1, label.long())
#                         #categorical loss:
#                         #test_categorical_loss = CrossEntropyLoss(pred, label.detach())
#                         pred = torch.clamp(pred, 1e-15, 1.0)
#                         loss = -(label*torch.log(pred) + (1 - label)*torch.log(1-pred))
#                         #loss = -(label*torch.log(pred_action_prob) + (1-label)*torch.log(1-pred_action_prob))
#                         #log_actions = -torch.log(action_label_prob)
#                         #loss = log_actions.mean()
#                         return loss.mean()

#                     def loss_f_valid(all_act_prob, valid_act_multi_hot):
#                         sigmoid_act = F.sigmoid(all_act_prob)
                        
#                         #hldr1 = 1-sigmoid_act
#                         #hldr2 = 1-valid_act_multi_hot
#                         #hldr3 = torch.log(sigmoid_act)

#                         valid_act_loss = -(torch.log(sigmoid_act) * valid_act_multi_hot + torch.log(1-sigmoid_act)*(1-valid_act_multi_hot))
#                         return valid_act_loss.mean()

#                     a_pred = a_probs #data[0]
#                     a = expert_a #data[1]
#                     action_loss = loss_f_actions(a_pred, a)
#                     loss2 = action_loss
#                     block_loss, vld_loss = None, None

#                     if args.ppo_heur_block:
#                         block_loss = 0.5*loss_f_blocking(blocking_pred, is_blocking)
#                         loss2 += block_loss

#                     if args.ppo_heur_valid_act:
#                         vld_loss = 0.5*loss_f_valid(a_pred, buff_valid_act_flat)
#                         #loss2 += vld_loss
                    
#                     if logger is not None:
#                         if block_loss is not None and vld_loss is not None:
#                             hldr = {"bc_action_loss": action_loss.item(),
#                                     "bc_block_loss": block_loss.item(),
#                                     "bc_valid_loss": vld_loss.item()}
#                             logger.log_bc(hldr)
#                     ppo.actors[0].optimizer.zero_grad()
#                     loss2.backward()
#                     torch.nn.utils.clip_grad_norm_(ppo.actors[0].parameters(), 1000)
#                     ppo.actors[0].optimizer.step()
#     del env








