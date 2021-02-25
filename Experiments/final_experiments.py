import main
from sklearn.model_selection import ParameterGrid
import argparse
from tabulate import tabulate
from Env.make_env import make_env
from Agents.PPO.utils import make_parallel_env
from gym import spaces
import config
import numpy as np
from PIL import Image
import time
import signal
import torch
import os
import copy
import itertools
from Agents.PPO.ppo import PPO
from Agents.PPO.PRIMAL_ppo import PPO_PRIMAL
import moviepy.editor as mpy

from Env.env import Independent_NavigationV8_1
from Env.env import Independent_NavigationV8_0


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


def benchmark_func(env_args, recurrent, model, num_episodes, render_len, device, greedy = False, valid_act = False):

    results = {"agents_on_goal": [],
                "all_done": [],
                "episode_length": [],
                "execution_time": [],
                "obstacle_collisions": [],
                "agent_collisions": []}
    env = make_parallel_env(env_args, np.random.randint(0, 10000), 1)
    render_frames = []
    model.actors[0].eval()
    terminal_info = []
    render_frames.append(env.render(indices = [0])[0])

    for ep in range(num_episodes):
        obs = env.reset()
        if recurrent:
            hx_cx_actr = [{i:model.init_hx_cx(device) for i in ob.keys()} for ob in obs]
            hx_cx_cr = [{i:model.init_hx_cx(device) for i in ob.keys()} for ob in obs]
        else:
            hx_cx_actr = [{i:(None,None) for i in ob.keys()} for ob in obs]
            hx_cx_cr = [{i:(None,None) for i in ob.keys()} for ob in obs]
        info2 = [{"valid_act":{i:None for i in ob.keys()}} for ob in obs]
        total_ep_time = 0
        for t in itertools.count():
            t1 = time.time()
            if valid_act:
                val_act_hldr = copy.deepcopy(env.return_valid_act())
                info2 = [{"valid_act":hldr} for hldr in val_act_hldr]

            a_probs, a_select, value, hx_cx_actr_n, hx_cx_cr_n, blocking = zip(*[model.forward(ob,ha,hc, dev=device, valid_act_heur = inf2["valid_act"], greedy = greedy) \
                                    for ob,ha,hc, inf2 in zip(obs, hx_cx_actr, hx_cx_cr, info2)])
            
            a_env_dict = [{key:val.item() for key,val in hldr.items()} for hldr in a_select]
            next_obs, r, dones, info = env.step(a_env_dict)
            total_ep_time += (time.time() - t1)

            hx_cx_actr = [{k:v for k,v in hldr.items()} for hldr in hx_cx_actr_n]
            hx_cx_cr = [{k:v for k,v in hldr.items()} for hldr in hx_cx_cr_n]
            if info[0]["terminate"]:
                results["agents_on_goal"].append(info[0]["agent_dones"])
                results["all_done"].append(info[0]["all_agents_on_goal"])
                results["episode_length"].append(t+1)
                results["obstacle_collisions"].append(info[0]["total_obstacle_collisions"])
                results["agent_collisions"].append(info[0]["total_agent_collisions"])
                results["execution_time"].append(total_ep_time)
                total_ep_time = 0

            if ep < render_len:
                if info[0]["terminate"]:
                    render_frames.append(info[0]["terminal_render"])
                else:
                    render_frames.append(env.render(indices = [0])[0])
            obs = copy.deepcopy(next_obs)
            if info[0]["terminate"]:
                terminal_info.append(info[0])
                break
    
    return render_frames, results


def run_PRIMAL(args, num_trials, render_len = 5):
    CHECKPOINT_PATH = '/home/james/Desktop/Gridworld/EXPERIMENTS/FINAL_COMPARISON/Checkpoint_Policies/primal/checkpoint_1800'
    env_name = "independent_navigation-v8_1"
    args.env_name = env_name
    env = Independent_NavigationV8_1(args)
    obs_space = env.observation_space[-1]
    ppo = PPO_PRIMAL(5, obs_space, "primal9", env.n_agents, True, True, 8, 512,recurrent=True)
    model_info = torch.load(CHECKPOINT_PATH)
    ppo.load(model_info["model"])
    DEVICE = 'cpu'
    render_frames, results = benchmark_func(args, True, ppo, num_trials,render_len, DEVICE, greedy = False, valid_act=False)
    return render_frames, results



def run_bc(args, num_trials, render_len = 5):
    CHECKPOINT_PATH = "/home/james/Desktop/Gridworld/EXPERIMENTS/FINAL_COMPARISON/Checkpoint_Policies/bc/checkpoint_20"
    env_name = "independent_navigation-v8_0"
    args.env_name = env_name
    env = Independent_NavigationV8_0(args)
    obs_space = env.observation_space[-1]
    ppo = PPO(5, obs_space, "primal7", env.n_agents, True, True, 8, 512,recurrent=False)
    model_info = torch.load(CHECKPOINT_PATH)
    ppo.load(model_info)
    DEVICE = 'cpu'
    render_frames, results = benchmark_func(args, True, ppo, num_trials, render_len, DEVICE)
    return render_frames, results

def run_mstar(args, num_trials, render_len):
    env = Independent_NavigationV8_0(args)
    TIME_LIMIT = 10*60 
    INFLATION = 1.1
    
    render_frames = []

    results = {"agents_on_goal": [],
                "all_done": [],
                "episode_length": [],
                "execution_time": [],
                "obstacle_collisions": [],
                "agent_collisions": []}


    def signal_handler(signum, frame):
        raise Exception("Time expired")

    handle = signal.signal(signal.SIGALRM, signal_handler)

    for i in range(num_trials):
        render_cntr = render_len
        _ = env.reset()
        start_pos = make_start_postion_list(env)
        end_pos = make_end_postion_list(env)
        all_actions, time_taken = None, None

        signal.alarm(TIME_LIMIT)   
        try:
            all_actions, time_taken =  env.graph.mstar_search4_OD(start_pos, end_pos, inflation=INFLATION, return_time_taken=True)
        except:
            print("ODMstar time ran out of time")
        finally:
            signal.signal(signal.SIGALRM, handle)
            signal.alarm(0)


        if all_actions is not None:
            results['agents_on_goal'].append(1)
            results["all_done"].append(1)
            results["episode_length"].append(len(all_actions))
        else:
            results['agents_on_goal'].append(0)
            results["all_done"].append(0)
            results["episode_length"].append(None)
        
        if time_taken is not None:
            results["execution_time"].append(float(time_taken))
        else:
            results["execution_time"].append(None)
        
        if render_cntr > 0 and all_actions is not None:
            render_cntr -= 1
            for a in all_actions:
                _, _, _, _ = env.step(a)
                frame = env.render()
                render_frames.append(frame)
    return render_frames, results

def save_render(render_frams, path, name):
    if len(render_frams) > 0:
        clip = mpy.ImageSequenceClip(render_frams, fps = 5)
        hldr2 = "render_" + name + '.mp4'
        hldr = os.path.join(path, hldr2)
        clip.write_videofile(hldr)
    

def run_final_1_0():
    FILE_PATH = "/home/james/Desktop/Gridworld/EXPERIMENTS/FINAL_COMPARISON/"
    RENDER_PATH = "/home/james/Desktop/Gridworld/EXPERIMENTS/FINAL_COMPARISON/RENDER"
    NUM_TRIALS = 20
    #TIME_LIMIT = 10*60

    name ="test_mstar-v0"
   # mode = "human"
    parser = argparse.ArgumentParser("Testing")

    #Environment:
    parser.add_argument("--map_shape", default = (32,32), type=object)
    parser.add_argument("--n_agents", default = 10, type=int)
    parser.add_argument("--view_d", default = 4, type=int)
    parser.add_argument("--env_name", default = name, type= str)
    parser.add_argument("--use_default_rewards", default=True, type=bool)
    parser.add_argument("--obj_density", default = 0.2, type=int)
    parser.add_argument("--use_custom_rewards", default = False, action='store_true')
    parser.add_argument("--custom_env_ind", default= 1, type=int)
    parser.add_argument("--ppo_heur_block", default=False, type=bool)

    args = parser.parse_args()

    #env = Independent_NavigationV8_0(args)

    # #10x10
    map_shape = (10,10)
    args.map_shape = map_shape
    for ob_density in [0.0, 0.1,0.2,0.3]:
        for n in [4,8,12,16,20,24,28]:
            args.n_agents = n
            args.obj_density = ob_density
            name =  "primalFull_" + "map_shape_" + str(map_shape[0]) + "_density_" + str(ob_density) + "_nagents_" + str(n) 
            data_dict = {"name": name,
                        "primalFull": None}#,
                        #"bc": None
                        #}

            args.view_d = 4
            render_frames, results = run_PRIMAL(args, NUM_TRIALS)
            data_dict["primalFull"] = copy.deepcopy(results)
            save_render(render_frames, RENDER_PATH, "primal_" + name)

            # args.view_d = 3
            # render_frames, results = run_bc(args, NUM_TRIALS)
            # data_dict["bc"] = copy.deepcopy(results)
            # save_render(render_frames, RENDER_PATH, "bc_" + name)

            # render_frames, results = run_mstar(args, NUM_TRIALS, render_len=10)
            # data_dict["mstar"] = copy.deepcopy(results)
            # save_render(render_frames, RENDER_PATH, "mstar_" + name)

            torch.save(data_dict, os.path.join(FILE_PATH, name +".pkl"))

    #32x32
    map_shape = (32,32)
    args.map_shape = map_shape
    for ob_density in [0.0, 0.1,0.2,0.3]:
        for n in [20,40,80]: #[10,20,30,40,50,60,70]:
            args.n_agents = n
            args.obj_density = ob_density
            name =  "primalFull_" + "map_shape_" + str(map_shape[0]) + "_density" + str(ob_density) + "_nagents_" + str(n) 
            data_dict = {"name": name,
                        "primalFull": None} #,
                        #"bc": None}

            args.view_d = 4
            render_frames, results = run_PRIMAL(args, NUM_TRIALS)
            data_dict["primal"] = copy.deepcopy(results)
            save_render(render_frames, RENDER_PATH, "primal_" + name)

            # args.view_d = 3
            # render_frames, results = run_bc(args, NUM_TRIALS)
            # data_dict["bc"] = copy.deepcopy(results)
            # save_render(render_frames, RENDER_PATH, "bc_" + name)

            # render_frames, results = run_mstar(args, NUM_TRIALS, render_len=10)
            # data_dict["mstar"] = copy.deepcopy(results)
            # save_render(render_frames, RENDER_PATH, "mstar_" + name)

            torch.save(data_dict, os.path.join(FILE_PATH, name +".pkl"))

    #50x50
    map_shape = (50,50)
    args.map_shape = map_shape
    for ob_density in [0.0, 0.1, 0.2, 0.3]:
        for n in [120,80,20]: #[10,20,30,40,50,60,70,80,90,100,110,120]:
            args.n_agents = n
            args.obj_density = ob_density
            name = "primalFull_" + "map_shape_" + str(map_shape[0]) + "_density" + str(ob_density) + "_nagents_" + str(n) 
            data_dict = {"name": name,
                        "primalFull": None} #,
                        #"bc": None}

            args.view_d = 4
            render_frames, results = run_PRIMAL(args, NUM_TRIALS)
            data_dict["primal"] = copy.deepcopy(results)
            save_render(render_frames, RENDER_PATH, "primal_" + name)

            # args.view_d = 3
            # render_frames, results = run_bc(args, NUM_TRIALS)
            # data_dict["bc"] = copy.deepcopy(results)
            # save_render(render_frames, RENDER_PATH, "bc_" + name)

            # render_frames, results = run_mstar(args, NUM_TRIALS, render_len=10)
            # data_dict["mstar"] = copy.deepcopy(results)
            # save_render(render_frames, RENDER_PATH, "mstar_" + name)

            torch.save(data_dict, os.path.join(FILE_PATH, name +".pkl"))
