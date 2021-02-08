# #For manual testing:
import argparse
from tabulate import tabulate
from Env.make_env import make_env
from gym import spaces
from sklearn.model_selection import ParameterGrid
from distutils.dir_util import copy_tree
# import config
# import numpy as np
# from PIL import Image
# import time
import os
import torch
import torch.nn.functional as F
from Agents.PPO.ppo import PPO
from Agents.Ind_PPO import benchmark_func
import math
import numpy as np
from utils.logger import Logger
import torch.optim as optim
from Agents.BC.data_generator import generate_data
import copy


def get_files_in_fldr(path):
    files_in_folder = None
    rt = None
    for (root, dirs, files) in os.walk(path, topdown=True):
        files_in_folder = files
        rt = root
        break
    files_in_folder = [os.path.join(rt, f) for f in files_in_folder]
    return files_in_folder

def get_file_names_in_fldr(path):
    files_in_folder = None
    rt = None
    for (root, dirs, files) in os.walk(path, topdown=True):
        files_in_folder = files
        rt = root
        break
    return files_in_folder

def get_data_files(fldr_path):
    all_files = get_files_in_fldr(fldr_path)
    data_str = ["train", "test", "validation"]
    dat_files = {}
    for f in all_files:
        for d_type in data_str:
            if d_type in f.split('/')[-1]:
                dat_files[d_type] = f
                break
    return dat_files["train"], dat_files["test"], dat_files["validation"]


def combine_all_data(folder_path):
    '''Combine all the data files in a folder and
    returns a list of the data and a list of the 
    file paths '''
    files_in_folder = None
    for (root, dirs, files) in os.walk(folder_path, topdown=True):
        files_in_folder = files
        break
    data = []
    files_in_folder = [os.path.join(folder_path, f) for f in files_in_folder]
    for file in files_in_folder:
        data.append(torch.load(file))
    
    all_data = []
    #all_obs = []
    #all_a = []
    for d in data:
        all_data.extend(d)
        #all_obs.extend(d["observations"])
        #all_a.extend(d["actions"])
    
    # for i in range(len(all_obs)):
    #     headers = ["Channels"]
    #     rows = [["Obstacle Channel"], ["Other Agent Channel"], ["Own Goals Channel"], \
    #         ["Own Position Channel"], ["Other Goal Channel"]]
    #     #for agnt in range(args.n_agents):
    #     #headers.append("Agent {}".format(agnt))
    #     rows[0].append(all_obs[i][0])
    #     rows[1].append(all_obs[i][1])
    #     rows[2].append(all_obs[i][2])
    #     rows[3].append(all_obs[i][4])
    #     rows[4].append(all_obs[i][3])
    #     print(tabulate(rows, tablefmt = 'fancy_grid'))
    #     print("Action: {}".format(all_a[i]))

    #data = {"observations":np.array(all_obs), "actions": np.array(all_a)}
    return all_data, files_in_folder

def delete_files(files):
    for f in files:
        os.remove(f)

def split_data(data, save_folder, name, ratio = "70:15:15", shuffle=True):
    '''Takes a list of the data and
    splits it into train, test and validation sets '''
    #assert len(data["observations"]) == len(data["actions"])
    #data_len = len(data["observations"])
    data_len = len(data)
    ratios = [int(r) for r in ratio.split(":")]
    ind = [(data_len / sum(ratios) )*r for r in ratios]
    ind = [math.floor(r) for r in ind[:-1]]
    ind_train = math.floor(ind[0])
    ind_validate = math.floor(ind[0] + ind[1])
    #ind.append(data_len-1)

    data_item_ind = np.arange(data_len)

    if shuffle:
        data_item_ind = np.random.choice(data_item_ind, data_len, replace = False)
    
    #train_data = (data["observations"][data_item_ind[:ind_train]] ,data["actions"][data_item_ind[:ind_train]])
    #validation_data = (data["observations"][data_item_ind[ind_train:ind_validate]] ,data["actions"][data_item_ind[ind_train:ind_validate]])
    #test_data = (data["observations"][data_item_ind[ind_validate:]] , data["actions"][data_item_ind[ind_validate:]])

    train_data = data[:ind_train]
    validation_data = data[ind_train:ind_validate]
    test_data = data[ind_validate:]
    # for i in range(0, data_len,4):
    #     headers = ["Channels"]
    #     rows = [["Obstacle Channel"], ["Other Agent Channel"], ["Own Goals Channel"], \
    #         ["Own Position Channel"], ["Other Goal Channel"]]
    #     #for agnt in range(args.n_agents):
    #     #headers.append("Agent {}".format(agnt))
    #     rows[0].append(train_data[i][0][0])
    #     rows[1].append(train_data[i][0][1])
    #     rows[2].append(train_data[i][0][2])
    #     rows[3].append(train_data[i][0][4])
    #     rows[4].append(train_data[i][0][3])
    #     print(tabulate(rows, tablefmt = 'fancy_grid'))
    #     print("Action: {}".format(train_data[i][1]))
    
    train_path = os.path.join(save_folder, name + "_train.pt")
    torch.save(train_data, train_path)
    validation_path = os.path.join(save_folder, name + "_validation.pt")
    torch.save(validation_data, validation_path)
    test_path = os.path.join(save_folder, name + "_test.pt")
    torch.save(test_data, test_path)
    return train_path, validation_path, test_path



def BC_train():
    #Helper functions
    def loss_f(pred, label):
        action_label_prob = torch.gather(pred,-1, label.long())
        log_actions = -torch.log(action_label_prob)
        loss = log_actions.mean()
        return loss

    def get_validation_loss(validate_loader, ppo_policy):
        with torch.no_grad():
            mae_like = 0
            total = 0
            valid_loss_hldr = []
            for ob,a in validate_loader:
                (a_pred, _, _, _) = ppo_policy.actors[0].forward(ob)
                valid_loss_hldr.append(loss_f(a_pred, a).item())
        return np.mean(valid_loss_hldr)
            #valid_loss = {"validation_loss": torch.mean(valid_loss_hldr).item()}
            #logger.plot_tensorboard(valid_loss)
            
    def save(model, logger, end):
        name = "checkpoint_" + str(end)
        checkpoint_path = os.path.join(logger.checkpoint_dir, name)
        model.save(checkpoint_path)


    experiment_group_name = "BC_5x5"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"
   # work_dir = experiment_group_name
  #  plot_dir = experiment_group_name + "_Central"

    # 
    parser = argparse.ArgumentParser("Generate Data")

    #
    parser.add_argument("--map_shape", default = (5,5), type=object)
    parser.add_argument("--n_agents", default = 4, type=int)
    parser.add_argument("--env_name", default = 'independent_navigation-v0', type= str)
    parser.add_argument("--use_default_rewards", default=True, type=bool)
    parser.add_argument("--obj_density", default = 0.2, type=int)
    parser.add_argument("--use_custom_rewards", default = False, action='store_true')
    parser.add_argument("--custom_env_ind", default= 1, type=int)
    parser.add_argument("--ppo_recurrent", default= False, action='store_true')

    parser.add_argument("--alternative_plot_dir", default="none")
    parser.add_argument("--working_directory", default="none")
    parser.add_argument("--name", default="NO_NAME", type=str, help="Experiment name")

    args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
    
    DEVICE = 'gpu'

    parmeter_grid1 = { #1:mbsize_32_lr_0.0001_epochs_100_weightdecay_0.0001  2:ize_32_lr_5e-05_epochs_40_weightdecay_0.0001
        "bc_mb_size": [32],
        "bc_lr": [0.000005], #0.0001
        "n_epoch": [150],
        "weight_decay": [0.0001]
    }
    grid1 = ParameterGrid(parmeter_grid1)

    data_folder_path = '/home/james/Desktop/Gridworld/BC_Data/5x5'

    combine_data = False

    if combine_data:
        data, files = combine_all_data(data_folder_path)
        delete_files(files)
        train_f, val_f, test_f = split_data(data, data_folder_path, "5x5")
    else:
        train_f, val_f, test_f = get_data_files(data_folder_path)
        
    #Get data from files:
    train_data = torch.load(train_f)
    val_data = torch.load(val_f)
    test_data = torch.load(test_f)

    for param in grid1:
        name = "BC_" + "5x5_" \
            + "mbsize_" + str(param["bc_mb_size"]) \
            + "_lr_" + str(param["bc_lr"]) \
            + "_epochs_" + str(param["n_epoch"]) \
            + "_weightdecay_" + str(param["weight_decay"]) \

        args.extend(["--name", name])
        args = parser.parse_args(args)

        ppo = PPO(5, spaces.Box(low=0, high=1, shape= (5,5,5), dtype=int),
                "primal6", 
                1, True, 
            True, 1, 
                1, 
                param["bc_lr"], 0.001, 
                120, 0.2, 0.01, False,
                False)

        logger = Logger(args, "5x5", "none", ppo)

        #Make training data loader
        (obs, actions) = zip(*train_data)
        (obs, actions) = (np.array(obs), np.array(actions))
        ppo.prep_device(DEVICE)
        obs = ppo.tens_to_dev(DEVICE, torch.from_numpy(obs).float())
        action_labels = ppo.tens_to_dev(DEVICE, torch.from_numpy(actions).reshape((-1,1)).float())
        train_dataset = torch.utils.data.TensorDataset(obs, action_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=param["bc_mb_size"], shuffle=True)

        #Make validation data_loader
        (obs, actions) = zip(*val_data)
        (obs, actions) = (np.array(obs), np.array(actions))
        obs = ppo.tens_to_dev(DEVICE, torch.from_numpy(obs).float())
        val_action_labels = ppo.tens_to_dev(DEVICE, torch.from_numpy(actions).reshape((-1,1)).float())
        valid_dataset = torch.utils.data.TensorDataset(obs, val_action_labels)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=param["bc_mb_size"], shuffle=True)


       # optimizer = ppo.actors[0].optimizer

        optimizer = optim.Adam(ppo.actors[0].parameters(), lr = param["bc_lr"], weight_decay=param["weight_decay"])

        #Train:
        #print("Nr iterations: {}".format(len(train_loader) / param["bc_mb_size"]))
        #print("dataset size: {}".format(len(train_data)))
        print(name)
        for epoch in range(param["n_epoch"]):
            epoch_loss_hldr = []
            iterations = 0
            for data in train_loader:
                ob, a = data
                (a_pred, _, _, _) = ppo.actors[0].forward(ob)
                loss = loss_f(a_pred, a)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss_hldr.append(loss.item())
                iterations += 1
            if (epoch+1) % 2 == 0:
                save(ppo, logger, epoch)

            if (epoch+1) % 10 == 0:
                ppo.extend_agent_indexes(args.n_agents)
                rend_frames, all_info = benchmark_func(args, ppo, 100, 30, DEVICE, greedy=True)
                logger.benchmark_info(all_info, rend_frames, epoch, end_str = "_greedy")
                rend_frames, all_info = benchmark_func(args, ppo, 100, 30, DEVICE, greedy=False)
                logger.benchmark_info(all_info, rend_frames, epoch, end_str = "_NotGreedy")

            print("iterations: {}".format(iterations))
            epoch_loss = np.mean(epoch_loss_hldr)
            valid_loss = get_validation_loss(valid_loader, ppo)
            log_info = {"train_loss": epoch_loss,
                        "validation_loss": valid_loss}
            logger.plot_tensorboard_custom_keys(log_info, external_iteration=epoch)
            print("Epoch: {}  Train Loss: {} Validation Loss: {}".format(epoch, epoch_loss, valid_loss))
        print("Done")


        #Save policy
        # name = "checkpoint_0"
        # checkpoint_path = os.path.join(logger.checkpoint_dir, name)
        # ppo.save(checkpoint_path)
        save(ppo, logger, "end")

        #Evaluate policy (benchmark)
        ppo.extend_agent_indexes(args.n_agents)
        rend_frames, all_info = benchmark_func(args, ppo, 100, 30, DEVICE)
        logger.benchmark_info(all_info, rend_frames, param["n_epoch"]+1)

def evaluate_checkpoint():

    #CHECKPOINT_PATH = "/home/james/Desktop/Gridworld/EXPERIMENTS/BC1AgentShortestPath_2_V0/BC_BC1AgentShortestPath_2_V0mbsize_32_lr_5e-05_epochs_50_weightdecay_0.0001_N0/checkpoint/checkpoint_40"
    #experiment_group_name = "Results_Shortest Path"

    #CHECKPOINT_PATH = "/home/james/Desktop/Gridworld/EXPERIMENTS/BC1AgentDirVec_2_V1/BC_BC1AgentDirVec_2_V1mbsize_32_lr_5e-05_epochs_50_weightdecay_0.0001_N0/checkpoint/checkpoint_31"
    CHECKPOINT_PATH = "/home/james/Desktop/Gridworld/EXPERIMENTS/9_0_0/BC_9_0_0mbsize_32_lr_5e-05_epochs_50_weightdecay_0.0001_N0/checkpoint/checkpoint_31"
    experiment_group_name = "9_0_0"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name 

    parser = argparse.ArgumentParser("Train arguments")
    parser.add_argument("--alternative_plot_dir", default="none")
    parser.add_argument("--working_directory", default="none")
    parser.add_argument("--name", default="9_0_0_benchmark", type=str, help="Experiment name")
    parser.add_argument("--replace_checkpoints", default=True, type=bool)
    #Placeholders:
    parser.add_argument("--env_name", default="independent_navigation-v8_0", type=str)
    parser.add_argument("--n_agents", default=1, type=int)
    parser.add_argument("--map_shape", default=(5,5), type=object)
    parser.add_argument("--obj_density", default=0.0, type=float)
    parser.add_argument("--view_d", default=3, type=int)
    parser.add_argument("--use_default_rewards", default=True, type=bool)
    parser.add_argument("--use_custom_rewards", default=False, type=bool)
    parser.add_argument("--base_path", default= "none", type=str)
    args = parser.parse_args()

    args.working_directory = work_dir
    args.alternative_plot_dir = plot_dir
##########
    args = make_env_args(args, {})
    env_hldr = make_env(args)
    observation_space = env_hldr.observation_space[-1]


    ppo = PPO(5, observation_space,
            "primal7", 
            1, True, 
        True, 1, 
            1, 
            0.001,  0.001, 
            120, 0.2, 0.01, False,
            False)

    ppo.load(torch.load(CHECKPOINT_PATH))
    logger = Logger(args, "NONE" , "none",ppo)


    # variable_args_dict = {
    #     "n_agents": [1],
    #     "obj_density": [0.0,0.1, 0.2, 0.3], 
    #     "map_shape": [(30, 30), (40, 40)]
    # }

    # evaluate_across_evs(ppo, logger, args, variable_args_dict, 1000, 30, 'gpu', greedy=False)
    
    # 32 x 32
    variable_args_dict = {
        "n_agents": [10,30,35,40,45,50, 60, 70],
        "obj_density": [0.0,0.1,0.2,0.3], 
        "map_shape": [(32, 32)]
    }
    evaluate_across_evs(ppo, logger, args, variable_args_dict, 1000, 30, 'gpu', greedy=False)

    # 40 x 40
    variable_args_dict = {
        "n_agents": [10,30,35,40,45,50, 60, 70],
        "obj_density": [0.0,0.1,0.2,0.3], 
        "map_shape": [(40,40)]
    }
    evaluate_across_evs(ppo, logger, args, variable_args_dict, 1000, 30, 'gpu', greedy=False)

    # 50 x 50
    variable_args_dict = {
        "n_agents": [10,30,35,40,45,50, 60, 70],
        "obj_density": [0.0,0.1,0.2,0.3], 
        "map_shape": [(50,50)]
    }
    evaluate_across_evs(ppo, logger, args, variable_args_dict, 1000, 30, 'gpu', greedy=False)



def evaluate_checkpoint2():

    #CHECKPOINT_PATH = "/home/james/Desktop/Gridworld/EXPERIMENTS/BC1AgentShortestPath_2_V0/BC_BC1AgentShortestPath_2_V0mbsize_32_lr_5e-05_epochs_50_weightdecay_0.0001_N0/checkpoint/checkpoint_40"
    #experiment_group_name = "Results_Shortest Path"

    #CHECKPOINT_PATH = "/home/james/Desktop/Gridworld/EXPERIMENTS/BC1AgentDirVec_2_V1/BC_BC1AgentDirVec_2_V1mbsize_32_lr_5e-05_epochs_50_weightdecay_0.0001_N0/checkpoint/checkpoint_31"
    #CHECKPOINT_PATH = "/home/james/Desktop/Gridworld/EXPERIMENTS/5_2_0_CL/ppo_arc_primal7_sr_-0.1_ocr_-0.4_acr_-0.4_grr_0.3_fer_2.0_viewd_3_disc_0.5_lambda_1.0_entropy_0.01_minibatch_512_rollouts_256_workers_4_kepochs_8_curr_ppo_cl_inc_size_seed_1_N4/checkpoint/checkpoint_17600"
    CHECKPOINT_PATH = "/home/james/Desktop/Gridworld/EXPERIMENTS/5_3_0_CL/ppo_arc_primal7_sr_-0.1_ocr_-0.4_acr_-0.4_grr_0.3_fer_2.0_viewd_3_disc_0.5_lambda_1.0_entropy_0.01_minibatch_512_rollouts_256_workers_4_kepochs_8_curr_ppo_cl_inc_size_seed_1_N0/checkpoint/checkpoint_5200"
    #CHECKPOINT_PATH = "/home/james/Desktop/Gridworld/EXPERIMENTS/9_0_0/BC_9_0_0mbsize_32_lr_5e-05_epochs_50_weightdecay_1e-05_N0/checkpoint/checkpoint_20"
    experiment_group_name = "5_3_0_CL_benchmark3"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name 

    parser = argparse.ArgumentParser("Train arguments")
    parser.add_argument("--alternative_plot_dir", default="none")
    parser.add_argument("--working_directory", default="none")
    parser.add_argument("--name", default="benchmark3", type=str, help="Experiment name")
    parser.add_argument("--replace_checkpoints", default=True, type=bool)
    #Placeholders:
    parser.add_argument("--env_name", default="independent_navigation-v0", type=str)
    parser.add_argument("--n_agents", default=1, type=int)
    parser.add_argument("--map_shape", default=(7,7), type=object)
    parser.add_argument("--obj_density", default=0.0, type=float)
    parser.add_argument("--view_d", default=3, type=int)
    parser.add_argument("--use_default_rewards", default=True, type=bool)
    parser.add_argument("--use_custom_rewards", default=False, type=bool)
    parser.add_argument("--base_path", default= "none", type=str)
    args = parser.parse_args()

    args.working_directory = work_dir
    args.alternative_plot_dir = plot_dir
##########
    args = make_env_args(args, {})
    env_hldr = make_env(args)
    observation_space = env_hldr.observation_space[-1]


    ppo = PPO(5, observation_space,
            "primal7", 
            1, True, 
        True, 1, 
            1, 
            0.001,  0.001, 
            120, 0.2, 0.01, False,
            False)

    ppo.load(torch.load(CHECKPOINT_PATH))
    logger = Logger(args, "NONE" , "none",ppo)


    # variable_args_dict = {
    #     "n_agents": [10,20,30,40],
    #     "obj_density": [0.0,0.1, 0.2, 0.3], 
    #     "map_shape": [(32, 32)]
    # }

    # evaluate_across_evs(ppo, logger, args, variable_args_dict, 200, 10, 'gpu', greedy=False)
    
    variable_args_dict = {
        "n_agents": [4],
        "obj_density": [0.0, 0.1,0.2,0.3], 
        "map_shape": [(7, 7)]
    }
    evaluate_across_evs(ppo, logger, args, variable_args_dict, 500, 30, 'gpu', greedy=False)


    # variable_args_dict = {
    #     "n_agents": [2,4,8,16],
    #     "obj_density": [0.0], 
    #     "map_shape": [(7, 7)]
    # }
    # evaluate_across_evs(ppo, logger, args, variable_args_dict, 500, 30, 'gpu', greedy=False)

    # # 40 x 40
    # variable_args_dict = {
    #     "n_agents": [10,30,35,40,45,50, 60, 70],
    #     "obj_density": [0.0,0.1,0.2,0.3], 
    #     "map_shape": [(40,40)]
    # }
    # evaluate_across_evs(ppo, logger, args, variable_args_dict, 1000, 30, 'gpu', greedy=False)

    # # 50 x 50
    # variable_args_dict = {
    #     "n_agents": [10,30,35,40,45,50, 60, 70],
    #     "obj_density": [0.0,0.1,0.2,0.3], 
    #     "map_shape": [(50,50)]
    # }
    # evaluate_across_evs(ppo, logger, args, variable_args_dict, 1000, 30, 'gpu', greedy=False)




def evaluate_across_evs(policy, logger, env_args, variable_args_dict, num_episodes, render_len, device, greedy=False): 
    param_grid = ParameterGrid(variable_args_dict)
    for param in param_grid:
        this_param_dict ={k: param[k] for k in variable_args_dict.keys()}
        this_env_args = make_env_args(env_args, this_param_dict)
        policy.extend_agent_indexes(this_env_args.n_agents)
        end_str = ""
        for k, v in this_param_dict.items():
            end_str += '_'
            end_str += str(k)
            end_str += "_"
            if k == "map_shape" and type(v)==tuple:
                end_str += str(v[-1])
            else:
                end_str += str(v)
        if greedy:
            end_str += "_Greedy"
        else:
            end_str += "_NotGreedy"
            
        rend_frames, all_info = benchmark_func(this_env_args, policy, num_episodes, render_len, device, greedy=greedy)
        logger.init_custom_benchmark_logging_fldr(end_str)
        logger.benchmark_info(all_info, rend_frames, 0, end_str = end_str, dont_init_ben = True)

    
def train_PO_FOV_data(custom_args = None):
    '''Check if data has been processed into train, val and test sets.
        If not, make copy of data, and process into diff sets.
        Then starts training with given parameters. '''
    #Helper functions
    def loss_f(pred, label):
        action_label_prob = torch.gather(pred,-1, label.long())
        log_actions = -torch.log(action_label_prob)
        loss = log_actions.mean()
        return loss

    def get_validation_loss(validate_loader, ppo_policy):
        with torch.no_grad():
            mae_like = 0
            total = 0
            valid_loss_hldr = []
            for data in validate_loader:
                if len(data) ==2:
                    ob, a = data
                elif len(data) == 3:
                    ob = (data[0], data[1])
                    a = data[-1]
                else:
                    raise Exception("Data incorrect length")
                (a_pred, _, _, _) = ppo_policy.actors[0].forward(ob)
                valid_loss_hldr.append(loss_f(F.softmax(a_pred), a).item())
        return np.mean(valid_loss_hldr)
            #valid_loss = {"validation_loss": torch.mean(valid_loss_hldr).item()}
            #logger.plot_tensorboard(valid_loss)
            
    def save(model, logger, end):
        name = "checkpoint_" + str(end)
        #checkpoint_path = os.path.join(logger.checkpoint_dir, name)
        #model.save(checkpoint_path)
        logger.make_checkpoint(False, str(end))

    def is_processesed(path):
        '''Returns bool whether or not there exists 
            three files for train, val and test '''
        files = get_file_names_in_fldr(path)
        file_markers = ["train", "test", "validation"]
        is_processed_flag = True
        for m in file_markers:
            sub_flag = False
            for f in files:
                if m in f:
                    sub_flag = True
            if sub_flag == False:
                is_processed_flag = False
                break
        return is_processed_flag
            

    parser = argparse.ArgumentParser("Train arguments")
    parser.add_argument("--folder_name", type= str)
    parser.add_argument("--mb_size", default=32, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--n_epoch", default=50, type=int)
    parser.add_argument("--weight_decay", default=0.0001, type=float)
    parser.add_argument("--env_name", default='none', type=str)
    parser.add_argument("--alternative_plot_dir", default="none")
    parser.add_argument("--working_directory", default="none")
    parser.add_argument("--name", default="NO_NAME", type=str, help="Experiment name")
    parser.add_argument("--replace_checkpoints", default=True, type=bool)
    #Placeholders:
    parser.add_argument("--n_agents", default=1, type=int)
    parser.add_argument("--map_shape", default=(5,5), type=object)
    parser.add_argument("--obj_density", default=0.0, type=float)
    parser.add_argument("--view_d", default=3, type=int)
    parser.add_argument("--use_default_rewards", default=True, type=bool)
    parser.add_argument("--use_custom_rewards", default=False, type=bool)
    parser.add_argument("--base_path", default= "none", type=str)

    #Best previous performing hyp param is: mb:32 lr:5e-5  weightdecay: 0.0001 

    if custom_args is None:
        args = parser.parse_args()
    else:
        args, unkn = parser.parse_known_args(custom_args)


    experiment_group_name = args.folder_name #"BC_5x5"
    if args.working_directory == "none":
        import __main__
        work_dir = os.path.join(os.path.dirname(__main__.__file__), '/EXPERIMENTS/', experiment_group_name)
    else:
        work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    
    if args.alternative_plot_dir == "none":
        import __main__
        plot_dir = os.path.join(os.path.dirname(__main__.__file__), '/CENTRAL_TENSORBOARD/', experiment_group_name)
    else:
        plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name 

    work_dir = "/home/jellis/workspace2/gridworld/EXPERIMENTS/" + args.folder_name
    plot_dir ="/home/jellis/workspace2/gridoworld/CENTRAL_TENSORBOARD/"+ args.folder_name
    args.working_directory = work_dir
    args.alternative_plot_dir = plot_dir

    #BASE_DATA_FOLDER_PATH = '/home/james/Desktop/Gridworld/BC_Data'

    if args.base_path == "none":
        import __main__
        BASE_DATA_FOLDER_PATH = os.path.dirname(__main__.__file__)
        BASE_DATA_FOLDER_PATH = os.path.join(BASE_DATA_FOLDER_PATH, "BC_Data")
    else:
        #base_path = args.base_path
        BASE_DATA_FOLDER_PATH = '/home/james/Desktop/Gridworld/BC_Data'

    data_fldr_path = os.path.join(BASE_DATA_FOLDER_PATH, args.folder_name)
    data_fldr_path = "/home/jellis/workspace2/BC_Data/"
    if is_processesed(data_fldr_path):
        train_f, val_f, test_f = get_data_files(data_fldr_path)
    else:
        #Copy data:
        to_dir = data_fldr_path + "_cpy"
        os.makedirs(to_dir)
        copy_tree(data_fldr_path, to_dir)
        #Split and save data:
        data, files = combine_all_data(data_fldr_path)
        delete_files(files)
        train_f, val_f, test_f = split_data(data, data_fldr_path, "data_")


    DEVICE = 'gpu'
    #Train on data: keep best policy
    #Get data from files:
    train_data = torch.load(train_f)
    val_data = torch.load(val_f)
    test_data = torch.load(test_f)

    env_hldr = make_env(args)
    observation_space = env_hldr.observation_space[-1]

    name = "BC_" + args.folder_name \
            + "mbsize_" + str(args.mb_size) \
            + "_lr_" + str(args.lr) \
            + "_epochs_" + str(args.n_epoch) \
            + "_weightdecay_" + str(args.weight_decay)

        #args.extend(["--name", name])
    #args = parser.parse_args(args)
    args.name = name

    ppo = PPO(5, observation_space,
            "primal7", 
            1, True, 
        True, 1, 
            1, 
            args.lr, 0.001, 
            120, 0.2, 0.01, False,
            False)

    logger = Logger(args, env_hldr.summary() , "none", ppo)

    #Make training data loader
    (obs, actions) = zip(*train_data)
    #(obs, actions) = (np.array(obs), np.array(actions))
    if type(observation_space) == tuple:
        (obs1, obs2) = zip(*obs)
        (obs, actions) = ((np.array(obs1), np.array(obs2)), np.array(actions))
    else:
        (obs, actions) = (np.array(obs), np.array(actions))
    ppo.prep_device(DEVICE)


    if type(observation_space) == tuple:
        obs = (ppo.tens_to_dev(DEVICE, torch.from_numpy(obs[0]).float()), \
            ppo.tens_to_dev(DEVICE, torch.from_numpy(obs[1]).float()))
    else:
        obs = [ ppo.tens_to_dev(DEVICE, torch.from_numpy(obs).float()) ]

   # obs = ppo.tens_to_dev(DEVICE, torch.from_numpy(obs).float())
    action_labels = ppo.tens_to_dev(DEVICE, torch.from_numpy(actions).reshape((-1,1)).float())
    train_dataset = torch.utils.data.TensorDataset(*obs, action_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.mb_size, shuffle=True)

    #Make validation data_loader
    (obs, actions) = zip(*val_data)
    #(obs, actions) = (np.array(obs), np.array(actions))
    if type(observation_space) == tuple:
        (obs1, obs2) = zip(*obs)
        (obs, actions) = ((np.array(obs1), np.array(obs2)), np.array(actions))
    else:
        (obs, actions) = (np.array(obs), np.array(actions))
    #obs = ppo.tens_to_dev(DEVICE, torch.from_numpy(obs).float())
    if type(observation_space) == tuple:
        obs = (ppo.tens_to_dev(DEVICE, torch.from_numpy(obs[0]).float()), \
            ppo.tens_to_dev(DEVICE, torch.from_numpy(obs[1]).float()))
    else:
        obs = [ppo.tens_to_dev(DEVICE, torch.from_numpy(obs).float())]

    val_action_labels = ppo.tens_to_dev(DEVICE, torch.from_numpy(actions).reshape((-1,1)).float())
    valid_dataset = torch.utils.data.TensorDataset(*obs, val_action_labels)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.mb_size, shuffle=True)

    #Make test data_loader
    (obs, actions) = zip(*test_data)
    #(obs, actions) = (np.array(obs), np.array(actions))
    if type(observation_space) == tuple:
        (obs1, obs2) = zip(*obs)
        (obs, actions) = ((np.array(obs1), np.array(obs2)), np.array(actions))
    else:
        (obs, actions) = (np.array(obs), np.array(actions))
    #obs = ppo.tens_to_dev(DEVICE, torch.from_numpy(obs).float())
    if type(observation_space) == tuple:
        obs = (ppo.tens_to_dev(DEVICE, torch.from_numpy(obs[0]).float()), \
            ppo.tens_to_dev(DEVICE, torch.from_numpy(obs[1]).float()))
    else:
        obs = [ppo.tens_to_dev(DEVICE, torch.from_numpy(obs).float())]

    test_action_labels = ppo.tens_to_dev(DEVICE, torch.from_numpy(actions).reshape((-1,1)).float())
    test_dataset = torch.utils.data.TensorDataset(*obs, test_action_labels)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.mb_size, shuffle=True)

    optimizer = optim.Adam(ppo.actors[0].parameters(), lr = args.lr, weight_decay=args.weight_decay)

    #Train:
    prev_val_loss = 1e6
    best_epoch_nr = None
    val_loss_is_greater_cntr = 0
    val_loss_is_greater_threshhold = 8
    best_policy = None
    print(name)
    for epoch in range(args.n_epoch):
        epoch_loss_hldr = []
        iterations = 0
        for data in train_loader:
            #ob, a = data
            if len(data) ==2:
                ob, a = data
            elif len(data) == 3:
                ob = (data[0], data[1])
                a = data[-1]
            else:
                raise Exception("Data incorrect length")
            (a_pred, _, _, _) = ppo.actors[0].forward(ob)
            loss = loss_f(F.softmax(a_pred), a)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_hldr.append(loss.item())
            iterations += 1

        # if (epoch+1) % 10 == 0:
        #     ppo.extend_agent_indexes(args.n_agents)
        #     rend_frames, all_info = benchmark_func(args, ppo, 100, 30, DEVICE, greedy=True)
        #     logger.benchmark_info(all_info, rend_frames, epoch, end_str = "_greedy")
        #     rend_frames, all_info = benchmark_func(args, ppo, 100, 30, DEVICE, greedy=False)
        #     logger.benchmark_info(all_info, rend_frames, epoch, end_str = "_NotGreedy")

        print("iterations: {}".format(iterations))
        epoch_loss = np.mean(epoch_loss_hldr)
        valid_loss = get_validation_loss(valid_loader, ppo)
        test_loss = get_validation_loss(test_loader, ppo)
        log_info = {"train_loss": epoch_loss,
                    "validation_loss": valid_loss,
                    "test_loss": test_loss}
                    
        logger.plot_tensorboard_custom_keys(log_info, external_iteration=epoch)
        if valid_loss < prev_val_loss:
            save(ppo, logger, epoch)
            best_policy = copy.deepcopy(ppo)
            prev_val_loss = valid_loss
            val_loss_is_greater_cntr = 0
            best_epoch_nr = epoch
        else:
            val_loss_is_greater_cntr +=1
        

        print("Epoch: {}  Train Loss: {} Validation Loss: {}".format(epoch, epoch_loss, valid_loss))
        if val_loss_is_greater_cntr > val_loss_is_greater_threshhold:
            print("Ending training")
            break
    print("Done")

    # free up memory:
    try:
        del train_data 
        del val_data 
        del test_data 
        del action_labels 
        del train_dataset
        del train_loader
        del val_action_labels
        del valid_dataset
        del valid_loader
    except:
        pass



    # #Evaluate best policy across all ens
    # assert not best_policy is None
    # print("Best epoch nr is {}".format(best_epoch_nr))
    # variable_args_dict = {
    #     "obj_density": [0.0,0.1,0.2,0.3], 
    #     "map_shape": [7,10,15,20,25,30]
    # }
    # variable_args_dict["map_shape"] = [(ms, ms) for ms in variable_args_dict["map_shape"]]
    # #evaluate_across_evs(best_policy, logger, args, variable_args_dict, 1000, 30, DEVICE, greedy=True)
    # evaluate_across_evs(best_policy, logger, args, variable_args_dict, 1000, 30, DEVICE, greedy=False)


    assert not best_policy is None
    print("Best epoch nr is {}".format(best_epoch_nr))

    # 32 x 32
    variable_args_dict = {
        "n_agents": [4, 10,30,35,40,45,50, 60, 70],
        "obj_density": [0.0,0.1,0.2,0.3], 
        "map_shape": [(32,32)]
    }
    evaluate_across_evs(best_policy, logger, args, variable_args_dict, 1000, 30, DEVICE, greedy=False)

    # 40 x 40
    variable_args_dict = {
        "n_agents": [4, 10,30,35,40,45,50, 60, 70],
        "obj_density": [0.0,0.1,0.2,0.3], 
        "map_shape": [(40,40)]
    }
    evaluate_across_evs(best_policy, logger, args, variable_args_dict, 1000, 30, DEVICE, greedy=False)


def make_env_args(args_parse, args_dict):
    '''Returns a class with same attributes as in  '''
    class Base():
        def __init__(self):
            self.env_name = "independent_navigation-v8_0"
            self.map_shape = (5,5)#(map_size, map_size)
            self.n_agents = 1 #n_agents
            self.view_d = 3
            self.obj_density = 0.0 #obj_density
            self.name = "NONAME" #"mapsize_" + str(map_size) + "nagents" + str(n_agents) + "_objdensity" + str(obj_density)
            self.use_custom_rewards = False
            self.custom_env_ind = -1
            self.ppo_recurrent = False
            self.ppo_heur_block = False
            self.ppo_heur_valid_act = False
            self.ppo_heur_no_prev_state = False

    env_args = Base()

    # Copy args in argparse
    for k, v in vars(args_parse).items():
        setattr(env_args, k, v)

    #Set args in Env args to that of args_dict
    for k, v in args_dict.items():
        if not hasattr(env_args, k):
            print("Adding new attribute: {}".format(k))
        setattr(env_args, k, v)
    
    return env_args

    


def generate_PO_FOV_data(custom_args = None): 
    '''Generates 5000 samples of each env '''
    parser = argparse.ArgumentParser("Generate PO FOV data")
    parser.add_argument("--folder_name", default='none', type=str)
    parser.add_argument("--env_name", default='independent_navigation-v8_0', type= str)
    parser.add_argument("--n_episodes", default=5000, type= int)
    parser.add_argument("--n_agents", default=1, type= int)
    if custom_args is None:
        gen_args, unkn = parser.parse_known_args()
    else:
        gen_args, unkn = parser.parse_known_args(custom_args)

    gen_dat_custom_args = []
    gen_dat_custom_args.extend(["--env_name", gen_args.env_name])
    gen_dat_custom_args.extend(["--folder_name", gen_args.folder_name])
    gen_dat_custom_args.extend(["--n_agents", str(gen_args.n_agents)])
    
    gen_dat_custom_args.extend(["--n_episodes", str(gen_args.n_episodes)])
    gen_dat_custom_args.extend(["--view_d", "3"])

    ob_dens = [0.0,0.1,0.2,0.3]
    env_sizes = [10,20,30]

    # ob_dens = [0.0]
    # env_sizes = [30]

    for e_s in env_sizes:
        for ob_d in ob_dens:
            gen_dat_custom_args.extend(["--map_shape", str(e_s)])
            gen_dat_custom_args.extend(["--obj_density", str(ob_d)])
            dat_name = "envsize_" + str(e_s) + "_obdensity_" + str(ob_d)
            gen_dat_custom_args.extend(["--data_name", dat_name])
            generate_data(custom_args=gen_dat_custom_args)


#### Experiments: Imitation Learning Approach
#############################################

def generate_9_0_0(custom_args = None): 
    '''Generates 5000 samples of each env '''
    parser = argparse.ArgumentParser("Generate PO FOV data")
    parser.add_argument("--folder_name", default='9_0_0', type=str)
    parser.add_argument("--env_name", default='independent_navigation-v8_0', type= str)
    parser.add_argument("--n_episodes", default=50, type= int)
    parser.add_argument("--n_agents", default=1, type= int)
    if custom_args is None:
        gen_args, unkn = parser.parse_known_args()
    else:
        gen_args, unkn = parser.parse_known_args(custom_args)

    gen_dat_custom_args = []
    gen_dat_custom_args.extend(["--env_name", gen_args.env_name])
    gen_dat_custom_args.extend(["--folder_name", gen_args.folder_name])
    gen_dat_custom_args.extend(["--n_agents", str(gen_args.n_agents)])
    
    gen_dat_custom_args.extend(["--n_episodes", str(gen_args.n_episodes)])
    gen_dat_custom_args.extend(["--view_d", "3"])

    env_sizes = [10]
    nagents = [4]
    ob_dens = [0.0,0.1,0.2,0.3]

    for e_s in env_sizes:
        for n in nagents:
            for ob_d in ob_dens:
                gen_dat_custom_args.extend(["--map_shape", str(e_s)])
                gen_dat_custom_args.extend(["--obj_density", str(ob_d)])
                dat_name = "envsize_" + str(e_s) + "_obdensity_" + str(ob_d)
                gen_dat_custom_args.extend(["--data_name", dat_name])
                generate_data(custom_args=gen_dat_custom_args)

def generate_9_0_1(custom_args = None): 
    '''Generates 5000 samples of each env '''
    parser = argparse.ArgumentParser("Generate PO FOV data")
    parser.add_argument("--folder_name", default='9_0_0', type=str)
    parser.add_argument("--env_name", default='independent_navigation-v8_0', type= str)
    parser.add_argument("--n_episodes", default=5000, type= int)
    parser.add_argument("--n_agents", default=1, type= int)
    if custom_args is None:
        gen_args, unkn = parser.parse_known_args()
    else:
        gen_args, unkn = parser.parse_known_args(custom_args)

    gen_dat_custom_args = []
    gen_dat_custom_args.extend(["--env_name", gen_args.env_name])
    gen_dat_custom_args.extend(["--folder_name", gen_args.folder_name])
    #gen_dat_custom_args.extend(["--n_agents", str(gen_args.n_agents)])
    
    gen_dat_custom_args.extend(["--n_episodes", str(gen_args.n_episodes)])
    gen_dat_custom_args.extend(["--view_d", "3"])

    # 10x10 Env:
    env_sizes = [10]
    nagents = [4,8,12,16,18]
    ob_dens = [0.0,0.1,0.2,0.3]

    for e_s in env_sizes:
        for n in nagents:
            for ob_d in ob_dens:
                gen_dat_custom_args.extend(["--map_shape", str(e_s)])
                gen_dat_custom_args.extend(["--obj_density", str(ob_d)])
                gen_dat_custom_args.extend(["--n_agents", str(gen_args.n_agents)])
                dat_name = "envsize_" + str(e_s) + "_nagents_" + str(n) + "_obdensity_" + str(ob_d)
                gen_dat_custom_args.extend(["--data_name", dat_name])
                generate_data(custom_args=gen_dat_custom_args)
    
    # 15 x 15
    env_sizes = [15]
    nagents = [10,15,20]
    ob_dens = [0.0,0.1,0.2,0.3]
    for e_s in env_sizes:
        for n in nagents:
            for ob_d in ob_dens:
                gen_dat_custom_args.extend(["--map_shape", str(e_s)])
                gen_dat_custom_args.extend(["--obj_density", str(ob_d)])
                gen_dat_custom_args.extend(["--n_agents", str(gen_args.n_agents)])
                dat_name = "envsize_" + str(e_s) + "_nagents_" + str(n) + "_obdensity_" + str(ob_d)
                gen_dat_custom_args.extend(["--data_name", dat_name])
                generate_data(custom_args=gen_dat_custom_args)
###############################################
    # 20 x 20
    env_sizes = [20]
    nagents = [20,25,30,35]
    ob_dens = [0.0,0.1,0.2,0.3]
    for e_s in env_sizes:
        for n in nagents:
            for ob_d in ob_dens:
                gen_dat_custom_args.extend(["--map_shape", str(e_s)])
                gen_dat_custom_args.extend(["--obj_density", str(ob_d)])
                gen_dat_custom_args.extend(["--n_agents", str(gen_args.n_agents)])
                dat_name = "envsize_" + str(e_s) + "_nagents_" + str(n) + "_obdensity_" + str(ob_d)
                gen_dat_custom_args.extend(["--data_name", dat_name])
                generate_data(custom_args=gen_dat_custom_args)

    # 25 x 25
    env_sizes = [25]
    nagents = [25,30,35,40]
    ob_dens = [0.0,0.1,0.2,0.3]
    for e_s in env_sizes:
        for n in nagents:
            for ob_d in ob_dens:
                gen_dat_custom_args.extend(["--map_shape", str(e_s)])
                gen_dat_custom_args.extend(["--obj_density", str(ob_d)])
                gen_dat_custom_args.extend(["--n_agents", str(gen_args.n_agents)])
                dat_name = "envsize_" + str(e_s) + "_nagents_" + str(n) + "_obdensity_" + str(ob_d)
                gen_dat_custom_args.extend(["--data_name", dat_name])
                generate_data(custom_args=gen_dat_custom_args)


    # 30 x 30
    env_sizes = [32]
    nagents = [30,35,40, 45,50]
    ob_dens = [0.0,0.1,0.2,0.3]
    for e_s in env_sizes:
        for n in nagents:
            for ob_d in ob_dens:
                gen_dat_custom_args.extend(["--map_shape", str(e_s)])
                gen_dat_custom_args.extend(["--obj_density", str(ob_d)])
                gen_dat_custom_args.extend(["--n_agents", str(gen_args.n_agents)])
                dat_name = "envsize_" + str(e_s) + "_nagents_" + str(n) + "_obdensity_" + str(ob_d)
                gen_dat_custom_args.extend(["--data_name", dat_name])
                generate_data(custom_args=gen_dat_custom_args)
