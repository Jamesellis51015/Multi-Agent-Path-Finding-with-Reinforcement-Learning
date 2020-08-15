# #For manual testing:
import argparse
from tabulate import tabulate
# from Env.make_env import make_env
from gym import spaces
from sklearn.model_selection import ParameterGrid
# import config
# import numpy as np
# from PIL import Image
# import time
import os
import torch
from Agents.PPO.ppo import PPO
from Agents.Ind_PPO import benchmark_func
import math
import numpy as np
from utils.logger import Logger
import torch.optim as optim


def get_files_in_fldr(path):
    files_in_folder = None
    rt = None
    for (root, dirs, files) in os.walk(path, topdown=True):
        files_in_folder = files
        rt = root
        break
    files_in_folder = [os.path.join(rt, f) for f in files_in_folder]
    return files_in_folder

def get_data_files(fldr_path):
    all_files = get_files_in_fldr(fldr_path)
    data_str = ["train", "test", "validation"]
    dat_files = {}
    for f in all_files:
        for d_type in data_str:
            if d_type in f:
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
        "bc_lr": [0.00005], #0.0001
        "n_epoch": [40],
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
                # #######################
                # headers = ["Channels"]
                # rows = [["Obstacle Channel"], ["Other Agent Channel"], ["Own Goals Channel"], \
                #     ["Own Position Channel"], ["Other Goal Channel"]]
                # #for agnt in range(args.n_agents):
                # #headers.append("Agent {}".format(agnt))
                # rows[0].append(ob[0][0])
                # rows[1].append(ob[0][1])
                # rows[2].append(ob[0][2])
                # rows[3].append(ob[0][4])
                # rows[4].append(ob[0][3])
                # print(tabulate(rows, tablefmt = 'fancy_grid'))
                # print("Action: {}".format(a[0].item()))
                # ##########################
                (a_pred, _, _, _) = ppo.actors[0].forward(ob)
                loss = loss_f(a_pred, a)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss_hldr.append(loss.item())
                iterations += 1
            if (epoch+1) % 2 == 0:
                save(ppo, logger, epoch)

            if (epoch+1) % 25 == 0:
                ppo.extend_agent_indexes(args.n_agents)
                rend_frames, all_info = benchmark_func(args, ppo, 100, 30, DEVICE, greedy=True)
                logger.benchmark_info(all_info, rend_frames, epoch)

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



    

    
    



