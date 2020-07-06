import main
from sklearn.model_selection import ParameterGrid




def cn_maac_ic3(alg = "all"):
    alg = 3
    experiment_group_name = "Exp0CN"
    exp_parameters = {
        "env": ["cooperative_navigation-v0"], #, "cooperative_navigation-v1"],
        "env_size": [5],
        "n_agents": [4],
        "obj_density":[0.0]
    }
    grid = ParameterGrid(exp_parameters)
    
    for p in grid:

        param = [experiment_group_name, p["n_agents"], \
            p["env"], p["env_size"], p["obj_density"]]

        if alg == "all":
            run_maac(*param)
            run_ic3(*param)
        elif alg == 0:
            run_Ind_ac(*param)
        elif alg == 1:
            run_Ind_ppo(*param)
        elif alg == 2:
            run_ic3(*param)
        elif alg == 3:
            run_maac(*param)
        else:
            raise Exception("Alorithm number not implemented")






def run_ic3(experiment_group_name, n_agents, env_name, env_size, obj_density):
    n_iterations = str(5000)
    #experiment_group_name = "TEST4_IC3"
    work_dir = experiment_group_name
    plot_dir = experiment_group_name + "_Central"

 
    parmeter_grid1 = {
        "seed": [2],
        "batch_sizes": [500],
        "entropy_coeff":[0.0],
        "discount":[0.99],
        #"lambda_": [1.0, 0.95,0.6],
        "value_ceoff": [0.01],
        "policy": ["IC3"],
        "n_agents": [n_agents],
        "recurrent": [False],
        "base_policy_type": ["mlp"],
        "comm_passes": [1]
    }


    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
        args.extend(["--iterations", n_iterations])
       # args.extend(["--n_workers", str(workers)])
       # args.extend(["--map_shape", tuple((5,5))])

        # parser.add_argument("--hidden_size", default= 128, type =int)
        # parser.add_argument("--recurrent", default= True, type =bool)
        # parser.add_argument("--detach_gap", default = 10, type = int)
        # parser.add_argument("--comm_passes", default= 1, type =int)
        # #parser.add_argument("--comm_mode", default = 1, type = int,
        # #help= "if mode == 0 -- no communication; mode==1--ic3net communication; mode ==2 -- commNet communication")
        # parser.add_argument("--comm_mode", default = "avg", type = str, help="Average or sum the hidden states to obtain the comm vector")
        # parser.add_argument("--hard_attn", default=True, type=bool, help="to communicate or not. If hard_attn == False, no comm")
        # parser.add_argument("--comm_mask_zero", default=False, type=bool, help="to communicate or not. If hard_attn == False, no comm")

        
        if "cooperative_navigation" in env_name:
            en = "cn"
        else:
            en = "unk"
        v = env_name[-2:]
        
        name = en + v + "_"+ "obj_d"+ str(obj_density) + "_envsz" + str(env_size) + param["policy"] + "_enc_type_" + \
            param["base_policy_type"] + "_disc" + str(param["discount"]) \
            + "_bat"+ str(param["batch_sizes"]) + \
            "_val_coeff_" + str(param["value_ceoff"]) \
            + "_ent_coeff_" + str(param["entropy_coeff"]) \
            + "_agnts" + str(param["n_agents"])

        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(env_size)])
        args.extend(["--policy", str(param["policy"])])
        args.extend(["--name", name])
        args.extend(["--batch_size", str(param["batch_sizes"])])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--entropy_coeff", str(param["entropy_coeff"])])
       # args.extend(["--c1", str(param["value_coeff"])])
        args.extend(["--discount", str(param["discount"])])
       # args.extend(["--lambda_", str(param["lambda_"])])
        args.extend(["--ic3_base_policy_type", str(param["base_policy_type"])])

        args.extend(["--render_rate", str(int(5e2))])
        args.extend(["--checkpoint_frequency", str(int(1e3))])
        args.extend(["--benchmark_frequency", str(int(1000))])
        args.extend(["--benchmark_num_episodes", str(int(500))])
        args.extend(["--benchmark_render_length", str(int(100))])
        args.extend(["--obj_density", str(obj_density)])

    
        main.main(args)







def run_maac(experiment_group_name, n_agents, env_name, env_size, obj_density):
    episodes = str(50000)
    #experiment_group_name = exp_group_name #"TestBases"
    work_dir = experiment_group_name
    plot_dir = experiment_group_name + "_Central" #"_Central_TestBases"

 
    parmeter_grid1 = {
        #"seed": [2],
        #"batch_sizes": [500,1000,2000],
       # "entropy_coeff":[0.0],
       # "discount":[1.0],
        #"lambda_": [1.0, 0.95,0.6],
        #"value_ceoff": [0.01, 0.1, 0.5, 1.0],
        "rollout_threads": [1],
        "reward_scale": [100],
        "policy": ["MAAC"],
        "base_policy_type": ["mlp"], #["cnn_old","cnn_new"],
        "n_agents": [n_agents]
    }

    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir,"--alternative_plot_dir", plot_dir]

        if "cooperative_navigation" in env_name:
            en = "cn"
        else:
            en = "unk"
        v = env_name[-2:]

        name = en + v + "_"+ "obj_d"+ str(obj_density)+ "_envsz" + str(env_size) +  param["policy"] + "_enc_type_" + \
            param["base_policy_type"] + "_reward_scale" + str(param["reward_scale"]) \
            + "_agnts" + str(param["n_agents"])
        # name = "reward_scale" + str(param["reward_scale"]) \
        #         + "_agnts" + str(param["n_agents"]) + "enc_type_" + param["base_policy_type"]
      #  name.replace('.', "_")

        args.extend(["--maac_buffer_length", str(int(5e5))])
        args.extend(["--map_shape", str(env_size)])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--policy", str(param["policy"])])
        args.extend(["--name", name])
        args.extend(["--maac_use_gpu"])
        args.extend(["--maac_share_actor"])
        args.extend(["--maac_reward_scale", str(param["reward_scale"])])
        args.extend(["--maac_n_rollout_threads", str(param["rollout_threads"])])
        args.extend(["--maac_n_episodes", episodes])
        args.extend(["--env_name", env_name])#"cooperative_navigation-v0"])
        args.extend(["--maac_base_policy_type", param["base_policy_type"]]) 

        args.extend(["--render_rate", str(int(3e2))])
        args.extend(["--checkpoint_frequency", str(int(10e3))])
        args.extend(["--obj_density", str(obj_density)])
        args.extend(["--benchmark_frequency", str(int(1000))])
        args.extend(["--benchmark_num_episodes", str(int(500))])
        args.extend(["--benchmark_render_length", str(int(100))])
        #args.extend(["--batch_size", str(param["batch_sizes"])])
        #args.extend(["--seed", str(param["seed"])])
        #args.extend(["--c2", str(param["entropy_coeff"])])
       # args.extend(["--c1", str(param["value_coeff"])])
        #args.extend(["--discount", str(param["discount"])])
       # args.extend(["--lambda_", str(param["lambda_"])])
    
        main.main(args)




def run_Ind_ac(experiment_group_name, n_agents, env_name, env_size, obj_density):
    pass

def run_Ind_ppo(experiment_group_name, n_agents, env_name, env_size, obj_density):
    pass
