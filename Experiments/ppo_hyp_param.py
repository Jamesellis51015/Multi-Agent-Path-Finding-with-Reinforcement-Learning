import main
from sklearn.model_selection import ParameterGrid

def ppo_hyp_param():
    n_iterations = str(1000) #str(3000)
    experiment_group_name = "ppo_hyp_param"
    work_dir = experiment_group_name
    plot_dir = experiment_group_name + "_Central"
 
    parmeter_grid1 = {
        "seed": [3],
        "workers": [4],
        "rollout_length": [250],
        "eps_clip": [0.2],
        "k_epochs": [1,2,4,8],
        "minibatch_size": [125,250,500,1000],
        "entropy_coeff":[0.01],
        "discount":[0.9],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["PPO"],
        "n_agents": [1,4],
        #"recurrent": [False],
        "base_policy_type": ["mlp"],
    }


    grid1 = ParameterGrid(parmeter_grid1)

    
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
        #args.extend(["--iterations", n_iterations])
        name = "n_agnts"+ str(param["n_agents"]) \
            + "_k_epochs" + str(param["k_epochs"]) \
            + "_minibatch_size" + str(param["minibatch_size"])
        # name = "disc" + str(param["discount"]) \
        #     + "_lambda" + str(param["lambda_"]) \
        #  + "_workers" + str(param["workers"]) + \
        #      "_nrollouts" + str(param["rollout_length"]) + \
        #      "_eps_clip" + str(param["eps_clip"]) + \
        #     "_val_coeff_" + str(param["value_coeff"]) \
        #     + "_ent" + str(param["entropy_coeff"]) \
        #     + "_agnts" + str(param["n_agents"]) + "_" + str(param["policy"])
        #"_ent" + str(param["entropy_coeff"]) \
        args.extend(["--env_name","cooperative_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(50))])
        args.extend(["--benchmark_frequency", str(int(200))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(25))])

        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(120)])
        args.extend(["--ppo_lr_a", str(0.001)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", "mlp"])
        args.extend(["--ppo_workers", str(param['workers'])])
        args.extend(["--ppo_rollout_length", str(param['rollout_length'])])
        args.extend(["--ppo_share_actor"])
        args.extend(["--ppo_share_value"])
        args.extend(["--ppo_iterations", n_iterations])
        args.extend(["--ppo_k_epochs", str(param['k_epochs'])])
        args.extend(["--ppo_eps_clip", str(param["eps_clip"])])
        args.extend(["--ppo_minibatch_size", str(param['minibatch_size'])])
        args.extend(["--ppo_entropy_coeff", str(param['entropy_coeff'])])
        args.extend(["--ppo_value_coeff", str(param['value_coeff'])])
        args.extend(["--ppo_discount", str(param['discount'])])
        args.extend(["--ppo_gae_lambda", str(param['lambda_'])])
      #  args.extend(["--ppo_use_gpu"])
    
        main.main(args)



def ppo_hyp_k_epochs():
    n_iterations = str(1000) #str(3000)
    experiment_group_name = "ppo_hyp_k_epochs"
    work_dir = experiment_group_name
    plot_dir = experiment_group_name + "_Central"
 
    parmeter_grid1 = {
        "seed": [3],
        "workers": [4],
        "rollout_length": [256],
        "eps_clip": [0.2],
        "k_epochs": [1,2,4,8,12,16,20],
        "minibatch_size": [256],
        "entropy_coeff":[0.01],
        "discount":[0.9],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["PPO"],
        "n_agents": [1,4],
        #"recurrent": [False],
        "base_policy_type": ["mlp"],
    }


    grid1 = ParameterGrid(parmeter_grid1)

    
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
        #args.extend(["--iterations", n_iterations])
        name = "n_agnts"+ str(param["n_agents"]) \
            + "_k_epochs" + str(param["k_epochs"]) \
            + "_minibatch_size" + str(param["minibatch_size"])
        # name = "disc" + str(param["discount"]) \
        #     + "_lambda" + str(param["lambda_"]) \
        #  + "_workers" + str(param["workers"]) + \
        #      "_nrollouts" + str(param["rollout_length"]) + \
        #      "_eps_clip" + str(param["eps_clip"]) + \
        #     "_val_coeff_" + str(param["value_coeff"]) \
        #     + "_ent" + str(param["entropy_coeff"]) \
        #     + "_agnts" + str(param["n_agents"]) + "_" + str(param["policy"])
        #"_ent" + str(param["entropy_coeff"]) \
        args.extend(["--env_name","cooperative_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(50))])
        args.extend(["--benchmark_frequency", str(int(200))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(25))])

        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(120)])
        args.extend(["--ppo_lr_a", str(0.001)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", "mlp"])
        args.extend(["--ppo_workers", str(param['workers'])])
        args.extend(["--ppo_rollout_length", str(param['rollout_length'])])
        args.extend(["--ppo_share_actor"])
        args.extend(["--ppo_share_value"])
        args.extend(["--ppo_iterations", n_iterations])
        args.extend(["--ppo_k_epochs", str(param['k_epochs'])])
        args.extend(["--ppo_eps_clip", str(param["eps_clip"])])
        args.extend(["--ppo_minibatch_size", str(param['minibatch_size'])])
        args.extend(["--ppo_entropy_coeff", str(param['entropy_coeff'])])
        args.extend(["--ppo_value_coeff", str(param['value_coeff'])])
        args.extend(["--ppo_discount", str(param['discount'])])
        args.extend(["--ppo_gae_lambda", str(param['lambda_'])])
      #  args.extend(["--ppo_use_gpu"])
    
        main.main(args)

def ppo_hyp_discount():
    n_iterations = str(1000) #str(3000)
    experiment_group_name = "ppo_hyp_discount"
    work_dir = experiment_group_name
    plot_dir = experiment_group_name + "_Central"
 
    parmeter_grid1 = {
        "seed": [3],
        "env": ["cooperative_navigation-v0"],
        "workers": [8],
        "rollout_length": [256],
        "eps_clip": [0.2],
        "k_epochs": [4],
        "minibatch_size": [512],
        "entropy_coeff":[0.01],
        "discount":[0.99, 0.95, 0.9,0.8,0.7],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["PPO"],
        "n_agents": [1,4],
        "map_size":[5,10],
        #"recurrent": [False],
        "base_policy_type": ["mlp"],
    }


    grid1 = ParameterGrid(parmeter_grid1)

    
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
        #args.extend(["--iterations", n_iterations])
        name = "n_agnts"+ str(param["n_agents"]) \
                +"_env"+ param["env"] \
            + "_map_size" + str(param["map_size"]) \
            + "_disc" + str(param["discount"]) \
            + "_lambda" + str(param["lambda_"]) 
        # name = "disc" + str(param["discount"]) \
        #     + "_lambda" + str(param["lambda_"]) \
        #  + "_workers" + str(param["workers"]) + \
        #      "_nrollouts" + str(param["rollout_length"]) + \
        #      "_eps_clip" + str(param["eps_clip"]) + \
        #     "_val_coeff_" + str(param["value_coeff"]) \
        #     + "_ent" + str(param["entropy_coeff"]) \
        #     + "_agnts" + str(param["n_agents"]) + "_" + str(param["policy"])
        #"_ent" + str(param["entropy_coeff"]) \
        args.extend(["--env_name",param["env"]])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(50))])
        args.extend(["--benchmark_frequency", str(int(200))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(25))])

        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(120)])
        args.extend(["--ppo_lr_a", str(0.001)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", "mlp"])
        args.extend(["--ppo_workers", str(param['workers'])])
        args.extend(["--ppo_rollout_length", str(param['rollout_length'])])
        args.extend(["--ppo_share_actor"])
        args.extend(["--ppo_share_value"])
        args.extend(["--ppo_iterations", n_iterations])
        args.extend(["--ppo_k_epochs", str(param['k_epochs'])])
        args.extend(["--ppo_eps_clip", str(param["eps_clip"])])
        args.extend(["--ppo_minibatch_size", str(param['minibatch_size'])])
        args.extend(["--ppo_entropy_coeff", str(param['entropy_coeff'])])
        args.extend(["--ppo_value_coeff", str(param['value_coeff'])])
        args.extend(["--ppo_discount", str(param['discount'])])
        args.extend(["--ppo_gae_lambda", str(param['lambda_'])])
      #  args.extend(["--ppo_use_gpu"])
    
        main.main(args)


def ppo_hyp_lambda():
    n_iterations = str(1000) #str(3000)
    experiment_group_name = "ppo_hyp_lambda"
    work_dir = experiment_group_name
    plot_dir = experiment_group_name + "_Central"
 
    parmeter_grid1 = {
        "seed": [3],
        "env": ["cooperative_navigation-v0"],
        "workers": [8],
        "rollout_length": [256],
        "eps_clip": [0.2],
        "k_epochs": [4],
        "minibatch_size": [512],
        "entropy_coeff":[0.01],
        "discount":[0.9],
        "lambda_": [1.0, 0.95,0.9,0.8,0.7,0.5,0.2,0.0],
        "value_coeff": [0.5],
        "policy": ["PPO"],
        "n_agents": [1,4],
        "map_size":[5,10],
        #"recurrent": [False],
        "base_policy_type": ["mlp"],
    }


    grid1 = ParameterGrid(parmeter_grid1)

    
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
        #args.extend(["--iterations", n_iterations])
        name = "n_agnts"+ str(param["n_agents"]) \
                +"_env"+ param["env"] \
            + "_map_size" + str(param["map_size"]) \
            + "_disc" + str(param["discount"]) \
            + "_lambda" + str(param["lambda_"]) 
        # name = "disc" + str(param["discount"]) \
        #     + "_lambda" + str(param["lambda_"]) \
        #  + "_workers" + str(param["workers"]) + \
        #      "_nrollouts" + str(param["rollout_length"]) + \
        #      "_eps_clip" + str(param["eps_clip"]) + \
        #     "_val_coeff_" + str(param["value_coeff"]) \
        #     + "_ent" + str(param["entropy_coeff"]) \
        #     + "_agnts" + str(param["n_agents"]) + "_" + str(param["policy"])
        #"_ent" + str(param["entropy_coeff"]) \
        args.extend(["--env_name",param["env"]])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(50))])
        args.extend(["--benchmark_frequency", str(int(200))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(25))])

        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(120)])
        args.extend(["--ppo_lr_a", str(0.001)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", "mlp"])
        args.extend(["--ppo_workers", str(param['workers'])])
        args.extend(["--ppo_rollout_length", str(param['rollout_length'])])
        args.extend(["--ppo_share_actor"])
        args.extend(["--ppo_share_value"])
        args.extend(["--ppo_iterations", n_iterations])
        args.extend(["--ppo_k_epochs", str(param['k_epochs'])])
        args.extend(["--ppo_eps_clip", str(param["eps_clip"])])
        args.extend(["--ppo_minibatch_size", str(param['minibatch_size'])])
        args.extend(["--ppo_entropy_coeff", str(param['entropy_coeff'])])
        args.extend(["--ppo_value_coeff", str(param['value_coeff'])])
        args.extend(["--ppo_discount", str(param['discount'])])
        args.extend(["--ppo_gae_lambda", str(param['lambda_'])])
      #  args.extend(["--ppo_use_gpu"])
    
        main.main(args)