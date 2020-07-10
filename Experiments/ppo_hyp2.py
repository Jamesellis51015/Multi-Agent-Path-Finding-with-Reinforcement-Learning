import main
from sklearn.model_selection import ParameterGrid

def ppo_A():
    n_iterations = str(500) #str(3000)
    experiment_group_name = "ppo_hyp_param"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"
 
    parmeter_grid1 = {
        "seed": [1],
        "workers": [4],
        "rollout_length": [256],
        "eps_clip": [0.2],
        "k_epochs": [8],
        "minibatch_size": [512],
        "entropy_coeff":[0.01],
        "discount":[0.4, 0.3, 0.1, 0.5, 0.7, 0.8, 0.9,0.95,0.99,1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["PPO"],
        "n_agents": [1],
        #"recurrent": [False],
        "base_policy_type": ["mlp"],
        "obj_density": [0.0],
        "env_size": [5]
    }

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        name = "A" + "_disc_" + str(param["discount"]) \
            + "_lambda_" + str(param["lambda_"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["minibatch_size"]) \
            + "_rollouts_" + str(param["rollout_length"]) \
            + "_workers_" + str(param["workers"]) \
            + "_kepochs_" + str(param["k_epochs"]) \
            + "_envsize_" + str(param["env_size"]) \
            + "_nagents_"+ str(param["n_agents"]) \
            + "_objdensity_"+ str(param["obj_density"]) \
            + "_seed_"+ str(param["seed"])

        args.extend(["--env_name","independent_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--obj_density", str(param["obj_density"])])
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
        args.extend(["--ppo_use_gpu"])
    
        main.main(args)


def ppo_1B1():
    n_iterations = str(500) #str(3000)
    experiment_group_name = "1B"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"
 
    parmeter_grid1 = {
        "seed": [1],
        "workers": [4],
        "rollout_length": [256],
        "eps_clip": [0.2],
        "k_epochs": [8],
        "minibatch_size": [512],
        "entropy_coeff":[0.01],
        "discount":[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [0.0, 0.3, 0.5, 0.9, 0.95, 1.0],
        "value_coeff": [0.5],
        "policy": ["PPO"],
        "n_agents": [2],
        #"recurrent": [False],
        "base_policy_type": ["mlp"],
        "obj_density": [0.2],
        "env_size": [5]
    }

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        name = "1B" + "_disc_" + str(param["discount"]) \
            + "_lambda_" + str(param["lambda_"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["minibatch_size"]) \
            + "_rollouts_" + str(param["rollout_length"]) \
            + "_workers_" + str(param["workers"]) \
            + "_kepochs_" + str(param["k_epochs"]) \
            + "_envsize_" + str(param["env_size"]) \
            + "_nagents_"+ str(param["n_agents"]) \
            + "_objdensity_"+ str(param["obj_density"]) \
            + "_seed_"+ str(param["seed"])

        args.extend(["--env_name","independent_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--obj_density", str(param["obj_density"])])
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
        args.extend(["--ppo_use_gpu"])
    
        main.main(args)


def ppo_1B2():
    n_iterations = str(1000) #str(3000)
    experiment_group_name = "1B"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"
 
    parmeter_grid1 = {
        "seed": [1],
        "workers": [4],
        "rollout_length": [256],
        "eps_clip": [0.2],
        "k_epochs": [8],
        "minibatch_size": [512],
        "entropy_coeff":[0.01],
        "discount":[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [0.0, 0.3, 0.5, 0.9, 0.95, 1.0],
        "value_coeff": [0.5],
        "policy": ["PPO"],
        "n_agents": [2],
        #"recurrent": [False],
        "base_policy_type": ["mlp"],
        "obj_density": [0.2],
        "env_size": [7]
    }

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        name = "1B" + "_disc_" + str(param["discount"]) \
            + "_lambda_" + str(param["lambda_"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["minibatch_size"]) \
            + "_rollouts_" + str(param["rollout_length"]) \
            + "_workers_" + str(param["workers"]) \
            + "_kepochs_" + str(param["k_epochs"]) \
            + "_envsize_" + str(param["env_size"]) \
            + "_nagents_"+ str(param["n_agents"]) \
            + "_objdensity_"+ str(param["obj_density"]) \
            + "_seed_"+ str(param["seed"])

        args.extend(["--env_name","independent_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--obj_density", str(param["obj_density"])])
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
        args.extend(["--ppo_use_gpu"])
    
        main.main(args)


def ppo_1B3():
    n_iterations = str(500) #str(3000)
    experiment_group_name = "1B"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"
 
    parmeter_grid1 = {
        "seed": [1],
        "workers": [4],
        "rollout_length": [256],
        "eps_clip": [0.2],
        "k_epochs": [8],
        "minibatch_size": [512],
        "entropy_coeff":[0.01],
        "discount":[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [0.0, 0.3, 0.5, 0.9, 0.95, 1.0],
        "value_coeff": [0.5],
        "policy": ["PPO"],
        "n_agents": [2],
        #"recurrent": [False],
        "base_policy_type": ["mlp"],
        "obj_density": [0.0],
        "env_size": [5]
    }

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        name = "1B" + "_disc_" + str(param["discount"]) \
            + "_lambda_" + str(param["lambda_"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["minibatch_size"]) \
            + "_rollouts_" + str(param["rollout_length"]) \
            + "_workers_" + str(param["workers"]) \
            + "_kepochs_" + str(param["k_epochs"]) \
            + "_envsize_" + str(param["env_size"]) \
            + "_nagents_"+ str(param["n_agents"]) \
            + "_objdensity_"+ str(param["obj_density"]) \
            + "_seed_"+ str(param["seed"])

        args.extend(["--env_name","independent_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--obj_density", str(param["obj_density"])])
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
        args.extend(["--ppo_use_gpu"])
    
        main.main(args)

def ppo_1B4():
    n_iterations = str(1000) #str(3000)
    experiment_group_name = "1B"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"
 
    parmeter_grid1 = {
        "seed": [1],
        "workers": [4],
        "rollout_length": [256],
        "eps_clip": [0.2],
        "k_epochs": [8],
        "minibatch_size": [512],
        "entropy_coeff":[0.01],
        "discount":[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [0.0, 0.3, 0.5, 0.9, 0.95, 1.0],
        "value_coeff": [0.5],
        "policy": ["PPO"],
        "n_agents": [6],
        #"recurrent": [False],
        "base_policy_type": ["mlp"],
        "obj_density": [0.0],
        "env_size": [5]
    }

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        name = "1B" + "_disc_" + str(param["discount"]) \
            + "_lambda_" + str(param["lambda_"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["minibatch_size"]) \
            + "_rollouts_" + str(param["rollout_length"]) \
            + "_workers_" + str(param["workers"]) \
            + "_kepochs_" + str(param["k_epochs"]) \
            + "_envsize_" + str(param["env_size"]) \
            + "_nagents_"+ str(param["n_agents"]) \
            + "_objdensity_"+ str(param["obj_density"]) \
            + "_seed_"+ str(param["seed"])

        args.extend(["--env_name","independent_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--obj_density", str(param["obj_density"])])
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
        args.extend(["--ppo_use_gpu"])
    
        main.main(args)