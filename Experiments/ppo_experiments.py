import main
from sklearn.model_selection import ParameterGrid


def ppo_test():
    '''Many agents no obstacles'''
    n_iterations = str(3000)
    experiment_group_name = "Test_PPO"
    work_dir = experiment_group_name
    plot_dir = experiment_group_name + "_Tensorboard"
 
    parmeter_grid1 = {
        "seed": [1],
        "workers": [4],
        "rollout_length": [256],
        "eps_clip": [0.2],
        "k_epochs": [8],
        "minibatch_size": [512],
        "entropy_coeff":[0.01],
        "discount":[0.5],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["PPO"],
        "n_agents": [4],
        #"recurrent": [False],
        "base_policy_type": ["primal7"],
        "obj_density": [0.1],
        "env_size": [7],
        "step_r": [-0.1],
        "obstacle_collision_r": [-0.4], 
        "agent_collision_r": [-0.4], 
        "goal_reached_r": [0.5], 
        "finish_episode_r": [2.0]
    
    }

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        name = "ppoTest" \
            + "_disc_" + str(param["discount"]) \
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

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])
        args.extend(["--env_name","independent_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(50))])
        args.extend(["--benchmark_frequency", str(int(1000))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(50))])
        args.extend(["--checkpoint_frequency", str(int(500))])

        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(120)])
        args.extend(["--ppo_lr_a", str(0.001)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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
        # Flag for training with GPU
        #args.extend(["--ppo_use_gpu"])
        main.main(args)



def primal_test():
    '''Partially observable with shortest path heur'''
    n_iterations = str(4000)
    experiment_group_name = "Test_PRIMAL"
    work_dir = experiment_group_name
    plot_dir = experiment_group_name + "_Tensorboard"
 
    parmeter_grid1 = {
        "seed": [1],
        "workers": [4],
        "rollout_length": [256],
        "eps_clip": [0.2],
        "k_epochs": [8],
        "minibatch_size": [512],
        "entropy_coeff":[0.01],
        "discount":[0.5], 
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["CURR_PPO"],
        "n_agents": [2],
        "base_policy_type": ["primal9"],
        "view_d": [4],
        "obj_density": [0.0],
        "env_size": [10],
        "step_r": [-0.1],
        "obstacle_collision_r": [-0.4], 
        "agent_collision_r": [-0.4],
        "goal_reached_r": [0.5],
        "finish_episode_r": [2.0],
        "block_r": [-2.0],
        "recurrent": [True],
        "blockheur": [False],
        "validact": [True],
        "noprevstate": [True]
    }
    grid1 = ParameterGrid(parmeter_grid1)
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        name = "primalTest" \
            + "_disc_" + str(param["discount"]) \
            + "_lambda_" + str(param["lambda_"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["minibatch_size"]) \
            + "_rollouts_" + str(param["rollout_length"]) \
            + "_workers_" + str(param["workers"]) \
            + "_kepochs_" + str(param["k_epochs"]) \
            + "_recurrent_" + str(param["recurrent"]) \
            + "_block_" + str(param["blockheur"]) \
            + "_validact_" + str(param["validact"])

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])
        args.extend(["--block_r", str(param["block_r"])])

        args.extend(["--env_name","independent_navigation-v8_0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--view_d", str(param["view_d"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(20))])
        args.extend(["--benchmark_frequency", str(int(100))])
        args.extend(["--benchmark_num_episodes", str(int(10))])
        args.extend(["--benchmark_render_length", str(int(10))])
        args.extend(["--checkpoint_frequency", str(int(30))])
        args.extend(["--ppo_bc_iteration_prob", str(float(0.5))])
       # args.extend(["--ppo_continue_from_checkpoint"])



        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(512)])
        args.extend(["--ppo_lr_a", str(0.00002)])
        args.extend(["--ppo_lr_v", str(0.00002)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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
        #args.extend(["--ppo_use_gpu"])

        if param["recurrent"]:
            args.extend(["--ppo_recurrent"])
        if param["blockheur"]:
            args.extend(["--ppo_heur_block"])
        if param["validact"]:
            args.extend(["--ppo_heur_valid_act"])
        if param["noprevstate"]:
            args.extend(["--ppo_heur_no_prev_state"])
    
        main.main(args)




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
        "discount":[0.1], #[0.4, 0.5, 0.7, 0.9, 1.0],
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
        "discount":[0.1],#[0.4, 0.5, 0.7, 0.9, 1.0],
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
        "discount":[0.1],#[0.4, 0.5, 0.7, 0.9, 1.0],
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

def ppo_1C1():
    n_iterations = str(1000) #str(3000)
    experiment_group_name = "ppo_1C"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"
 
    parmeter_grid1 = {
        "seed": [1],
        "workers": [4],
        "rollout_length": [256],
        "eps_clip": [0.2],
        "k_epochs": [8],
        "minibatch_size": [512],
        "entropy_coeff":[0.0, 0.01, 0.1, 0.5],
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
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

def ppo_1C2():
    n_iterations = str(1000) #str(3000)
    experiment_group_name = "ppo_1C"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"
 
    parmeter_grid1 = {
        "seed": [1],
        "workers": [4],
        "rollout_length": [256],
        "eps_clip": [0.2],
        "k_epochs": [8],
        "minibatch_size": [512],
        "entropy_coeff":[0.0, 0.01, 0.1, 0.5],
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["PPO"],
        "n_agents": [6],
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

def ppo_1D1():
    n_iterations = str(1000) #str(3000)
    experiment_group_name = "ppo_1D"
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
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["PPO"],
        "n_agents": [2],
        #"recurrent": [False],
        "base_policy_type": ["mlp"],
        "obj_density": [0.2],
        "env_size": [7],
        "lr":[0.1, 0.001, 0.0001,0.00005]
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
            + "_lr_" + str(param["lr"]) \
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
        args.extend(["--ppo_lr_a", str(param['lr'])])
        args.extend(["--ppo_lr_v", str(param['lr'])])
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

def ppo_1D2():
    n_iterations = str(1000) #str(3000)
    experiment_group_name = "ppo_1D"
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
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["PPO"],
        "n_agents": [6],
        #"recurrent": [False],
        "base_policy_type": ["mlp"],
        "obj_density": [0.2],
        "env_size": [7],
        "lr":[0.1, 0.001, 0.0001,0.00005]
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
            + "_lr_" + str(param["lr"]) \
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
        args.extend(["--ppo_lr_a", str(param['lr'])])
        args.extend(["--ppo_lr_v", str(param['lr'])])
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



def ppo_1D3():
    n_iterations = str(1000) #str(3000)
    experiment_group_name = "ppo_1D"
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
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["PPO"],
        "n_agents": [2],
        #"recurrent": [False],
        "base_policy_type": ["mlp"],
        "obj_density": [0.0],
        "env_size": [7],
        "lr":[0.1, 0.001, 0.0001,0.00005]
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
            + "_lr_" + str(param["lr"]) \
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
        args.extend(["--ppo_lr_a", str(param['lr'])])
        args.extend(["--ppo_lr_v", str(param['lr'])])
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

def ppo_1D4():
    n_iterations = str(1000) #str(3000)
    experiment_group_name = "ppo_1D"
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
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["PPO"],
        "n_agents": [6],
        #"recurrent": [False],
        "base_policy_type": ["mlp"],
        "obj_density": [0.0],
        "env_size": [7],
        "lr":[0.1, 0.001, 0.0001,0.00005]
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
            + "_lr_" + str(param["lr"]) \
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
        args.extend(["--ppo_lr_a", str(param['lr'])])
        args.extend(["--ppo_lr_v", str(param['lr'])])
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
    # n_iterations = str(1000) #str(3000)
    # experiment_group_name = "1B"
    # work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    # plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"
 
    # parmeter_grid1 = {
    #     "seed": [1],
    #     "workers": [4],
    #     "rollout_length": [256],
    #     "eps_clip": [0.2],
    #     "k_epochs": [8],
    #     "minibatch_size": [512],
    #     "entropy_coeff":[0.01],
    #     "discount":[0.1],#[0.4, 0.5, 0.7, 0.9, 1.0],
    #     "lambda_": [0.0, 0.3, 0.5, 0.9, 0.95, 1.0],
    #     "value_coeff": [0.5],
    #     "policy": ["PPO"],
    #     "n_agents": [2],
    #     #"recurrent": [False],
    #     "base_policy_type": ["mlp"],
    #     "obj_density": [0.2],
    #     "env_size": [7]
    # }

    # grid1 = ParameterGrid(parmeter_grid1)

    # for param in grid1:
    #     args = []
    #     args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

    #     name = "1B" + "_disc_" + str(param["discount"]) \
    #         + "_lambda_" + str(param["lambda_"]) \
    #         + "_entropy_" + str(param["entropy_coeff"]) \
    #         + "_minibatch_" + str(param["minibatch_size"]) \
    #         + "_rollouts_" + str(param["rollout_length"]) \
    #         + "_workers_" + str(param["workers"]) \
    #         + "_kepochs_" + str(param["k_epochs"]) \
    #         + "_envsize_" + str(param["env_size"]) \
    #         + "_nagents_"+ str(param["n_agents"]) \
    #         + "_objdensity_"+ str(param["obj_density"]) \
    #         + "_seed_"+ str(param["seed"])

    #     args.extend(["--env_name","independent_navigation-v0"])
    #     args.extend(["--n_agents", str(param["n_agents"])])
    #     args.extend(["--map_shape", str(param["env_size"])])
    #     args.extend(["--obj_density", str(param["obj_density"])])
    #     args.extend(["--policy", param["policy"]])
    #     args.extend(["--name", name])
    #     args.extend(["--seed", str(param["seed"])])
    #     args.extend(["--render_rate", str(int(50))])
    #     args.extend(["--benchmark_frequency", str(int(200))])
    #     args.extend(["--benchmark_num_episodes", str(int(100))])
    #     args.extend(["--benchmark_render_length", str(int(25))])

    #     #           PPO Paramares:
    #     args.extend(["--ppo_hidden_dim", str(120)])
    #     args.extend(["--ppo_lr_a", str(0.001)])
    #     args.extend(["--ppo_lr_v", str(0.001)])
    #     args.extend(["--ppo_base_policy_type", "mlp"])
    #     args.extend(["--ppo_workers", str(param['workers'])])
    #     args.extend(["--ppo_rollout_length", str(param['rollout_length'])])
    #     args.extend(["--ppo_share_actor"])
    #     args.extend(["--ppo_share_value"])
    #     args.extend(["--ppo_iterations", n_iterations])
    #     args.extend(["--ppo_k_epochs", str(param['k_epochs'])])
    #     args.extend(["--ppo_eps_clip", str(param["eps_clip"])])
    #     args.extend(["--ppo_minibatch_size", str(param['minibatch_size'])])
    #     args.extend(["--ppo_entropy_coeff", str(param['entropy_coeff'])])
    #     args.extend(["--ppo_value_coeff", str(param['value_coeff'])])
    #     args.extend(["--ppo_discount", str(param['discount'])])
    #     args.extend(["--ppo_gae_lambda", str(param['lambda_'])])
    #     args.extend(["--ppo_use_gpu"])
    
    #     main.main(args)





def ppo_seeds():
    n_iterations = str(500) #str(3000)
    experiment_group_name = "seeds"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"
 
    parmeter_grid1 = {
        "seed": [1,456,789],
        "workers": [4],
        "rollout_length": [256],
        "eps_clip": [0.2],
        "k_epochs": [8],
        "minibatch_size": [512],
        "entropy_coeff":[0.01],
        "discount":[0.5, 1.0],
        "lambda_": [1.0],
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
def ppo_seeds2():
    n_iterations = str(500) #str(3000)
    experiment_group_name = "seeds"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"
 
    parmeter_grid1 = {
        "seed": [1,456,789],
        "workers": [4],
        "rollout_length": [256],
        "eps_clip": [0.2],
        "k_epochs": [8],
        "minibatch_size": [512],
        "entropy_coeff":[0.01],
        "discount":[0.9],
        "lambda_": [1.0],
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




def ppo_TEST():
    n_iterations = str(1500) #str(3000)
    experiment_group_name = "ppo_TEST3"
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
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["PPO"],
        "n_agents": [4],
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
            + "_seed_"+ str(param["seed"]) \
            + "Normalize_False"

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
        args.extend(["--checkpoint_frequency", str(int(10))])
        args.extend(["--ppo_use_gpu"])
    
        main.main(args)




def ppo_primalArc1():
    n_iterations = str(3000) #str(3000)
    experiment_group_name = "3A"
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
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["PPO"],
        "n_agents": [4],
        #"recurrent": [False],
        "base_policy_type": ["primal7"],
        "obj_density": [0.2],
        "env_size": [7] #7
    }

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        name = "1C" + "_arc_" + param["base_policy_type"] \
            + "_disc_" + str(param["discount"]) \
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
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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


def ppo_mlp():
    n_iterations = str(3000) #str(3000)
    experiment_group_name = "3A"
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
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["PPO"],
        "n_agents": [4],
        #"recurrent": [False],
        "base_policy_type": ["mlp"],
        "obj_density": [0.2],
        "env_size": [5]
    }

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        name = "1C" + "_arc_" + param["base_policy_type"] \
            + "_disc_" + str(param["discount"]) \
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
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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


# def ppo_primalArc1():
#     n_iterations = str(3000) #str(3000)
#     experiment_group_name = "3A"
#     work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
#     plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"
 
#     parmeter_grid1 = {
#         "seed": [1],
#         "workers": [4],
#         "rollout_length": [256],
#         "eps_clip": [0.2],
#         "k_epochs": [8],
#         "minibatch_size": [512],
#         "entropy_coeff":[0.01],
#         "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
#         "lambda_": [1.0],
#         "value_coeff": [0.5],
#         "policy": ["PPO"],
#         "n_agents": [4],
#         #"recurrent": [False],
#         "base_policy_type": ["primal7"],
#         "obj_density": [0.2],
#         "env_size": [7] #7
#     }

#     grid1 = ParameterGrid(parmeter_grid1)

#     for param in grid1:
#         args = []
#         args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

#         name = "1C" + "_arc_" + param["base_policy_type"] \
#             + "_disc_" + str(param["discount"]) \
#             + "_lambda_" + str(param["lambda_"]) \
#             + "_entropy_" + str(param["entropy_coeff"]) \
#             + "_minibatch_" + str(param["minibatch_size"]) \
#             + "_rollouts_" + str(param["rollout_length"]) \
#             + "_workers_" + str(param["workers"]) \
#             + "_kepochs_" + str(param["k_epochs"]) \
#             + "_envsize_" + str(param["env_size"]) \
#             + "_nagents_"+ str(param["n_agents"]) \
#             + "_objdensity_"+ str(param["obj_density"]) \
#             + "_seed_"+ str(param["seed"])

#         args.extend(["--env_name","independent_navigation-v0"])
#         args.extend(["--n_agents", str(param["n_agents"])])
#         args.extend(["--map_shape", str(param["env_size"])])
#         args.extend(["--obj_density", str(param["obj_density"])])
#         args.extend(["--policy", param["policy"]])
#         args.extend(["--name", name])
#         args.extend(["--seed", str(param["seed"])])
#         args.extend(["--render_rate", str(int(50))])
#         args.extend(["--benchmark_frequency", str(int(200))])
#         args.extend(["--benchmark_num_episodes", str(int(100))])
#         args.extend(["--benchmark_render_length", str(int(25))])

#         #           PPO Paramares:
#         args.extend(["--ppo_hidden_dim", str(120)])
#         args.extend(["--ppo_lr_a", str(0.001)])
#         args.extend(["--ppo_lr_v", str(0.001)])
#         args.extend(["--ppo_base_policy_type", param['base_policy_type']])
#         args.extend(["--ppo_workers", str(param['workers'])])
#         args.extend(["--ppo_rollout_length", str(param['rollout_length'])])
#         args.extend(["--ppo_share_actor"])
#         args.extend(["--ppo_share_value"])
#         args.extend(["--ppo_iterations", n_iterations])
#         args.extend(["--ppo_k_epochs", str(param['k_epochs'])])
#         args.extend(["--ppo_eps_clip", str(param["eps_clip"])])
#         args.extend(["--ppo_minibatch_size", str(param['minibatch_size'])])
#         args.extend(["--ppo_entropy_coeff", str(param['entropy_coeff'])])
#         args.extend(["--ppo_value_coeff", str(param['value_coeff'])])
#         args.extend(["--ppo_discount", str(param['discount'])])
#         args.extend(["--ppo_gae_lambda", str(param['lambda_'])])
#         args.extend(["--ppo_use_gpu"])
    
#         main.main(args)


def ppo_2A2_1():
    '''Experiment for individual rewards structure '''
    n_iterations = str(3000) #str(3000)
    experiment_group_name = "2A2"
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
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["PPO"],
        "n_agents": [4],
        #"recurrent": [False],
        "base_policy_type": ["primal6"],
        "obj_density": [0.2],
        "env_size": [5],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [1.0],
        "finish_episode_r": [0.0]
    
    }

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        name = "2A2_ppo" + "_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_disc_" + str(param["discount"]) \
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

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])

        args.extend(["--env_name","independent_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(50))])
        args.extend(["--benchmark_frequency", str(int(1500))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(50))])
        args.extend(["--checkpoint_frequency", str(int(500))])


        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(120)])
        args.extend(["--ppo_lr_a", str(0.001)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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


def ppo_2A2_2():
    '''Experiment for mixed rewards structure with finish ep reward'''
    n_iterations = str(10000) #str(3000)
    experiment_group_name = "2A2"
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
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["PPO"],
        "n_agents": [4],
        #"recurrent": [False],
        "base_policy_type": ["primal6"],
        "obj_density": [0.2],
        "env_size": [5],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.1],
        "finish_episode_r": [2.0]
    
    }

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        name = "2A2_ppo_mixedr" + "_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_disc_" + str(param["discount"]) \
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

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])

        args.extend(["--env_name","independent_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(50))])
        args.extend(["--benchmark_frequency", str(int(1500000))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(50))])
        args.extend(["--checkpoint_frequency", str(int(500))])


        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(120)])
        args.extend(["--ppo_lr_a", str(0.001)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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

def ppo_2A2_3():
    '''Experiment for global rewards structure '''
    n_iterations = str(10000) #str(3000)
    experiment_group_name = "2A2"
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
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["PPO"],
        "n_agents": [4],
        #"recurrent": [False],
        "base_policy_type": ["primal6"],
        "obj_density": [0.2],
        "env_size": [5],
        "env": ["independent_navigation-v5_1","independent_navigation-v5_2"],
        # "step_r": [-0.1, -0.4],
        # "obstacle_collision_r": [-0.015],
        # "agent_collision_r": [-0.4],
        # "goal_reached_r": [0.1],
        # "finish_episode_r": [0.0]
    
    }

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        name = "2A2_ppo_globalr" + "_arc_" + param["base_policy_type"] \
            + "_env_" + str(param["env"]) \
            + "_disc_" + str(param["discount"]) \
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

           # + "_sr_" + str(param["step_r"]) \
            # + "_ocr_" + str(param["obstacle_collision_r"]) \
            # + "_acr_" + str(param["agent_collision_r"]) \
            # + "_grr_" + str(param["goal_reached_r"]) \
            # + "_fer_" + str(param["finish_episode_r"]) \

        # args.extend(["--use_custom_rewards"])
        # args.extend(["--step_r", str(param["step_r"])])
        # args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        # args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        # args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        # args.extend(["--finish_episode_r", str(param["finish_episode_r"])])

        args.extend(["--env_name", str(param["env"])])#"independent_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(50))])
        args.extend(["--benchmark_frequency", str(int(3000))])
        args.extend(["--benchmark_num_episodes", str(int(200))])
        args.extend(["--benchmark_render_length", str(int(50))])
        args.extend(["--checkpoint_frequency", str(int(500))])


        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(120)])
        args.extend(["--ppo_lr_a", str(0.001)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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

def ppo_2A2_3t1():
    '''Experiment for global rewards structure '''
    n_iterations = str(10000) #str(3000)
    experiment_group_name = "2A2t"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"
   # work_dir = experiment_group_name
   # plot_dir = experiment_group_name + "_Central"
    
    parmeter_grid1 = {
        "seed": [1],
        "workers": [4],
        "rollout_length": [256],
        "eps_clip": [0.2],
        "k_epochs": [8],
        "minibatch_size": [512],
        "entropy_coeff":[0.01],
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["PPO"],
        "n_agents": [4],
        #"recurrent": [False],
        "base_policy_type": ["primal6"],
        "obj_density": [0.2],
        "env_size": [5],
        "env": ["independent_navigation-v5_1"],#,"independent_navigation-v5_2"],
        # "step_r": [-0.1, -0.4],
        # "obstacle_collision_r": [-0.015],
        # "agent_collision_r": [-0.4],
        # "goal_reached_r": [0.1],
        # "finish_episode_r": [0.0]
    
    }

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        name = "2A2_ppo_globalr" + "_arc_" + param["base_policy_type"] \
            + "_env_" + str(param["env"]) \
            + "_disc_" + str(param["discount"]) \
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

           # + "_sr_" + str(param["step_r"]) \
            # + "_ocr_" + str(param["obstacle_collision_r"]) \
            # + "_acr_" + str(param["agent_collision_r"]) \
            # + "_grr_" + str(param["goal_reached_r"]) \
            # + "_fer_" + str(param["finish_episode_r"]) \

        # args.extend(["--use_custom_rewards"])
        # args.extend(["--step_r", str(param["step_r"])])
        # args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        # args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        # args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        # args.extend(["--finish_episode_r", str(param["finish_episode_r"])])

        args.extend(["--env_name", str(param["env"])])#"independent_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(50))])
        args.extend(["--benchmark_frequency", str(int(3000))])
        args.extend(["--benchmark_num_episodes", str(int(200))])
        args.extend(["--benchmark_render_length", str(int(50))])
        args.extend(["--checkpoint_frequency", str(int(500))])


        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(120)])
        args.extend(["--ppo_lr_a", str(0.001)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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


def ppo_2A2_3t2():
    '''Experiment for global rewards structure '''
    n_iterations = str(10000) #str(3000)
    experiment_group_name = "2A2t"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"
    work_dir = experiment_group_name
    plot_dir = experiment_group_name + "_Central"
 
    parmeter_grid1 = {
        "seed": [1],
        "workers": [4],
        "rollout_length": [256],
        "eps_clip": [0.2],
        "k_epochs": [8],
        "minibatch_size": [512],
        "entropy_coeff":[0.01],
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["PPO"],
        "n_agents": [4],
        #"recurrent": [False],
        "base_policy_type": ["primal6"],
        "obj_density": [0.2],
        "env_size": [5],
        "env": ["independent_navigation-v5_2"],
        # "step_r": [-0.1, -0.4],
        # "obstacle_collision_r": [-0.015],
        # "agent_collision_r": [-0.4],
        # "goal_reached_r": [0.1],
        # "finish_episode_r": [0.0]
    
    }

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        name = "2A2_ppo_globalr" + "_arc_" + param["base_policy_type"] \
            + "_env_" + str(param["env"]) \
            + "_disc_" + str(param["discount"]) \
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

           # + "_sr_" + str(param["step_r"]) \
            # + "_ocr_" + str(param["obstacle_collision_r"]) \
            # + "_acr_" + str(param["agent_collision_r"]) \
            # + "_grr_" + str(param["goal_reached_r"]) \
            # + "_fer_" + str(param["finish_episode_r"]) \

        # args.extend(["--use_custom_rewards"])
        # args.extend(["--step_r", str(param["step_r"])])
        # args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        # args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        # args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        # args.extend(["--finish_episode_r", str(param["finish_episode_r"])])

        args.extend(["--env_name", str(param["env"])])#"independent_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(50))])
        args.extend(["--benchmark_frequency", str(int(3000))])
        args.extend(["--benchmark_num_episodes", str(int(200))])
        args.extend(["--benchmark_render_length", str(int(50))])
        args.extend(["--checkpoint_frequency", str(int(500))])


        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(120)])
        args.extend(["--ppo_lr_a", str(0.001)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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


def ppo_2A2_3_1():
    '''Experiment for global rewards structure ppo lambda = 0 '''
    n_iterations = str(10000) #str(3000)
    experiment_group_name = "2A2"
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
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [0.0],
        "value_coeff": [0.5],
        "policy": ["PPO"],
        "n_agents": [4],
        #"recurrent": [False],
        "base_policy_type": ["primal6"],
        "obj_density": [0.2],
        "env_size": [5],
        "env": ["independent_navigation-v6_1"] #, "independent_navigation-v5_1","independent_navigation-v5_2"],
        # "step_r": [-0.1, -0.4],
        # "obstacle_collision_r": [-0.015],
        # "agent_collision_r": [-0.4],
        # "goal_reached_r": [0.1],
        # "finish_episode_r": [0.0]
    
    }

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        name = "2A2_ppo_globalr" + "_arc_" + param["base_policy_type"] \
            + "_env_" + str(param["env"]) \
            + "_disc_" + str(param["discount"]) \
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

           # + "_sr_" + str(param["step_r"]) \
            # + "_ocr_" + str(param["obstacle_collision_r"]) \
            # + "_acr_" + str(param["agent_collision_r"]) \
            # + "_grr_" + str(param["goal_reached_r"]) \
            # + "_fer_" + str(param["finish_episode_r"]) \

        # args.extend(["--use_custom_rewards"])
        # args.extend(["--step_r", str(param["step_r"])])
        # args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        # args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        # args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        # args.extend(["--finish_episode_r", str(param["finish_episode_r"])])

        args.extend(["--env_name", str(param["env"])])#"independent_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(50))])
        args.extend(["--benchmark_frequency", str(int(3000))])
        args.extend(["--benchmark_num_episodes", str(int(200))])
        args.extend(["--benchmark_render_length", str(int(50))])
        args.extend(["--checkpoint_frequency", str(int(500))])


        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(120)])
        args.extend(["--ppo_lr_a", str(0.001)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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

def ppo_2A2_4():
    '''Mixed reward indiv agent and obstacle collisions'''
    n_iterations = str(10000) #str(3000)
    experiment_group_name = "2A2"
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
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["PPO"],
        "n_agents": [4],
        #"recurrent": [False],
        "base_policy_type": ["primal6"],
        "obj_density": [0.2],
        "env_size": [5],
        "env": ["independent_navigation-v7_1"] #, "independent_navigation-v5_1","independent_navigation-v5_2"],
        # "step_r": [-0.1, -0.4],
        # "obstacle_collision_r": [-0.015],
        # "agent_collision_r": [-0.4],
        # "goal_reached_r": [0.1],
        # "finish_episode_r": [0.0]
    
    }

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        name = "2A2_ppo_mixedr" + "_arc_" + param["base_policy_type"] \
            + "_env_" + str(param["env"]) \
            + "_disc_" + str(param["discount"]) \
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

           # + "_sr_" + str(param["step_r"]) \
            # + "_ocr_" + str(param["obstacle_collision_r"]) \
            # + "_acr_" + str(param["agent_collision_r"]) \
            # + "_grr_" + str(param["goal_reached_r"]) \
            # + "_fer_" + str(param["finish_episode_r"]) \

        # args.extend(["--use_custom_rewards"])
        # args.extend(["--step_r", str(param["step_r"])])
        # args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        # args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        # args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        # args.extend(["--finish_episode_r", str(param["finish_episode_r"])])

        args.extend(["--env_name", str(param["env"])])#"independent_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(50))])
        args.extend(["--benchmark_frequency", str(int(300000))])
        args.extend(["--benchmark_num_episodes", str(int(200))])
        args.extend(["--benchmark_render_length", str(int(50))])
        args.extend(["--checkpoint_frequency", str(int(500))])


        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(120)])
        args.extend(["--ppo_lr_a", str(0.001)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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



def ppo_4_1():
    '''Single agent '''
    n_iterations = str(5000) #str(3000)
    experiment_group_name = "4_1"
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
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["PPO"],
        "n_agents": [1],
        #"recurrent": [False],
        "base_policy_type": ["primal7"],
        "obj_density": [0.2,0.4,0.6],
        "env_size": [7],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.1],
        "finish_episode_r": [2.0]
    
    }

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        name = "ppo41" + "_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_disc_" + str(param["discount"]) \
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

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])

        args.extend(["--env_name","independent_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(50))])
        args.extend(["--benchmark_frequency", str(int(150000))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(50))])
        args.extend(["--checkpoint_frequency", str(int(500))])


        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(120)])
        args.extend(["--ppo_lr_a", str(0.001)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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


def ppo_4_2():
    '''Many agents no obstacles'''
    n_iterations = str(6000) #str(3000)
    experiment_group_name = "4_2"
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
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["PPO"],
        "n_agents": [16,8,4,2],
        #"recurrent": [False],
        "base_policy_type": ["primal7"],
        "obj_density": [0.0],
        "env_size": [7],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.5], #[0.1],
        "finish_episode_r": [2.0]
    
    }

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        name = "ppo41" + "_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_disc_" + str(param["discount"]) \
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

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])

        args.extend(["--env_name","independent_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(50))])
        args.extend(["--benchmark_frequency", str(int(150000))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(50))])
        args.extend(["--checkpoint_frequency", str(int(500))])


        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(120)])
        args.extend(["--ppo_lr_a", str(0.001)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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

def ppo_4_2_2():
    '''Many agents no obstacles'''
    n_iterations = str(6000) #str(3000)
    experiment_group_name = "4_2"
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
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["PPO"],
        "n_agents": [8,4,2],
        #"recurrent": [False],
        "base_policy_type": ["primal6"],
        "obj_density": [0.0],
        "env_size": [5],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.5], #[0.1],
        "finish_episode_r": [2.0]
    
    }

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        name = "ppo41" + "_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_disc_" + str(param["discount"]) \
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

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])

        args.extend(["--env_name","independent_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(50))])
        args.extend(["--benchmark_frequency", str(int(150000))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(50))])
        args.extend(["--checkpoint_frequency", str(int(500))])


        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(120)])
        args.extend(["--ppo_lr_a", str(0.001)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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


def ppo_4_3_1():
    '''Many agents no obstacles'''
    n_iterations = str(5000) #str(3000)
    experiment_group_name = "4_3"
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
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["PPO"],
        "n_agents": [4],
        #"recurrent": [False],
        "base_policy_type": ["primal7"],
        "obj_density": [0.1],
        "env_size": [7],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.5], #[0.1],
        "finish_episode_r": [2.0]
    
    }

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        name = "ppo41" + "_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_disc_" + str(param["discount"]) \
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

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])

        args.extend(["--env_name","independent_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(50))])
        args.extend(["--benchmark_frequency", str(int(150000))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(50))])
        args.extend(["--checkpoint_frequency", str(int(500))])


        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(120)])
        args.extend(["--ppo_lr_a", str(0.001)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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


def ppo_4_3_2():
    '''Many agents no obstacles'''
    n_iterations = str(5000) #str(3000)
    experiment_group_name = "4_3"
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
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["PPO"],
        "n_agents": [4],
        #"recurrent": [False],
        "base_policy_type": ["primal6"],
        "obj_density": [0.1, 0.2,0.3],
        "env_size": [5],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.5], #[0.1],
        "finish_episode_r": [2.0]
    
    }

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        name = "ppo41" + "_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_disc_" + str(param["discount"]) \
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

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])

        args.extend(["--env_name","independent_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(50))])
        args.extend(["--benchmark_frequency", str(int(150000))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(50))])
        args.extend(["--checkpoint_frequency", str(int(500))])


        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(120)])
        args.extend(["--ppo_lr_a", str(0.001)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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

def ppo_4_4():#Nr 3
    '''Many agents no obstacles'''
    n_iterations = str(5000) #str(3000)
    experiment_group_name = "4_4_3"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"
   # work_dir = experiment_group_name
  #  plot_dir = experiment_group_name + "_Central"
 
    parmeter_grid1 = {
        "seed": [1],
        "workers": [4],
        "rollout_length": [256],
        "eps_clip": [0.2],
        "k_epochs": [8],
        "minibatch_size": [512],
        "entropy_coeff":[0.01],
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["PPO"],
        "n_agents": [4],
        #"recurrent": [False],
        "base_policy_type": ["primal6"],
        "obj_density": [0.0],
        "env_size": [5],
        "env": ["ind_navigation_custom-v0"], #env ind from 0 to 6
        "custom_env_ind": [1] #,1,2,3,4,5,6]
        # "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        # "obstacle_collision_r": [-0.4], #[-0.015],
        # "agent_collision_r": [-0.4], #[-1.0], 
        # "goal_reached_r": [0.5], #[0.1],
        # "finish_episode_r": [2.0]

    
    }

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        name = "ppo" + "_arc_" + param["base_policy_type"] \
            + "_envind_" + str(param["custom_env_ind"]) \
            + "_disc_" + str(param["discount"]) \
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

            # + "_sr_" + str(param["step_r"]) \
            # + "_ocr_" + str(param["obstacle_collision_r"]) \
            # + "_acr_" + str(param["agent_collision_r"]) \
            # + "_grr_" + str(param["goal_reached_r"]) \
            # + "_fer_" + str(param["finish_episode_r"]) \

        # args.extend(["--use_custom_rewards"])
        # args.extend(["--step_r", str(param["step_r"])])
        # args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        # args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        # args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        # args.extend(["--finish_episode_r", str(param["finish_episode_r"])])

        args.extend(["--env_name",param["env"]])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--custom_env_ind", str(param["custom_env_ind"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(50))])
        args.extend(["--benchmark_frequency", str(int(150000))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(50))])
        args.extend(["--checkpoint_frequency", str(int(500))])


        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(120)])
        args.extend(["--ppo_lr_a", str(0.001)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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



def ppo_6_1():
    '''Partially observable with shortest path her'''
    n_iterations = str(6000) #str(3000)
    experiment_group_name = "6_1"
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
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["PPO"],
        "n_agents": [4],
        #"recurrent": [False],
        "base_policy_type": ["primal7"],
        "view_d": [3],
        "obj_density": [0.2],
        "env_size": [10],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.5], #[0.1],
        "finish_episode_r": [2.0]
    
    }

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        name = "ppo" + "_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_viewd_" + str(param["view_d"]) \
            + "_disc_" + str(param["discount"]) \
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

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])

        args.extend(["--env_name","independent_navigation-v8_0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--view_d", str(param["view_d"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(50))])
        args.extend(["--benchmark_frequency", str(int(150000))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(50))])
        args.extend(["--checkpoint_frequency", str(int(500))])


        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(120)])
        args.extend(["--ppo_lr_a", str(0.001)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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


def ppo_6_1_dirvec():
    '''Partially observable with shortest path her'''
    n_iterations = str(6000) #str(3000)
    experiment_group_name = "6_1"
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
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["PPO"],
        "n_agents": [4],
        #"recurrent": [False],
        "base_policy_type": ["primal7"],
        "view_d": [3],
        "obj_density": [0.2],
        "env_size": [10],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.5], #[0.1],
        "finish_episode_r": [2.0]
    
    }

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        name = "ppo" + "_dirvec" "_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_viewd_" + str(param["view_d"]) \
            + "_disc_" + str(param["discount"]) \
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

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])

        args.extend(["--env_name","independent_navigation-v8_1"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--view_d", str(param["view_d"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(50))])
        args.extend(["--benchmark_frequency", str(int(150000))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(50))])
        args.extend(["--checkpoint_frequency", str(int(500))])


        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(120)])
        args.extend(["--ppo_lr_a", str(0.001)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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


def ppo_6_2(): # CL with new shortest path observation
    '''Partially observable with shortest path heur'''
    n_iterations = str(12000) #str(3000)
    experiment_group_name = "6_2"
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
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["CURR_PPO"],
        "n_agents": [4],
        #"recurrent": [False],
        "base_policy_type": ["primal7"],
        "view_d": [3],
        "obj_density": [0.2],
        "env_size": [10],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.3], #[0.1],
        "finish_episode_r": [2.0]
    
    }

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        name = "ppo" + "_shortestpathobs" +"_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_viewd_" + str(param["view_d"]) \
            + "_disc_" + str(param["discount"]) \
            + "_lambda_" + str(param["lambda_"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["minibatch_size"]) \
            + "_rollouts_" + str(param["rollout_length"]) \
            + "_workers_" + str(param["workers"]) \
            + "_kepochs_" + str(param["k_epochs"]) \
            + "_curr_ppo_cl_inc_size" \
            + "_seed_"+ str(param["seed"])
            #+ "_envsize_" + str(param["env_size"]) \
            #+ "_nagents_"+ str(param["n_agents"]) \
            #+ "_objdensity_"+ str(param["obj_density"]) \

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])

        args.extend(["--env_name","independent_navigation-v8_0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--view_d", str(param["view_d"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(50))])
        args.extend(["--benchmark_frequency", str(int(500))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(50))])
        args.extend(["--checkpoint_frequency", str(int(400))])


        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(120)])
        args.extend(["--ppo_lr_a", str(0.001)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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
    

def ppo_6_2_dirvec(): # CL with direction vector
    '''Partially observable with shortest path her'''
    n_iterations = str(12000) #str(3000)
    experiment_group_name = "6_2"
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
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["CURR_PPO"],
        "n_agents": [4],
        #"recurrent": [False],
        "base_policy_type": ["primal7"],
        "view_d": [3],
        "obj_density": [0.2],
        "env_size": [10],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.3], #[0.1],
        "finish_episode_r": [2.0]
    
    }

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        name = "ppo" + "_dirvec" +"_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_viewd_" + str(param["view_d"]) \
            + "_disc_" + str(param["discount"]) \
            + "_lambda_" + str(param["lambda_"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["minibatch_size"]) \
            + "_rollouts_" + str(param["rollout_length"]) \
            + "_workers_" + str(param["workers"]) \
            + "_kepochs_" + str(param["k_epochs"]) \
            + "_curr_ppo_cl_inc_size" \
            + "_seed_"+ str(param["seed"])
            #+ "_envsize_" + str(param["env_size"]) \
            #+ "_nagents_"+ str(param["n_agents"]) \
            #+ "_objdensity_"+ str(param["obj_density"]) \

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])

        args.extend(["--env_name","independent_navigation-v8_1"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--view_d", str(param["view_d"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(50))])
        args.extend(["--benchmark_frequency", str(int(500))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(50))])
        args.extend(["--checkpoint_frequency", str(int(400))])


        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(120)])
        args.extend(["--ppo_lr_a", str(0.001)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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

    



def ppo_8_0(): # CL with new shortest path observation
    '''Partially observable with shortest path heur'''
    n_iterations = str(12000) #str(3000)
    experiment_group_name = "8_0"
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
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["CURR_PPO"],
        "n_agents": [4],
        #"recurrent": [False],
        "base_policy_type": ["primal7"],
        "view_d": [3],
        "obj_density": [0.2],
        "env_size": [10],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.3], #[0.1],
        "finish_episode_r": [2.0]
    
    }

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        name = "ppo" + "_shortestpathobs" +"_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_viewd_" + str(param["view_d"]) \
            + "_disc_" + str(param["discount"]) \
            + "_lambda_" + str(param["lambda_"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["minibatch_size"]) \
            + "_rollouts_" + str(param["rollout_length"]) \
            + "_workers_" + str(param["workers"]) \
            + "_kepochs_" + str(param["k_epochs"]) \
            + "_curr_ppo_cl_inc_size" \
            + "_seed_"+ str(param["seed"])
            #+ "_envsize_" + str(param["env_size"]) \
            #+ "_nagents_"+ str(param["n_agents"]) \
            #+ "_objdensity_"+ str(param["obj_density"]) \

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])

        args.extend(["--env_name","independent_navigation-v8_0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--view_d", str(param["view_d"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(50))])
        args.extend(["--benchmark_frequency", str(int(500))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(20))])
        args.extend(["--checkpoint_frequency", str(int(200))])
        args.extend(["--ppo_continue_from_checkpoint"])



        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(120)])
        args.extend(["--ppo_lr_a", str(0.001)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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


def ppo_8_2_0(): # CL with new shortest path observation
    '''Partially observable with shortest path heur'''
    n_iterations = str(12000) #str(3000)
    experiment_group_name = "8_3_0"
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
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["CURR_PPO"],
        "n_agents": [4],
        #"recurrent": [False],
        "base_policy_type": ["primal7"],
        "view_d": [3],
        "obj_density": [0.2],
        "env_size": [10],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.3], #[0.1],
        "finish_episode_r": [2.0]
    
    }

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        name = "ppo" + "_shortestpathobs" +"_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_viewd_" + str(param["view_d"]) \
            + "_disc_" + str(param["discount"]) \
            + "_lambda_" + str(param["lambda_"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["minibatch_size"]) \
            + "_rollouts_" + str(param["rollout_length"]) \
            + "_workers_" + str(param["workers"]) \
            + "_kepochs_" + str(param["k_epochs"]) \
            + "_curr_ppo_cl_inc_size" \
            + "_seed_"+ str(param["seed"])
            #+ "_envsize_" + str(param["env_size"]) \
            #+ "_nagents_"+ str(param["n_agents"]) \
            #+ "_objdensity_"+ str(param["obj_density"]) \

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])

        args.extend(["--env_name","independent_navigation-v8_0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--view_d", str(param["view_d"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(50))])
        args.extend(["--benchmark_frequency", str(int(500))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(20))])
        args.extend(["--checkpoint_frequency", str(int(200))])
       # args.extend(["--ppo_continue_from_checkpoint"])



        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(120)])
        args.extend(["--ppo_lr_a", str(0.00002)])
        args.extend(["--ppo_lr_v", str(0.00002)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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

def ppo_8_4_0(): # CL with dir vector #NB: REDO in 8_5_0 ... env was not changed in lessons.py, therefore independent_navigation-v8_0 was used`
    '''Partially observable with shortest path heur'''
    n_iterations = str(12000) #str(3000)
    experiment_group_name = "8_4_0"
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
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["CURR_PPO"],
        "n_agents": [4],
        #"recurrent": [False],
        "base_policy_type": ["primal7"],
        "view_d": [3],
        "obj_density": [0.2],
        "env_size": [10],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.3], #[0.1],
        "finish_episode_r": [2.0]
    
    }

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        name = "ppo" + "_shortestpathobs" +"_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_viewd_" + str(param["view_d"]) \
            + "_disc_" + str(param["discount"]) \
            + "_lambda_" + str(param["lambda_"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["minibatch_size"]) \
            + "_rollouts_" + str(param["rollout_length"]) \
            + "_workers_" + str(param["workers"]) \
            + "_kepochs_" + str(param["k_epochs"]) \
            + "_curr_ppo_cl_inc_size" \
            + "_seed_"+ str(param["seed"])
            #+ "_envsize_" + str(param["env_size"]) \
            #+ "_nagents_"+ str(param["n_agents"]) \
            #+ "_objdensity_"+ str(param["obj_density"]) \

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])

        args.extend(["--env_name","independent_navigation-v8_1"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--view_d", str(param["view_d"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(50))])
        args.extend(["--benchmark_frequency", str(int(500))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(20))])
        args.extend(["--checkpoint_frequency", str(int(200))])
       # args.extend(["--ppo_continue_from_checkpoint"])



        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(120)])
        args.extend(["--ppo_lr_a", str(0.001)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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


def ppo_8_5_0(): # Direction vec
    '''Partially observable with shortest path heur'''
    n_iterations = str(12000) #str(3000)
    experiment_group_name = "8_5_0_test3_2"
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
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["PPO"],
        "n_agents": [2],
        #"recurrent": [False],
        "base_policy_type": ["primal7"],
        "view_d": [3],
        "obj_density": [0.0],
        "env_size": [10],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.3], #[0.1],
        "finish_episode_r": [2.0]
    
    }

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        name = "ppo" + "_dirvec" +"_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_viewd_" + str(param["view_d"]) \
            + "_disc_" + str(param["discount"]) \
            + "_lambda_" + str(param["lambda_"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["minibatch_size"]) \
            + "_rollouts_" + str(param["rollout_length"]) \
            + "_workers_" + str(param["workers"]) \
            + "_kepochs_" + str(param["k_epochs"]) \
            + "_seed_"+ str(param["seed"])
            #+ "_envsize_" + str(param["env_size"]) \
            #+ "_nagents_"+ str(param["n_agents"]) \
            #+ "_objdensity_"+ str(param["obj_density"]) \

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])

        args.extend(["--env_name","independent_navigation-v8_1"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--view_d", str(param["view_d"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(30))])
        args.extend(["--benchmark_frequency", str(int(500))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(20))])
        args.extend(["--checkpoint_frequency", str(int(200))])
       # args.extend(["--ppo_continue_from_checkpoint"])



        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(120)])
        args.extend(["--ppo_lr_a", str(0.001)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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

def ppo_8_6_0(): # PRIMAL test
    '''Partially observable with shortest path heur'''
    n_iterations = str(12000) #str(3000)
    experiment_group_name = "8_6_0_testPRIMAL"
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
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["CURR_PPO"],
        "n_agents": [2],
        #"recurrent": [False],
        "base_policy_type": ["primal7"],
        "view_d": [3],
        "obj_density": [0.0],
        "env_size": [10],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.3], #[0.1],
        "finish_episode_r": [2.0]
    
    }

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        name = "ppo" +"_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_viewd_" + str(param["view_d"]) \
            + "_disc_" + str(param["discount"]) \
            + "_lambda_" + str(param["lambda_"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["minibatch_size"]) \
            + "_rollouts_" + str(param["rollout_length"]) \
            + "_workers_" + str(param["workers"]) \
            + "_kepochs_" + str(param["k_epochs"]) \
            + "_seed_"+ str(param["seed"])
            #+ "_envsize_" + str(param["env_size"]) \
            #+ "_nagents_"+ str(param["n_agents"]) \
            #+ "_objdensity_"+ str(param["obj_density"]) \

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])

        args.extend(["--env_name","independent_navigation-v8_1"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--view_d", str(param["view_d"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(30))])
        args.extend(["--benchmark_frequency", str(int(500))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(20))])
        args.extend(["--checkpoint_frequency", str(int(200))])
        args.extend(["--ppo_bc_iteration_prob", str(float(1.0))])
       # args.extend(["--ppo_continue_from_checkpoint"])



        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(120)])
        args.extend(["--ppo_lr_a", str(0.001)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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


def ppo_8_7_0():
    '''Partially observable with dir vec'''
    n_iterations = str(12000) #str(3000)
    experiment_group_name = "8_7_0_V8_0Curr"
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
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["CURR_PPO"],
        "n_agents": [2],
        #"recurrent": [False],
        "base_policy_type": ["primal7"],
        "view_d": [3],
        "obj_density": [0.0],
        "env_size": [10],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.3], #[0.1],
        "finish_episode_r": [2.0]
    
    }

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        name = "ppo" +"_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_viewd_" + str(param["view_d"]) \
            + "_disc_" + str(param["discount"]) \
            + "_lambda_" + str(param["lambda_"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["minibatch_size"]) \
            + "_rollouts_" + str(param["rollout_length"]) \
            + "_workers_" + str(param["workers"]) \
            + "_kepochs_" + str(param["k_epochs"]) \
            + "_seed_"+ str(param["seed"])
            #+ "_envsize_" + str(param["env_size"]) \
            #+ "_nagents_"+ str(param["n_agents"]) \
            #+ "_objdensity_"+ str(param["obj_density"]) \

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])

        args.extend(["--env_name","independent_navigation-v8_1"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--view_d", str(param["view_d"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(30))])
        args.extend(["--benchmark_frequency", str(int(500))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(20))])
        args.extend(["--checkpoint_frequency", str(int(200))])
        args.extend(["--ppo_bc_iteration_prob", str(float(0.5))])
       # args.extend(["--ppo_continue_from_checkpoint"])



        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(120)])
        args.extend(["--ppo_lr_a", str(0.001)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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

def ppo_8_7_1():
    '''Partially observable with shortest path heur'''
    n_iterations = str(12000) #str(3000)
    experiment_group_name = "8_7_1_V8_1Curr"
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
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["CURR_PPO"],
        "n_agents": [2],
        #"recurrent": [False],
        "base_policy_type": ["primal7"],
        "view_d": [3],
        "obj_density": [0.0],
        "env_size": [10],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.3], #[0.1],
        "finish_episode_r": [2.0]
    
    }

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        name = "ppo" +"_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_viewd_" + str(param["view_d"]) \
            + "_disc_" + str(param["discount"]) \
            + "_lambda_" + str(param["lambda_"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["minibatch_size"]) \
            + "_rollouts_" + str(param["rollout_length"]) \
            + "_workers_" + str(param["workers"]) \
            + "_kepochs_" + str(param["k_epochs"]) \
            + "_seed_"+ str(param["seed"])
            #+ "_envsize_" + str(param["env_size"]) \
            #+ "_nagents_"+ str(param["n_agents"]) \
            #+ "_objdensity_"+ str(param["obj_density"]) \

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])

        args.extend(["--env_name","independent_navigation-v8_1"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--view_d", str(param["view_d"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(30))])
        args.extend(["--benchmark_frequency", str(int(500))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(20))])
        args.extend(["--checkpoint_frequency", str(int(200))])
        args.extend(["--ppo_bc_iteration_prob", str(float(0.5))])
       # args.extend(["--ppo_continue_from_checkpoint"])



        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(120)])
        args.extend(["--ppo_lr_a", str(0.001)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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



def ppo_10_0_0(): # PRIMAL
    '''Partially observable with shortest path heur'''
    n_iterations = str(12000) #str(3000)
    experiment_group_name = "10_1_0"#"10_0_0_PRIMAL" #"test_validact"# #"test_NoLSTM_4Agnts_sz_15_ob_0_3ShortestPath" # "10_0_0_PRIMAL"
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
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["CURR_PPO"],
        "n_agents": [2],
        #"recurrent": [False],
        "base_policy_type": ["primal9"],
        "view_d": [4],
        "obj_density": [0.0],
        "env_size": [10],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.3], #[0.1],
        "finish_episode_r": [2.0],
        "block_r": [-0.2],
        "recurrent": [True],
        "blockheur": [True],
        "validact": [True],
        "noprevstate": [True]
    
    }

    # "step_r": [-0.1], #[-0.01, -0.1, -0.4],
    #     "obstacle_collision_r": [-0.4], #[-0.015],
    #     "agent_collision_r": [-0.4], #[-1.0], 
    #     "goal_reached_r": [0.3], #[0.1],
    #     "finish_episode_r": [2.0],
    #     "block_r": [-0.2],

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        id = "_1ownrownh_"
        name = "ppo" + id +"_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_viewd_" + str(param["view_d"]) \
            + "_disc_" + str(param["discount"]) \
            + "_lambda_" + str(param["lambda_"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["minibatch_size"]) \
            + "_rollouts_" + str(param["rollout_length"]) \
            + "_workers_" + str(param["workers"]) \
            + "_kepochs_" + str(param["k_epochs"]) \
            + "_recurrent_" + str(param["recurrent"]) \
            + "_block_" + str(param["blockheur"]) \
            + "_validact_" + str(param["validact"]) \
            + "_noprevstate_" + str(param["noprevstate"]) \
            + "_seed_"+ str(param["seed"])
            #+ "_envsize_" + str(param["env_size"]) \
            #+ "_nagents_"+ str(param["n_agents"]) \
            #+ "_objdensity_"+ str(param["obj_density"]) \

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])
        args.extend(["--block_r", str(param["block_r"])])

        args.extend(["--env_name","independent_navigation-v8_1"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--view_d", str(param["view_d"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(10))])
        args.extend(["--benchmark_frequency", str(int(500))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(10))])
        args.extend(["--checkpoint_frequency", str(int(200))])
        args.extend(["--ppo_bc_iteration_prob", str(float(0.5))])
       # args.extend(["--ppo_continue_from_checkpoint"])



        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(512)])
        args.extend(["--ppo_lr_a", str(0.00002)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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

        if param["recurrent"]:
            args.extend(["--ppo_recurrent"])
        if param["blockheur"]:
            args.extend(["--ppo_heur_block"])
        if param["validact"]:
            args.extend(["--ppo_heur_valid_act"])
        if param["noprevstate"]:
            args.extend(["--ppo_heur_no_prev_state"])
    
        main.main(args)


def ppo_10_1_1(): # PRIMAL: OWN r-structure; own H-parameters
    '''Partially observable with shortest path heur'''
    n_iterations = str(12000) #str(3000)
    experiment_group_name = "10_2_0"#"10_0_0_PRIMAL" #"test_validact"# #"test_NoLSTM_4Agnts_sz_15_ob_0_3ShortestPath" # "10_0_0_PRIMAL"
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
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["CURR_PPO"],
        "n_agents": [2],
        #"recurrent": [False],
        "base_policy_type": ["primal9"],
        "view_d": [4],
        "obj_density": [0.0],
        "env_size": [10],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.3], #[0.1],
        "finish_episode_r": [2.0],
        "block_r": [-1.0],
        "recurrent": [True],
        "blockheur": [True],
        "validact": [True],
        "noprevstate": [True]
    
    }

    # "step_r": [-0.1], #[-0.01, -0.1, -0.4],
    #     "obstacle_collision_r": [-0.4], #[-0.015],
    #     "agent_collision_r": [-0.4], #[-1.0], 
    #     "goal_reached_r": [0.3], #[0.1],
    #     "finish_episode_r": [2.0],
    #     "block_r": [-0.2],

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        id = "_1ownrownh_"
        name = "ppo" + id +"_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_viewd_" + str(param["view_d"]) \
            + "_disc_" + str(param["discount"]) \
            + "_lambda_" + str(param["lambda_"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["minibatch_size"]) \
            + "_rollouts_" + str(param["rollout_length"]) \
            + "_workers_" + str(param["workers"]) \
            + "_kepochs_" + str(param["k_epochs"]) \
            + "_recurrent_" + str(param["recurrent"]) \
            + "_block_" + str(param["blockheur"]) \
            + "_validact_" + str(param["validact"]) \
            + "_noprevstate_" + str(param["noprevstate"]) \
            + "_seed_"+ str(param["seed"])
            #+ "_envsize_" + str(param["env_size"]) \
            #+ "_nagents_"+ str(param["n_agents"]) \
            #+ "_objdensity_"+ str(param["obj_density"]) \

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])
        args.extend(["--block_r", str(param["block_r"])])

        args.extend(["--env_name","independent_navigation-v8_1"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--view_d", str(param["view_d"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(10))])
        args.extend(["--benchmark_frequency", str(int(500))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(10))])
        args.extend(["--checkpoint_frequency", str(int(200))])
        args.extend(["--ppo_bc_iteration_prob", str(float(0.5))])
       # args.extend(["--ppo_continue_from_checkpoint"])



        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(512)])
        args.extend(["--ppo_lr_a", str(0.00002)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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

        if param["recurrent"]:
            args.extend(["--ppo_recurrent"])
        if param["blockheur"]:
            args.extend(["--ppo_heur_block"])
        if param["validact"]:
            args.extend(["--ppo_heur_valid_act"])
        if param["noprevstate"]:
            args.extend(["--ppo_heur_no_prev_state"])
    
        main.main(args)


def ppo_10_1_2(): # PRIMAL: OWN r-structure; thier similar hyperparam H-parameters
    '''Partially observable with shortest path heur'''
    n_iterations = str(12000) #str(3000)
    experiment_group_name = "10_2_0"#"10_0_0_PRIMAL" #"test_validact"# #"test_NoLSTM_4Agnts_sz_15_ob_0_3ShortestPath" # "10_0_0_PRIMAL"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"
 
    parmeter_grid1 = {
        "seed": [1],
        "workers": [4],
        "rollout_length": [256],
        "eps_clip": [0.2],
        "k_epochs": [8],
        "minibatch_size": [128],
        "entropy_coeff":[0.01],
        "discount":[0.95],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["CURR_PPO"],
        "n_agents": [2],
        #"recurrent": [False],
        "base_policy_type": ["primal9"],
        "view_d": [4],
        "obj_density": [0.0],
        "env_size": [10],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.3], #[0.1],
        "finish_episode_r": [2.0],
        "block_r": [-1.0],
        "recurrent": [True],
        "blockheur": [True],
        "validact": [True],
        "noprevstate": [True]
    
    }

    # "step_r": [-0.1], #[-0.01, -0.1, -0.4],
    #     "obstacle_collision_r": [-0.4], #[-0.015],
    #     "agent_collision_r": [-0.4], #[-1.0], 
    #     "goal_reached_r": [0.3], #[0.1],
    #     "finish_episode_r": [2.0],
    #     "block_r": [-0.2],

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        id = "_1ownr_theirsimilarh_"
        name = "ppo" + id +"_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_viewd_" + str(param["view_d"]) \
            + "_disc_" + str(param["discount"]) \
            + "_lambda_" + str(param["lambda_"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["minibatch_size"]) \
            + "_rollouts_" + str(param["rollout_length"]) \
            + "_workers_" + str(param["workers"]) \
            + "_kepochs_" + str(param["k_epochs"]) \
            + "_recurrent_" + str(param["recurrent"]) \
            + "_block_" + str(param["blockheur"]) \
            + "_validact_" + str(param["validact"]) \
            + "_noprevstate_" + str(param["noprevstate"]) \
            + "_seed_"+ str(param["seed"])
            #+ "_envsize_" + str(param["env_size"]) \
            #+ "_nagents_"+ str(param["n_agents"]) \
            #+ "_objdensity_"+ str(param["obj_density"]) \

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])
        args.extend(["--block_r", str(param["block_r"])])

        args.extend(["--env_name","independent_navigation-v8_1"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--view_d", str(param["view_d"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(10))])
        args.extend(["--benchmark_frequency", str(int(500))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(10))])
        args.extend(["--checkpoint_frequency", str(int(200))])
        args.extend(["--ppo_bc_iteration_prob", str(float(0.5))])
       # args.extend(["--ppo_continue_from_checkpoint"])



        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(512)])
        args.extend(["--ppo_lr_a", str(0.00002)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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

        if param["recurrent"]:
            args.extend(["--ppo_recurrent"])
        if param["blockheur"]:
            args.extend(["--ppo_heur_block"])
        if param["validact"]:
            args.extend(["--ppo_heur_valid_act"])
        if param["noprevstate"]:
            args.extend(["--ppo_heur_no_prev_state"])
    
        main.main(args)

    
def ppo_10_1_3(): # PRIMAL: Their r-structure; thier similar hyperparam H-parameters
    '''Partially observable with shortest path heur'''
    n_iterations = str(12000) #str(3000)
    experiment_group_name = "10_2_0"#"10_0_0_PRIMAL" #"test_validact"# #"test_NoLSTM_4Agnts_sz_15_ob_0_3ShortestPath" # "10_0_0_PRIMAL"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"
 
    parmeter_grid1 = {
        "seed": [1],
        "workers": [4],
        "rollout_length": [256],
        "eps_clip": [0.2],
        "k_epochs": [8],
        "minibatch_size": [128],
        "entropy_coeff":[0.01],
        "discount":[0.95],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["CURR_PPO"],
        "n_agents": [2],
        #"recurrent": [False],
        "base_policy_type": ["primal9"],
        "view_d": [4],
        "obj_density": [0.0],
        "env_size": [10],
        "step_r": [-0.3], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.0], #[-0.015],
        "agent_collision_r": [-2.0], #[-1.0], 
        "goal_reached_r": [0.3], #[0.1],
        "finish_episode_r": [20.0],
        "block_r": [-2.0],
        "recurrent": [True],
        "blockheur": [True],
        "validact": [True],
        "noprevstate": [True]
    
    }

    # "step_r": [-0.1], #[-0.01, -0.1, -0.4],
    #     "obstacle_collision_r": [-0.4], #[-0.015],
    #     "agent_collision_r": [-0.4], #[-1.0], 
    #     "goal_reached_r": [0.3], #[0.1],
    #     "finish_episode_r": [2.0],
    #     "block_r": [-0.2],

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        id = "_3theirr_theirsimilarh_"
        name = "ppo" + id +"_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_viewd_" + str(param["view_d"]) \
            + "_disc_" + str(param["discount"]) \
            + "_lambda_" + str(param["lambda_"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["minibatch_size"]) \
            + "_rollouts_" + str(param["rollout_length"]) \
            + "_workers_" + str(param["workers"]) \
            + "_kepochs_" + str(param["k_epochs"]) \
            + "_recurrent_" + str(param["recurrent"]) \
            + "_block_" + str(param["blockheur"]) \
            + "_validact_" + str(param["validact"]) \
            + "_noprevstate_" + str(param["noprevstate"]) \
            + "_seed_"+ str(param["seed"])
            #+ "_envsize_" + str(param["env_size"]) \
            #+ "_nagents_"+ str(param["n_agents"]) \
            #+ "_objdensity_"+ str(param["obj_density"]) \

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])
        args.extend(["--block_r", str(param["block_r"])])

        args.extend(["--env_name","independent_navigation-v8_1"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--view_d", str(param["view_d"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(10))])
        args.extend(["--benchmark_frequency", str(int(500))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(10))])
        args.extend(["--checkpoint_frequency", str(int(200))])
        args.extend(["--ppo_bc_iteration_prob", str(float(0.5))])
       # args.extend(["--ppo_continue_from_checkpoint"])



        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(512)])
        args.extend(["--ppo_lr_a", str(0.00002)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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

        if param["recurrent"]:
            args.extend(["--ppo_recurrent"])
        if param["blockheur"]:
            args.extend(["--ppo_heur_block"])
        if param["validact"]:
            args.extend(["--ppo_heur_valid_act"])
        if param["noprevstate"]:
            args.extend(["--ppo_heur_no_prev_state"])
    
        main.main(args)





def ppo_10_4_0(): #TEST different # PRIMAL: OWN r-structure; own H-parameters
    '''Partially observable with shortest path heur'''
    n_iterations = str(12000) #str(3000)
    experiment_group_name = "test10_2_0"#"10_0_0_PRIMAL" #"test_validact"# #"test_NoLSTM_4Agnts_sz_15_ob_0_3ShortestPath" # "10_0_0_PRIMAL"
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
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["CURR_PPO"],
        "n_agents": [2],
        #"recurrent": [False],
        "base_policy_type": ["primal9"],
        "view_d": [4],
        "obj_density": [0.0],
        "env_size": [10],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.3], #[0.1],
        "finish_episode_r": [2.0],
        "block_r": [-1.0],
        "recurrent": [True],
        "blockheur": [True],
        "validact": [True],
        "noprevstate": [True]
    
    }

    # "step_r": [-0.1], #[-0.01, -0.1, -0.4],
    #     "obstacle_collision_r": [-0.4], #[-0.015],
    #     "agent_collision_r": [-0.4], #[-1.0], 
    #     "goal_reached_r": [0.3], #[0.1],
    #     "finish_episode_r": [2.0],
    #     "block_r": [-0.2],

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        id = "_TestBCnoLoss1ownrownh2e-5_"
        name = "ppo" + id +"_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_viewd_" + str(param["view_d"]) \
            + "_disc_" + str(param["discount"]) \
            + "_lambda_" + str(param["lambda_"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["minibatch_size"]) \
            + "_rollouts_" + str(param["rollout_length"]) \
            + "_workers_" + str(param["workers"]) \
            + "_kepochs_" + str(param["k_epochs"]) \
            + "_recurrent_" + str(param["recurrent"]) \
            + "_block_" + str(param["blockheur"]) \
            + "_validact_" + str(param["validact"]) \
            + "_noprevstate_" + str(param["noprevstate"]) \
            + "_seed_"+ str(param["seed"])
            #+ "_envsize_" + str(param["env_size"]) \
            #+ "_nagents_"+ str(param["n_agents"]) \
            #+ "_objdensity_"+ str(param["obj_density"]) \

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])
        args.extend(["--block_r", str(param["block_r"])])

        args.extend(["--env_name","independent_navigation-v8_1"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--view_d", str(param["view_d"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(10))])
        args.extend(["--benchmark_frequency", str(int(500))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(10))])
        args.extend(["--checkpoint_frequency", str(int(200))])
        args.extend(["--ppo_bc_iteration_prob", str(float(0.5))])
       # args.extend(["--ppo_continue_from_checkpoint"])



        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(512)])
        args.extend(["--ppo_lr_a", str(0.00002)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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

        if param["recurrent"]:
            args.extend(["--ppo_recurrent"])
        if param["blockheur"]:
            args.extend(["--ppo_heur_block"])
        if param["validact"]:
            args.extend(["--ppo_heur_valid_act"])
        if param["noprevstate"]:
            args.extend(["--ppo_heur_no_prev_state"])
    
        main.main(args)


def ppo_10_5_0(): #PRIMAL with shortest path heuristic
    '''Partially observable with shortest path heur'''
    n_iterations = str(12000) #str(3000)
    experiment_group_name = "10_2_0"#"10_0_0_PRIMAL" #"test_validact"# #"test_NoLSTM_4Agnts_sz_15_ob_0_3ShortestPath" # "10_0_0_PRIMAL"
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
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["CURR_PPO"],
        "n_agents": [2],
        #"recurrent": [False],
        "base_policy_type": ["primal9"],
        "view_d": [4],
        "obj_density": [0.0],
        "env_size": [10],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.3], #[0.1],
        "finish_episode_r": [2.0],
        "block_r": [-1.0],
        "recurrent": [True],
        "blockheur": [True],
        "validact": [True],
        "noprevstate": [True]
    }

    # "step_r": [-0.1], #[-0.01, -0.1, -0.4],
    #     "obstacle_collision_r": [-0.4], #[-0.015],
    #     "agent_collision_r": [-0.4], #[-1.0], 
    #     "goal_reached_r": [0.3], #[0.1],
    #     "finish_episode_r": [2.0],
    #     "block_r": [-0.2],

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        id = "_ShortPathBCnoLoss1ownrownh2e-5FullPrimal_"
        name = "ppo" + id +"_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_viewd_" + str(param["view_d"]) \
            + "_disc_" + str(param["discount"]) \
            + "_lambda_" + str(param["lambda_"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["minibatch_size"]) \
            + "_rollouts_" + str(param["rollout_length"]) \
            + "_workers_" + str(param["workers"]) \
            + "_kepochs_" + str(param["k_epochs"]) \
            + "_recurrent_" + str(param["recurrent"]) \
            + "_block_" + str(param["blockheur"]) \
            + "_validact_" + str(param["validact"]) \
            + "_noprevstate_" + str(param["noprevstate"]) \
            + "_seed_"+ str(param["seed"])
            #+ "_envsize_" + str(param["env_size"]) \
            #+ "_nagents_"+ str(param["n_agents"]) \
            #+ "_objdensity_"+ str(param["obj_density"]) \

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])
        args.extend(["--block_r", str(param["block_r"])])

        args.extend(["--env_name","independent_navigation-v8_0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--view_d", str(param["view_d"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(10))])
        args.extend(["--benchmark_frequency", str(int(500))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(10))])
        args.extend(["--checkpoint_frequency", str(int(200))])
        args.extend(["--ppo_bc_iteration_prob", str(float(0.5))])
       # args.extend(["--ppo_continue_from_checkpoint"])



        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(512)])
        args.extend(["--ppo_lr_a", str(0.00002)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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

        if param["recurrent"]:
            args.extend(["--ppo_recurrent"])
        if param["blockheur"]:
            args.extend(["--ppo_heur_block"])
        if param["validact"]:
            args.extend(["--ppo_heur_valid_act"])
        if param["noprevstate"]:
            args.extend(["--ppo_heur_no_prev_state"])
    
        main.main(args)


def ppo_10_6_0(): #PRIMAL with shortest path heuristic
    '''Partially observable with shortest path heur'''
    n_iterations = str(12000) #str(3000)
    experiment_group_name = "10_2_0"#"10_0_0_PRIMAL" #"test_validact"# #"test_NoLSTM_4Agnts_sz_15_ob_0_3ShortestPath" # "10_0_0_PRIMAL"
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
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["CURR_PPO"],
        "n_agents": [2],
        #"recurrent": [False],
        "base_policy_type": ["primal9"],
        "view_d": [4],
        "obj_density": [0.0],
        "env_size": [10],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.3], #[0.1],
        "finish_episode_r": [2.0],
        "block_r": [-1.0],
        "recurrent": [True],
        "blockheur": [True],
        "validact": [False],
        "noprevstate": [False]
    }

    # "step_r": [-0.1], #[-0.01, -0.1, -0.4],
    #     "obstacle_collision_r": [-0.4], #[-0.015],
    #     "agent_collision_r": [-0.4], #[-1.0], 
    #     "goal_reached_r": [0.3], #[0.1],
    #     "finish_episode_r": [2.0],
    #     "block_r": [-0.2],

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        id = "_ShortPathBCnoLoss1ownrownh2e-5NoVld_"
        name = "ppo" + id +"_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_viewd_" + str(param["view_d"]) \
            + "_disc_" + str(param["discount"]) \
            + "_lambda_" + str(param["lambda_"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["minibatch_size"]) \
            + "_rollouts_" + str(param["rollout_length"]) \
            + "_workers_" + str(param["workers"]) \
            + "_kepochs_" + str(param["k_epochs"]) \
            + "_recurrent_" + str(param["recurrent"]) \
            + "_block_" + str(param["blockheur"]) \
            + "_validact_" + str(param["validact"]) \
            + "_noprevstate_" + str(param["noprevstate"]) \
            + "_seed_"+ str(param["seed"])
            #+ "_envsize_" + str(param["env_size"]) \
            #+ "_nagents_"+ str(param["n_agents"]) \
            #+ "_objdensity_"+ str(param["obj_density"]) \

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])
        args.extend(["--block_r", str(param["block_r"])])

        args.extend(["--env_name","independent_navigation-v8_0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--view_d", str(param["view_d"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(10))])
        args.extend(["--benchmark_frequency", str(int(500))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(10))])
        args.extend(["--checkpoint_frequency", str(int(200))])
        args.extend(["--ppo_bc_iteration_prob", str(float(0.5))])
       # args.extend(["--ppo_continue_from_checkpoint"])



        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(512)])
        args.extend(["--ppo_lr_a", str(0.00002)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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

        if param["recurrent"]:
            args.extend(["--ppo_recurrent"])
        if param["blockheur"]:
            args.extend(["--ppo_heur_block"])
        if param["validact"]:
            args.extend(["--ppo_heur_valid_act"])
        if param["noprevstate"]:
            args.extend(["--ppo_heur_no_prev_state"])
    
        main.main(args)


def ppo_10_7_0(): #Full Primal (dir vec) again
    '''Partially observable with shortest path heur'''
    n_iterations = str(12000) #str(3000)
    experiment_group_name = "10_2_0" # "10_0_0_PRIMAL" #"test_validact"# #"test_NoLSTM_4Agnts_sz_15_ob_0_3ShortestPath" # "10_0_0_PRIMAL"
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
        "discount":[0.5], #[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["CURR_PPO"],
        "n_agents": [2],
        #"recurrent": [False],
        "base_policy_type": ["primal9"],
        "view_d": [4],
        "obj_density": [0.0],
        "env_size": [10],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.3], #[0.1],
        "finish_episode_r": [2.0],
        "block_r": [-1.0],
        "recurrent": [True],
        "blockheur": [True],
        "validact": [True],
        "noprevstate": [True]
    }

    # "step_r": [-0.1], #[-0.01, -0.1, -0.4],
    #     "obstacle_collision_r": [-0.4], #[-0.015],
    #     "agent_collision_r": [-0.4], #[-1.0], 
    #     "goal_reached_r": [0.3], #[0.1],
    #     "finish_episode_r": [2.0],
    #     "block_r": [-0.2],

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        id = "_10_7_0FullPrimalBCnoLoss1ownrownh2e-5_"
        name = "ppo" + id +"_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_viewd_" + str(param["view_d"]) \
            + "_disc_" + str(param["discount"]) \
            + "_lambda_" + str(param["lambda_"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["minibatch_size"]) \
            + "_rollouts_" + str(param["rollout_length"]) \
            + "_workers_" + str(param["workers"]) \
            + "_kepochs_" + str(param["k_epochs"]) \
            + "_recurrent_" + str(param["recurrent"]) \
            + "_block_" + str(param["blockheur"]) \
            + "_validact_" + str(param["validact"]) \
            + "_noprevstate_" + str(param["noprevstate"]) \
            + "_seed_"+ str(param["seed"])
            #+ "_envsize_" + str(param["env_size"]) \
            #+ "_nagents_"+ str(param["n_agents"]) \
            #+ "_objdensity_"+ str(param["obj_density"]) \

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])
        args.extend(["--block_r", str(param["block_r"])])

        args.extend(["--env_name","independent_navigation-v8_1"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--view_d", str(param["view_d"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(10))])
        args.extend(["--benchmark_frequency", str(int(500))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(10))])
        args.extend(["--checkpoint_frequency", str(int(200))])
        args.extend(["--ppo_bc_iteration_prob", str(float(0.5))])
       # args.extend(["--ppo_continue_from_checkpoint"])



        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(512)])
        args.extend(["--ppo_lr_a", str(0.00002)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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

        if param["recurrent"]:
            args.extend(["--ppo_recurrent"])
        if param["blockheur"]:
            args.extend(["--ppo_heur_block"])
        if param["validact"]:
            args.extend(["--ppo_heur_valid_act"])
        if param["noprevstate"]:
            args.extend(["--ppo_heur_no_prev_state"])
    
        main.main(args)


def ppo_10_7_2(): #Full Primal (dir vec) again
    '''Partially observable with shortest path heur'''
    n_iterations = str(12000) #str(3000)
    experiment_group_name = "10_2_0" # "10_0_0_PRIMAL" #"test_validact"# #"test_NoLSTM_4Agnts_sz_15_ob_0_3ShortestPath" # "10_0_0_PRIMAL"
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
        "discount":[0.5], #[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["CURR_PPO"],
        "n_agents": [2],
        #"recurrent": [False],
        "base_policy_type": ["primal9"],
        "view_d": [4],
        "obj_density": [0.0],
        "env_size": [10],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.3], #[0.1],
        "finish_episode_r": [2.0],
        "block_r": [-1.0],
        "recurrent": [True],
        "blockheur": [False],
        "validact": [True],
        "noprevstate": [True]
    }

    # "step_r": [-0.1], #[-0.01, -0.1, -0.4],
    #     "obstacle_collision_r": [-0.4], #[-0.015],
    #     "agent_collision_r": [-0.4], #[-1.0], 
    #     "goal_reached_r": [0.3], #[0.1],
    #     "finish_episode_r": [2.0],
    #     "block_r": [-0.2],

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        id = "_10_7_2_FullPrimalBCnoLoss1ownrownh2e-5_"
        name = "ppo" + id +"_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_viewd_" + str(param["view_d"]) \
            + "_disc_" + str(param["discount"]) \
            + "_lambda_" + str(param["lambda_"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["minibatch_size"]) \
            + "_rollouts_" + str(param["rollout_length"]) \
            + "_workers_" + str(param["workers"]) \
            + "_kepochs_" + str(param["k_epochs"]) \
            + "_recurrent_" + str(param["recurrent"]) \
            + "_block_" + str(param["blockheur"]) \
            + "_validact_" + str(param["validact"]) \
            + "_noprevstate_" + str(param["noprevstate"]) \
            + "_seed_"+ str(param["seed"])
            #+ "_envsize_" + str(param["env_size"]) \
            #+ "_nagents_"+ str(param["n_agents"]) \
            #+ "_objdensity_"+ str(param["obj_density"]) \

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])
        args.extend(["--block_r", str(param["block_r"])])

        args.extend(["--env_name","independent_navigation-v8_1"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--view_d", str(param["view_d"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(10))])
        args.extend(["--benchmark_frequency", str(int(500))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(10))])
        args.extend(["--checkpoint_frequency", str(int(200))])
        args.extend(["--ppo_bc_iteration_prob", str(float(0.5))])
       # args.extend(["--ppo_continue_from_checkpoint"])



        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(512)])
        args.extend(["--ppo_lr_a", str(0.00002)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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

        if param["recurrent"]:
            args.extend(["--ppo_recurrent"])
        if param["blockheur"]:
            args.extend(["--ppo_heur_block"])
        if param["validact"]:
            args.extend(["--ppo_heur_valid_act"])
        if param["noprevstate"]:
            args.extend(["--ppo_heur_no_prev_state"])
    
        main.main(args)


def ppo_10_7_3(): #(dir vec) no env sampling
    '''Partially observable with shortest path heur'''
    n_iterations = str(12000) #str(3000)
    experiment_group_name = "10_3_0" # "10_0_0_PRIMAL" #"test_validact"# #"test_NoLSTM_4Agnts_sz_15_ob_0_3ShortestPath" # "10_0_0_PRIMAL"
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
        "discount":[0.5], #[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["CURR_PPO"],
        "n_agents": [2],
        #"recurrent": [False],
        "base_policy_type": ["primal9"],
        "view_d": [4],
        "obj_density": [0.0],
        "env_size": [10],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.3], #[0.1],
        "finish_episode_r": [2.0],
        "block_r": [-1.0],
        "recurrent": [True],
        "blockheur": [True],
        "validact": [True],
        "noprevstate": [True]
    }

    # "step_r": [-0.1], #[-0.01, -0.1, -0.4],
    #     "obstacle_collision_r": [-0.4], #[-0.015],
    #     "agent_collision_r": [-0.4], #[-1.0], 
    #     "goal_reached_r": [0.3], #[0.1],
    #     "finish_episode_r": [2.0],
    #     "block_r": [-0.2],

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        id = "_10_7_3_NoEnvSamplingBCnoLoss1ownrownh2e-5_"
        name = "ppo" + id +"_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_viewd_" + str(param["view_d"]) \
            + "_disc_" + str(param["discount"]) \
            + "_lambda_" + str(param["lambda_"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["minibatch_size"]) \
            + "_rollouts_" + str(param["rollout_length"]) \
            + "_workers_" + str(param["workers"]) \
            + "_kepochs_" + str(param["k_epochs"]) \
            + "_recurrent_" + str(param["recurrent"]) \
            + "_block_" + str(param["blockheur"]) \
            + "_validact_" + str(param["validact"]) \
            + "_noprevstate_" + str(param["noprevstate"]) \
            + "_seed_"+ str(param["seed"])
            #+ "_envsize_" + str(param["env_size"]) \
            #+ "_nagents_"+ str(param["n_agents"]) \
            #+ "_objdensity_"+ str(param["obj_density"]) \

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])
        args.extend(["--block_r", str(param["block_r"])])

        args.extend(["--env_name","independent_navigation-v8_1"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--view_d", str(param["view_d"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(10))])
        args.extend(["--benchmark_frequency", str(int(500))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(10))])
        args.extend(["--checkpoint_frequency", str(int(200))])
        args.extend(["--ppo_bc_iteration_prob", str(float(0.5))])
       # args.extend(["--ppo_continue_from_checkpoint"])



        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(512)])
        args.extend(["--ppo_lr_a", str(0.00002)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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

        if param["recurrent"]:
            args.extend(["--ppo_recurrent"])
        if param["blockheur"]:
            args.extend(["--ppo_heur_block"])
        if param["validact"]:
            args.extend(["--ppo_heur_valid_act"])
        if param["noprevstate"]:
            args.extend(["--ppo_heur_no_prev_state"])
    
        main.main(args)


def ppo_10_7_4(): # (dir vec) again
    '''Partially observable with shortest path heur'''
    n_iterations = str(12000) #str(3000)
    experiment_group_name = "10_3_0" # "10_0_0_PRIMAL" #"test_validact"# #"test_NoLSTM_4Agnts_sz_15_ob_0_3ShortestPath" # "10_0_0_PRIMAL"
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
        "discount":[0.5], #[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["CURR_PPO"],
        "n_agents": [2],
        #"recurrent": [False],
        "base_policy_type": ["primal9"],
        "view_d": [4],
        "obj_density": [0.0],
        "env_size": [10],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.3], #[0.1],
        "finish_episode_r": [2.0],
        "block_r": [-1.0],
        "recurrent": [True],
        "blockheur": [False],
        "validact": [False],
        "noprevstate": [False]
    }

    # "step_r": [-0.1], #[-0.01, -0.1, -0.4],
    #     "obstacle_collision_r": [-0.4], #[-0.015],
    #     "agent_collision_r": [-0.4], #[-1.0], 
    #     "goal_reached_r": [0.3], #[0.1],
    #     "finish_episode_r": [2.0],
    #     "block_r": [-0.2],

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        id = "_10_7_4_NvldNblkBCnoLoss1ownrownh2e-5_"
        name = "ppo" + id +"_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_viewd_" + str(param["view_d"]) \
            + "_disc_" + str(param["discount"]) \
            + "_lambda_" + str(param["lambda_"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["minibatch_size"]) \
            + "_rollouts_" + str(param["rollout_length"]) \
            + "_workers_" + str(param["workers"]) \
            + "_kepochs_" + str(param["k_epochs"]) \
            + "_recurrent_" + str(param["recurrent"]) \
            + "_block_" + str(param["blockheur"]) \
            + "_validact_" + str(param["validact"]) \
            + "_noprevstate_" + str(param["noprevstate"]) \
            + "_seed_"+ str(param["seed"])
            #+ "_envsize_" + str(param["env_size"]) \
            #+ "_nagents_"+ str(param["n_agents"]) \
            #+ "_objdensity_"+ str(param["obj_density"]) \

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])
        args.extend(["--block_r", str(param["block_r"])])

        args.extend(["--env_name","independent_navigation-v8_1"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--view_d", str(param["view_d"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(10))])
        args.extend(["--benchmark_frequency", str(int(500))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(10))])
        args.extend(["--checkpoint_frequency", str(int(200))])
        args.extend(["--ppo_bc_iteration_prob", str(float(0.5))])
       # args.extend(["--ppo_continue_from_checkpoint"])



        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(512)])
        args.extend(["--ppo_lr_a", str(0.00002)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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

        if param["recurrent"]:
            args.extend(["--ppo_recurrent"])
        if param["blockheur"]:
            args.extend(["--ppo_heur_block"])
        if param["validact"]:
            args.extend(["--ppo_heur_valid_act"])
        if param["noprevstate"]:
            args.extend(["--ppo_heur_no_prev_state"])
    
        main.main(args)


def ppo_10_8_0(): #NO IL:
    '''Partially observable with shortest path heur'''
    n_iterations = str(12000) #str(3000)
    experiment_group_name = "10_2_0" # "10_0_0_PRIMAL" #"test_validact"# #"test_NoLSTM_4Agnts_sz_15_ob_0_3ShortestPath" # "10_0_0_PRIMAL"
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
        "discount":[0.5], #[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["CURR_PPO"],
        "n_agents": [2],
        #"recurrent": [False],
        "base_policy_type": ["primal9"],
        "view_d": [4],
        "obj_density": [0.0],
        "env_size": [10],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.3], #[0.1],
        "finish_episode_r": [2.0],
        "block_r": [-1.0],
        "recurrent": [True],
        "blockheur": [True],
        "validact": [True],
        "noprevstate": [True]
    }

    # "step_r": [-0.1], #[-0.01, -0.1, -0.4],
    #     "obstacle_collision_r": [-0.4], #[-0.015],
    #     "agent_collision_r": [-0.4], #[-1.0], 
    #     "goal_reached_r": [0.3], #[0.1],
    #     "finish_episode_r": [2.0],
    #     "block_r": [-0.2],

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        id = "_No_IL_PrimalBCnoLoss1ownrownh2e-5_"
        name = "ppo" + id +"_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_viewd_" + str(param["view_d"]) \
            + "_disc_" + str(param["discount"]) \
            + "_lambda_" + str(param["lambda_"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["minibatch_size"]) \
            + "_rollouts_" + str(param["rollout_length"]) \
            + "_workers_" + str(param["workers"]) \
            + "_kepochs_" + str(param["k_epochs"]) \
            + "_recurrent_" + str(param["recurrent"]) \
            + "_block_" + str(param["blockheur"]) \
            + "_validact_" + str(param["validact"]) \
            + "_noprevstate_" + str(param["noprevstate"]) \
            + "_seed_"+ str(param["seed"])
            #+ "_envsize_" + str(param["env_size"]) \
            #+ "_nagents_"+ str(param["n_agents"]) \
            #+ "_objdensity_"+ str(param["obj_density"]) \

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])
        args.extend(["--block_r", str(param["block_r"])])

        args.extend(["--env_name","independent_navigation-v8_1"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--view_d", str(param["view_d"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(10))])
        args.extend(["--benchmark_frequency", str(int(500))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(10))])
        args.extend(["--checkpoint_frequency", str(int(200))])
        args.extend(["--ppo_bc_iteration_prob", str(float(0.0))])
       # args.extend(["--ppo_continue_from_checkpoint"])



        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(512)])
        args.extend(["--ppo_lr_a", str(0.00002)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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

        if param["recurrent"]:
            args.extend(["--ppo_recurrent"])
        if param["blockheur"]:
            args.extend(["--ppo_heur_block"])
        if param["validact"]:
            args.extend(["--ppo_heur_valid_act"])
        if param["noprevstate"]:
            args.extend(["--ppo_heur_no_prev_state"])
    
        main.main(args)




def ppo_10_9_0(): #Full Primal (dir vec) again
    '''Partially observable with shortest path heur'''
    n_iterations = str(12000) #str(3000)
    experiment_group_name = "10_3_0" # "10_0_0_PRIMAL" #"test_validact"# #"test_NoLSTM_4Agnts_sz_15_ob_0_3ShortestPath" # "10_0_0_PRIMAL"
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
        "discount":[0.5], #[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["CURR_PPO"],
        "n_agents": [2],
        #"recurrent": [False],
        "base_policy_type": ["primal9"],
        "view_d": [4],
        "obj_density": [0.0],
        "env_size": [10],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.3], #[0.1],
        "finish_episode_r": [2.0],
        "block_r": [-1.0],
        "recurrent": [True],
        "blockheur": [True],
        "validact": [True],
        "noprevstate": [True]
    }

    # "step_r": [-0.1], #[-0.01, -0.1, -0.4],
    #     "obstacle_collision_r": [-0.4], #[-0.015],
    #     "agent_collision_r": [-0.4], #[-1.0], 
    #     "goal_reached_r": [0.3], #[0.1],
    #     "finish_episode_r": [2.0],
    #     "block_r": [-0.2],

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        id = "_No_EnvSampling_PrimalBCnoLoss1ownrownh2e-5_"
        name = "ppo" + id +"_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_viewd_" + str(param["view_d"]) \
            + "_minibatch_" + str(param["minibatch_size"]) \
            + "_rollouts_" + str(param["rollout_length"]) \
            + "_workers_" + str(param["workers"]) \
            + "_kepochs_" + str(param["k_epochs"]) \
            + "_recurrent_" + str(param["recurrent"]) \
            + "_block_" + str(param["blockheur"]) \
            + "_validact_" + str(param["validact"]) \
            + "_noprevstate_" + str(param["noprevstate"]) \
            + "_seed_"+ str(param["seed"])


            # + "_disc_" + str(param["discount"]) \
            # + "_lambda_" + str(param["lambda_"]) \
            # + "_entropy_" + str(param["entropy_coeff"]) \


            #+ "_envsize_" + str(param["env_size"]) \
            #+ "_nagents_"+ str(param["n_agents"]) \
            #+ "_objdensity_"+ str(param["obj_density"]) \

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])
        args.extend(["--block_r", str(param["block_r"])])

        args.extend(["--env_name","independent_navigation-v8_1"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--view_d", str(param["view_d"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(10))])
        args.extend(["--benchmark_frequency", str(int(500))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(10))])
        args.extend(["--checkpoint_frequency", str(int(200))])
        args.extend(["--ppo_bc_iteration_prob", str(float(0.5))])
       # args.extend(["--ppo_continue_from_checkpoint"])



        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(512)])
        args.extend(["--ppo_lr_a", str(0.00002)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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

        if param["recurrent"]:
            args.extend(["--ppo_recurrent"])
        if param["blockheur"]:
            args.extend(["--ppo_heur_block"])
        if param["validact"]:
            args.extend(["--ppo_heur_valid_act"])
        if param["noprevstate"]:
            args.extend(["--ppo_heur_no_prev_state"])
    
        main.main(args)



def ppo_10_10_0(): #PRIMAL with shortest path heuristic and Not-recurrent
    '''Partially observable with shortest path heur'''
    n_iterations = str(12000) #str(3000)
    experiment_group_name = "10_2_0"#"10_0_0_PRIMAL" #"test_validact"# #"test_NoLSTM_4Agnts_sz_15_ob_0_3ShortestPath" # "10_0_0_PRIMAL"
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
        "discount":[0.5],#[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["CURR_PPO"],
        "n_agents": [2],
        #"recurrent": [False],
        "base_policy_type": ["primal9"],
        "view_d": [4],
        "obj_density": [0.0],
        "env_size": [10],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.3], #[0.1],
        "finish_episode_r": [2.0],
        "block_r": [-1.0],
        "recurrent": [False],
        "blockheur": [True],
        "validact": [True],
        "noprevstate": [True]
    }

    # "step_r": [-0.1], #[-0.01, -0.1, -0.4],
    #     "obstacle_collision_r": [-0.4], #[-0.015],
    #     "agent_collision_r": [-0.4], #[-1.0], 
    #     "goal_reached_r": [0.3], #[0.1],
    #     "finish_episode_r": [2.0],
    #     "block_r": [-0.2],

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        id = "_10_10_0ShortPathBCnoLoss1ownrownh2e-5NotRecurrent_"
        name = "ppo" + id +"_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_minibatch_" + str(param["minibatch_size"]) \
            + "_rollouts_" + str(param["rollout_length"]) \
            + "_workers_" + str(param["workers"]) \
            + "_kepochs_" + str(param["k_epochs"]) \
            + "_recurrent_" + str(param["recurrent"]) \
            + "_block_" + str(param["blockheur"]) \
            + "_validact_" + str(param["validact"]) \
            + "_noprevstate_" + str(param["noprevstate"]) #\

            # + "_viewd_" + str(param["view_d"]) \
            # + "_disc_" + str(param["discount"]) \
            # + "_lambda_" + str(param["lambda_"]) \
            # + "_entropy_" + str(param["entropy_coeff"]) \


            #+ "_seed_"+ str(param["seed"])
            #+ "_envsize_" + str(param["env_size"]) \
            #+ "_nagents_"+ str(param["n_agents"]) \
            #+ "_objdensity_"+ str(param["obj_density"]) \

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])
        args.extend(["--block_r", str(param["block_r"])])

        args.extend(["--env_name","independent_navigation-v8_0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--view_d", str(param["view_d"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(10))])
        args.extend(["--benchmark_frequency", str(int(500))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(10))])
        args.extend(["--checkpoint_frequency", str(int(200))])
        args.extend(["--ppo_bc_iteration_prob", str(float(0.5))])
       # args.extend(["--ppo_continue_from_checkpoint"])



        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(512)])
        args.extend(["--ppo_lr_a", str(0.00002)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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

        if param["recurrent"]:
            args.extend(["--ppo_recurrent"])
        if param["blockheur"]:
            args.extend(["--ppo_heur_block"])
        if param["validact"]:
            args.extend(["--ppo_heur_valid_act"])
        if param["noprevstate"]:
            args.extend(["--ppo_heur_no_prev_state"])
    
        main.main(args)


def ppo_10_11_0(): #Dir vec without blocking trial 2
    '''Partially observable with shortest path heur'''
    n_iterations = str(12000) #str(3000)
    experiment_group_name = "10_3_0" # "10_0_0_PRIMAL" #"test_validact"# #"test_NoLSTM_4Agnts_sz_15_ob_0_3ShortestPath" # "10_0_0_PRIMAL"
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
        "discount":[0.7], #[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [0.95],
        "value_coeff": [0.5],
        "policy": ["CURR_PPO"],
        "n_agents": [2],
        #"recurrent": [False],
        "base_policy_type": ["primal9"],
        "view_d": [4],
        "obj_density": [0.0],
        "env_size": [10],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.5], #[0.1],
        "finish_episode_r": [0.0],
        "block_r": [-1.0],
        "recurrent": [True],
        "blockheur": [False],
        "validact": [True],
        "noprevstate": [True]
    }

    # "step_r": [-0.1], #[-0.01, -0.1, -0.4],
    #     "obstacle_collision_r": [-0.4], #[-0.015],
    #     "agent_collision_r": [-0.4], #[-1.0], 
    #     "goal_reached_r": [0.3], #[0.1],
    #     "finish_episode_r": [2.0],
    #     "block_r": [-0.2],

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        id = "_10_11_0NoBlk_nr3_BCnoLoss1ownrownh2e-5_"
        name = "ppo" + id +"_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_viewd_" + str(param["view_d"]) \
            + "_disc_" + str(param["discount"]) \
            + "_lambda_" + str(param["lambda_"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["minibatch_size"]) \
            + "_rollouts_" + str(param["rollout_length"]) \
            + "_workers_" + str(param["workers"]) \
            + "_kepochs_" + str(param["k_epochs"]) \
            + "_recurrent_" + str(param["recurrent"]) \
            + "_block_" + str(param["blockheur"]) \
            + "_validact_" + str(param["validact"]) \
            + "_noprevstate_" + str(param["noprevstate"]) \
            + "_seed_"+ str(param["seed"])
            #+ "_envsize_" + str(param["env_size"]) \
            #+ "_nagents_"+ str(param["n_agents"]) \
            #+ "_objdensity_"+ str(param["obj_density"]) \

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])
        args.extend(["--block_r", str(param["block_r"])])

        args.extend(["--env_name","independent_navigation-v8_1"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--view_d", str(param["view_d"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(10))])
        args.extend(["--benchmark_frequency", str(int(100))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(10))])
        args.extend(["--checkpoint_frequency", str(int(50))])
        args.extend(["--ppo_bc_iteration_prob", str(float(0.5))])
       # args.extend(["--ppo_continue_from_checkpoint"])



        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(512)])
        args.extend(["--ppo_lr_a", str(0.00002)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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

        if param["recurrent"]:
            args.extend(["--ppo_recurrent"])
        if param["blockheur"]:
            args.extend(["--ppo_heur_block"])
        if param["validact"]:
            args.extend(["--ppo_heur_valid_act"])
        if param["noprevstate"]:
            args.extend(["--ppo_heur_no_prev_state"])
    
        main.main(args)


def ppo_10_12_0(): #Shortest path no blocking
    '''Partially observable with shortest path heur'''
    n_iterations = str(12000) #str(3000)
    experiment_group_name = "10_4_0" # "10_0_0_PRIMAL" #"test_validact"# #"test_NoLSTM_4Agnts_sz_15_ob_0_3ShortestPath" # "10_0_0_PRIMAL"
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
        "discount":[0.5], #[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["CURR_PPO"],
        "n_agents": [2],
        #"recurrent": [False],
        "base_policy_type": ["primal9"],
        "view_d": [4],
        "obj_density": [0.0],
        "env_size": [10],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.3], #[0.1],
        "finish_episode_r": [2.0],
        "block_r": [-1.0],
        "recurrent": [True],
        "blockheur": [False],
        "validact": [True],
        "noprevstate": [True]
    }

    # "step_r": [-0.1], #[-0.01, -0.1, -0.4],
    #     "obstacle_collision_r": [-0.4], #[-0.015],
    #     "agent_collision_r": [-0.4], #[-1.0], 
    #     "goal_reached_r": [0.3], #[0.1],
    #     "finish_episode_r": [2.0],
    #     "block_r": [-0.2],

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        id = "_10_12_0NoBlk_ShortPathBCnoLoss1ownrownh2e-5_"
        name = "ppo" + id +"_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_viewd_" + str(param["view_d"]) \
            + "_minibatch_" + str(param["minibatch_size"]) \
            + "_rollouts_" + str(param["rollout_length"]) \
            + "_workers_" + str(param["workers"]) \
            + "_kepochs_" + str(param["k_epochs"]) \
            + "_recurrent_" + str(param["recurrent"]) \
            + "_block_" + str(param["blockheur"]) \
            + "_validact_" + str(param["validact"]) \
            + "_noprevstate_" + str(param["noprevstate"]) \
            + "_seed_"+ str(param["seed"])


            # + "_disc_" + str(param["discount"]) \
            # + "_lambda_" + str(param["lambda_"]) \
            # + "_entropy_" + str(param["entropy_coeff"]) \

            #+ "_envsize_" + str(param["env_size"]) \
            #+ "_nagents_"+ str(param["n_agents"]) \
            #+ "_objdensity_"+ str(param["obj_density"]) \

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])
        args.extend(["--block_r", str(param["block_r"])])

        args.extend(["--env_name","independent_navigation-v8_0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--view_d", str(param["view_d"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(10))])
        args.extend(["--benchmark_frequency", str(int(500))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(10))])
        args.extend(["--checkpoint_frequency", str(int(200))])
        args.extend(["--ppo_bc_iteration_prob", str(float(0.5))])
       # args.extend(["--ppo_continue_from_checkpoint"])

        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(512)])
        args.extend(["--ppo_lr_a", str(0.00002)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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

        if param["recurrent"]:
            args.extend(["--ppo_recurrent"])
        if param["blockheur"]:
            args.extend(["--ppo_heur_block"])
        if param["validact"]:
            args.extend(["--ppo_heur_valid_act"])
        if param["noprevstate"]:
            args.extend(["--ppo_heur_no_prev_state"])
    
        main.main(args)


def ppo_10_13_0(): #Shortest path no blocking + Non-primal architecture not recurrent
    '''Partially observable with shortest path heur'''
    n_iterations = str(12000) #str(3000)
    experiment_group_name = "10_4_0" # "10_0_0_PRIMAL" #"test_validact"# #"test_NoLSTM_4Agnts_sz_15_ob_0_3ShortestPath" # "10_0_0_PRIMAL"
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
        "discount":[0.5], #[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["CURR_PPO"],
        "n_agents": [2],
        #"recurrent": [False],
        "base_policy_type": ["primal9"],
        "view_d": [4],
        "obj_density": [0.0],
        "env_size": [10],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.3], #[0.1],
        "finish_episode_r": [2.0],
        "block_r": [-1.0],
        "recurrent": [False],
        "blockheur": [False],
        "validact": [True],
        "noprevstate": [True]
    }

    # "step_r": [-0.1], #[-0.01, -0.1, -0.4],
    #     "obstacle_collision_r": [-0.4], #[-0.015],
    #     "agent_collision_r": [-0.4], #[-1.0], 
    #     "goal_reached_r": [0.3], #[0.1],
    #     "finish_episode_r": [2.0],
    #     "block_r": [-0.2],

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        id = "_10_13_0NoPrimalNoBlk_ShortPathBCnoLoss1ownrownh2e-5_"
        name = "ppo" + id +"_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_viewd_" + str(param["view_d"]) \
            + "_minibatch_" + str(param["minibatch_size"]) \
            + "_rollouts_" + str(param["rollout_length"]) \
            + "_workers_" + str(param["workers"]) \
            + "_kepochs_" + str(param["k_epochs"]) \
            + "_recurrent_" + str(param["recurrent"]) \
            + "_block_" + str(param["blockheur"]) \
            + "_validact_" + str(param["validact"]) \
            + "_noprevstate_" + str(param["noprevstate"]) \
            + "_seed_"+ str(param["seed"])


            # + "_disc_" + str(param["discount"]) \
            # + "_lambda_" + str(param["lambda_"]) \
            # + "_entropy_" + str(param["entropy_coeff"]) \

            #+ "_envsize_" + str(param["env_size"]) \
            #+ "_nagents_"+ str(param["n_agents"]) \
            #+ "_objdensity_"+ str(param["obj_density"]) \

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])
        args.extend(["--block_r", str(param["block_r"])])

        args.extend(["--env_name","independent_navigation-v8_0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--view_d", str(param["view_d"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(10))])
        args.extend(["--benchmark_frequency", str(int(500))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(10))])
        args.extend(["--checkpoint_frequency", str(int(200))])
        args.extend(["--ppo_bc_iteration_prob", str(float(0.5))])
       # args.extend(["--ppo_continue_from_checkpoint"])

        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(512)])
        args.extend(["--ppo_lr_a", str(0.00002)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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

        if param["recurrent"]:
            args.extend(["--ppo_recurrent"])
        if param["blockheur"]:
            args.extend(["--ppo_heur_block"])
        if param["validact"]:
            args.extend(["--ppo_heur_valid_act"])
        if param["noprevstate"]:
            args.extend(["--ppo_heur_no_prev_state"])
    
        main.main(args)




def ppo_10_14_0(): #Dir vec IL only
    '''Partially observable with shortest path heur'''
    n_iterations = str(12000) #str(3000)
    experiment_group_name = "10_3_0" # "10_0_0_PRIMAL" #"test_validact"# #"test_NoLSTM_4Agnts_sz_15_ob_0_3ShortestPath" # "10_0_0_PRIMAL"
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
        "discount":[0.5], #[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["CURR_PPO"],
        "n_agents": [2],
        #"recurrent": [False],
        "base_policy_type": ["primal9"],
        "view_d": [4],
        "obj_density": [0.0],
        "env_size": [10],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.5], #[0.1],
        "finish_episode_r": [0.0],
        "block_r": [-1.0],
        "recurrent": [True],
        "blockheur": [True],
        "validact": [True],
        "noprevstate": [True]
    }

    # "step_r": [-0.1], #[-0.01, -0.1, -0.4],
    #     "obstacle_collision_r": [-0.4], #[-0.015],
    #     "agent_collision_r": [-0.4], #[-1.0], 
    #     "goal_reached_r": [0.3], #[0.1],
    #     "finish_episode_r": [2.0],
    #     "block_r": [-0.2],

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        id = "_10_14_0_IL_Only_BCnoLoss1ownrownh2e-5_"
        name = "ppo" + id +"_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_viewd_" + str(param["view_d"]) \
            + "_disc_" + str(param["discount"]) \
            + "_lambda_" + str(param["lambda_"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["minibatch_size"]) \
            + "_rollouts_" + str(param["rollout_length"]) \
            + "_workers_" + str(param["workers"]) \
            + "_kepochs_" + str(param["k_epochs"]) \
            + "_recurrent_" + str(param["recurrent"]) \
            + "_block_" + str(param["blockheur"]) \
            + "_validact_" + str(param["validact"]) \
            + "_noprevstate_" + str(param["noprevstate"]) \
            + "_seed_"+ str(param["seed"])
            #+ "_envsize_" + str(param["env_size"]) \
            #+ "_nagents_"+ str(param["n_agents"]) \
            #+ "_objdensity_"+ str(param["obj_density"]) \

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])
        args.extend(["--block_r", str(param["block_r"])])

        args.extend(["--env_name","independent_navigation-v8_1"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--view_d", str(param["view_d"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(10))])
        args.extend(["--benchmark_frequency", str(int(100))])
        args.extend(["--benchmark_num_episodes", str(int(100))])
        args.extend(["--benchmark_render_length", str(int(10))])
        args.extend(["--checkpoint_frequency", str(int(50))])
        args.extend(["--ppo_bc_iteration_prob", str(float(1.0))])
       # args.extend(["--ppo_continue_from_checkpoint"])



        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(512)])
        args.extend(["--ppo_lr_a", str(0.00002)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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

        if param["recurrent"]:
            args.extend(["--ppo_recurrent"])
        if param["blockheur"]:
            args.extend(["--ppo_heur_block"])
        if param["validact"]:
            args.extend(["--ppo_heur_valid_act"])
        if param["noprevstate"]:
            args.extend(["--ppo_heur_no_prev_state"])
    
        main.main(args)




def ppo_10_16_0(): #Dir vec
    '''Partially observable with shortest path heur'''
    n_iterations = str(12000) #str(3000)
    experiment_group_name = "10_3_0" # "10_0_0_PRIMAL" #"test_validact"# #"test_NoLSTM_4Agnts_sz_15_ob_0_3ShortestPath" # "10_0_0_PRIMAL"
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
        "discount":[0.5], #[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["CURR_PPO"],
        "n_agents": [2],
        #"recurrent": [False],
        "base_policy_type": ["primal9"],
        "view_d": [4],
        "obj_density": [0.0],
        "env_size": [10],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.5], #[0.1],
        "finish_episode_r": [0.0],
        "block_r": [-1.0],
        "recurrent": [True],
        "blockheur": [True],
        "validact": [True],
        "noprevstate": [True]
    }

    # "step_r": [-0.1], #[-0.01, -0.1, -0.4],
    #     "obstacle_collision_r": [-0.4], #[-0.015],
    #     "agent_collision_r": [-0.4], #[-1.0], 
    #     "goal_reached_r": [0.3], #[0.1],
    #     "finish_episode_r": [2.0],
    #     "block_r": [-0.2],

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        id = "_10_17_1_BC_LrDecayNadamBcOnly1ownrownh2e-5_"
        name = "ppo" + id +"_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_viewd_" + str(param["view_d"]) \
            + "_disc_" + str(param["discount"]) \
            + "_lambda_" + str(param["lambda_"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["minibatch_size"]) \
            + "_rollouts_" + str(param["rollout_length"]) \
            + "_workers_" + str(param["workers"]) \
            + "_kepochs_" + str(param["k_epochs"]) \
            + "_recurrent_" + str(param["recurrent"]) \
            + "_block_" + str(param["blockheur"]) \
            + "_validact_" + str(param["validact"]) # \
            #+ "_noprevstate_" + str(param["noprevstate"]) #\
           # + "_seed_"+ str(param["seed"])
            #+ "_envsize_" + str(param["env_size"]) \
            #+ "_nagents_"+ str(param["n_agents"]) \
            #+ "_objdensity_"+ str(param["obj_density"]) \

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])
        args.extend(["--block_r", str(param["block_r"])])

        args.extend(["--env_name","independent_navigation-v8_1"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--view_d", str(param["view_d"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(10))])
        args.extend(["--benchmark_frequency", str(int(100))])
        args.extend(["--benchmark_num_episodes", str(int(10))])
        args.extend(["--benchmark_render_length", str(int(10))])
        args.extend(["--checkpoint_frequency", str(int(30))])
        args.extend(["--ppo_bc_iteration_prob", str(float(1.0))])
       # args.extend(["--ppo_continue_from_checkpoint"])



        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(512)])
        args.extend(["--ppo_lr_a", str(0.00002)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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

        if param["recurrent"]:
            args.extend(["--ppo_recurrent"])
        if param["blockheur"]:
            args.extend(["--ppo_heur_block"])
        if param["validact"]:
            args.extend(["--ppo_heur_valid_act"])
        if param["noprevstate"]:
            args.extend(["--ppo_heur_no_prev_state"])
    
        main.main(args)



def ppo_10_18_0(): #Dir vec #Full Primal again
    '''Partially observable with shortest path heur'''
    n_iterations = str(12000) #str(3000)
    experiment_group_name = "10_6_0" # "10_0_0_PRIMAL" #"test_validact"# #"test_NoLSTM_4Agnts_sz_15_ob_0_3ShortestPath" # "10_0_0_PRIMAL"
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
        "discount":[0.5], #[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["CURR_PPO"],
        "n_agents": [2],
        #"recurrent": [False],
        "base_policy_type": ["primal9"],
        "view_d": [4],
        "obj_density": [0.0],
        "env_size": [10],
        "step_r": [-0.3], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.0], #[0.1],
        "finish_episode_r": [20.0],
        "block_r": [-2.0],
        "recurrent": [True],
        "blockheur": [True],
        "validact": [True],
        "noprevstate": [True]
    }

    # "step_r": [-0.1], #[-0.01, -0.1, -0.4],
    #     "obstacle_collision_r": [-0.4], #[-0.015],
    #     "agent_collision_r": [-0.4], #[-1.0], 
    #     "goal_reached_r": [0.3], #[0.1],
    #     "finish_episode_r": [2.0],
    #     "block_r": [-0.2],

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        id = "test_10_6_0_FullPrimal_BC_LrDecayOwnrownh2e-5_"
        name = "ppo" + id +"_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_viewd_" + str(param["view_d"]) \
            + "_disc_" + str(param["discount"]) \
            + "_lambda_" + str(param["lambda_"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["minibatch_size"]) \
            + "_rollouts_" + str(param["rollout_length"]) \
            + "_workers_" + str(param["workers"]) \
            + "_kepochs_" + str(param["k_epochs"]) \
            + "_recurrent_" + str(param["recurrent"]) \
            + "_block_" + str(param["blockheur"]) \
            + "_validact_" + str(param["validact"]) # \
            #+ "_noprevstate_" + str(param["noprevstate"]) #\
           # + "_seed_"+ str(param["seed"])
            #+ "_envsize_" + str(param["env_size"]) \
            #+ "_nagents_"+ str(param["n_agents"]) \
            #+ "_objdensity_"+ str(param["obj_density"]) \

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])
        args.extend(["--block_r", str(param["block_r"])])

        args.extend(["--env_name","independent_navigation-v8_1"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--view_d", str(param["view_d"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(20))])
        args.extend(["--benchmark_frequency", str(int(100))])
        args.extend(["--benchmark_num_episodes", str(int(10))])
        args.extend(["--benchmark_render_length", str(int(10))])
        args.extend(["--checkpoint_frequency", str(int(30))])
        args.extend(["--ppo_bc_iteration_prob", str(float(0.5))])
       # args.extend(["--ppo_continue_from_checkpoint"])



        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(512)])
        args.extend(["--ppo_lr_a", str(0.00002)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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

        if param["recurrent"]:
            args.extend(["--ppo_recurrent"])
        if param["blockheur"]:
            args.extend(["--ppo_heur_block"])
        if param["validact"]:
            args.extend(["--ppo_heur_valid_act"])
        if param["noprevstate"]:
            args.extend(["--ppo_heur_no_prev_state"])
    
        main.main(args)


def ppo_10_19_0(): #Dir vec
    '''Partially observable with shortest path heur'''
    n_iterations = str(12000) #str(3000)
    experiment_group_name = "10_6_0" # "10_0_0_PRIMAL" #"test_validact"# #"test_NoLSTM_4Agnts_sz_15_ob_0_3ShortestPath" # "10_0_0_PRIMAL"
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
        "discount":[0.5], #[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["CURR_PPO"],
        "n_agents": [2],
        #"recurrent": [False],
        "base_policy_type": ["primal9"],
        "view_d": [4],
        "obj_density": [0.0],
        "env_size": [10],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.5], #[0.1],
        "finish_episode_r": [2.0],
        "block_r": [-2.0],
        "recurrent": [True],
        "blockheur": [False],
        "validact": [True],
        "noprevstate": [True]
    }

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        id = "10_22_0_BC_LrDecayNoBlk1ownrownh2e-5_"
        name = "ppo" + id +"_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_viewd_" + str(param["view_d"]) \
            + "_disc_" + str(param["discount"]) \
            + "_lambda_" + str(param["lambda_"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["minibatch_size"]) \
            + "_rollouts_" + str(param["rollout_length"]) \
            + "_workers_" + str(param["workers"]) \
            + "_kepochs_" + str(param["k_epochs"]) \
            + "_recurrent_" + str(param["recurrent"]) \
            + "_block_" + str(param["blockheur"]) \
            + "_validact_" + str(param["validact"]) # \
            #+ "_noprevstate_" + str(param["noprevstate"]) #\
           # + "_seed_"+ str(param["seed"])
            #+ "_envsize_" + str(param["env_size"]) \
            #+ "_nagents_"+ str(param["n_agents"]) \
            #+ "_objdensity_"+ str(param["obj_density"]) \

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])
        args.extend(["--block_r", str(param["block_r"])])

        args.extend(["--env_name","independent_navigation-v8_1"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--view_d", str(param["view_d"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(20))])
        args.extend(["--benchmark_frequency", str(int(100))])
        args.extend(["--benchmark_num_episodes", str(int(10))])
        args.extend(["--benchmark_render_length", str(int(10))])
        args.extend(["--checkpoint_frequency", str(int(30))])
        args.extend(["--ppo_bc_iteration_prob", str(float(0.5))])
       # args.extend(["--ppo_continue_from_checkpoint"])



        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(512)])
        args.extend(["--ppo_lr_a", str(0.00002)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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

        if param["recurrent"]:
            args.extend(["--ppo_recurrent"])
        if param["blockheur"]:
            args.extend(["--ppo_heur_block"])
        if param["validact"]:
            args.extend(["--ppo_heur_valid_act"])
        if param["noprevstate"]:
            args.extend(["--ppo_heur_no_prev_state"])
    
        main.main(args)





def ppo_10_20_0(): #Shortest path, no recurrent, FF NN
    '''Partially observable with shortest path heur'''
    n_iterations = str(12000) #str(3000)
    experiment_group_name = "10_6_0" # "10_0_0_PRIMAL" #"test_validact"# #"test_NoLSTM_4Agnts_sz_15_ob_0_3ShortestPath" # "10_0_0_PRIMAL"
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
        "discount":[0.5], #[0.4, 0.5, 0.7, 0.9, 1.0],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["CURR_PPO"],
        "n_agents": [2],
        #"recurrent": [False],
        "base_policy_type": ["primal9"],
        "view_d": [4],
        "obj_density": [0.0],
        "env_size": [10],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.5], #[0.1],
        "finish_episode_r": [2.0],
        "block_r": [-2.0],
        "recurrent": [False],
        "blockheur": [False],
        "validact": [False],
        "noprevstate": [False]
    }

    # "step_r": [-0.1], #[-0.01, -0.1, -0.4],
    #     "obstacle_collision_r": [-0.4], #[-0.015],
    #     "agent_collision_r": [-0.4], #[-1.0], 
    #     "goal_reached_r": [0.3], #[0.1],
    #     "finish_episode_r": [2.0],
    #     "block_r": [-0.2],

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        id = "11_0_0_BC_NotPRIMAL_ShortPath_1ownrownh2e-5_"
        name = "ppo" + id +"_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_viewd_" + str(param["view_d"]) \
            + "_disc_" + str(param["discount"]) \
            + "_lambda_" + str(param["lambda_"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["minibatch_size"]) \
            + "_rollouts_" + str(param["rollout_length"]) \
            + "_workers_" + str(param["workers"]) \
            + "_kepochs_" + str(param["k_epochs"]) \
            + "_recurrent_" + str(param["recurrent"]) \
            + "_block_" + str(param["blockheur"]) \
            + "_validact_" + str(param["validact"]) # \
            #+ "_noprevstate_" + str(param["noprevstate"]) #\
           # + "_seed_"+ str(param["seed"])
            #+ "_envsize_" + str(param["env_size"]) \
            #+ "_nagents_"+ str(param["n_agents"]) \
            #+ "_objdensity_"+ str(param["obj_density"]) \

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])
        args.extend(["--block_r", str(param["block_r"])])

        args.extend(["--env_name","independent_navigation-v8_0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--view_d", str(param["view_d"])])
        args.extend(["--obj_density", str(param["obj_density"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(20))])
        args.extend(["--benchmark_frequency", str(int(100))])
        args.extend(["--benchmark_num_episodes", str(int(10))])
        args.extend(["--benchmark_render_length", str(int(10))])
        args.extend(["--checkpoint_frequency", str(int(30))])
        args.extend(["--ppo_bc_iteration_prob", str(float(0.5))])
       # args.extend(["--ppo_continue_from_checkpoint"])



        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(512)])
        args.extend(["--ppo_lr_a", str(0.00002)])
        args.extend(["--ppo_lr_v", str(0.00002)])
        args.extend(["--ppo_base_policy_type", param['base_policy_type']])
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

        if param["recurrent"]:
            args.extend(["--ppo_recurrent"])
        if param["blockheur"]:
            args.extend(["--ppo_heur_block"])
        if param["validact"]:
            args.extend(["--ppo_heur_valid_act"])
        if param["noprevstate"]:
            args.extend(["--ppo_heur_no_prev_state"])
    
        main.main(args)






