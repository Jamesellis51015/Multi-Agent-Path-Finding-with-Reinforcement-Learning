import main
from sklearn.model_selection import ParameterGrid

def exp_ppo_curr():
    n_iterations = str(1000) #str(3000)
    experiment_group_name = "TEST_ppo_curr"
    work_dir = experiment_group_name
    plot_dir = experiment_group_name + "_Central"
 
    parmeter_grid1 = {
        "seed": [3],
        "workers": [2],
        "rollout_length": [100],
        "eps_clip": [0.2],
        "k_epochs": [1],
        "minibatch_size": [50],
        "entropy_coeff":[0.01],
        "discount":[0.99],
        "lambda_": [1.0],
        "value_coeff": [0.5],
        "policy": ["CURR_PPO"],
        "n_agents": [1],
        #"recurrent": [False],
        "base_policy_type": ["mlp"],
    }


    grid1 = ParameterGrid(parmeter_grid1)

    
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
        #args.extend(["--iterations", n_iterations])
      
        name = "disc" + str(param["discount"]) \
            + "_lambda" + str(param["lambda_"]) \
         + "_workers" + str(param["workers"]) + \
             "_nrollouts" + str(param["rollout_length"]) + \
             "_eps_clip" + str(param["eps_clip"]) + \
            "_val_coeff_" + str(param["value_coeff"]) \
            + "_ent" + str(param["entropy_coeff"]) \
            + "_agnts" + str(param["n_agents"]) + "_" + str(param["policy"])
        #"_ent" + str(param["entropy_coeff"]) \
        

        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(10))])
        args.extend(["--benchmark_frequency", str(int(10))])
        args.extend(["--benchmark_num_episodes", str(int(50))])
        args.extend(["--benchmark_render_length", str(int(25))])

        args.extend(["--checkpoint_frequency", str(10)])

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
      #  args.extend(["--ppo_continue_from_checkpoint"])
    
        main.main(args)

def exp_ppo_curr2():
    n_iterations = str(1000) #str(3000)
    experiment_group_name = "5_ppo"
    #work_dir = experiment_group_name
    #plot_dir = experiment_group_name + "_Central"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"
 
 
    # parmeter_grid1 = {
    #     "seed": [1],
    #     "workers": [2],
    #     "rollout_length": [100],
    #     "eps_clip": [0.2],
    #     "k_epochs": [1],
    #     "minibatch_size": [50],
    #     "entropy_coeff":[0.01],
    #     "discount":[0.99],
    #     "lambda_": [1.0],
    #     "value_coeff": [0.5],
    #     "policy": ["CURR_PPO"],
    #     "n_agents": [1],
    #     #"recurrent": [False],
    #     "base_policy_type": ["mlp"],
    # }

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
        "base_policy_type": ["primal6"],
        "obj_density": [0.0],
        "env_size": [5],
       #"env": ["ind_navigation_custom-v0"], #env ind from 0 to 6
       # "custom_env_ind": [0,1,2,3,4,5,6]
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
        #args.extend(["--iterations", n_iterations])
      
        name = "CL5x5+Custom_disc" + str(param["discount"]) \
            + "_lambda" + str(param["lambda_"]) \
         + "_workers" + str(param["workers"]) + \
             "_nrollouts" + str(param["rollout_length"]) + \
             "_eps_clip" + str(param["eps_clip"]) + \
            "_val_coeff_" + str(param["value_coeff"]) \
            + "_ent" + str(param["entropy_coeff"]) \
            + "_" + str(param["policy"])
        #"_ent" + str(param["entropy_coeff"]) \
        

        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--policy", param["policy"]])
        args.extend(["--name", name])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--render_rate", str(int(20))])
        args.extend(["--benchmark_frequency", str(int(250))])
        args.extend(["--benchmark_num_episodes", str(int(50))])
        args.extend(["--benchmark_render_length", str(int(25))])

        args.extend(["--checkpoint_frequency", str(100)])

        #           PPO Paramares:
        args.extend(["--ppo_hidden_dim", str(120)])
        args.extend(["--ppo_lr_a", str(0.001)])
        args.extend(["--ppo_lr_v", str(0.001)])
        args.extend(["--ppo_base_policy_type", param["base_policy_type"]])
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
      #  args.extend(["--ppo_continue_from_checkpoint"])
    
        main.main(args)