import main
from sklearn.model_selection import ParameterGrid


def test_maac_new():
    episodes = str(1000)
    experiment_group_name = "TestBases"
    work_dir = experiment_group_name
    plot_dir = experiment_group_name + "_Central_TestBases"

 
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
        "base_policy_type": ["mlp"],
        "n_agents": [2]
    }

#     parser.add_argument("--maac_buffer_length", default=int(1e6), type=int)
#     parser.add_argument("--maac_n_episodes", default=50000, type=int)
#     parser.add_argument("--maac_n_rollout_threads", default=6, type=int)
#    # parser.add_argument("--maac_episode_length", default=25, type=int)
#     parser.add_argument("--maac_steps_per_update", default=100, type=int)
#     parser.add_argument("--maac_num_updates", default=4, type=int,
#                         help="Number of updates per update cycle")
#     parser.add_argument("--maac_batch_size",
#                         default=1024, type=int,
#                         help="Batch size for training")
#    # parser.add_argument("--maac_save_interval", default=1000, type=int)
#     parser.add_argument("--maac_pol_hidden_dim", default=128, type=int)
#     parser.add_argument("--maac_critic_hidden_dim", default=128, type=int)
#     parser.add_argument("--maac_attend_heads", default=4, type=int)
#     parser.add_argument("--maac_pi_lr", default=0.001, type=float)
#     parser.add_argument("--maac_q_lr", default=0.001, type=float)
#     parser.add_argument("--maac_tau", default=0.001, type=float)
#     parser.add_argument("--maac_gamma", default=0.99, type=float)
#     parser.add_argument("--maac_reward_scale", default=100., type=float)
#     parser.add_argument("--maac_use_gpu", action='store_true')

    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir,"--alternative_plot_dir", plot_dir]
       # args.extend(["--iterations", n_iterations])
       # args.extend(["--n_workers", str(workers)])
       # args.extend(["--map_shape", tuple((5,5))])

        # name = "disc" + str(param["discount"]) \
        #  + "_bat" + str(param["batch_sizes"]) + \
        # "_ent" + str(param["entropy_coeff"]) \
        #     + "_agnts" + str(param["n_agents"]) + "_" + str(param["policy"]) + "_sd" + str(param["seed"])

        name = "reward_scale" + str(param["reward_scale"]) \
                + "_agnts" + str(param["n_agents"]) + "enc_type_" + param["base_policy_type"]
      #  name.replace('.', "_")

        args.extend(["--maac_buffer_length", str(int(1e2))])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--policy", str(param["policy"])])
        args.extend(["--name", name])
       # args.extend(["--maac_use_gpu"])
        args.extend(["--maac_share_actor"])
        args.extend(["--maac_reward_scale", str(param["reward_scale"])])
        args.extend(["--maac_n_rollout_threads", str(param["rollout_threads"])])
        args.extend(["--maac_n_episodes", episodes])
        args.extend(["--maac_batch_size", str(80)])
        args.extend(["--env_name", "cooperative_navigation-v0"])
        args.extend(["--maac_base_policy_type", param["base_policy_type"]]) 
        args.extend(["--benchmark_frequency", str(int(1000))])
        args.extend(["--benchmark_num_episodes", str(int(50))])
        args.extend(["--benchmark_render_length", str(int(25))])
        #args.extend(["--batch_size", str(param["batch_sizes"])])
        #args.extend(["--seed", str(param["seed"])])
        #args.extend(["--c2", str(param["entropy_coeff"])])
       # args.extend(["--c1", str(param["value_coeff"])])
        #args.extend(["--discount", str(param["discount"])])
       # args.extend(["--lambda_", str(param["lambda_"])])
    
        main.main(args)