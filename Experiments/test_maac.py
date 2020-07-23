import main
from sklearn.model_selection import ParameterGrid


def test_maac_new():
    episodes = str(10000)
    # experiment_group_name = "TestMaac"
    # work_dir = experiment_group_name
    # plot_dir = experiment_group_name + "_Central_TestBases"

    experiment_group_name = "testMaac2"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"

 
    parmeter_grid1 = {
        "seed": [1],
        #"batch_sizes": [500,1000,2000],
       # "entropy_coeff":[0.0],
        "discount":[0.5],
        "rollout_threads": [1],
        "reward_scale": [100],
        "policy": ["MAAC"],
        "base_policy_type": ["mlp"],
        "n_agents": [4],
        "obj_density": [0.0],
        "env_size": [5],
        ###
        "maac_steps_per_update":[100],
        "maac_num_updates":[4],
        "maac_attend_heads":[4],
        "maac_batch_size": [1024]
    }
#  ################################################################################################
#     #           MAAC parampeters from : https://github.com/shariqiqbal2810/MAAC
#     #
#     parser.add_argument("--maac_buffer_length", default=int(1e6), type=int)
#     parser.add_argument("--maac_n_episodes", default=50000, type=int)
#     parser.add_argument("--maac_n_rollout_threads", default=6, type=int)
#     parser.add_argument("--maac_steps_per_update", default=100, type=int)
#     parser.add_argument("--maac_num_updates", default=4, type=int,
#                         help="Number of updates per update cycle")
#     parser.add_argument("--maac_batch_size",
#                         default=1024, type=int, 
#                         help="Batch size for training")
#     parser.add_argument("--maac_pol_hidden_dim", default=128, type=int)
#     parser.add_argument("--maac_critic_hidden_dim", default=128, type=int)
#     parser.add_argument("--maac_attend_heads", default=4, type=int)
#     parser.add_argument("--maac_pi_lr", default=0.001, type=float)
#     parser.add_argument("--maac_q_lr", default=0.001, type=float)
#     parser.add_argument("--maac_tau", default=0.001, type=float)
#     parser.add_argument("--maac_gamma", default=0.99, type=float)
#     parser.add_argument("--maac_reward_scale", default=100., type=float)
#     parser.add_argument("--maac_use_gpu", action='store_true')
#     parser.add_argument("--maac_share_actor", action='store_true')
#     parser.add_argument("--maac_base_policy_type", default = 'cnn_old') #Types: mlp, cnn_old, cnn_new

    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir,"--alternative_plot_dir", plot_dir]

        name = "MAAC" + "_disc_" + str(param["discount"]) \
            + "_rewardscale_" + str(param["reward_scale"]) \
            + "_minibatch_" + str(param["maac_batch_size"]) \
            + "_nupdates_" + str(param["maac_steps_per_update"]) \
            + "_attheads_" + str(param["maac_attend_heads"]) \
            + "_envsize_" + str(param["env_size"]) \
            + "_nagents_"+ str(param["n_agents"]) \
            + "_objdensity_"+ str(param["obj_density"]) \
            + "_seed_"+ str(param["seed"])


        args.extend(["--env_name","independent_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--obj_density", str(param["obj_density"])])

        args.extend(["--maac_buffer_length", str(int(1e5))])
        args.extend(["--policy", str(param["policy"])])
        args.extend(["--name", name])
        args.extend(["--maac_use_gpu"])
        args.extend(["--maac_share_actor"])
        args.extend(["--maac_reward_scale", str(param["reward_scale"])])
        args.extend(["--maac_n_rollout_threads", str(param["rollout_threads"])])
        args.extend(["--maac_steps_per_update", str(param["maac_steps_per_update"])])
        args.extend(["--maac_num_updates", str(param["maac_num_updates"])])
        args.extend(["--maac_attend_heads", str(param["maac_attend_heads"])])
        args.extend(["--maac_n_episodes", episodes])
        args.extend(["--maac_batch_size", str(param["maac_batch_size"])])
        args.extend(["--maac_base_policy_type", param["base_policy_type"]]) 
        args.extend(["--benchmark_frequency", str(int(1000))])
        args.extend(["--benchmark_num_episodes", str(int(50))])
        args.extend(["--benchmark_render_length", str(int(25))])
        args.extend(["--render_rate", str(int(50))])

        #args.extend(["--batch_size", str(param["batch_sizes"])])
        args.extend(["--seed", str(param["seed"])])
        #args.extend(["--c2", str(param["entropy_coeff"])])
       # args.extend(["--c1", str(param["value_coeff"])])
        #args.extend(["--discount", str(param["discount"])])
       # args.extend(["--lambda_", str(param["lambda_"])])
    
        main.main(args)






# def run_maac():
#     #experiment_group_name, n_agents, env_name, env_size, obj_density
#     episodes = str(50000)
#     #experiment_group_name = exp_group_name #"TestBases"
#     work_dir = experiment_group_name
#     plot_dir = experiment_group_name + "_Central" #"_Central_TestBases"

 
#     parmeter_grid1 = {
#         #"seed": [2],
#         #"batch_sizes": [500,1000,2000],
#        # "entropy_coeff":[0.0],
#        # "discount":[1.0],
#         #"lambda_": [1.0, 0.95,0.6],
#         #"value_ceoff": [0.01, 0.1, 0.5, 1.0],
#         "rollout_threads": [1],
#         "reward_scale": [100],
#         "policy": ["MAAC"],
#         "base_policy_type": ["mlp"], #["cnn_old","cnn_new"],
#         "n_agents": [n_agents]
#     }

#     grid1 = ParameterGrid(parmeter_grid1)
    
#     for param in grid1:
#         args = []
#         args = ["--working_directory", work_dir,"--alternative_plot_dir", plot_dir]

#         if "cooperative_navigation" in env_name:
#             en = "cn"
#         else:
#             en = "unk"
#         v = env_name[-2:]

#         name = en + v + "_"+ "obj_d"+ str(obj_density)+ "_envsz" + str(env_size) +  param["policy"] + "_enc_type_" + \
#             param["base_policy_type"] + "_reward_scale" + str(param["reward_scale"]) \
#             + "_agnts" + str(param["n_agents"])
#         # name = "reward_scale" + str(param["reward_scale"]) \
#         #         + "_agnts" + str(param["n_agents"]) + "enc_type_" + param["base_policy_type"]
#       #  name.replace('.', "_")

#         args.extend(["--maac_buffer_length", str(int(5e5))])
#         args.extend(["--map_shape", str(env_size)])
#         args.extend(["--n_agents", str(param["n_agents"])])
#         args.extend(["--policy", str(param["policy"])])
#         args.extend(["--name", name])
#         args.extend(["--maac_use_gpu"])
#         args.extend(["--maac_share_actor"])
#         args.extend(["--maac_reward_scale", str(param["reward_scale"])])
#         args.extend(["--maac_n_rollout_threads", str(param["rollout_threads"])])
#         args.extend(["--maac_n_episodes", episodes])
#         args.extend(["--env_name", env_name])#"cooperative_navigation-v0"])
#         args.extend(["--maac_base_policy_type", param["base_policy_type"]]) 

#         args.extend(["--render_rate", str(int(3e2))])
#         args.extend(["--checkpoint_frequency", str(int(10e3))])
#         args.extend(["--obj_density", str(obj_density)])
#         args.extend(["--benchmark_frequency", str(int(1000))])
#         args.extend(["--benchmark_num_episodes", str(int(500))])
#         args.extend(["--benchmark_render_length", str(int(100))])
#         #args.extend(["--batch_size", str(param["batch_sizes"])])
#         #args.extend(["--seed", str(param["seed"])])
#         #args.extend(["--c2", str(param["entropy_coeff"])])
#        # args.extend(["--c1", str(param["value_coeff"])])
#         #args.extend(["--discount", str(param["discount"])])
#        # args.extend(["--lambda_", str(param["lambda_"])])
    
#         main.main(args)