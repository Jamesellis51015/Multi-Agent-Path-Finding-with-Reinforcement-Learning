import main
from sklearn.model_selection import ParameterGrid

def test_ic3():
    n_iterations = str(3000)
    #experiment_group_name = "TEST4_IC3"
    experiment_group_name = "testIc3"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"

 
    parmeter_grid1 = {
        "seed": [1],
        "batch_sizes": [500],
        "entropy_coeff":[0.0],
        "discount":[0.9],
        #"lambda_": [1.0, 0.95,0.6],
        "value_ceoff": [0.01],
        "policy": ["IC3"],
        "n_agents": [4],
        "obj_density": [0.2],
        "env_size": [7],
        "recurrent": [False],
        "base_policy_type": ["mlp"],
        "comm_passes": [1],
        "comm_mask_zero":[False],
        "hid_size":[120],
        "share_weights":[False]
    }


    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
        args.extend(["--iterations", n_iterations])

    #     parser.add_argument("--hid_size", default= 120, type =int)
    # parser.add_argument("--recurrent", default= False, action = 'store_true')
    # parser.add_argument("--detach_gap", default = 10, type = int)
    # parser.add_argument("--comm_passes", default= 1, type =int)
    # parser.add_argument('--share_weights', default=False, action='store_true', 
    #                 help='Share weights for hops')
    # #parser.add_argument("--comm_mode", default = 1, type = int,
    # #help= "if mode == 0 -- no communication; mode==1--ic3net communication; mode ==2 -- commNet communication")
    # parser.add_argument("--comm_mode", default = "avg", type = str, help="Average or sum the hidden states to obtain the comm vector")
    # parser.add_argument("--hard_attn", default=True, type=bool, help="to communicate or not. If hard_attn == False, no comm")
    # parser.add_argument("--comm_mask_zero", default=False,action='store_true', help="to communicate or not. If hard_attn == False, no comm")
    # parser.add_argument('--comm_action_one', default=False, action='store_true',
    #                 help='Whether to always talk, sanity check for hard attention.')
    # parser.add_argument("--ic3_base_policy_type", default= "mlp", type =str)


        
        name = "IC3" + "_disc_" + str(param["discount"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["batch_sizes"]) \
            + "_commpasses_" + str(param["comm_passes"]) \
            + "_commzero_" + str(param["comm_mask_zero"]) \
            + "_recurrent_" + str(param["recurrent"]) \
            + "_envsize_" + str(param["env_size"]) \
            + "_nagents_"+ str(param["n_agents"]) \
            + "_objdensity_"+ str(param["obj_density"]) \
            + "_seed_"+ str(param["seed"])

        args.extend(["--env_name","independent_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--policy", str(param["policy"])])
        args.extend(["--name", name])
        args.extend(["--batch_size", str(param["batch_sizes"])])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--entropy_coeff", str(param["entropy_coeff"])])
        args.extend(["--discount", str(param["discount"])])
       # args.extend(["--lambda_", str(param["lambda_"])])
        args.extend(["--ic3_base_policy_type", str(param["base_policy_type"])])
        args.extend(["--comm_passes", str(param["comm_passes"])])
        args.extend(["--hid_size", str(param["hid_size"])])

        if param["share_weights"]:
            args.extend(["--share_weights"])
        if param["comm_mask_zero"]:
            args.extend(["--comm_mask_zero"])
        if param["recurrent"]:
            args.extend(["--recurrent"])

            

        args.extend(["--render_rate", str(int(5e2))])
        args.extend(["--checkpoint_frequency", str(int(2e3))])
        args.extend(["--benchmark_frequency", str(int(1000))])
        args.extend(["--benchmark_num_episodes", str(int(500))])
        args.extend(["--benchmark_render_length", str(int(20))])
        args.extend(["--obj_density", str(param["obj_density"])])

        main.main(args)


# def test_ic3():
#     n_iterations = str(5) #str(3000)
#     experiment_group_name = "TEST4_IC3"
#     work_dir = experiment_group_name
#     plot_dir = experiment_group_name + "_Central"

 
#     parmeter_grid1 = {
#         "seed": [2],
#         "batch_sizes": [100],
#         "entropy_coeff":[0.0],
#         "discount":[1.0],
#         #"lambda_": [1.0, 0.95,0.6],
#         "value_ceoff": [0.01],
#         "policy": ["IC3"],
#         "n_agents": [4],
#         "recurrent": [False],
#         "base_policy_type": ["cnn_old"],
#         "comm_passes": [1]
#     }


#     grid1 = ParameterGrid(parmeter_grid1)
    
#     for param in grid1:
#         args = []
#         args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
#         args.extend(["--iterations", n_iterations])
#        # args.extend(["--n_workers", str(workers)])
#        # args.extend(["--map_shape", tuple((5,5))])

#         # parser.add_argument("--hidden_size", default= 128, type =int)
#         # parser.add_argument("--recurrent", default= True, type =bool)
#         # parser.add_argument("--detach_gap", default = 10, type = int)
#         # parser.add_argument("--comm_passes", default= 1, type =int)
#         # #parser.add_argument("--comm_mode", default = 1, type = int,
#         # #help= "if mode == 0 -- no communication; mode==1--ic3net communication; mode ==2 -- commNet communication")
#         # parser.add_argument("--comm_mode", default = "avg", type = str, help="Average or sum the hidden states to obtain the comm vector")
#         # parser.add_argument("--hard_attn", default=True, type=bool, help="to communicate or not. If hard_attn == False, no comm")
#         # parser.add_argument("--comm_mask_zero", default=False, type=bool, help="to communicate or not. If hard_attn == False, no comm")

#         name = "disc" + str(param["discount"]) \
#          + "_bat" + str(param["batch_sizes"]) + \
#             "_val_coeff_" + str(param["value_ceoff"]) \
#             + "_agnts" + str(param["n_agents"]) + "_" + str(param["policy"]) + "_sd" + str(param["seed"])
#         #"_ent" + str(param["entropy_coeff"]) \
        

#         name.replace('.', "_")

#         args.extend(["--n_agents", str(param["n_agents"])])
#         args.extend(["--policy", str(param["policy"])])
#         args.extend(["--name", name])
#         args.extend(["--batch_size", str(param["batch_sizes"])])
#         args.extend(["--seed", str(param["seed"])])
#         args.extend(["--entropy_coeff", str(param["entropy_coeff"])])
#        # args.extend(["--c1", str(param["value_coeff"])])
#         args.extend(["--discount", str(param["discount"])])
#        # args.extend(["--lambda_", str(param["lambda_"])])
#         args.extend(["--ic3_base_policy_type", str(param["base_policy_type"])])
#         args.extend(["--benchmark_frequency", str(int(3))])
#         args.extend(["--benchmark_num_episodes", str(int(50))])
#         args.extend(["--benchmark_render_length", str(int(25))])
    
#         main.main(args)
