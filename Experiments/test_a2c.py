import main
from sklearn.model_selection import ParameterGrid

def test_a2c():
    n_iterations = str(7000) #str(3000)
    experiment_group_name = "TEST_a2c"
    work_dir = experiment_group_name
    plot_dir = experiment_group_name + "_Central"

 
    parmeter_grid1 = {
        "seed": [2],
        "batch_sizes": [128],
        "entropy_coeff":[0.0],
        "discount":[0.99],
        #"lambda_": [1.0, 0.95,0.6],
        "value_coeff": [0.5],
        "policy": ["A2C"],
        "n_agents": [1],
        #"recurrent": [False],
        "base_policy_type": ["mlp"],
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

        name = "disc" + str(param["discount"]) \
         + "_bat" + str(param["batch_sizes"]) + \
            "_val_coeff_" + str(param["value_coeff"]) \
            + "_ent" + str(param["entropy_coeff"]) \
            + "_agnts" + str(param["n_agents"]) + "_" + str(param["policy"]) + "_sd" + str(param["seed"])
        #"_ent" + str(param["entropy_coeff"]) \
        

        name.replace('.', "_")

        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--policy", str(param["policy"])])
        args.extend(["--name", name])
        args.extend(["--a2c_batch_size", str(param["batch_sizes"])])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--a2c_entropy_coeff", str(param["entropy_coeff"])])
        args.extend(["--a2c_value_coeff", str(param["value_coeff"])])
        args.extend(["--a2c_discount", str(param["discount"])])
       # args.extend(["--lambda_", str(param["lambda_"])])
        args.extend(["--a2c_base_policy_type", str(param["base_policy_type"])])
        args.extend(["--benchmark_frequency", str(int(1000))])
        args.extend(["--benchmark_num_episodes", str(int(50))])
        args.extend(["--benchmark_render_length", str(int(25))])
        #args.extend(["--a2c_use_gpu"])
    
        main.main(args)