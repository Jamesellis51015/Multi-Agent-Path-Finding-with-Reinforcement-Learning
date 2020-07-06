import main
from sklearn.model_selection import ParameterGrid
import manual_test


def T0_1_p1():
    n_iterations = str(2500)
    experiment_group_name = "T0_1"
    work_dir = "/home/james/Desktop/Gridworld/EXPERIMENTS/" + experiment_group_name
    plot_dir = "/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/" + experiment_group_name

 
    parmeter_grid1 = {
        "seed": [2],
        "batch_sizes": [250,500, 1000],
        "entropy_coeff":[0.0, 0.001],
        "discount":[1.0],
        "lambda_": [1.0],
        "value_ceoff": [0.01, 0.1, 0.5, 1.0],
        "policy": ["AC_Shared"],
        "n_agents": [1]
    }


    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
        args.extend(["--iterations", n_iterations])
       # args.extend(["--n_workers", str(workers)])
        args.extend(["--map_shape", tuple((5,5))])

        name = "disc" + str(param["discount"]) \
         + "_bat" + str(param["batch_sizes"]) + \
        "_ent" + str(param["entropy_coeff"]) + "_val" + str(param["value_ceoff"]) \
            + "_agnts" + str(param["n_agents"]) + "_" + str(param["policy"]) + "_sd" + str(param["seed"])

        name.replace('.', "_")

        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--policy", str(param["policy"])])
        args.extend(["--name", name])
        args.extend(["--batch_size", str(param["batch_sizes"])])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--c2", str(param["entropy_coeff"])])
        args.extend(["--c1", str(param["value_coeff"])])
        args.extend(["--discount", str(param["discount"])])
    
        main.main(args)



def T0_1_p2():
    n_iterations = str(2500)
    experiment_group_name = "T0_1"
    work_dir = "/home/james/Desktop/Gridworld/EXPERIMENTS/" + experiment_group_name
    plot_dir = "/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/" + experiment_group_name

 
    parmeter_grid1 = {
        "seed": [2],
        "batch_sizes": [250,500, 1000],
        "entropy_coeff":[0.0, 0.001],
        "discount":[1.0],
        "lambda_": [1.0],
        "value_ceoff": [0.01, 0.1, 0.5, 1.0],
        "policy": ["PG_Shared"],
        "n_agents": [1]
    }


    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
        args.extend(["--iterations", n_iterations])
       # args.extend(["--n_workers", str(workers)])
        args.extend(["--map_shape", tuple((5,5))])

        name = "disc" + str(param["discount"]) \
         + "_bat" + str(param["batch_sizes"]) + \
        "_ent" + str(param["entropy_coeff"]) + "_val" + str(param["value_ceoff"]) \
            + "_agnts" + str(param["n_agents"]) + "_" + str(param["policy"]) + "_sd" + str(param["seed"])

        name.replace('.', "_")

        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--policy", str(param["policy"])])
        args.extend(["--name", name])
        args.extend(["--batch_size", str(param["batch_sizes"])])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--c2", str(param["entropy_coeff"])])
        args.extend(["--c1", str(param["value_coeff"])])
        args.extend(["--discount", str(param["discount"])])
    
        main.main(args)

def T0_1_p3():
    n_iterations = str(2500)
    experiment_group_name = "T0_1"
    work_dir = "/home/james/Desktop/Gridworld/EXPERIMENTS/" + experiment_group_name
    plot_dir = "/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/" + experiment_group_name

 
    parmeter_grid1 = {
        "seed": [2],
        "batch_sizes": [250,500, 1000],
        "entropy_coeff":[0.0, 0.001],
        "discount":[1.0],
        "lambda_": [1.0],
        #"value_ceoff": [0.01, 0.1, 0.5, 1.0],
        "policy": ["AC_Separate", "PG_Separate"],
        "n_agents": [1]
    }


    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
        args.extend(["--iterations", n_iterations])
       # args.extend(["--n_workers", str(workers)])
        args.extend(["--map_shape", tuple((5,5))])

        name = "disc" + str(param["discount"]) \
         + "_bat" + str(param["batch_sizes"]) + \
        "_ent" + str(param["entropy_coeff"]) \
            + "_agnts" + str(param["n_agents"]) + "_" + str(param["policy"]) + "_sd" + str(param["seed"])

        name.replace('.', "_")

        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--policy", str(param["policy"])])
        args.extend(["--name", name])
        args.extend(["--batch_size", str(param["batch_sizes"])])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--c2", str(param["entropy_coeff"])])
        #args.extend(["--c1", str(param["value_coeff"])])
        args.extend(["--discount", str(param["discount"])])
        
    
        main.main(args)



def T0_2_p1():
    n_iterations = str(2500)
    experiment_group_name = "T0_2"
    work_dir = "/home/james/Desktop/Gridworld/EXPERIMENTS/" + experiment_group_name
    plot_dir = "/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/" + experiment_group_name

 
    parmeter_grid1 = {
        "seed": [2],
        "batch_sizes": [500,1000,2000],
        "entropy_coeff":[0.0],
        "discount":[1.0, 0.95, 0.6],
        "lambda_": [1.0, 0.95,0.6],
        #"value_ceoff": [0.01, 0.1, 0.5, 1.0],
        "policy": ["AC_Separate"],
        "n_agents": [1,2,3,4]
    }


    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
        args.extend(["--iterations", n_iterations])
       # args.extend(["--n_workers", str(workers)])
       # args.extend(["--map_shape", tuple((5,5))])

        name = "disc" + str(param["discount"]) \
         + "_bat" + str(param["batch_sizes"]) + \
        "_ent" + str(param["entropy_coeff"]) + "_lambda" + str(param["lambda_"]) \
            + "_agnts" + str(param["n_agents"]) + "_" + str(param["policy"]) + "_sd" + str(param["seed"])

        name.replace('.', "_")

        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--policy", str(param["policy"])])
        args.extend(["--name", name])
        args.extend(["--batch_size", str(param["batch_sizes"])])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--c2", str(param["entropy_coeff"])])
       # args.extend(["--c1", str(param["value_coeff"])])
        args.extend(["--discount", str(param["discount"])])
        args.extend(["--lambda_", str(param["lambda_"])])
    
        main.main(args)

def T0_2_p2():
    n_iterations = str(2500)
    experiment_group_name = "T0_2"
    work_dir = "/home/james/Desktop/Gridworld/EXPERIMENTS/" + experiment_group_name
    plot_dir = "/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/" + experiment_group_name

 
    parmeter_grid1 = {
        "seed": [2],
        "batch_sizes": [500,1000,2000],
        "entropy_coeff":[0.001],
        "discount":[1.0, 0.95, 0.6],
        "lambda_": [1.0, 0.95,0.6],
        #"value_ceoff": [0.01, 0.1, 0.5, 1.0],
        "policy": ["PG_Separate"],
        "n_agents": [1,2,3,4]
    }


    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
        args.extend(["--iterations", n_iterations])
       # args.extend(["--n_workers", str(workers)])
       # args.extend(["--map_shape", tuple((5,5))])

        name = "disc" + str(param["discount"]) \
         + "_bat" + str(param["batch_sizes"]) + \
        "_ent" + str(param["entropy_coeff"]) + "_lambda" + str(param["lambda_"]) \
            + "_agnts" + str(param["n_agents"]) + "_" + str(param["policy"]) + "_sd" + str(param["seed"])

        name.replace('.', "_")

        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--policy", str(param["policy"])])
        args.extend(["--name", name])
        args.extend(["--batch_size", str(param["batch_sizes"])])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--c2", str(param["entropy_coeff"])])
       # args.extend(["--c1", str(param["value_coeff"])])
        args.extend(["--discount", str(param["discount"])])
        args.extend(["--lambda_", str(param["lambda_"])])
    
        main.main(args)

def T0_2_p3():
    n_iterations = str(2500)
    experiment_group_name = "T0_2"
    work_dir = "/home/james/Desktop/Gridworld/EXPERIMENTS/" + experiment_group_name
    plot_dir = "/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/" + experiment_group_name

 
    parmeter_grid1 = {
        "seed": [2],
        "batch_sizes": [500,1000,2000],
        "entropy_coeff":[0.01],
        "discount":[1.0, 0.95, 0.6],
        "lambda_": [1.0, 0.95,0.6],
        #"value_ceoff": [0.01, 0.1, 0.5, 1.0],
        "policy": ["PG_Separate"],
        "n_agents": [1,2,3,4]
    }


    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
        args.extend(["--iterations", n_iterations])
       # args.extend(["--n_workers", str(workers)])
       # args.extend(["--map_shape", tuple((5,5))])

        name = "disc" + str(param["discount"]) \
         + "_bat" + str(param["batch_sizes"]) + \
        "_ent" + str(param["entropy_coeff"]) + "_lambda" + str(param["lambda_"]) \
            + "_agnts" + str(param["n_agents"]) + "_" + str(param["policy"]) + "_sd" + str(param["seed"])

        name.replace('.', "_")

        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--policy", str(param["policy"])])
        args.extend(["--name", name])
        args.extend(["--batch_size", str(param["batch_sizes"])])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--c2", str(param["entropy_coeff"])])
       # args.extend(["--c1", str(param["value_coeff"])])
        args.extend(["--discount", str(param["discount"])])
        args.extend(["--lambda_", str(param["lambda_"])])
    
        main.main(args)




def T0_3_p1():
    n_iterations = str(2500)
    experiment_group_name = "T0_3"
    work_dir = "/home/james/Desktop/Gridworld/EXPERIMENTS/" + experiment_group_name
    plot_dir = "/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/" + experiment_group_name

 
    parmeter_grid1 = {
        "seed": [2],
        "batch_sizes": [500,1000,2000],
        "entropy_coeff":[0.0],
        "discount":[1.0],
        #"lambda_": [1.0, 0.95,0.6],
        #"value_ceoff": [0.01, 0.1, 0.5, 1.0],
        "policy": ["PG_Separate"],
        "n_agents": [1,2,3,4]
    }


    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
        args.extend(["--iterations", n_iterations])
       # args.extend(["--n_workers", str(workers)])
       # args.extend(["--map_shape", tuple((5,5))])

        name = "disc" + str(param["discount"]) \
         + "_bat" + str(param["batch_sizes"]) + \
        "_ent" + str(param["entropy_coeff"]) \
            + "_agnts" + str(param["n_agents"]) + "_" + str(param["policy"]) + "_sd" + str(param["seed"])

        name.replace('.', "_")

        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--policy", str(param["policy"])])
        args.extend(["--name", name])
        args.extend(["--batch_size", str(param["batch_sizes"])])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--c2", str(param["entropy_coeff"])])
       # args.extend(["--c1", str(param["value_coeff"])])
        args.extend(["--discount", str(param["discount"])])
       # args.extend(["--lambda_", str(param["lambda_"])])
    
        main.main(args)

def T0_3_p2():
    n_iterations = str(2500)
    experiment_group_name = "T0_3"
    work_dir = "/home/james/Desktop/Gridworld/EXPERIMENTS/" + experiment_group_name
    plot_dir = "/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/" + experiment_group_name

 
    parmeter_grid1 = {
        "seed": [2],
        "batch_sizes": [500,1000,2000],
        "entropy_coeff":[0.0],
        "discount":[0.95],
        #"lambda_": [1.0, 0.95,0.6],
        #"value_ceoff": [0.01, 0.1, 0.5, 1.0],
        "policy": ["PG_Separate"],
        "n_agents": [1,2,3,4]
    }


    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
        args.extend(["--iterations", n_iterations])
       # args.extend(["--n_workers", str(workers)])
       # args.extend(["--map_shape", tuple((5,5))])

        name = "disc" + str(param["discount"]) \
         + "_bat" + str(param["batch_sizes"]) + \
        "_ent" + str(param["entropy_coeff"]) \
            + "_agnts" + str(param["n_agents"]) + "_" + str(param["policy"]) + "_sd" + str(param["seed"])

        name.replace('.', "_")

        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--policy", str(param["policy"])])
        args.extend(["--name", name])
        args.extend(["--batch_size", str(param["batch_sizes"])])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--c2", str(param["entropy_coeff"])])
       # args.extend(["--c1", str(param["value_coeff"])])
        args.extend(["--discount", str(param["discount"])])
       # args.extend(["--lambda_", str(param["lambda_"])])
    
        main.main(args)



def T0_3_p3():
    n_iterations = str(2500)
    experiment_group_name = "T0_3"
    work_dir = "/home/james/Desktop/Gridworld/EXPERIMENTS/" + experiment_group_name
    plot_dir = "/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/" + experiment_group_name

 
    parmeter_grid1 = {
        "seed": [2],
        "batch_sizes": [500,1000,2000],
        "entropy_coeff":[0.0],
        "discount":[0.7],
        #"lambda_": [1.0, 0.95,0.6],
        #"value_ceoff": [0.01, 0.1, 0.5, 1.0],
        "policy": ["PG_Separate"],
        "n_agents": [1,2,3,4]
    }


    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
        args.extend(["--iterations", n_iterations])
       # args.extend(["--n_workers", str(workers)])
       # args.extend(["--map_shape", tuple((5,5))])

        name = "disc" + str(param["discount"]) \
         + "_bat" + str(param["batch_sizes"]) + \
        "_ent" + str(param["entropy_coeff"]) \
            + "_agnts" + str(param["n_agents"]) + "_" + str(param["policy"]) + "_sd" + str(param["seed"])

        name.replace('.', "_")

        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--policy", str(param["policy"])])
        args.extend(["--name", name])
        args.extend(["--batch_size", str(param["batch_sizes"])])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--entropy_coeff", str(param["entropy_coeff"])])
       # args.extend(["--c1", str(param["value_coeff"])])
        args.extend(["--discount", str(param["discount"])])
       # args.extend(["--lambda_", str(param["lambda_"])])
    
        main.main(args)

def TEST():
    n_iterations = str(2500)
    experiment_group_name = "TEST"
    work_dir = "/home/james/Desktop/Gridworld/EXPERIMENTS/" + experiment_group_name
    plot_dir = "/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/" + experiment_group_name

 
    parmeter_grid1 = {
        "seed": [2],
        "batch_sizes": [500,1000,2000],
        "entropy_coeff":[0.0],
        "discount":[0.7],
        #"lambda_": [1.0, 0.95,0.6],
        #"value_ceoff": [0.01, 0.1, 0.5, 1.0],
        "policy": ["PG_Separate"],
        "n_agents": [1,2,3,4]
    }


    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
        args.extend(["--iterations", n_iterations])
       # args.extend(["--n_workers", str(workers)])
       # args.extend(["--map_shape", tuple((5,5))])

        name = "disc" + str(param["discount"]) \
         + "_bat" + str(param["batch_sizes"]) + \
        "_ent" + str(param["entropy_coeff"]) \
            + "_agnts" + str(param["n_agents"]) + "_" + str(param["policy"]) + "_sd" + str(param["seed"])

        name.replace('.', "_")

        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--policy", str(param["policy"])])
        args.extend(["--name", name])
        args.extend(["--batch_size", str(param["batch_sizes"])])
        args.extend(["--seed", str(param["seed"])])
        args.extend(["--entropy_coeff", str(param["entropy_coeff"])])
       # args.extend(["--c1", str(param["value_coeff"])])
        args.extend(["--discount", str(param["discount"])])
       # args.extend(["--lambda_", str(param["lambda_"])])
    
        main.main(args)

# def test_ic3():
#     n_iterations = str(3000)
#     experiment_group_name = "TEST2_IC3"
#     work_dir = "/home/james/Desktop/Gridworld/EXPERIMENTS/" + experiment_group_name
#     plot_dir = "/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/" + experiment_group_name

 
#     parmeter_grid1 = {
#         "seed": [2],
#         "batch_sizes": [500],
#         "entropy_coeff":[0.0],
#         "discount":[1.0],
#         #"lambda_": [1.0, 0.95,0.6],
#         "value_ceoff": [0.01],
#         "policy": ["IC3"],
#         "n_agents": [4],
#         "recurrent": [True],
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
    
#         main.main(args)











# def parameter_tuning():
#     n_iterations = str(1000)
#     workers = 1
#     experiment_name = "PT0_1"
#     work_dir = "/home/james/Desktop/Gridworld/test5"
#     plot_dir = "/home/james/Desktop/Gridworld/CENTRAL_PLOT_Test5"

#     # parmeter_grid = {
#     #     "seed": [2,4],
#     #     "batch_sizes": [200, 500, 1000, 2000],
#     #     "entropy_coeff":[0.01, 0.05, 0.1, 0.5],
#     #     "value_coeff":[0.01, 0.1, 0.5, 0.9],
#     #     "discount":[1.0, 0.9]
#     # }

#     #For debugging/testing
#     parmeter_grid = {
#         "seed": [2],
#         "batch_sizes": [500,1000],
#         "entropy_coeff":[0.005, 0.01, 0.05, 0.1],
#         #"value_coeff":[0.001,0.5],
#         "discount":[1.0],
#         "policy": ["PG_Seperate", "AC_Seperate"]
#     }

#     grid = ParameterGrid(parmeter_grid)
    
#     for param in grid:
#         args = []
#         args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
#         args.extend(["--iterations", n_iterations])
#         args.extend(["--n_workers", str(workers)])

#         name = "disc" + str(param["discount"]) +"_val" \
#         + str(param["value_coeff"]) + "_bat" + str(param["batch_sizes"]) + \
#         "_ent" + str(param["entropy_coeff"]) + "_" + str(param["policy"]) + "_sd" + str(param["seed"])

#         name.replace('.', "_")

#         args.extend(["--policy", str(param["policy"])])
#         args.extend(["--name", name])
#         args.extend(["--batch_size", str(param["batch_sizes"])])
#         args.extend(["--seed", str(param["seed"])])
#         args.extend(["--c2", str(param["entropy_coeff"])])
#         #args.extend(["--c1", str(param["value_coeff"])])
#         args.extend(["--discount", str(param["discount"])])
    
#         main.main(args)

# def test():
#     n_iterations = str(1000)
#     workers = 1
#     experiment_group_name = "test_2_agents"
#     work_dir = "/home/james/Desktop/Gridworld/TEST/" + experiment_group_name
#     plot_dir = "/home/james/Desktop/Gridworld/CENTRAL_TEST/" + experiment_group_name

#     # parmeter_grid = {
#     #     "seed": [2,4],
#     #     "batch_sizes": [200, 500, 1000, 2000],
#     #     "entropy_coeff":[0.01, 0.05, 0.1, 0.5],
#     #     "value_coeff":[0.01, 0.1, 0.5, 0.9],
#     #     "discount":[1.0, 0.9]
#     # }

#     #For debugging/testing
#     parmeter_grid = {
#         "seed": [2],
#         "batch_sizes": [500],
#         "entropy_coeff":[0.01],
#         #"value_coeff":[0.001,0.5],
#         "discount":[1.0],
#         "policy": ["PG_Separate", "AC_Separate"],
#         "n_agents": [2]
#     }

#     grid = ParameterGrid(parmeter_grid)
    
#     for param in grid:
#         args = []
#         args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
#         args.extend(["--iterations", n_iterations])
#         args.extend(["--n_workers", str(workers)])

#         name = "disc" + str(param["discount"])  \
#          + "_bat" + str(param["batch_sizes"]) + \
#         "_ent" + str(param["entropy_coeff"]) + "_agnts" + str(param["n_agents"]) + "_" + str(param["policy"]) + "_sd" + str(param["seed"])

#         name.replace('.', "_")

#         args.extend(["--n_agents", str(param["n_agents"])])
#         args.extend(["--policy", str(param["policy"])])
#         args.extend(["--name", name])
#         args.extend(["--batch_size", str(param["batch_sizes"])])
#         args.extend(["--seed", str(param["seed"])])
#         args.extend(["--c2", str(param["entropy_coeff"])])
#         #args.extend(["--c1", str(param["value_coeff"])])
#         args.extend(["--discount", str(param["discount"])])
    
#         main.main(args)

# def single_agent_comp_shared_value_f():
#     n_iterations = str(4000)
#     workers = 1
#     experiment_group_name = "T0_1"
#     work_dir = "/home/james/Desktop/Gridworld/EXPERIMENTS/" + experiment_group_name
#     plot_dir = "/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/" + experiment_group_name

#     # parmeter_grid = {
#     #     "seed": [2,4],
#     #     "batch_sizes": [200, 500, 1000, 2000],
#     #     "entropy_coeff":[0.01, 0.05, 0.1, 0.5],
#     #     "value_coeff":[0.01, 0.1, 0.5, 0.9],
#     #     "discount":[1.0, 0.9]
#     # }

#     #For debugging/testing
#     parmeter_grid1 = {
#         "seed": [2],
#         "batch_sizes": [500],
#         "entropy_coeff":[0.01],
#         "value_coeff":[0.001,0.01,0.1],
#         "discount":[1.0],
#         "policy": ["PG_Shared"],
#         "n_agents": [1]
#     }
#     parmeter_grid2 = {
#         "seed": [2],
#         "batch_sizes": [500],
#         "entropy_coeff":[0.01],
#         "value_coeff":[1.0],
#         "discount":[1.0],
#         "policy": ["PG_Separate"],
#         "n_agents": [1]
#     }

#     grid1 = ParameterGrid(parmeter_grid1)
#     grid2 = ParameterGrid(parmeter_grid2)
    
#     for param in grid1:
#         args = []
#         args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
#         args.extend(["--iterations", n_iterations])
#         args.extend(["--n_workers", str(workers)])

#         name = "disc" + str(param["discount"]) \
#          + "_bat" + str(param["batch_sizes"]) + \
#         "_ent" + str(param["entropy_coeff"]) + "_val" + str(param["value_coeff"])+ "_agnts" + str(param["n_agents"]) + "_" + str(param["policy"]) + "_sd" + str(param["seed"])

#         name.replace('.', "_")

#         args.extend(["--n_agents", str(param["n_agents"])])
#         args.extend(["--policy", str(param["policy"])])
#         args.extend(["--name", name])
#         args.extend(["--batch_size", str(param["batch_sizes"])])
#         args.extend(["--seed", str(param["seed"])])
#         args.extend(["--c2", str(param["entropy_coeff"])])
#         args.extend(["--c1", str(param["value_coeff"])])
#         args.extend(["--discount", str(param["discount"])])
    
#         main.main(args)
    
#     for param in grid2:
#         args = []
#         args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
#         args.extend(["--iterations", n_iterations])
#         args.extend(["--n_workers", str(workers)])

#         name = "disc" + str(param["discount"]) \
#          + "_bat" + str(param["batch_sizes"]) + \
#         "_ent" + str(param["entropy_coeff"]) + "_val" + str(param["value_coeff"])+ "_agnts" + str(param["n_agents"]) + "_" + str(param["policy"]) + "_sd" + str(param["seed"])

#         name.replace('.', "_")

#         args.extend(["--n_agents", str(param["n_agents"])])
#         args.extend(["--policy", str(param["policy"])])
#         args.extend(["--name", name])
#         args.extend(["--batch_size", str(param["batch_sizes"])])
#         args.extend(["--seed", str(param["seed"])])
#         args.extend(["--c2", str(param["entropy_coeff"])])
#         args.extend(["--c1", str(param["value_coeff"])])
#         args.extend(["--discount", str(param["discount"])])
    
#         main.main(args)
    

# def multi_agent_comparison_1():
#     n_iterations = str(1000)
#     workers = 1
#     experiment_group_name = "P0_1"
#     work_dir = "/home/james/Desktop/Gridworld/EXPERIMENTS/" + experiment_group_name
#     plot_dir = "/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/" + experiment_group_name

#     # parmeter_grid = {
#     #     "seed": [2,4],
#     #     "batch_sizes": [200, 500, 1000, 2000],
#     #     "entropy_coeff":[0.01, 0.05, 0.1, 0.5],
#     #     "value_coeff":[0.01, 0.1, 0.5, 0.9],
#     #     "discount":[1.0, 0.9]
#     # }

#     #For debugging/testing
#     parmeter_grid = {
#         "seed": [2],
#         "batch_sizes": [500,1000],
#         "entropy_coeff":[0.005, 0.01, 0.05],
#         #"value_coeff":[0.001,0.5],
#         "discount":[1.0],
#         "policy": ["PG_Separate", "AC_Separate"],
#         "n_agents": [1,2,3]
#     }

#     grid = ParameterGrid(parmeter_grid)
    
#     for param in grid:
#         args = []
#         args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
#         args.extend(["--iterations", n_iterations])
#         args.extend(["--n_workers", str(workers)])

#         name = "disc" + str(param["discount"]) \
#          + "_bat" + str(param["batch_sizes"]) + \
#         "_ent" + str(param["entropy_coeff"]) + "_agnts" + str(param["n_agents"]) + "_" + str(param["policy"]) + "_sd" + str(param["seed"])

#         name.replace('.', "_")

#         args.extend(["--n_agents", str(param["n_agents"])])
#         args.extend(["--policy", str(param["policy"])])
#         args.extend(["--name", name])
#         args.extend(["--batch_size", str(param["batch_sizes"])])
#         args.extend(["--seed", str(param["seed"])])
#         args.extend(["--c2", str(param["entropy_coeff"])])
#         #args.extend(["--c1", str(param["value_coeff"])])
#         args.extend(["--discount", str(param["discount"])])
    
#         main.main(args)
#     n_iterations = str(3000)
#     workers = 1
#     experiment_group_name = "P0_2"
#     work_dir = "/home/james/Desktop/Gridworld/EXPERIMENTS/" + experiment_group_name
#     plot_dir = "/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/" + experiment_group_name

#     # parmeter_grid = {
#     #     "seed": [2,4],
#     #     "batch_sizes": [200, 500, 1000, 2000],
#     #     "entropy_coeff":[0.01, 0.05, 0.1, 0.5],
#     #     "value_coeff":[0.01, 0.1, 0.5, 0.9],
#     #     "discount":[1.0, 0.9]
#     # }

#     #For debugging/testing
#     parmeter_grid = {
#         "seed": [2],
#         "batch_sizes": [250, 500],
#         "entropy_coeff":[0.01, 0.05],
#         #"value_coeff":[0.001,0.5],
#         "discount":[1.0, 0.9],
#         "policy": ["PG_Separate", "AC_Separate"],
#         "n_agents": [2,3]
#     }

#     grid = ParameterGrid(parmeter_grid)
    
#     for param in grid:
#         args = []
#         args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
#         args.extend(["--iterations", n_iterations])
#         args.extend(["--n_workers", str(workers)])
#         args.extend(["--lambda_", str(0.9)])

#         name = "disc" + str(param["discount"]) \
#          + "_bat" + str(param["batch_sizes"]) + \
#         "_ent" + str(param["entropy_coeff"]) + "_agnts" + str(param["n_agents"]) + "_" + str(param["policy"]) + "_sd" + str(param["seed"])

#         name.replace('.', "_")

#         args.extend(["--n_agents", str(param["n_agents"])])
#         args.extend(["--policy", str(param["policy"])])
#         args.extend(["--name", name])
#         args.extend(["--batch_size", str(param["batch_sizes"])])
#         args.extend(["--seed", str(param["seed"])])
#         args.extend(["--c2", str(param["entropy_coeff"])])
#         #args.extend(["--c1", str(param["value_coeff"])])
#         args.extend(["--discount", str(param["discount"])])
    
#         main.main(args)


# def multi_agent_comparison_2():
#     n_iterations = str(3000)
#     workers = 1
#     experiment_group_name = "P0_2"
#     work_dir = "/home/james/Desktop/Gridworld/EXPERIMENTS/" + experiment_group_name
#     plot_dir = "/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/" + experiment_group_name

#     # parmeter_grid = {
#     #     "seed": [2,4],
#     #     "batch_sizes": [200, 500, 1000, 2000],
#     #     "entropy_coeff":[0.01, 0.05, 0.1, 0.5],
#     #     "value_coeff":[0.01, 0.1, 0.5, 0.9],
#     #     "discount":[1.0, 0.9]
#     # }

#     #For debugging/testing
#     parmeter_grid = {
#         "seed": [2],
#         "batch_sizes": [250, 500],
#         "entropy_coeff":[0.01, 0.05],
#         #"value_coeff":[0.001,0.5],
#         "discount":[1.0, 0.9],
#         "policy": ["PG_Separate", "AC_Separate"],
#         "n_agents": [2,3]
#     }

#     grid = ParameterGrid(parmeter_grid)
    
#     for param in grid:
#         args = []
#         args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
#         args.extend(["--iterations", n_iterations])
#         args.extend(["--n_workers", str(workers)])
#         args.extend(["--lambda_", str(0.9)])

#         name = "disc" + str(param["discount"]) \
#          + "_bat" + str(param["batch_sizes"]) + \
#         "_ent" + str(param["entropy_coeff"]) + "_agnts" + str(param["n_agents"]) + "_" + str(param["policy"]) + "_sd" + str(param["seed"])

#         name.replace('.', "_")

#         args.extend(["--n_agents", str(param["n_agents"])])
#         args.extend(["--policy", str(param["policy"])])
#         args.extend(["--name", name])
#         args.extend(["--batch_size", str(param["batch_sizes"])])
#         args.extend(["--seed", str(param["seed"])])
#         args.extend(["--c2", str(param["entropy_coeff"])])
#         #args.extend(["--c1", str(param["value_coeff"])])
#         args.extend(["--discount", str(param["discount"])])
    
#         main.main(args)


# def singel_agentP0_3_1():
#     n_iterations = str(10000)
#     workers = 1
#     experiment_group_name = "P0_3"
#     work_dir = "/home/james/Desktop/Gridworld/EXPERIMENTS/" + experiment_group_name
#     plot_dir = "/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/" + experiment_group_name

#     parmeter_grid1 = {
#         "seed": [3],
#         "batch_sizes": [100,250,500],
#         "entropy_coeff":[0.00001, 0.0001,0.001, 0.005],
#         "step_r": [-0.01],
#         "discount":[1.0],
#         "policy": ["PG_Separate", "AC_Separate"],
#         "n_agents": [1]
#     }
    

#     grid1 = ParameterGrid(parmeter_grid1)

    
#     for param in grid1:
#         args = []
#         args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
#         args.extend(["--iterations", n_iterations])
#         args.extend(["--n_workers", str(workers)])
#         #args.extend(["--map_shape", str((5,5))])

#         name = "disc" + str(param["discount"]) \
#          + "_bat" + str(param["batch_sizes"]) + \
#         "_ent" + str(param["entropy_coeff"]) +"_step_r" + str(param["step_r"]) \
#         + "_agnts" + str(param["n_agents"]) + "_" + str(param["policy"]) + "_sd" + str(param["seed"])

#         name.replace('.', "_")

#         args.extend(["--n_agents", str(param["n_agents"])])
#         args.extend(["--policy", str(param["policy"])])
#         args.extend(["--name", name])
#         args.extend(["--batch_size", str(param["batch_sizes"])])
#         args.extend(["--seed", str(param["seed"])])
#         args.extend(["--c2", str(param["entropy_coeff"])])
#        # args.extend(["--c1", str(param["value_coeff"])])
#         args.extend(["--discount", str(param["discount"])])
    
#         main.main(args)


# def singel_agentP0_3_part1():
#     n_iterations = str(4000)
#     workers = 1
#     experiment_group_name = "P0_3"
#     work_dir = "/home/james/Desktop/Gridworld/EXPERIMENTS/" + experiment_group_name
#     plot_dir = "/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/" + experiment_group_name

#     parmeter_grid1 = {
#         "seed": [2],
#         "batch_sizes": [100,250,500],
#         "entropy_coeff":[0.001, 0.005, 0.01],
#         "step_r": [-0.01, -0.08],
#         "discount":[1.0],
#         "policy": ["PG_Separate"],
#         "n_agents": [1]
#     }
    

#     grid1 = ParameterGrid(parmeter_grid1)

    
#     for param in grid1:
#         args = []
#         args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
#         args.extend(["--iterations", n_iterations])
#         args.extend(["--n_workers", str(workers)])
#         #args.extend(["--map_shape", str((5,5))])

#         name = "disc" + str(param["discount"]) \
#          + "_bat" + str(param["batch_sizes"]) + \
#         "_ent" + str(param["entropy_coeff"]) +"_step_r" + str(param["step_r"]) \
#         + "_agnts" + str(param["n_agents"]) + "_" + str(param["policy"]) + "_sd" + str(param["seed"])

#         name.replace('.', "_")

#         args.extend(["--n_agents", str(param["n_agents"])])
#         args.extend(["--policy", str(param["policy"])])
#         args.extend(["--name", name])
#         args.extend(["--batch_size", str(param["batch_sizes"])])
#         args.extend(["--seed", str(param["seed"])])
#         args.extend(["--c2", str(param["entropy_coeff"])])
#        # args.extend(["--c1", str(param["value_coeff"])])
#         args.extend(["--discount", str(param["discount"])])
    
#         main.main(args)
    
# def singel_agentP0_3_part3():
#     n_iterations = str(4000)
#     workers = 1
#     experiment_group_name = "P0_3"
#     work_dir = "/home/james/Desktop/Gridworld/EXPERIMENTS/" + experiment_group_name
#     plot_dir = "/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/" + experiment_group_name

#     parmeter_grid1 = {
#         "seed": [2],
#         "batch_sizes": [100,250,500],
#         "entropy_coeff":[0.001, 0.005, 0.01],
#         "step_r": [-0.01, -0.08],
#         "discount":[1.0],
#         "policy": ["AC_One_Step"],
#         "n_agents": [1]
#     }
    

#     grid1 = ParameterGrid(parmeter_grid1)

    
#     for param in grid1:
#         args = []
#         args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
#         args.extend(["--iterations", n_iterations])
#         args.extend(["--n_workers", str(workers)])
#         #args.extend(["--map_shape", str((5,5))])

#         name = "disc" + str(param["discount"]) \
#          + "_bat" + str(param["batch_sizes"]) + \
#         "_ent" + str(param["entropy_coeff"]) +"_step_r" + str(param["step_r"]) \
#         + "_agnts" + str(param["n_agents"]) + "_" + str(param["policy"]) + "_sd" + str(param["seed"])

#         name.replace('.', "_")

#         args.extend(["--n_agents", str(param["n_agents"])])
#         args.extend(["--policy", str(param["policy"])])
#         args.extend(["--name", name])
#         args.extend(["--batch_size", str(param["batch_sizes"])])
#         args.extend(["--seed", str(param["seed"])])
#         args.extend(["--c2", str(param["entropy_coeff"])])
#        # args.extend(["--c1", str(param["value_coeff"])])
#         args.extend(["--discount", str(param["discount"])])
    
#         main.main(args)

# def singel_agentP0_3_part2():
#     n_iterations = str(4000)
#     workers = 1
#     experiment_group_name = "P0_3"
#     work_dir = "/home/james/Desktop/Gridworld/EXPERIMENTS/" + experiment_group_name
#     plot_dir = "/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/" + experiment_group_name

 
#     parmeter_grid2 = {
#         "seed": [2],
#         "batch_sizes": [100,250,500],
#         "entropy_coeff":[0.001, 0.005, 0.01],
#         "step_r": [-0.01, -0.08],
#         "discount":[1.0],
#         "lambda_": [1.0, 0.5, 0.25],
#         "policy": ["AC_Separate"],
#         "n_agents": [1]
#     }


#     grid2 = ParameterGrid(parmeter_grid2)
    
#     for param in grid2:
#         args = []
#         args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
#         args.extend(["--iterations", n_iterations])
#         args.extend(["--n_workers", str(workers)])
#         #args.extend(["--map_shape", str((5,5))])

#         name = "disc" + str(param["discount"]) \
#          + "_bat" + str(param["batch_sizes"]) + \
#         "_ent" + str(param["entropy_coeff"]) + "_lamb" + str(param["lambda_"])  +"_step_r" + str(param["step_r"]) \
#             + "_agnts" + str(param["n_agents"]) + "_" + str(param["policy"]) + "_sd" + str(param["seed"])

#         name.replace('.', "_")

#         args.extend(["--n_agents", str(param["n_agents"])])
#         args.extend(["--policy", str(param["policy"])])
#         args.extend(["--name", name])
#         args.extend(["--batch_size", str(param["batch_sizes"])])
#         args.extend(["--seed", str(param["seed"])])
#         args.extend(["--c2", str(param["entropy_coeff"])])
#       #  args.extend(["--c1", str(param["value_coeff"])])
#         args.extend(["--discount", str(param["discount"])])
    
#         main.main(args)


# def test2():

#     n_iterations = str(10000)
#     workers = 1
#     experiment_group_name = "test_AC_nets"
#     work_dir = "/home/james/Desktop/Gridworld/TEST/" + experiment_group_name
#     plot_dir = "/home/james/Desktop/Gridworld/CENTRAL_TEST/" + experiment_group_name

#     # parmeter_grid = {
#     #     "seed": [2,4],
#     #     "batch_sizes": [200, 500, 1000, 2000],
#     #     "entropy_coeff":[0.01, 0.05, 0.1, 0.5],
#     #     "value_coeff":[0.01, 0.1, 0.5, 0.9],
#     #     "discount":[1.0, 0.9]
#     # }

#     #For debugging/testing
#     parmeter_grid = {
#         "seed": [2],
#         "batch_sizes": [1000],
#         "entropy_coeff":[0.001],
#         #"value_coeff":[0.001,0.5],
#         "discount":[1.0],
#         "policy": ["AC_Test"],
#         "n_agents": [1]
#     }

#     grid = ParameterGrid(parmeter_grid)
    
#     for param in grid:
#         args = []
#         args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
#         args.extend(["--iterations", n_iterations])
#         args.extend(["--n_workers", str(workers)])
#         args.extend(["--lambda_", str(0.9)])

#         name = "disc" + str(param["discount"])  \
#          + "_bat" + str(param["batch_sizes"]) + \
#         "_ent" + str(param["entropy_coeff"]) + "_agnts" + str(param["n_agents"]) + "_" + str(param["policy"]) + "_sd" + str(param["seed"])

#         name.replace('.', "_")

#         args.extend(["--n_agents", str(param["n_agents"])])
#         args.extend(["--policy", str(param["policy"])])
#         args.extend(["--name", name])
#         args.extend(["--batch_size", str(param["batch_sizes"])])
#         args.extend(["--seed", str(param["seed"])])
#         args.extend(["--c2", str(param["entropy_coeff"])])
#         #args.extend(["--c1", str(param["value_coeff"])])
#         args.extend(["--discount", str(param["discount"])])
    
#         main.main(args)

#  def test3():
#     n_iterations = str(2000)
#     workers = 1
#     experiment_group_name = "T0_3_1"
#     work_dir = "/home/james/Desktop/Gridworld/EXPERIMENTS/" + experiment_group_name
#     plot_dir = "/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/" + experiment_group_name

 
#     parmeter_grid2 = {
#         "seed": [2],
#         "batch_sizes": [250,500, 1000],
#         "entropy_coeff":[0.0, 0.001,0.1],
#         "discount":[1.0],
#         "lambda_": [1.0],
#         "policy": ["AC_Separate"],
#         "n_agents": [1]
#     }


#     grid2 = ParameterGrid(parmeter_grid2)
    
#     for param in grid2:
#         args = []
#         args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
#         args.extend(["--iterations", n_iterations])
#         args.extend(["--n_workers", str(workers)])
#         #args.extend(["--map_shape", str((5,5))])

#         name = "disc" + str(param["discount"]) \
#          + "_bat" + str(param["batch_sizes"]) + \
#         "_ent" + str(param["entropy_coeff"]) + "_lamb" + str(param["lambda_"]) \
#             + "_agnts" + str(param["n_agents"]) + "_" + str(param["policy"]) + "_sd" + str(param["seed"])

#         name.replace('.', "_")

#         args.extend(["--n_agents", str(param["n_agents"])])
#         args.extend(["--policy", str(param["policy"])])
#         args.extend(["--name", name])
#         args.extend(["--batch_size", str(param["batch_sizes"])])
#         args.extend(["--seed", str(param["seed"])])
#         args.extend(["--c2", str(param["entropy_coeff"])])
#       #  args.extend(["--c1", str(param["value_coeff"])])
#         args.extend(["--discount", str(param["discount"])])
    
#         main.main(args)