import main
from sklearn.model_selection import ParameterGrid

def ic3net_test():
    n_iterations = str(2500)
    experiment_group_name = "Ic3Net_Test"
    work_dir = experiment_group_name
    plot_dir = experiment_group_name + "_Tensorboard"

    parmeter_grid1 = {
        "seed": [1],
        "batch_sizes": [500],
        "entropy_coeff":[0.01],
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
        "comm_mask_zero":[False], #If true, all communication is zero
        "hid_size":[120],
        "share_weights":[False],
        "comm_action_one": [False] #Usually false for ic3; if true, then its commNet
    }

    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
        args.extend(["--iterations", n_iterations])

        name = "IC3Net_Test" + "_disc_" + str(param["discount"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["batch_sizes"]) \
            + "_commpasses_" + str(param["comm_passes"]) \
            + "_commzero_" + str(param["comm_mask_zero"]) \
            + "_commNet_" + str(param["comm_action_one"]) \
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
        if param["comm_action_one"]:
            args.extend(["--comm_action_one"])

        args.extend(["--render_rate", str(int(1e2))])
        args.extend(["--checkpoint_frequency", str(int(5e2))])
        args.extend(["--benchmark_frequency", str(int(3000))])
        args.extend(["--benchmark_num_episodes", str(int(10))])
        args.extend(["--benchmark_render_length", str(int(20))])
        args.extend(["--obj_density", str(param["obj_density"])])

        main.main(args)


def ic3_1A1():
    n_iterations = str(2500)
    experiment_group_name = "Ic3_1A"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"

    parmeter_grid1 = {
        "seed": [1],
        "batch_sizes": [500],
        "entropy_coeff":[0.01,0.001],
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

            

        args.extend(["--render_rate", str(int(1e2))])
        args.extend(["--checkpoint_frequency", str(int(5e2))])
        args.extend(["--benchmark_frequency", str(int(3000))])
        args.extend(["--benchmark_num_episodes", str(int(10))])
        args.extend(["--benchmark_render_length", str(int(20))])
        args.extend(["--obj_density", str(param["obj_density"])])

        main.main(args)


def ic3_1B1():
    n_iterations = str(2500)
  #  experiment_group_name = "IC3"
    experiment_group_name = "Ic3_1B"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"

 
    parmeter_grid1 = {
        "seed": [1],
        "batch_sizes": [500],
        "entropy_coeff":[0.01],
        "discount":[0.95, 0.5, 0.1],
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

        args.extend(["--render_rate", str(int(1e2))])
        args.extend(["--checkpoint_frequency", str(int(5e2))])
        args.extend(["--benchmark_frequency", str(int(3000))])
        args.extend(["--benchmark_num_episodes", str(int(10))])
        args.extend(["--benchmark_render_length", str(int(20))])
        args.extend(["--obj_density", str(param["obj_density"])])

        main.main(args)

def ic3_1C1():
    n_iterations = str(2500)
    experiment_group_name = "Ic3_1C"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"

 
    parmeter_grid1 = {
        "seed": [1],
        "batch_sizes": [500],
        "entropy_coeff":[0.01],
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
        "share_weights":[False],
        "comm_action_one": [True] #Usually false for ic3; if true, then its commNet
    }


    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
        args.extend(["--iterations", n_iterations])

        name = "IC3" + "_disc_" + str(param["discount"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["batch_sizes"]) \
            + "_commpasses_" + str(param["comm_passes"]) \
            + "_commzero_" + str(param["comm_mask_zero"]) \
            + "_commNet_" + str(param["comm_action_one"]) \
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
        if param["comm_action_one"]:
            args.extend(["--comm_action_one"])

            

        args.extend(["--render_rate", str(int(1e2))])
        args.extend(["--checkpoint_frequency", str(int(5e2))])
        args.extend(["--benchmark_frequency", str(int(3000))])
        args.extend(["--benchmark_num_episodes", str(int(10))])
        args.extend(["--benchmark_render_length", str(int(20))])
        args.extend(["--obj_density", str(param["obj_density"])])

        main.main(args)

def ic3_1D1():
    n_iterations = str(2500)
  #  experiment_group_name = "IC3"
    experiment_group_name = "Ic3_1D"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"

 
    parmeter_grid1 = {
        "seed": [1],
        "batch_sizes": [500],
        "entropy_coeff":[0.01],
        "discount":[0.9],
        #"lambda_": [1.0, 0.95,0.6],
        "value_ceoff": [0.01],
        "policy": ["IC3"],
        "n_agents": [4],
        "obj_density": [0.2],
        "env_size": [5],
        "recurrent": [False],
        "base_policy_type": ["mlp"],
        "comm_passes": [1],
        "comm_mask_zero":[False],
        "hid_size":[120],
        "share_weights":[False],
        "comm_action_one": [True, False] #Usually false for ic3; if true, then its commNet
       # "hard_attention":[False] #Usually this parameter is true. If false communication is always present (commNet)
    }


    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
        args.extend(["--iterations", n_iterations])

        name = "IC3" + "_disc_" + str(param["discount"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["batch_sizes"]) \
            + "_commpasses_" + str(param["comm_passes"]) \
            + "_commzero_" + str(param["comm_mask_zero"]) \
            + "_commNet_" + str(param["comm_action_one"]) \
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
        if param["comm_action_one"]:
            args.extend(["--comm_action_one"])

            

        args.extend(["--render_rate", str(int(1e2))])
        args.extend(["--checkpoint_frequency", str(int(5e2))])
        args.extend(["--benchmark_frequency", str(int(3000))])
        args.extend(["--benchmark_num_episodes", str(int(10))])
        args.extend(["--benchmark_render_length", str(int(20))])
        args.extend(["--obj_density", str(param["obj_density"])])

        main.main(args)

def ic3_1D2():
    n_iterations = str(2500)
  #  experiment_group_name = "IC3"
    experiment_group_name = "Ic3_1D"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"

 
    parmeter_grid1 = {
        "seed": [1],
        "batch_sizes": [500],
        "entropy_coeff":[0.01],
        "discount":[0.9],
        #"lambda_": [1.0, 0.95,0.6],
        "value_ceoff": [0.01],
        "policy": ["IC3"],
        "n_agents": [4],
        "obj_density": [0.2],
        "env_size": [5],
        "recurrent": [False],
        "base_policy_type": ["mlp"],
        "comm_passes": [1],
        "comm_mask_zero":[True],
        "hid_size":[120],
        "share_weights":[False],
        "comm_action_one": [False] #Usually false for ic3; if true, then its commNet
       # "hard_attention":[False] #Usually this parameter is true. If false communication is always present (commNet)
    }


    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
        args.extend(["--iterations", n_iterations])

        name = "IC3" + "_disc_" + str(param["discount"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["batch_sizes"]) \
            + "_commpasses_" + str(param["comm_passes"]) \
            + "_commzero_" + str(param["comm_mask_zero"]) \
            + "_commNet_" + str(param["comm_action_one"]) \
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
        if param["comm_action_one"]:
            args.extend(["--comm_action_one"])

            

        args.extend(["--render_rate", str(int(1e2))])
        args.extend(["--checkpoint_frequency", str(int(5e2))])
        args.extend(["--benchmark_frequency", str(int(3000))])
        args.extend(["--benchmark_num_episodes", str(int(10))])
        args.extend(["--benchmark_render_length", str(int(20))])
        args.extend(["--obj_density", str(param["obj_density"])])

        main.main(args)

def ic3_1E1():
    n_iterations = str(2500)
  #  experiment_group_name = "IC3"
    experiment_group_name = "Ic3_1E"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"

 
    parmeter_grid1 = {
        "seed": [1],
        "batch_sizes": [500],
        "entropy_coeff":[0.01],
        "discount":[0.9],
        #"lambda_": [1.0, 0.95,0.6],
        "value_ceoff": [0.01],
        "policy": ["IC3"],
        "n_agents": [4],
        "obj_density": [0.2],
        "env_size": [5],
        "recurrent": [False],
        "base_policy_type": ["mlp"],
        "comm_passes": [1],
        "comm_mask_zero":[False],
        "hid_size":[120],
        "share_weights":[False],
        "comm_action_one": [True, False] #Usually false for ic3; if true, then its commNet
       # "hard_attention":[False] #Usually this parameter is true. If false communication is always present (commNet)
    }


    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
        args.extend(["--iterations", n_iterations])

        name = "IC3" + "_disc_" + str(param["discount"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["batch_sizes"]) \
            + "_commpasses_" + str(param["comm_passes"]) \
            + "_commzero_" + str(param["comm_mask_zero"]) \
            + "_commNet_" + str(param["comm_action_one"]) \
            + "_recurrent_" + str(param["recurrent"]) \
            + "_envsize_" + str(param["env_size"]) \
            + "_nagents_"+ str(param["n_agents"]) \
            + "_objdensity_"+ str(param["obj_density"]) \
            + "_seed_"+ str(param["seed"])

        args.extend(["--env_name","independent_navigation-v3"])
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
        if param["comm_action_one"]:
            args.extend(["--comm_action_one"])

            

        args.extend(["--render_rate", str(int(1e2))])
        args.extend(["--checkpoint_frequency", str(int(5e2))])
        args.extend(["--benchmark_frequency", str(int(3000))])
        args.extend(["--benchmark_num_episodes", str(int(10))])
        args.extend(["--benchmark_render_length", str(int(20))])
        args.extend(["--obj_density", str(param["obj_density"])])

        main.main(args)

def ic3_1E2():
    n_iterations = str(2500)
  #  experiment_group_name = "IC3"
    experiment_group_name = "Ic3_1E"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"

 
    parmeter_grid1 = {
        "seed": [1],
        "batch_sizes": [500],
        "entropy_coeff":[0.01],
        "discount":[0.9],
        #"lambda_": [1.0, 0.95,0.6],
        "value_ceoff": [0.01],
        "policy": ["IC3"],
        "n_agents": [4],
        "obj_density": [0.2],
        "env_size": [5],
        "recurrent": [False],
        "base_policy_type": ["mlp"],
        "comm_passes": [1],
        "comm_mask_zero":[True],
        "hid_size":[120],
        "share_weights":[False],
        "comm_action_one": [False] #Usually false for ic3; if true, then its commNet
       # "hard_attention":[False] #Usually this parameter is true. If false communication is always present (commNet)
    }


    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
        args.extend(["--iterations", n_iterations])

        name = "IC3" + "_disc_" + str(param["discount"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["batch_sizes"]) \
            + "_commpasses_" + str(param["comm_passes"]) \
            + "_commzero_" + str(param["comm_mask_zero"]) \
            + "_commNet_" + str(param["comm_action_one"]) \
            + "_recurrent_" + str(param["recurrent"]) \
            + "_envsize_" + str(param["env_size"]) \
            + "_nagents_"+ str(param["n_agents"]) \
            + "_objdensity_"+ str(param["obj_density"]) \
            + "_seed_"+ str(param["seed"])

        args.extend(["--env_name","independent_navigation-v3"])
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
        if param["comm_action_one"]:
            args.extend(["--comm_action_one"])

            

        args.extend(["--render_rate", str(int(1e2))])
        args.extend(["--checkpoint_frequency", str(int(5e2))])
        args.extend(["--benchmark_frequency", str(int(3000))])
        args.extend(["--benchmark_num_episodes", str(int(10))])
        args.extend(["--benchmark_render_length", str(int(20))])
        args.extend(["--obj_density", str(param["obj_density"])])

        main.main(args)

def ic3_1E4(): #Test PPO again 
    experiment_group_name = "Ic3_1E"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"

 
    n_iterations = str(2500) #str(3000)
    # experiment_group_name = "ppo_1D"
    # work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    # plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"

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
        "env_size": [5],
        "lr":[0.001]
    }

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        name = "PPO_CN1" + "_disc_" + str(param["discount"]) \
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

        args.extend(["--env_name","independent_navigation-v3"])
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


def ic3_1F1(): #How valuable is communication: Same as E, but with cooperative navigation-v1: agents cant see each other
    n_iterations = str(10000)
  #  experiment_group_name = "IC3"
    experiment_group_name = "Ic3_1F"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"

 
    parmeter_grid1 = {
        "seed": [1],
        "batch_sizes": [500],
        "entropy_coeff":[0.01],
        "discount":[0.9],
        #"lambda_": [1.0, 0.95,0.6],
        "value_ceoff": [0.01],
        "policy": ["IC3"],
        "n_agents": [4],
        "obj_density": [0.2],
        "env_size": [5],
        "recurrent": [False],
        "base_policy_type": ["mlp"],
        "comm_passes": [1],
        "comm_mask_zero":[False],
        "hid_size":[120],
        "share_weights":[False],
        "comm_action_one": [True, False] #Usually false for ic3; if true, then its commNet
       # "hard_attention":[False] #Usually this parameter is true. If false communication is always present (commNet)
    }


    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
        args.extend(["--iterations", n_iterations])

        name = "IC3" + "_disc_" + str(param["discount"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["batch_sizes"]) \
            + "_commpasses_" + str(param["comm_passes"]) \
            + "_commzero_" + str(param["comm_mask_zero"]) \
            + "_commNet_" + str(param["comm_action_one"]) \
            + "_recurrent_" + str(param["recurrent"]) \
            + "_envsize_" + str(param["env_size"]) \
            + "_nagents_"+ str(param["n_agents"]) \
            + "_objdensity_"+ str(param["obj_density"]) \
            + "_seed_"+ str(param["seed"])

        args.extend(["--env_name","cooperative_navigation-v1"])
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
        if param["comm_action_one"]:
            args.extend(["--comm_action_one"])

            

        args.extend(["--render_rate", str(int(1e2))])
        args.extend(["--checkpoint_frequency", str(int(5e2))])
        args.extend(["--benchmark_frequency", str(int(3000))])
        args.extend(["--benchmark_num_episodes", str(int(10))])
        args.extend(["--benchmark_render_length", str(int(20))])
        args.extend(["--obj_density", str(param["obj_density"])])

        main.main(args)

def ic3_1F2():
    n_iterations = str(10000)
  #  experiment_group_name = "IC3"
    experiment_group_name = "Ic3_1F"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"

 
    parmeter_grid1 = {
        "seed": [1],
        "batch_sizes": [500],
        "entropy_coeff":[0.01],
        "discount":[0.9],
        #"lambda_": [1.0, 0.95,0.6],
        "value_ceoff": [0.01],
        "policy": ["IC3"],
        "n_agents": [4],
        "obj_density": [0.2],
        "env_size": [5],
        "recurrent": [False],
        "base_policy_type": ["mlp"],
        "comm_passes": [1],
        "comm_mask_zero":[True],
        "hid_size":[120],
        "share_weights":[False],
        "comm_action_one": [False] #Usually false for ic3; if true, then its commNet
       # "hard_attention":[False] #Usually this parameter is true. If false communication is always present (commNet)
    }


    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
        args.extend(["--iterations", n_iterations])

        name = "IC3" + "_disc_" + str(param["discount"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["batch_sizes"]) \
            + "_commpasses_" + str(param["comm_passes"]) \
            + "_commzero_" + str(param["comm_mask_zero"]) \
            + "_commNet_" + str(param["comm_action_one"]) \
            + "_recurrent_" + str(param["recurrent"]) \
            + "_envsize_" + str(param["env_size"]) \
            + "_nagents_"+ str(param["n_agents"]) \
            + "_objdensity_"+ str(param["obj_density"]) \
            + "_seed_"+ str(param["seed"])

        args.extend(["--env_name","cooperative_navigation-v1"])
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
        if param["comm_action_one"]:
            args.extend(["--comm_action_one"])

            

        args.extend(["--render_rate", str(int(1e2))])
        args.extend(["--checkpoint_frequency", str(int(5e2))])
        args.extend(["--benchmark_frequency", str(int(3000))])
        args.extend(["--benchmark_num_episodes", str(int(10))])
        args.extend(["--benchmark_render_length", str(int(20))])
        args.extend(["--obj_density", str(param["obj_density"])])

        main.main(args)

def ic3_1F5():
    n_iterations = str(10000)
  #  experiment_group_name = "IC3"
    experiment_group_name = "Ic3_1F"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"

 
    parmeter_grid1 = {
        "seed": [1],
        "batch_sizes": [500],
        "entropy_coeff":[0.01],
        "discount":[0.9],
        #"lambda_": [1.0, 0.95,0.6],
        "value_ceoff": [0.01],
        "policy": ["IC3"],
        "n_agents": [4],
        "obj_density": [0.2],
        "env_size": [5],
        "recurrent": [False],
        "base_policy_type": ["mlp"],
        "comm_passes": [1],
        "comm_mask_zero":[True],
        "hid_size":[120],
        "share_weights":[False],
        "comm_action_one": [False] #Usually false for ic3; if true, then its commNet
       # "hard_attention":[False] #Usually this parameter is true. If false communication is always present (commNet)
    }


    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
        args.extend(["--iterations", n_iterations])

        name = "IC3_CN0" + "_disc_" + str(param["discount"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["batch_sizes"]) \
            + "_commpasses_" + str(param["comm_passes"]) \
            + "_commzero_" + str(param["comm_mask_zero"]) \
            + "_commNet_" + str(param["comm_action_one"]) \
            + "_recurrent_" + str(param["recurrent"]) \
            + "_envsize_" + str(param["env_size"]) \
            + "_nagents_"+ str(param["n_agents"]) \
            + "_objdensity_"+ str(param["obj_density"]) \
            + "_seed_"+ str(param["seed"])

        args.extend(["--env_name","cooperative_navigation-v0"])
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
        if param["comm_action_one"]:
            args.extend(["--comm_action_one"])

            

        args.extend(["--render_rate", str(int(1e2))])
        args.extend(["--checkpoint_frequency", str(int(5e2))])
        args.extend(["--benchmark_frequency", str(int(3000))])
        args.extend(["--benchmark_num_episodes", str(int(10))])
        args.extend(["--benchmark_render_length", str(int(20))])
        args.extend(["--obj_density", str(param["obj_density"])])

        main.main(args)


def ic3_1F3(): #Samity check for env
   # n_iterations = str(2500)
  #  experiment_group_name = "IC3"
    experiment_group_name = "Ic3_1F"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"

 
    n_iterations = str(10000) #str(3000)
    # experiment_group_name = "ppo_1D"
    # work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    # plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"

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
        "env_size": [5],
        "lr":[0.001]
    }

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        name = "PPO_CN1" + "_disc_" + str(param["discount"]) \
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

        args.extend(["--env_name","cooperative_navigation-v1"])
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

def ic3_1F4(): #Samity check for env
   # n_iterations = str(2500)
  #  experiment_group_name = "IC3"
    experiment_group_name = "Ic3_1F"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"

 
    n_iterations = str(10000) #str(3000)
    # experiment_group_name = "ppo_1D"
    # work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    # plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"

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
        "env_size": [5],
        "lr":[0.001]
    }

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        name = "PPO_CN0" + "_disc_" + str(param["discount"]) \
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

        args.extend(["--env_name","cooperative_navigation-v0"])
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


################################################
def ic3_1G1(): #How valuable is communication: Same as E, but with cooperative navigation-v1: agents cant see each other
    n_iterations = str(5000)
  #  experiment_group_name = "IC3"
    experiment_group_name = "Ic3_1G"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"

 
    parmeter_grid1 = {
        "seed": [1],
        "batch_sizes": [500],
        "entropy_coeff":[0.01],
        "discount":[0.9],
        #"lambda_": [1.0, 0.95,0.6],
        "value_ceoff": [0.01],
        "policy": ["IC3"],
        "n_agents": [4],
        "obj_density": [0.0],
        "env_size": [5],
        "recurrent": [False],
        "base_policy_type": ["mlp"],
        "comm_passes": [1],
        "comm_mask_zero":[False],
        "hid_size":[120],
        "share_weights":[False],
        "comm_action_one": [True, False] #Usually false for ic3; if true, then its commNet
       # "hard_attention":[False] #Usually this parameter is true. If false communication is always present (commNet)
    }


    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
        args.extend(["--iterations", n_iterations])

        name = "IC3" + "_disc_" + str(param["discount"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["batch_sizes"]) \
            + "_commpasses_" + str(param["comm_passes"]) \
            + "_commzero_" + str(param["comm_mask_zero"]) \
            + "_commNet_" + str(param["comm_action_one"]) \
            + "_recurrent_" + str(param["recurrent"]) \
            + "_envsize_" + str(param["env_size"]) \
            + "_nagents_"+ str(param["n_agents"]) \
            + "_objdensity_"+ str(param["obj_density"]) \
            + "_seed_"+ str(param["seed"])

        args.extend(["--env_name","cooperative_navigation-v1"])
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
        if param["comm_action_one"]:
            args.extend(["--comm_action_one"])

            

        args.extend(["--render_rate", str(int(1e2))])
        args.extend(["--checkpoint_frequency", str(int(5e2))])
        args.extend(["--benchmark_frequency", str(int(1500))])
        args.extend(["--benchmark_num_episodes", str(int(200))])
        args.extend(["--benchmark_render_length", str(int(25))])
        args.extend(["--obj_density", str(param["obj_density"])])

        main.main(args)

def ic3_1G2():
    n_iterations = str(5000)
  #  experiment_group_name = "IC3"
    experiment_group_name = "Ic3_1G"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"

 
    parmeter_grid1 = {
        "seed": [1],
        "batch_sizes": [500],
        "entropy_coeff":[0.01],
        "discount":[0.9],
        #"lambda_": [1.0, 0.95,0.6],
        "value_ceoff": [0.01],
        "policy": ["IC3"],
        "n_agents": [4],
        "obj_density": [0.0],
        "env_size": [5],
        "recurrent": [False],
        "base_policy_type": ["mlp"],
        "comm_passes": [1],
        "comm_mask_zero":[True],
        "hid_size":[120],
        "share_weights":[False],
        "comm_action_one": [False] #Usually false for ic3; if true, then its commNet
       # "hard_attention":[False] #Usually this parameter is true. If false communication is always present (commNet)
    }


    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
        args.extend(["--iterations", n_iterations])

        name = "IC3" + "_disc_" + str(param["discount"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["batch_sizes"]) \
            + "_commpasses_" + str(param["comm_passes"]) \
            + "_commzero_" + str(param["comm_mask_zero"]) \
            + "_commNet_" + str(param["comm_action_one"]) \
            + "_recurrent_" + str(param["recurrent"]) \
            + "_envsize_" + str(param["env_size"]) \
            + "_nagents_"+ str(param["n_agents"]) \
            + "_objdensity_"+ str(param["obj_density"]) \
            + "_seed_"+ str(param["seed"])

        args.extend(["--env_name","cooperative_navigation-v1"])
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
        if param["comm_action_one"]:
            args.extend(["--comm_action_one"])

            

        args.extend(["--render_rate", str(int(1e2))])
        args.extend(["--checkpoint_frequency", str(int(5e2))])
        args.extend(["--benchmark_frequency", str(int(3000))])
        args.extend(["--benchmark_num_episodes", str(int(10))])
        args.extend(["--benchmark_render_length", str(int(20))])
        args.extend(["--obj_density", str(param["obj_density"])])

        main.main(args)


def ic3_1G3(): #Samity check for env
   # n_iterations = str(2500)
  #  experiment_group_name = "IC3"
    experiment_group_name = "Ic3_1G"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"

 
    n_iterations = str(5000) #str(3000)
    # experiment_group_name = "ppo_1D"
    # work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    # plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"

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
        "obj_density": [0.0],
        "env_size": [5],
        "lr":[0.001]
    }

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        name = "PPO_CN1" + "_disc_" + str(param["discount"]) \
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

        args.extend(["--env_name","cooperative_navigation-v1"])
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

def ic3_1G4(): #Samity check for env
   # n_iterations = str(2500)
  #  experiment_group_name = "IC3"
    experiment_group_name = "Ic3_1G"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"

 
    n_iterations = str(5000) #str(3000)
    # experiment_group_name = "ppo_1D"
    # work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    # plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"

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
        "obj_density": [0.0],
        "env_size": [5],
        "lr":[0.001]
    }

    grid1 = ParameterGrid(parmeter_grid1)

    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]

        name = "PPO_CN0" + "_disc_" + str(param["discount"]) \
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

        args.extend(["--env_name","cooperative_navigation-v0"])
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


def ic3_1H1():
    n_iterations = str(2500)
    experiment_group_name = "Ic3_1H"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"

    parmeter_grid1 = {
        "seed": [1],
        "batch_sizes": [500],
        "entropy_coeff":[0.01],
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
        "comm_mask_zero":[False], #If true, all communication is zero
        "hid_size":[120],
        "share_weights":[False],
        "comm_action_one": [False] #Usually false for ic3; if true, then its commNet
       # "hard_attention":[False] #Usually this parameter is true. If false communication is always present (commNet)
    }

    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
        args.extend(["--iterations", n_iterations])

        name = "IC3" + "_disc_" + str(param["discount"]) \
            + "_entropy_" + str(param["entropy_coeff"]) \
            + "_minibatch_" + str(param["batch_sizes"]) \
            + "_commpasses_" + str(param["comm_passes"]) \
            + "_commzero_" + str(param["comm_mask_zero"]) \
            + "_commNet_" + str(param["comm_action_one"]) \
            + "_recurrent_" + str(param["recurrent"]) \
            + "_envsize_" + str(param["env_size"]) \
            + "_nagents_"+ str(param["n_agents"]) \
            + "_objdensity_"+ str(param["obj_density"]) \
            + "_commentropy_"+ "0.0005" \
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
        if param["comm_action_one"]:
            args.extend(["--comm_action_one"])

            

        args.extend(["--render_rate", str(int(1e2))])
        args.extend(["--checkpoint_frequency", str(int(5e2))])
        args.extend(["--benchmark_frequency", str(int(3000))])
        args.extend(["--benchmark_num_episodes", str(int(10))])
        args.extend(["--benchmark_render_length", str(int(20))])
        args.extend(["--obj_density", str(param["obj_density"])])

        main.main(args)